"""Balatron training dashboard.

Single-file, stdlib-only. Reads logs/ and serves an auto-refreshing page.

    python dashboard.py            # serves http://localhost:8777
    python dashboard.py --port N

Every section answers a specific question:
  Outcomes      — is he getting deeper into runs? (mean ante, deep-run rates)
  Learning      — is the policy moving healthily? (R, entropy, KL/CF vs
                  target_kl, BC loss vs the flat-line decision threshold)
  Diagnosis     — where does he die, and which jokers correlate with wins?
  Ops           — heartbeat, steps/min, trainer restarts in the last 24h
"""

import argparse
import glob
import html
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
CHUNK = 200          # games per trend bucket
ERA_START = "2026-06-08"  # start of the current training era for trend charts
TARGET_KL = 0.03     # early-stop threshold in ppo.py
BC_FLAT_LINE = 3.7   # pre-committed decision: BC loss flat here after 50 clean updates -> raise bc_coef

# Regime boundaries: PPO updates where a metric's LEVEL shifted for a reason
# that is NOT gameplay (a measurement/methodology change). Hard-won lesson —
# this project has caught four display metrics "lying" (KL/CF read 0, R was a
# cumulative mean, R under-counted ~6x under N=3, entropy carried a ln(19)
# artifact). A trend read ACROSS one of these is meaningless, so the per-update
# charts draw a dashed vertical line here and the "Metric trust" panel explains
# why. Append new boundaries whenever a metric's accounting changes.
#   (update_no, short chart label, long caveat)
REGIMES = [
    (333, "ent artifact",
     "Entropy gating fix removed a ~ln(19) phantom term from unused heads — "
     "pre-U333 entropy (read ~2.6) was an artifact, not real exploration."),
    (379, "mask + prior-KL",
     "Binary legality mask + annealing prior-KL deployed — entropy unfloored "
     "(~0.24 -> ~0.5). Entropy is not comparable across this line."),
    (435, "R x6 fix",
     "R stepped ~6x: episode-tracker stats-isolation fix (N=3 had been "
     "under-counting per-episode reward). Gameplay was unchanged (ante/WR flat) "
     "— so compare R only WITHIN a regime, never across this line."),
]

UPDATE_RE = re.compile(
    r"Update\s+(\d+) \| Step\s+([\d,]+) \| FPS\s+(\d+) \| Ep\s+(\d+) \| "
    r"R\s+([-\d.]+) \| Ante\s+([\d.]+) \| WR\s+([\d.]+)% \| PL\s+([-\d.]+) \| "
    r"VL\s+([\d.]+) \| Ent\s+([\d.]+) \| KL\s+([\d.]+) \| CF\s+([\d.]+) \| "
    r"BC\s+([\d.]+)@([\d.]+)\((\d+)%\)"
    # Prior-KL field (06-14) and EV (06-16) are optional so older logs parse.
    r"(?: \| Pr\s+([\d.]+)@([\d.]+))? \| LR\s+([\d.e+-]+)"
    r"(?: \| EV\s+([-\d.]+))?"
)
SUPERVISOR_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\] trainer started")

# previous heartbeat sample for steps/min between page loads
_last_hb = {"t": None, "step": None}


def read_lifetime():
    try:
        with open(os.path.join(LOGS, "lifetime_stats.json")) as f:
            return json.load(f)
    except Exception:
        return {"wins": 0, "episodes": 0, "highest_ante": 0}


def read_heartbeat():
    try:
        with open(os.path.join(LOGS, "heartbeat")) as f:
            ts, step = f.read().split()
        return float(ts), int(step)
    except Exception:
        return None, None


def read_history():
    rows = []
    try:
        with open(os.path.join(LOGS, "game_history.jsonl")) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def read_updates():
    """Parse all PPO update lines from recent trainer logs.

    Restart-resumes can replay an update number; dedupe keeps the latest
    occurrence so the curve reflects what actually trained.
    """
    # Read the persistent, never-pruned metrics history FIRST (so the curve
    # survives supervisor log-pruning and the restart fragmentation that
    # otherwise left only ~2 updates parseable), then overlay ALL current
    # trainer logs (latest occurrence of each update number wins).
    logs = sorted(glob.glob(os.path.join(LOGS, "trainer_*.log")), key=os.path.getmtime)
    sources = []
    hist = os.path.join(LOGS, "metrics_history.log")
    if os.path.exists(hist):
        sources.append(hist)
    sources.extend(logs)
    by_no = {}
    for path in sources:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = UPDATE_RE.search(line)
                    if m:
                        g = m.groups()
                        by_no[int(g[0])] = {
                            "no": int(g[0]), "step": int(g[1].replace(",", "")),
                            "r": float(g[4]), "ante": float(g[5]),
                            "ent": float(g[9]), "kl": float(g[10]), "cf": float(g[11]),
                            "bc": float(g[12]), "bc_coef": float(g[13]),
                            "bc_frac": int(g[14]),
                            # Pr fields are None for pre-06-14 logs (optional group)
                            "pr_kl": float(g[15]) if g[15] is not None else None,
                            "pr_coef": float(g[16]) if g[16] is not None else None,
                            "lr": g[17],
                            "ev": float(g[18]) if g[18] is not None else None,
                            "vl": float(g[8]),
                        }
        except OSError:
            pass
    return [by_no[k] for k in sorted(by_no)]


def trend(values, window=30):
    """Significance-aware trend over the last `window` points.

    Returns (mean, std, drift, n, verdict). 'flat' when the total drift across
    the window is smaller than the run-to-run noise (std) — so a single high or
    low point can NEVER read as a real trend. This is the core anti-"tiny
    snippet" guard: it answers "is this actually moving?" instead of leaving
    you to eyeball a noisy line."""
    v = [x for x in values[-window:] if x is not None]
    if len(v) < 6:
        return None
    n = len(v)
    mean = sum(v) / n
    std = (sum((x - mean) ** 2 for x in v) / n) ** 0.5
    xm = (n - 1) / 2.0
    den = sum((i - xm) ** 2 for i in range(n)) or 1.0
    slope = sum((i - xm) * (v[i] - mean) for i in range(n)) / den
    drift = slope * (n - 1)                       # total change across window
    if std < 1e-9 or abs(drift) < 0.6 * std:      # drift buried in the noise
        verdict = "flat"
    else:
        verdict = "rising" if drift > 0 else "falling"
    return mean, std, drift, n, verdict


def read_status():
    """(state, detail) from logs/supervisor_status.txt, or (None, '')."""
    try:
        with open(os.path.join(LOGS, "supervisor_status.txt"),
                  encoding="utf-8") as f:
            parts = [p.strip() for p in f.read().strip().split("|")]
        if len(parts) >= 2:
            return parts[1], (parts[2] if len(parts) > 2 else "")
    except Exception:
        pass
    return None, ""


def read_restarts_24h():
    cutoff = datetime.now() - timedelta(hours=24)
    count = 0
    try:
        with open(os.path.join(LOGS, "supervisor.log"), encoding="utf-8", errors="replace") as f:
            for line in f:
                m = SUPERVISOR_TS_RE.match(line)
                if m and datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S") >= cutoff:
                    count += 1
    except OSError:
        return None
    return count


def chunk_trend(rows):
    recent = [r for r in rows if r.get("ts", "") >= ERA_START]
    chunks = []
    for i in range(0, len(recent), CHUNK):
        g = recent[i:i + CHUNK]
        if len(g) < 50:
            break
        chunks.append({
            "mean_ante": sum(x["ante"] for x in g) / len(g),
            "a6": 100.0 * sum(1 for x in g if x["ante"] >= 6) / len(g),
            "a8": 100.0 * sum(1 for x in g if x["ante"] >= 8) / len(g),
        })
    return chunks


def wins_by_day(rows):
    days = {}
    for r in rows:
        d = r.get("ts", "")[:10]
        if not d:
            continue
        s = days.setdefault(d, {"games": 0, "wins": 0, "ante": 0.0})
        s["games"] += 1
        s["wins"] += 1 if r.get("won") else 0
        s["ante"] += r["ante"]
    return sorted(days.items())


def last_win_info(rows):
    """(hours since last win, games played since) or None."""
    last_idx = None
    for i, r in enumerate(rows):
        if r.get("won"):
            last_idx = i
    if last_idx is None:
        return None
    try:
        ts = datetime.strptime(rows[last_idx]["ts"], "%Y-%m-%dT%H:%M:%S")
    except (KeyError, ValueError):
        return None
    hours = (datetime.now() - ts).total_seconds() / 3600.0
    return hours, len(rows) - 1 - last_idx


def joker_win_table(rows, top_n=8, min_held=20):
    """Jokers most common in winning runs, with lift vs overall hold rate.

    Lift > 1 means the joker shows up in wins more often than in games
    generally — a (correlational) hint at what builds work.
    """
    wins = [r for r in rows if r.get("won")]
    if len(wins) < 5:
        return [], len(wins)
    held_all = Counter(j for r in rows for j in set(r.get("jokers") or []))
    held_win = Counter(j for r in wins for j in set(r.get("jokers") or []))
    out = []
    for joker, wc in held_win.most_common():
        tc = held_all[joker]
        if tc < min_held:
            continue
        lift = (wc / len(wins)) / (tc / len(rows))
        out.append((joker, wc, tc, lift))
        if len(out) >= top_n:
            break
    return out, len(wins)


def rolling(vals, window=5, resets=()):
    """Rolling mean that RESTARTS at each reset index — so the average never
    smears across a regime boundary (a window spanning the U435 R-step would
    turn an honest discontinuity into a fake gradual climb)."""
    resets = set(resets)
    out = []
    seg_start = 0
    for i in range(len(vals)):
        if i in resets:
            seg_start = i
        w = vals[max(seg_start, i - window + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def regime_vlines(updates):
    """Map each regime boundary (by PPO update number) to an x-index in the
    per-update charts. Only returns boundaries that fall strictly inside the
    visible range (there must be data on both sides to mislead)."""
    nums = [u["no"] for u in updates]
    out = []
    for uno, short, _ in REGIMES:
        idx = next((i for i, n in enumerate(nums) if n >= uno), None)
        if idx is not None and 0 < idx < len(nums):
            out.append((idx, short))
    return out


def svg_line(series, w=860, h=200, pad=34, fmt="{:.2f}", hlines=(), vlines=()):
    """series: list of (name, color, [values]). hlines: list of (value, label).
    vlines: list of (x_index, label) — dashed vertical regime markers."""
    n = max((len(v) for _, _, v in series), default=0)
    if n < 2:
        return "<p class='dim'>not enough data yet</p>"
    vals = [x for _, _, v in series for x in v] + [hv for hv, _ in hlines]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    lo, hi = lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08

    def y_of(val):
        return h - pad + (pad - h + 18) * (val - lo) / (hi - lo)

    def pt(i, val):
        x = pad + (w - pad - 8) * i / (n - 1)
        return f"{x:.1f},{y_of(val):.1f}"

    out = [f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px'>"]
    for frac in (0.0, 0.5, 1.0):
        val = lo + (hi - lo) * frac
        y = y_of(val)
        out.append(f"<line x1='{pad}' y1='{y:.0f}' x2='{w-8}' y2='{y:.0f}' stroke='#333' stroke-width='1'/>")
        out.append(f"<text x='2' y='{y+4:.0f}' fill='#888' font-size='11'>{fmt.format(val)}</text>")
    for hval, hlabel in hlines:
        y = y_of(hval)
        out.append(f"<line x1='{pad}' y1='{y:.1f}' x2='{w-8}' y2='{y:.1f}' stroke='#777' stroke-width='1' stroke-dasharray='5,4'/>")
        out.append(f"<text x='{w-10}' y='{y-4:.0f}' fill='#999' font-size='11' text-anchor='end'>{html.escape(hlabel)}</text>")
    for vi, vlabel in vlines:
        if not (0 <= vi <= n - 1):
            continue
        x = pad + (w - pad - 8) * vi / (n - 1)
        out.append(f"<line x1='{x:.1f}' y1='18' x2='{x:.1f}' y2='{h-pad}' stroke='#8a6a3a' stroke-width='1' stroke-dasharray='2,3'/>")
        out.append(f"<text x='{x+3:.1f}' y='28' fill='#b98a4a' font-size='9'>{html.escape(vlabel)}</text>")
    for name, color, v in series:
        pts = " ".join(pt(i, val) for i, val in enumerate(v))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/>")
    legend_x = pad
    for name, color, _ in series:
        out.append(f"<rect x='{legend_x}' y='4' width='10' height='10' fill='{color}'/>")
        out.append(f"<text x='{legend_x+14}' y='13' fill='#ccc' font-size='12'>{html.escape(name)}</text>")
        legend_x += 14 + 8 * len(name) + 24
    out.append("</svg>")
    return "".join(out)


def svg_bars(pairs, w=430, h=190, pad=30, color="#e0a93e"):
    """pairs: list of (label, value)."""
    if not pairs:
        return "<p class='dim'>not enough data yet</p>"
    hi = max(v for _, v in pairs) or 1
    bw = (w - pad - 8) / len(pairs)
    out = [f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px'>"]
    for i, (label, v) in enumerate(pairs):
        bh = (h - pad - 22) * v / hi
        x = pad + i * bw
        y = h - pad - bh
        out.append(f"<rect x='{x+2:.1f}' y='{y:.1f}' width='{bw-4:.1f}' height='{bh:.1f}' fill='{color}' rx='2'/>")
        out.append(f"<text x='{x+bw/2:.1f}' y='{y-4:.0f}' fill='#ccc' font-size='11' text-anchor='middle'>{v}</text>")
        out.append(f"<text x='{x+bw/2:.1f}' y='{h-pad+14:.0f}' fill='#888' font-size='11' text-anchor='middle'>{html.escape(str(label))}</text>")
    out.append("</svg>")
    return "".join(out)


def render():
    life = read_lifetime()
    rows = read_history()
    chunks = chunk_trend(rows)
    updates = read_updates()
    hb_t, hb_step = read_heartbeat()
    restarts = read_restarts_24h()
    lw = last_win_info(rows)

    rate = None
    now = time.time()
    if hb_t is not None:
        if _last_hb["t"] and hb_step is not None and hb_step >= _last_hb["step"] and now - _last_hb["t"] > 5:
            rate = (hb_step - _last_hb["step"]) / ((now - _last_hb["t"]) / 60.0)
        _last_hb["t"], _last_hb["step"] = now, hb_step
    hb_age = (now - hb_t) if hb_t else None

    win_rate = 100.0 * life["wins"] / max(1, life["episodes"])

    # --- live training context (latest update + supervisor + velocity) ---
    state, state_detail = read_status()
    last = updates[-1] if updates else {}
    pr_coef = last.get("pr_coef")
    ev_now = last.get("ev")
    lr_now = last.get("lr")
    STEPS_PER_UPDATE = 2048 * 3                  # rollout_steps * num_envs
    upd_per_hr = (rate * 60.0 / STEPS_PER_UPDATE) if rate else None
    teacher_pct = f"{100*(1-pr_coef/0.5):.0f}%" if pr_coef is not None else "—"

    # significance-aware trends (the anti-"tiny snippet" verdicts)
    ante_tr = trend([u["ante"] for u in updates], 30)
    kl_tr = trend([u["kl"] for u in updates], 10)

    recent_best = max((r["ante"] for r in rows[-500:]), default=0) if rows else 0
    cards = [
        ("Wins (lifetime)", f"{life['wins']}"),
        ("Episodes", f"{life['episodes']:,}"),
        ("Best ante (all-time)", f"{life['highest_ante']}"),
        ("Best ante (last 500)", f"{recent_best}" if recent_best else "—"),
        ("Update", f"{last.get('no','?')}"),
        ("Teacher released", teacher_pct),
        ("Value fit EV", f"{ev_now:.2f}" if ev_now is not None else "—"),
        ("LR", lr_now or "—"),
        ("Updates/hr", f"{upd_per_hr:.1f}" if upd_per_hr else "(refresh)"),
        ("Steps/min", f"{rate:.0f}" if rate else "(refresh)"),
        ("Status", state or "—"),
        ("Restarts 24h", f"{restarts}" if restarts is not None else "?"),
        ("Heartbeat", f"{hb_age:.0f}s" if hb_age is not None else "MISSING"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='num'>{v}</div><div class='lbl'>{k}</div></div>"
        for k, v in cards
    )
    hb_warn = ""
    if hb_age is not None and hb_age > 300:
        hb_warn = "<p class='warn'>heartbeat stale &gt;5min — trainer may be down</p>"

    # --- Current read: honest, noise-aware one-glance summary ---
    def _verdict(label, tr, unit=""):
        if tr is None:
            return f"<li><b>{label}:</b> <span class='dim'>not enough data yet</span></li>"
        mean, std, drift, n, verd = tr
        col = {"rising": "#7dd87d", "falling": "#e05a5a", "flat": "#c9a227"}[verd]
        return (f"<li><b>{label}:</b> {mean:.2f}{unit} "
                f"<span class='dim'>(±{std:.2f} run-to-run, n={n})</span> — "
                f"<b style='color:{col}'>{verd.upper()}</b> "
                f"<span class='dim'>over last {n} updates (drift {drift:+.2f}{unit})</span></li>")
    read_lines = [_verdict("Ante (ground truth)", ante_tr)]
    # Per-RUN distribution (what you SEE watching live). The ante mean above is
    # bust-skewed: a run dying at ante 1 (often a disconnect/recycle artifact)
    # cancels a run reaching ante 5, so the mean reads ~1 ante below the typical
    # run. Median + reach-rates track what you actually watch far better.
    rant = sorted(r["ante"] for r in rows[-100:] if r.get("ante") is not None)
    if rant:
        nr = len(rant)
        med = rant[nr // 2] if nr % 2 else (rant[nr // 2 - 1] + rant[nr // 2]) / 2
        pct5 = 100.0 * sum(1 for a in rant if a >= 5) / nr
        pct4 = 100.0 * sum(1 for a in rant if a >= 4) / nr
        bust = 100.0 * sum(1 for a in rant if a <= 2) / nr
        read_lines.append(
            f"<li><b>Typical run (last {nr}):</b> median ante <b>{med:g}</b>, "
            f"<b>{pct5:.0f}%</b> reach 5+, {pct4:.0f}% reach 4+, {bust:.0f}% bust ≤2 "
            f"<span class='dim'>(median tracks what you watch; the mean above is "
            f"bust-skewed downward)</span></li>")
    if kl_tr is not None:
        klm = kl_tr[0]
        mv = "moving" if klm > 0.004 else "near-frozen (not learning)"
        read_lines.append(
            f"<li><b>Policy movement:</b> KL {klm:.4f} (last 10) — {mv} "
            f"<span class='dim'>(healthy 0.01–0.05; &lt;0.003 = stuck)</span></li>")
    if pr_coef is not None:
        read_lines.append(
            f"<li><b>Teacher (prior-KL):</b> coef {pr_coef:.2f}, {teacher_pct} released "
            f"<span class='dim'>(→0 ≈ U629 = full policy autonomy; can't truly "
            f"surpass the heuristic until then)</span></li>")
    if ev_now is not None:
        evq = "healthy" if ev_now > 0.5 else ("weak" if ev_now > 0.2 else "POOR")
        read_lines.append(
            f"<li><b>Value fit (EV):</b> {ev_now:.2f} — {evq} "
            f"<span class='dim'>(near 1 = good advantages; &lt;0.2 = noisy gradient)</span></li>")
    vel = f"{upd_per_hr:.1f} updates/hr" if upd_per_hr else "computing"
    st = (state or "?") + (f" — {state_detail}" if state_detail else "")
    read_lines.append(
        f"<li><b>Throughput / health:</b> {vel}; supervisor says "
        f"<b>{html.escape(st)}</b></li>")
    current_read = "".join(read_lines)

    # --- Outcomes ---
    ante_chart = svg_line([("mean ante / 200 games", "#e0a93e", [c["mean_ante"] for c in chunks])])
    deep_chart = svg_line([
        ("ante ≥ 6 %", "#5ab0f0", [c["a6"] for c in chunks]),
        ("ante ≥ 8 %", "#e05a5a", [c["a8"] for c in chunks]),
    ], fmt="{:.0f}%")

    # --- Learning health (per PPO update) ---
    small = dict(w=430, h=180, pad=34)
    vl = regime_vlines(updates)             # dashed "methodology changed here" markers
    reset_idx = [i for i, _ in vl]          # rolling mean restarts at each boundary
    r_chart = svg_line([("R rolling-5 (shaped — accounting-sensitive)", "#7dd87d",
                         rolling([u["r"] for u in updates], resets=reset_idx))],
                       vlines=vl, **small)
    ent_chart = svg_line([("entropy", "#c08ae0", [u["ent"] for u in updates])],
                         vlines=vl, **small)
    kl_chart = svg_line([
        ("KL", "#5ab0f0", [u["kl"] for u in updates]),
        ("clip frac", "#e0a93e", [u["cf"] for u in updates]),
    ], fmt="{:.3f}", hlines=[(TARGET_KL, f"target_kl {TARGET_KL}")], vlines=vl, **small)
    bc_chart = svg_line(
        [("BC loss", "#e05a5a", [u["bc"] for u in updates])],
        hlines=[(BC_FLAT_LINE, f"flat-line watch {BC_FLAT_LINE}")], vlines=vl, **small)
    # Per-update ante: raw (faint) vs rolling-10 (bold) so the eye follows the
    # SIGNAL not the per-update noise. Ante is gameplay -> comparable across regimes.
    ante_u = [u["ante"] for u in updates]
    ante_u_chart = svg_line([
        ("ante / update (noisy)", "#5a6a7a", ante_u),
        ("rolling-10 (the signal)", "#e0a93e", rolling(ante_u, window=10)),
    ], vlines=vl, **small)
    ev_series = [u.get("ev") for u in updates if u.get("ev") is not None]
    ev_chart = (svg_line([("explained var", "#7dd8c0", ev_series)],
                         hlines=[(0.5, "weak below")], **small)
                if len(ev_series) >= 2 else "<p class='dim'>EV logged from U547 on</p>")

    # Metric-trust panel — the "verify the path, don't read across a regime" lesson.
    regime_caveats = "".join(
        f"<li><b>U{uno}</b> — {html.escape(long)}</li>" for uno, _, long in REGIMES
    )

    # --- Diagnosis ---
    recent_antes = Counter(r["ante"] for r in rows[-1000:])
    max_ante = max(recent_antes) if recent_antes else 8
    death_hist = svg_bars([(a, recent_antes.get(a, 0)) for a in range(1, max_ante + 1)])
    jokers, n_wins = joker_win_table(rows)
    joker_rows = "".join(
        f"<tr><td>{html.escape(j)}</td><td>{wc}/{n_wins}</td><td>{tc}</td><td>{lift:.1f}×</td></tr>"
        for j, wc, tc, lift in jokers
    ) or "<tr><td colspan='4' class='dim'>not enough wins in window yet</td></tr>"

    def _pr_cell(u):
        if u.get("pr_kl") is None:
            return "<td class='dim'>—</td>"
        return f"<td>{u['pr_kl']:.3f}@{u['pr_coef']:.2f}</td>"

    def _ev_cell(u):
        return f"<td>{u['ev']:.2f}</td>" if u.get("ev") is not None else "<td class='dim'>—</td>"

    upd_rows = "".join(
        f"<tr><td>{u['no']}</td><td>{u['step']:,}</td><td>{u['r']:.2f}</td><td>{u['ante']:.1f}</td>"
        f"<td>{u['ent']:.3f}</td><td>{u['kl']:.4f}</td><td>{u['cf']:.3f}</td>"
        f"<td>{u['bc']:.2f}@{u['bc_coef']:.2f} ({u['bc_frac']}%)</td>{_pr_cell(u)}<td>{u['lr']}</td>{_ev_cell(u)}</tr>"
        for u in reversed(updates[-10:])
    ) or "<tr><td colspan='11' class='dim'>no update lines parsed yet</td></tr>"

    day_rows = "".join(
        f"<tr><td>{d}</td><td>{s['games']}</td><td>{s['wins']}</td>"
        f"<td>{100.0*s['wins']/s['games']:.2f}%</td><td>{s['ante']/s['games']:.2f}</td></tr>"
        for d, s in reversed(wins_by_day(rows))
    )

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='30'>
<title>Balatron</title>
<style>
 body {{ background:#16161c; color:#ddd; font:14px/1.5 'Segoe UI',sans-serif; margin:24px; max-width:920px; }}
 h1 {{ font-size:20px; }} h2 {{ font-size:15px; color:#aaa; margin:26px 0 8px; }}
 h3 {{ font-size:13px; color:#999; margin:10px 0 2px; font-weight:500; }}
 .cards {{ display:flex; gap:12px; flex-wrap:wrap; }}
 .card {{ background:#1f1f29; border-radius:8px; padding:12px 18px; min-width:96px; text-align:center; }}
 .num {{ font-size:20px; font-weight:600; color:#fff; white-space:nowrap; }} .lbl {{ font-size:11px; color:#999; }}
 .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 24px; }}
 table {{ border-collapse:collapse; }} td,th {{ padding:3px 12px; border-bottom:1px solid #2a2a35; text-align:right; }}
 th {{ color:#999; font-weight:500; }} td:first-child, th:first-child {{ text-align:left; }}
 .dim {{ color:#777; }} .warn {{ color:#e05a5a; font-weight:600; }}
 .note {{ color:#888; font-size:12px; }}
 .read {{ background:#1b1f1b; border-left:3px solid #5a7a5a; border-radius:6px; padding:10px 14px 10px 26px; margin:6px 0; list-style:disc; }}
 .read li {{ margin:3px 0; }}
</style></head><body>
<h1>Balatron — live training dashboard</h1>
<p class='note'>auto-refreshes every 30s · reads logs/ directly · trend buckets = {CHUNK} games since {ERA_START}</p>
{hb_warn}
<div class='cards'>{card_html}</div>

<h2>Current read <span class='note'>— honest one-glance summary, computed &amp; noise-aware</span></h2>
<ul class='read'>{current_read}</ul>
<p class='note'>A single update is noisy — ante swings ±0.5 and R ±5 update-to-update. Each verdict compares the trend's <em>drift</em> to its <em>run-to-run noise</em>: <b>FLAT</b> means the move is inside the noise, so don't read a real change into one or two data points. Sample sizes (n=) are shown so you know how much data backs each number.</p>

<h2>Outcomes — is he getting deeper? <span class='note'>(ground truth — no accounting change moves these)</span></h2>
{ante_chart}
{deep_chart}

<h2>Learning health — per PPO update ({len(updates)} parsed)</h2>
<p class='note'>Dashed amber lines mark <b>regime boundaries</b> — points where a metric's level shifted for a measurement reason, not gameplay. Do not read a trend across one.</p>
<div class='grid'>
 <div><h3>Ante per update — raw (noisy) vs rolling-10 (the signal)</h3>{ante_u_chart}</div>
 <div><h3>Episode reward — shaped &amp; accounting-sensitive (not ground truth)</h3>{r_chart}</div>
 <div><h3>Policy movement — healthy: KL 0.01–0.05, CF 0.1–0.2</h3>{kl_chart}</div>
 <div><h3>Entropy — gradual decline ok, cliff = exploration collapse</h3>{ent_chart}</div>
 <div><h3>Value fit (explained var) — near 1 good, &lt;0.2 = noisy gradient</h3>{ev_chart}</div>
 <div><h3>BC imitation loss — flat at watch line after 50 updates → raise bc_coef</h3>{bc_chart}</div>
</div>

<h2>Metric trust — read <em>levels</em> with care</h2>
<p class='note'>This project has caught four display metrics "lying" (KL/CF read 0, R was a cumulative mean, R under-counted ~6× under N=3, entropy carried a ln(19) artifact). A metric's level can jump for reasons that aren't gameplay — the amber lines above mark them. <b>Ground truth is ante</b> (Outcomes), which accounting can't move.</p>
<ul class='note'>{regime_caveats}</ul>

<h2>Diagnosis</h2>
<div class='grid'>
 <div><h3>Final ante, last 1,000 games — where the wall is</h3>{death_hist}</div>
 <div><h3>Jokers held in winning runs (lift = vs overall hold rate)</h3>
  <table><tr><th>Joker</th><th>In wins</th><th>Held overall</th><th>Lift</th></tr>{joker_rows}</table>
  <p class='note'>correlation, not causation — small win sample</p></div>
</div>

<h2>Recent PPO updates</h2>
<table><tr><th>Update</th><th>Step</th><th>R</th><th>Ante</th><th>Ent</th><th>KL</th><th>CF</th><th>BC</th><th>Pr</th><th>LR</th><th>EV</th></tr>{upd_rows}</table>
<p class='note'>One row = one update. Each is noisy — use the Current read above for the trend, not any single row.</p>

<h2>By day (rolling 5,000-game window)</h2>
<table><tr><th>Date</th><th>Games</th><th>Wins</th><th>Win %</th><th>Mean ante</th></tr>{day_rows}</table>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    print(f"Balatron dashboard: http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
