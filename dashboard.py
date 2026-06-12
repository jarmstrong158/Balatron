"""Balatron training dashboard.

Single-file, stdlib-only. Reads logs/ and serves an auto-refreshing page.

    python dashboard.py            # serves http://localhost:8777
    python dashboard.py --port N
"""

import argparse
import glob
import html
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
CHUNK = 200          # games per trend bucket
ERA_START = "2026-06-08"  # start of the current training era for trend charts

UPDATE_RE = re.compile(
    r"Update\s+(\d+) \| Step\s+([\d,]+) \| FPS\s+(\d+) \| Ep\s+(\d+) \| "
    r"R\s+([-\d.]+) \| Ante\s+([\d.]+) \| WR\s+([\d.]+)% \| PL\s+([-\d.]+) \| "
    r"VL\s+([\d.]+) \| Ent\s+([\d.]+) \| KL\s+([\d.]+) \| CF\s+([\d.]+) \| "
    r"BC\s+([\d.]+)@([\d.]+)\((\d+)%\) \| LR\s+([\d.e+-]+)"
)

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


def read_updates(limit=12):
    logs = sorted(glob.glob(os.path.join(LOGS, "trainer_*.log")), key=os.path.getmtime)
    updates = []
    for path in logs[-3:]:  # trainer restarts split the stream across files
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = UPDATE_RE.search(line)
                    if m:
                        updates.append(m.groups())
        except OSError:
            pass
    return updates[-limit:]


def chunk_trend(rows):
    recent = [r for r in rows if r.get("ts", "") >= ERA_START]
    chunks = []
    for i in range(0, len(recent), CHUNK):
        g = recent[i:i + CHUNK]
        if len(g) < 50:
            break
        chunks.append({
            "label": g[0]["ts"][5:16],
            "n": len(g),
            "mean_ante": sum(x["ante"] for x in g) / len(g),
            "a6": 100.0 * sum(1 for x in g if x["ante"] >= 6) / len(g),
            "a8": 100.0 * sum(1 for x in g if x["ante"] >= 8) / len(g),
            "wins": sum(1 for x in g if x.get("won")),
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


def svg_line(series, w=860, h=200, pad=34, fmt="{:.2f}"):
    """series: list of (name, color, [values]). All same length, shared x."""
    n = max(len(v) for _, _, v in series)
    if n < 2:
        return "<p class='dim'>not enough data yet</p>"
    lo = min(min(v) for _, _, v in series)
    hi = max(max(v) for _, _, v in series)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    lo, hi = lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08

    def pt(i, val):
        x = pad + (w - pad - 8) * i / (n - 1)
        y = h - pad + (pad - h + 8) * (val - lo) / (hi - lo)
        return f"{x:.1f},{y:.1f}"

    out = [f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px'>"]
    for frac in (0.0, 0.5, 1.0):
        val = lo + (hi - lo) * frac
        y = h - pad + (pad - h + 8) * frac
        out.append(f"<line x1='{pad}' y1='{y:.0f}' x2='{w-8}' y2='{y:.0f}' stroke='#333' stroke-width='1'/>")
        out.append(f"<text x='2' y='{y+4:.0f}' fill='#888' font-size='11'>{fmt.format(val)}</text>")
    for name, color, vals in series:
        pts = " ".join(pt(i, v) for i, v in enumerate(vals))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/>")
    legend_x = pad
    for name, color, _ in series:
        out.append(f"<rect x='{legend_x}' y='4' width='10' height='10' fill='{color}'/>")
        out.append(f"<text x='{legend_x+14}' y='13' fill='#ccc' font-size='12'>{html.escape(name)}</text>")
        legend_x += 14 + 8 * len(name) + 24
    out.append("</svg>")
    return "".join(out)


def render():
    life = read_lifetime()
    rows = read_history()
    chunks = chunk_trend(rows)
    updates = read_updates()
    hb_t, hb_step = read_heartbeat()

    rate = None
    now = time.time()
    if hb_t is not None:
        if _last_hb["t"] and hb_step is not None and hb_step >= _last_hb["step"] and now - _last_hb["t"] > 5:
            rate = (hb_step - _last_hb["step"]) / ((now - _last_hb["t"]) / 60.0)
        _last_hb["t"], _last_hb["step"] = now, hb_step
    hb_age = (now - hb_t) if hb_t else None

    win_rate = 100.0 * life["wins"] / max(1, life["episodes"])
    cards = [
        ("Wins", f"{life['wins']}"),
        ("Episodes", f"{life['episodes']:,}"),
        ("Win rate", f"{win_rate:.2f}%"),
        ("Best ante", f"{life['highest_ante']}"),
        ("Global step", f"{hb_step:,}" if hb_step else "?"),
        ("Steps/min", f"{rate:.0f}" if rate else "(refresh)"),
        ("Heartbeat", f"{hb_age:.0f}s ago" if hb_age is not None else "MISSING"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='num'>{v}</div><div class='lbl'>{k}</div></div>"
        for k, v in cards
    )
    hb_warn = ""
    if hb_age is not None and hb_age > 300:
        hb_warn = "<p class='warn'>heartbeat stale &gt;5min — trainer may be down</p>"

    ante_chart = svg_line([("mean ante / 200 games", "#e0a93e", [c["mean_ante"] for c in chunks])])
    deep_chart = svg_line([
        ("ante ≥ 6 %", "#5ab0f0", [c["a6"] for c in chunks]),
        ("ante ≥ 8 %", "#e05a5a", [c["a8"] for c in chunks]),
    ], fmt="{:.0f}%")

    upd_rows = "".join(
        f"<tr><td>{u[0]}</td><td>{u[1]}</td><td>{u[4]}</td><td>{u[5]}</td>"
        f"<td>{u[10]}</td><td>{u[11]}</td><td>{u[12]}@{u[13]} ({u[14]}%)</td><td>{u[15]}</td></tr>"
        for u in reversed(updates)
    ) or "<tr><td colspan='8' class='dim'>no update lines parsed yet</td></tr>"

    day_rows = "".join(
        f"<tr><td>{d}</td><td>{s['games']}</td><td>{s['wins']}</td>"
        f"<td>{100.0*s['wins']/s['games']:.2f}%</td><td>{s['ante']/s['games']:.2f}</td></tr>"
        for d, s in reversed(wins_by_day(rows))
    )

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='30'>
<title>Balatron</title>
<style>
 body {{ background:#16161c; color:#ddd; font:14px/1.5 'Segoe UI',sans-serif; margin:24px; }}
 h1 {{ font-size:20px; }} h2 {{ font-size:15px; color:#aaa; margin:26px 0 8px; }}
 .cards {{ display:flex; gap:12px; flex-wrap:wrap; }}
 .card {{ background:#1f1f29; border-radius:8px; padding:12px 18px; min-width:96px; text-align:center; }}
 .num {{ font-size:22px; font-weight:600; color:#fff; }} .lbl {{ font-size:11px; color:#999; }}
 table {{ border-collapse:collapse; }} td,th {{ padding:3px 12px; border-bottom:1px solid #2a2a35; text-align:right; }}
 th {{ color:#999; font-weight:500; }} td:first-child, th:first-child {{ text-align:left; }}
 .dim {{ color:#777; }} .warn {{ color:#e05a5a; font-weight:600; }}
 .note {{ color:#888; font-size:12px; }}
</style></head><body>
<h1>Balatron — live training dashboard</h1>
<p class='note'>auto-refreshes every 30s · reads logs/ directly · trend buckets = {CHUNK} games since {ERA_START}</p>
{hb_warn}
<div class='cards'>{card_html}</div>
<h2>Mean ante (is he getting deeper?)</h2>{ante_chart}
<h2>Deep-run rates (leading indicators for wins)</h2>{deep_chart}
<h2>Recent PPO updates</h2>
<table><tr><th>Update</th><th>Step</th><th>R</th><th>Ante</th><th>KL</th><th>CF</th><th>BC</th><th>LR</th></tr>{upd_rows}</table>
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
