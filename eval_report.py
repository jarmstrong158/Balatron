"""Evaluation reporting — the measurement instrument (foundation 1).

Computes the conditional-advance curve with Wilson 95% confidence intervals from
a results file (game_history.jsonl format: one JSON object per game with at least
`ante` and `won`, optionally `seed`). Supports a PAIRED A/B comparison between two
result files run over the SAME seed bank (McNemar-style), which removes seed-luck
variance so a change's effect can actually be attributed.

Why advance-rate, not win-rate: at ~0.5% win rate a 500-game sample expects ~2.5
wins (CI includes 0) — win rate is statistically invisible. The conditional-advance
curve (reach ante A -> advance to A+1) has tight CIs at the shallow antes, so a real
skill change shows up in ~5k games instead of ~30k. Validated (dec-043): realized
xmult VALUE predicts advance; engine COUNT does not.

Usage:
    python eval_report.py results.jsonl                 # single-run curve + CIs
    python eval_report.py baseline.jsonl candidate.jsonl  # A/B (paired if seeds present)
"""
import json
import math
import sys


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(point, lo, hi) Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, center - half), min(1.0, center + half))


def load_games(path: str, seed_set: set = None) -> list[dict]:
    """Load game records; if seed_set is given, keep only games whose seed is in
    it (isolates an eval run's games from training rows in a shared history)."""
    games = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            if seed_set is not None and g.get("seed") not in seed_set:
                continue
            games.append(g)
    return games


def advance_curve(games: list[dict]) -> list[dict]:
    """Per-ante: of games reaching >=A, what fraction advanced to >=A+1, with CI."""
    rows = []
    for a in range(1, 9):
        reached = [g for g in games if g.get("ante", 0) >= a]
        if not reached:
            continue
        adv = [g for g in reached if g.get("ante", 0) >= a + 1]
        p, lo, hi = wilson(len(adv), len(reached))
        rows.append({"ante": a, "reached": len(reached), "advanced": len(adv),
                     "rate": p, "lo": lo, "hi": hi})
    return rows


def win_rate(games: list[dict]) -> tuple[float, float, float, int, int]:
    n = len(games)
    w = sum(1 for g in games if g.get("won") or g.get("ante", 0) >= 9)
    p, lo, hi = wilson(w, n)
    return (p, lo, hi, w, n)


def print_single(path: str, seed_set: set = None):
    games = load_games(path, seed_set)
    print(f"=== {path}: {len(games)} games ===")
    print(f"{'reach>=A':>9} {'n':>6} {'advance':>8} {'95% CI':>16}")
    for r in advance_curve(games):
        if r["ante"] < 2:
            continue
        print(f"{'>= '+str(r['ante']):>9} {r['reached']:>6} {100*r['rate']:>7.1f}% "
              f"[{100*r['lo']:>5.1f},{100*r['hi']:>5.1f}]")
    p, lo, hi, w, n = win_rate(games)
    print(f"WIN: {100*p:.2f}% [{100*lo:.2f},{100*hi:.2f}]  ({w}/{n})")


def compare(path_a: str, path_b: str, seed_set: set = None):
    """A/B. Paired McNemar on per-seed advance when both files share seeds;
    otherwise unpaired curve comparison with CIs."""
    ga, gb = load_games(path_a, seed_set), load_games(path_b, seed_set)
    print_single(path_a, seed_set)
    print()
    print_single(path_b, seed_set)
    sa = {g["seed"] for g in ga if "seed" in g}
    sb = {g["seed"] for g in gb if "seed" in g}
    shared = sa & sb
    print(f"\n=== A/B: {path_a}  vs  {path_b} ===")
    if shared:
        # paired: per shared seed, did each reach >=6 (the deep gate)?
        amap = {g["seed"]: g.get("ante", 0) for g in ga if g.get("seed") in shared}
        bmap = {g["seed"]: g.get("ante", 0) for g in gb if g.get("seed") in shared}
        gate = 6
        b_better = sum(1 for s in shared if bmap[s] >= gate and amap[s] < gate)
        a_better = sum(1 for s in shared if amap[s] >= gate and bmap[s] < gate)
        print(f"paired on {len(shared)} shared seeds, gate = reach ante {gate}:")
        print(f"  candidate better (B reached, A didn't): {b_better}")
        print(f"  baseline better  (A reached, B didn't): {a_better}")
        # McNemar exact-ish: significant if discordant pairs lopsided
        disc = a_better + b_better
        if disc:
            p_b = b_better / disc
            pp, lo, hi = wilson(b_better, disc)
            verdict = ("B>A" if lo > 0.5 else "A>B" if hi < 0.5 else "inconclusive")
            print(f"  discordant {disc}, P(B better|discordant)={100*pp:.0f}% "
                  f"[{100*lo:.0f},{100*hi:.0f}] -> {verdict}")
        else:
            print("  no discordant pairs (identical on the gate)")
    else:
        print("no shared seeds -> unpaired; compare the curves/CIs above "
              "(overlapping CIs = not distinguishable at this N).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Balatron eval report")
    ap.add_argument("files", nargs="+", help="1 file = single curve; 2 = A/B")
    ap.add_argument("--seeds", help="seed-bank file: analyze ONLY games whose "
                    "seed is in it (isolates an eval run from training rows)")
    args = ap.parse_args()
    seed_set = None
    if args.seeds:
        with open(args.seeds) as f:
            seed_set = {s.strip() for s in f if s.strip()}
    if len(args.files) == 1:
        print_single(args.files[0], seed_set)
    elif len(args.files) == 2:
        compare(args.files[0], args.files[1], seed_set)
    else:
        ap.error("pass 1 or 2 files")
