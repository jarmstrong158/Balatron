"""THE SHIP GATE for the rollout evaluator (dec-083).

Answers one question: does the Monte-Carlo rollout predict `beaten` better than
the analytical leaf it would replace? Reports AUC per ante for both, on the SAME
held-out rows, so the comparison is apples-to-apples.

Run it ONLY on post-dec-082 rows (`start.at_blind_start == True`). Pre-fix
snapshots were captured at the LAST hand of the blind rather than the first, which
turns deck state into an outcome proxy: on contaminated rows the rollout reads
AUC 0.774, but normalizing deck size drops it to 0.669 — a +0.105 mirage. The
`--allow-contaminated` flag exists only to reproduce that finding, never to
justify shipping.

    python tools/validate_rollout.py                 # clean rows only (the gate)
    python tools/validate_rollout.py --min-rows 500
    python tools/validate_rollout.py --allow-contaminated   # diagnostic ONLY

DECISION RULE (pre-registered): ship the rollout into the planner only if it beats
the leaf by >= +0.05 AUC at the antes that matter (4-6), on clean rows. Anything
less is not worth its cost — it is ~100x slower than the leaf, and dec-079/081
showed that acting harder on a near-chance evaluator is negative EV.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.rollout import p_clear   # noqa: E402

BASE = {"Pair": (10, 2), "Two Pair": (20, 2), "Three of a Kind": (30, 3),
        "Straight": (30, 4), "Flush": (35, 4), "Full House": (40, 4),
        "Four of a Kind": (60, 7), "Straight Flush": (100, 8), "High Card": (5, 1),
        "Five of a Kind": (120, 12), "Flush House": (140, 14), "Flush Five": (160, 16)}


def gs_from(snap, ante):
    hands = {}
    for h, info in (snap.get("hand_levels") or {}).items():
        bc, bm = BASE.get(h, (5, 1))
        if isinstance(info, dict):
            hands[h] = {"chips": info.get("chips", bc), "mult": info.get("mult", bm),
                        "level": info.get("level", 1)}
        else:
            hands[h] = {"chips": bc, "mult": bm, "level": 1}
    return {"ante_num": ante, "ante": ante, "hands": hands, "blinds": {},
            "jokers": {"cards": snap.get("jokers", [])}}


def auc(pairs):
    """Rank-based AUC with tie handling."""
    pairs = sorted(pairs)
    pos = sum(l for _, l in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    i, s = 0, 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j + 1) / 2
        for k in range(i, j):
            if pairs[k][1]:
                s += rank
        i = j
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/blind_results.jsonl")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--allow-contaminated", action="store_true",
                    help="diagnostic only — pre-dec-082 rows leak deck state")
    args = ap.parse_args()

    rows = []
    with open(args.log) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("start")
            if not s or not s.get("jokers") or not r.get("target"):
                continue
            if r["blind"] in ("Small Blind", "Big Blind"):
                continue
            if not args.allow_contaminated and not s.get("at_blind_start"):
                continue
            rows.append(r)

    tag = "CONTAMINATED (diagnostic)" if args.allow_contaminated else "clean (post-dec-082)"
    print(f"usable boss-blind rows: {len(rows)}  [{tag}]")
    if len(rows) < args.min_rows:
        print(f"\nNOT ENOUGH CLEAN DATA YET (need >= {args.min_rows}).")
        print("The dec-082 capture-once fix only tags rows written after a trainer")
        print("restart. Let training run, then re-run this gate.")
        return 1

    random.seed(0)
    if len(rows) > args.max_rows:
        rows = random.sample(rows, args.max_rows)

    per_ante = {}
    for r in rows:
        snap, ante = r["start"], r["ante"]
        lab = 1 if r["beaten"] else 0
        try:
            p = p_clear(snap, snap["jokers"], gs_from(snap, ante), r["target"],
                        samples=args.samples, seed=0)
        except Exception:
            continue
        leaf = (r.get("proj_power") or 0) / max(r["target"], 1.0)
        per_ante.setdefault(ante, {"roll": [], "leaf": []})
        per_ante[ante]["roll"].append((p, lab))
        per_ante[ante]["leaf"].append((leaf, lab))

    print("\nAUC predicting `beaten` (higher = better discrimination)")
    print("ante |    n | analytical leaf | ROLLOUT | delta")
    verdict_deltas = []
    for ante in sorted(per_ante):
        d = per_ante[ante]
        if len(d["roll"]) < 40:
            continue
        la, ra = auc(d["leaf"]), auc(d["roll"])
        if la != la or ra != ra:
            continue
        delta = ra - la
        if ante in (4, 5, 6):
            verdict_deltas.append(delta)
        print(f"  {ante}  | {len(d['roll']):4d} |      {la:.3f}      |  {ra:.3f}  | {delta:+.3f}")

    if verdict_deltas:
        mean_d = sum(verdict_deltas) / len(verdict_deltas)
        print(f"\nmean delta at antes 4-6: {mean_d:+.3f}")
        if mean_d >= 0.05:
            print("VERDICT: PASSES the pre-registered gate (>= +0.05). "
                  "Proceed to a paired A/B with BALATRON_ROLLOUT=1.")
        else:
            print("VERDICT: FAILS the pre-registered gate (< +0.05). "
                  "Do NOT wire it into the planner — it costs ~100x the leaf's "
                  "runtime and dec-079/081 showed acting on a weak evaluator hurts.")
    else:
        print("\nnot enough rows at antes 4-6 to judge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
