"""dec-097: does levelling the WRONG hand actually cost blinds?

pick_best_planet deliberately levels the build's COMMITTED archetype, not the
hand being played -- its docstring says Jupiter beats Mercury "even if you've
played more Pairs", and Pillar 3b dismisses play frequency as "lagging". On clean
rows the committed hand is the most-played hand only 50.4% of the time.

build_progression could NOT answer whether that costs anything, because its
`margin` is power/target where `power = estimate_score_for_hand_type(COMMITTED)`.
It measures how strong the committed hand looks, so it structurally cannot see
the loss from playing a different hand. That is the wrong instrument, and reading
it gave a misleading ~1.00-1.06 ratio.

blind_results now carries committed_ht / played_ht / their levels next to the
binary `beaten` flag -- the trustworthy outcome label (con-001).

PRE-REGISTERED READING, fixed before any data exists:
  * aligned blinds clear MEANINGFULLY more often at antes 4-6, stratified ->
    pick_best_planet's committed-archetype bias is costing runs, and the fix is
    in the PICKER (a heuristic we control), not in the reward: the policy never
    chooses the planet, so there is no action for PPO to reinforce.
  * clear rates within noise -> levelling the committed hand is fine, the 50%
    misalignment is harmless, and this surface closes like the others.

CONFOUND, stated up front: alignment is not randomly assigned. A run that found
its archetype early is both more likely to be aligned and stronger for unrelated
reasons, so any raw gap OVERSTATES the causal effect. Stratify by ante and read
the result as an upper bound.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import collections
import json
import math
import os
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join("logs", "blind_results.jsonl")


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), c + h)


def main() -> None:
    if not os.path.exists(LOG):
        print(f"missing {LOG}")
        return

    rows = []
    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    tagged = [r for r in rows if r.get("ht_aligned") is not None]
    print(f"{len(rows)} blinds, {len(tagged)} carrying alignment "
          f"({len(tagged)/max(len(rows),1):.1%})")
    if not tagged:
        print("\nNo tagged rows yet — dec-097 logging starts at the next trainer\n"
              "restart, and only fills in once a hand has actually been played.\n"
              "Re-run after the trainer has played some blinds.")
        return

    print(f"\n{'ante':<6}{'n aligned':>10}{'clear%':>9}{'n mis':>8}{'clear%':>9}"
          f"{'diff':>8}   {'aligned 95% CI':>22}")
    by = collections.defaultdict(lambda: ([], []))
    for r in tagged:
        by[r.get("ante", 0)][0 if r["ht_aligned"] else 1].append(bool(r.get("beaten")))
    for a in sorted(by):
        A, M = by[a]
        if len(A) < 30 or len(M) < 30:
            continue
        pa, pm = sum(A) / len(A), sum(M) / len(M)
        lo, hi = wilson(sum(A), len(A))
        print(f"{a:<6}{len(A):>10}{pa:>9.1%}{len(M):>8}{pm:>9.1%}"
              f"{pa-pm:>+8.1%}   [{lo:.1%}, {hi:.1%}]")

    # Does the LEVEL gap matter, independent of identity?
    print("\nby level of the hand actually played (aligned rows excluded so the\n"
          "two effects are not confounded):")
    lv = collections.defaultdict(list)
    for r in tagged:
        if not r["ht_aligned"] and r.get("played_level"):
            lv[min(int(r["played_level"]), 5)].append(bool(r.get("beaten")))
    for level in sorted(lv):
        v = lv[level]
        if len(v) < 30:
            continue
        lo, hi = wilson(sum(v), len(v))
        print(f"  played-hand level {level}: n={len(v):5d} "
              f"clear {sum(v)/len(v):.1%}  [{lo:.1%}, {hi:.1%}]")

    print("\nAlignment is NOT randomly assigned — read any gap as an UPPER BOUND.")


if __name__ == "__main__":
    main()
