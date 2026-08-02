"""dec-099: do engines actually win, and what distinguishes the runs that build one?

dec-093 left the engine hypothesis explicitly untestable: the agent holds ~0.67
engines per run, never leaves the 0-2 band, and no shop policy could force it
there -- both forcing arms came back worse. This answers it OBSERVATIONALLY.

THE CONFOUND, and why the stratification is not optional: raw win-rate-by-engine
count is severely survivorship-biased, because surviving longer means more shops
means more jokers. Mean final ante rises monotonically with engine count, 3.63 at
zero engines to 6.56 at four -- engines and survival feed each other circularly.
Conditioning on REACHED ante fixes the dominant part of it: every run in a
stratum has had roughly the same number of shops.

Residual confounding remains (a run strong for unrelated reasons both survives
and accumulates engines), so this is strong observational evidence, not an
experiment.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import collections
import json
import math
import os
import statistics
import sys

import engine_forcing as ef

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join("logs", "game_history.jsonl")


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), c + h)


def engines(run):
    return sum(ef._tier(j) == 5 for j in (run.get("jokers") or []))


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
    # con-001: a real win is ante advancing PAST 8, never the `won` flag alone.
    rows = [r for r in rows if not r.get("from_curriculum") and r.get("jokers")]
    wins = [r for r in rows if r["ante"] > 8]
    print(f"{len(rows)} runs, {len(wins)} real wins (ante>8) = {len(wins)/len(rows):.2%}")

    print("\nRAW win rate by engine count — SURVIVORSHIP-CONFOUNDED, do not cite:")
    by = collections.defaultdict(list)
    for r in rows:
        by[engines(r)].append(r["ante"] > 8)
    for k in sorted(by):
        v = by[k]
        print(f"   {k} engines: n={len(v):5d}  win {sum(v)/len(v):6.2%}   "
              f"mean final ante {statistics.mean([x['ante'] for x in rows if engines(x)==k]):.2f}")
    print("   ^ mean ante rises with engine count: more antes = more shops = more"
          " jokers.")

    print("\nSTRATIFIED by REACHED ante — every run in a row had ~the same shops:")
    for floor in (6, 7, 8):
        sub = [r for r in rows if r["ante"] >= floor]
        b = collections.defaultdict(list)
        for r in sub:
            b[min(engines(r), 3)].append(r["ante"] > 8)
        print(f"  reached ante {floor}+  (n={len(sub)})")
        for k in sorted(b):
            v = b[k]
            if len(v) < 15:
                print(f"     {k}{'+' if k == 3 else ' '} engines: n={len(v):4d}  (too few)")
                continue
            lo, hi = wilson(sum(v), len(v))
            print(f"     {k}{'+' if k == 3 else ' '} engines: n={len(v):4d}  "
                  f"win {sum(v)/len(v):6.1%}  [{lo:.1%}, {hi:.1%}]")

    # What distinguishes the builds? Compare NON-engine composition only --
    # comparing engine counts between engine-rich and engine-free runs is
    # circular and says nothing.
    sub = [r for r in rows if r["ante"] >= 6]
    hi_g = [r for r in sub if engines(r) >= 3]
    lo_g = [r for r in sub if engines(r) == 0]
    if hi_g and lo_g:
        print(f"\nComposition at ante 6+: {len(hi_g)} runs with 3+ engines vs "
              f"{len(lo_g)} with none")
        print(f"  {'tier':<34}{'3+ eng':>9}{'0 eng':>9}")
        names = {4: "copier/retrigger", 3: "additive scaling",
                 2: "economy", 0: "NO engine value"}
        for t in (4, 3, 2, 0):
            a = statistics.mean(sum(ef._tier(j) == t for j in r["jokers"]) for r in hi_g)
            b_ = statistics.mean(sum(ef._tier(j) == t for j in r["jokers"]) for r in lo_g)
            print(f"  {t} {names[t]:<32}{a:>9.2f}{b_:>9.2f}")
        print(f"  {'total jokers held':<34}"
              f"{statistics.mean(len(r['jokers']) for r in hi_g):>9.2f}"
              f"{statistics.mean(len(r['jokers']) for r in lo_g):>9.2f}")
        print("\n  The engine builds do not hold MORE of anything else — they hold\n"
              "  LESS JUNK. Both have near-full slots, so the constraint is SLOT\n"
              "  ALLOCATION, not acquisition: engine-free runs spend their slots on\n"
              "  tier-0 cards they never sell.")


if __name__ == "__main__":
    main()
