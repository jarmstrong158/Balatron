"""dec-098: per-BLIND clear rate — the metric every prior A/B should have used.

WHY THE OLD METRIC COULD NOT WORK
---------------------------------
A run must clear ~24 blinds, so win rate is per-blind clear rate raised to the
24th. At the observed 0.858 average that compounds to ~1-2%, which matches the
measured 0.83-1.18%. Two consequences follow, and both are load-bearing:

1. NO SINGLE LEVER CAN FIX THE PLATEAU. Reaching a 10% win rate needs 0.909 per
   blind -- a +5pp improvement at EVERY ante at once. No shop policy or evaluator
   tweak does that. The search for one broken component was the wrong frame.

2. EVERY PRIOR A/B WAS UNDERPOWERED BY CONSTRUCTION. A genuinely good lever is
   worth perhaps +1pp per blind -- a 1.3x win-rate improvement, real and worth
   having. Detecting THAT as win rate needs ~18,627 runs per arm at 80% power.
   dec-079/081/083/093 ran 60-600. Those results are NON-RESULTS, not evidence
   of no effect, and should not be cited as "measured null" for realistic effect
   sizes.

Per-blind clear rate needs ~18,537 blinds per arm for the same +1pp -- and a run
is ~24 blinds, so the win-rate route costs ~447,000 blinds of play for the same
answer. That is ~24x more sample-efficient, and the ratio is not a coincidence: a
run yields 24 blinds of evidence but only ONE bit of win/loss.

(This docstring said "~7x / 62,134 blinds" until 08-03. That figure compared an
ante-4 base rate at delta 0.005 against a 0.858 base at delta 0.01 -- varying BOTH
the base rate and the effect size across the two sides. The decision entry was
corrected; this file was not, for a day. Corrections have to reach the code, not
just the prose.)

con-014: regime boundaries are respected. A metric whose accounting changed must
never be read across one, so rows are filtered by `step` and the boundary list is
printed with the result.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import argparse
import collections
import json
import math
import os

LOG = os.path.join("logs", "blind_results.jsonl")
PLATEAU = (4, 5, 6)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), c + h)


def n_needed(p1, delta, power_z=0.84, alpha_z=1.96):
    """Blinds per arm to detect +delta on a base rate of p1, 80% power."""
    p2 = min(p1 + delta, 0.999)
    pb = (p1 + p2) / 2
    return ((alpha_z * math.sqrt(2 * pb * (1 - pb))
             + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p2 - p1) ** 2)


def load(min_step=0, max_step=None):
    rows = []
    if not os.path.exists(LOG):
        return rows
    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("step", 0) or 0
            if s < min_step or (max_step is not None and s > max_step):
                continue
            rows.append(r)
    return rows


def rates(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("ante", 0)].append(bool(r.get("beaten")))
    return by


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-step", type=int, default=0,
                    help="ignore blinds before this global_step (con-014)")
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--split-step", type=int, default=None,
                    help="A/B: rows below vs at-or-above this step")
    args = ap.parse_args()

    rows = load(args.min_step, args.max_step)
    if not rows:
        print("no rows in range")
        return

    if args.split_step:
        A = [r for r in rows if (r.get("step", 0) or 0) < args.split_step]
        B = [r for r in rows if (r.get("step", 0) or 0) >= args.split_step]
        ra, rb = rates(A), rates(B)
        print(f"A: {len(A)} blinds (step < {args.split_step})")
        print(f"B: {len(B)} blinds (step >= {args.split_step})")
        print(f"\n{'ante':<6}{'nA':>7}{'clearA':>9}{'nB':>7}{'clearB':>9}"
              f"{'diff':>9}{'detectable?':>14}")
        for a in sorted(set(ra) | set(rb)):
            if a < 1 or a > 8:
                continue
            va, vb = ra.get(a, []), rb.get(a, [])
            if len(va) < 100 or len(vb) < 100:
                continue
            pa, pb_ = sum(va) / len(va), sum(vb) / len(vb)
            need = n_needed(pa, max(abs(pb_ - pa), 0.001))
            ok = "yes" if min(len(va), len(vb)) >= need else f"need {need:,.0f}/arm"
            print(f"{a:<6}{len(va):>7}{pa:>9.3f}{len(vb):>7}{pb_:>9.3f}"
                  f"{pb_-pa:>+9.3f}{ok:>14}")
        print("\nA 'diff' whose arm sizes are below the needed n is NOT a result —\n"
              "it is noise, in either direction.")
        return

    by = rates(rows)
    print(f"{len(rows)} blinds, steps "
          f"{min((r.get('step',0) or 0) for r in rows):,} .. "
          f"{max((r.get('step',0) or 0) for r in rows):,}")
    print(f"\n{'ante':<6}{'n':>8}{'clear':>9}{'95% CI':>20}")
    prod = 1.0
    have = []
    for a in sorted(by):
        if a < 1 or a > 8 or len(by[a]) < 100:
            continue
        v = by[a]
        p = sum(v) / len(v)
        lo, hi = wilson(sum(v), len(v))
        have.append(p)
        prod *= p ** 3
        print(f"{a:<6}{len(v):>8}{p:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}")

    if have:
        avg = sum(have) / len(have)
        print(f"\nmean per-blind clear {avg:.3f}   compounded over 24 blinds "
              f"-> predicted win {prod:.2%}")
        print("\nWIN RATE IS THIS NUMBER TO THE 24th POWER:")
        for target in (0.05, 0.10, 0.25):
            print(f"   to win {target:>4.0%} of runs, per-blind must reach "
                  f"{target ** (1/24):.3f}  (+{target ** (1/24) - avg:.3f})")
        print("\nblinds/arm needed to DETECT a change at the plateau "
              f"(base {avg:.3f}, 80% power):")
        for d in (0.005, 0.01, 0.02, 0.03):
            print(f"   +{d:.1%} per blind -> {n_needed(avg, d):>10,.0f} blinds/arm"
                  f"   (win rate {avg**24:.2%} -> {(avg+d)**24:.2%})")

    print("\ncon-014: this reads across whatever regimes are in range. Use\n"
          "--min-step / --split-step to keep a comparison inside one.")


if __name__ == "__main__":
    main()
