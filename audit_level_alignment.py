"""Do hand LEVELS land on the hand the agent actually PLAYS?

pick_best_planet deliberately levels the build's COMMITTED archetype rather than
the most-played hand -- its own docstring says Jupiter beats Mercury "even if
you've played more Pairs", and the Pillar 3b comment calls play frequency
"lagging frequency" worth ignoring so a pack concentrates levels on one hand.

That is defensible IF the committed hand is the one that ends up being played.
build_progression logs `committed_is_played`, and it is 0 about two thirds of the
time -- so levels may be going to a hand the agent does not play.

This asks whether that costs anything measurable, using the fields already
logged: `power` (projected score for the committed hand) and `margin`
(power/target). If misalignment were harmless, aligned and misaligned rows would
score the same.

CONFOUND, stated up front: alignment is not randomly assigned. A run that found
its archetype early is both more likely to be aligned AND stronger for unrelated
reasons, so a raw gap OVERSTATES the causal effect. Comparisons are therefore
stratified by ante, and the result is read as an upper bound, not an effect size.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import collections
import json
import os
import statistics
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join("logs", "build_progression.jsonl")
TAIL = 200_000


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
    rows = rows[-TAIL:]
    rows = [r for r in rows if r.get("committed_is_played") is not None]
    print(f"{len(rows)} rows with an alignment flag")

    al = [r for r in rows if r["committed_is_played"]]
    mis = [r for r in rows if not r["committed_is_played"]]
    print(f"  aligned    {len(al):7d}  ({len(al)/max(len(rows),1):.1%})")
    print(f"  MISaligned {len(mis):7d}  ({len(mis)/max(len(rows),1):.1%})")

    def med(v, k):
        x = [float(r[k]) for r in v if r.get(k) is not None]
        return statistics.median(x) if x else float("nan")

    print(f"\n{'ante':<6}{'n aligned':>10}{'n mis':>8}"
          f"{'median margin A':>18}{'median margin M':>18}{'ratio':>8}")
    by = collections.defaultdict(lambda: ([], []))
    for r in rows:
        by[r.get("ante", 0)][0 if r["committed_is_played"] else 1].append(r)
    for a in sorted(by):
        A, M = by[a]
        if len(A) < 100 or len(M) < 100:
            continue
        ma, mm = med(A, "margin"), med(M, "margin")
        print(f"{a:<6}{len(A):>10}{len(M):>8}{ma:>18.3f}{mm:>18.3f}"
              f"{(ma/mm if mm else float('nan')):>8.2f}")

    # What is the agent committing to, versus what it actually plays?
    print("\ncommitted hand vs most-played, where they DISAGREE:")
    pairs = collections.Counter(
        (r.get("ht"), r.get("most_played")) for r in mis)
    for (c, p), n in pairs.most_common(10):
        print(f"  commits {str(c):<16} but plays {str(p):<16} {n:6d}")

    # Is the committed hand even levelled higher?
    lv = [float(r["ht_level"]) for r in rows if r.get("ht_level") is not None]
    if lv:
        print(f"\ncommitted hand level: median {statistics.median(lv):.1f} "
              f"mean {statistics.mean(lv):.2f}")
        lva = [float(r["ht_level"]) for r in al if r.get("ht_level") is not None]
        lvm = [float(r["ht_level"]) for r in mis if r.get("ht_level") is not None]
        if lva and lvm:
            print(f"  aligned    median {statistics.median(lva):.1f}")
            print(f"  MISaligned median {statistics.median(lvm):.1f}")

    print("\nAlignment is NOT randomly assigned — a run that found its archetype\n"
          "early is both more likely to be aligned and stronger for other reasons.\n"
          "Read any gap as an UPPER BOUND on the cost of misalignment.")


if __name__ == "__main__":
    main()
