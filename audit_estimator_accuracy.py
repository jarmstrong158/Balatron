"""dec-096: how far off is the estimator from what the real game actually scores?

dec-095 found hand_eval implements NEITHER `xmult_scaling` NOR
`rotating_condition`, so 17 jokers (14 of them nominal tier-5 engines -- the
whole scaling archetype) are scored at ~zero by the agent's own evaluator. Before
implementing them, size the gap against ground truth.

`blind_results.jsonl` carries both sides per blind:
    proj_power         what the estimator projected
    realized           what the game actually scored
    realized_vs_proj   the ratio

CENSORING: when a blind is BEATEN the game stops scoring at the target, so
`realized` is truncated and `realized_censored` is set. Those rows understate
true capability and must be excluded from any bias estimate -- including them
would drag the ratio toward 1.0 and hide exactly the under-prediction being
looked for.

A ratio persistently > 1 means the estimator UNDER-predicts: the build scores
more than the agent believes it can, which is the signature of scoring effects
the evaluator cannot see.

Named audit_* so the committed ruff.toml excludes apply (con-019).
"""
import json
import os
import statistics
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join("logs", "blind_results.jsonl")


def main() -> None:
    if not os.path.exists(LOG):
        print(f"missing {LOG}")
        return

    rows = []
    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    usable = [r for r in rows
              if r.get("proj_power") and r.get("realized") is not None
              and float(r.get("proj_power") or 0) > 0]
    uncens = [r for r in usable if not r.get("realized_censored")]
    print(f"{len(rows)} blinds, {len(usable)} with both sides, "
          f"{len(uncens)} UNCENSORED ({len(uncens)/max(len(usable),1):.0%})")

    def ratio(r):
        return float(r["realized"]) / float(r["proj_power"])

    for label, sub in (("ALL (censored included, BIASED)", usable),
                       ("UNCENSORED only", uncens)):
        if not sub:
            continue
        v = sorted(ratio(r) for r in sub)
        print(f"\n{label}: n={len(v)}")
        print(f"  median realized/proj {statistics.median(v):.3f}   "
              f"mean {statistics.mean(v):.3f}")
        for p in (10, 25, 50, 75, 90):
            print(f"    p{p:<3}{v[min(int(len(v)*p/100), len(v)-1)]:.3f}")

    # by ante -- the plateau is at 4-5, so a widening gap there is the signal
    print("\nUNCENSORED, by ante:")
    print(f"  {'ante':<6}{'n':>6}{'median r/p':>12}{'% under-predicted':>20}")
    byante = {}
    for r in uncens:
        byante.setdefault(r.get("ante", 0), []).append(ratio(r))
    for a in sorted(byante):
        v = byante[a]
        if len(v) < 20:
            continue
        under = sum(x > 1.0 for x in v) / len(v)
        print(f"  {a:<6}{len(v):>6}{statistics.median(v):>12.3f}{under:>19.0%}")

    print("\nratio > 1 = the game scored MORE than the estimator projected, i.e.\n"
          "scoring power the evaluator cannot see. Persistent under-prediction is\n"
          "the signature dec-095 predicts from the missing scaling implementations.")


if __name__ == "__main__":
    main()
