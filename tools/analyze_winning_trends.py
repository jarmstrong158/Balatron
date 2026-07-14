"""Winning-trend miner (dec-066).

Mines logs/build_progression.jsonl for the build features that CAUSALLY predict
reaching the ante-8 boss. The key design choice — the one that separates signal
from survivorship — is that every rate is CONDITIONED ON REACHING the ante it is
measured at. Comparing runs that all reached ante N and asking which went on to
reach ante 8 controls for luck: both groups got equally deep, so a feature that
still separates them is a causal candidate, not a correlate of survival.

Outputs:
  1. A human-readable report of reach-8 survival curves per feature, per ante.
  2. logs/trend_calibration.json — the empirical P(reach 8 | margin at ante N)
     curve, so the planner's margin->survivability mapping can be recalibrated
     against real outcomes (the dec-038 REALIZATION_FACTOR, but as a full curve).

Read-only. Safe to run anytime; use it to VALIDATE that a change actually moved
the causal features before waiting ~150 updates for mean-ante to budge.

Usage:  python -m tools.analyze_winning_trends [--reach 8] [--min-run-ante 2]
"""
import argparse
import collections
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BP_PATH = REPO / "logs" / "build_progression.jsonl"
CALIB_OUT = REPO / "logs" / "trend_calibration.json"


def reconstruct_runs(bp_path: Path, min_run_ante: int):
    """Split build_progression into runs: per env, ordered by step, a new run
    begins whenever ante drops below the previous record's ante."""
    by_env = collections.defaultdict(list)
    for line in bp_path.open():
        try:
            d = json.loads(line)
        except Exception:
            continue
        by_env[d.get("env")].append(d)
    runs = []
    for _env, recs in by_env.items():
        recs.sort(key=lambda r: r.get("step", 0))
        cur, prev = [], 0
        for r in recs:
            a = r.get("ante", 0)
            if a < prev and cur:
                runs.append(cur)
                cur = []
            cur.append(r)
            prev = a
        if cur:
            runs.append(cur)
    return [r for r in runs if max(x.get("ante", 0) for x in r) >= min_run_ante]


def _max_ante(run):
    return max(x.get("ante", 0) for x in run)


def _snap_at(run, ante):
    got = [x for x in run if x.get("ante", 0) == ante]
    return got[-1] if got else None


def stratified_reach_rate(runs, ante, feature, buckets, reach):
    """Among runs that reached `ante`, P(reach `reach`) per feature bucket."""
    present = [r for r in runs if _snap_at(r, ante) is not None and _max_ante(r) >= ante]
    out = []
    for label, pred in buckets:
        grp = [r for r in present if pred(_snap_at(r, ante).get(feature))]
        rate = (100.0 * sum(1 for r in grp if _max_ante(r) >= reach) / len(grp)
                if grp else float("nan"))
        out.append((label, len(grp), rate))
    return len(present), out


def _bar(rate):
    return "#" * int(rate / 2) if rate == rate else ""


def print_curve(title, runs, feature, buckets, antes, reach):
    print(f"\n=== {title} ===")
    for ante in antes:
        n, rows = stratified_reach_rate(runs, ante, feature, buckets, reach)
        print(f"  at ante {ante} (n={n}):")
        for label, cnt, rate in rows:
            print(f"    {label:12s} n={cnt:6d}  reach-{reach}: {rate:5.1f}%  {_bar(rate)}")


def spread(runs, ante, feature, buckets, reach):
    """Max-min reach-rate across buckets at an ante = crude effect size."""
    _n, rows = stratified_reach_rate(runs, ante, feature, buckets, reach)
    rates = [r for _l, _c, r in rows if r == r]
    return (max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=int, default=8, help="success = reached this ante")
    ap.add_argument("--min-run-ante", type=int, default=2)
    ap.add_argument("--bp", default=str(BP_PATH))
    args = ap.parse_args()

    runs = reconstruct_runs(Path(args.bp), args.min_run_ante)
    reach = args.reach
    print(f"reconstructed runs: {len(runs)}  |  reach-{reach} success bar")

    xb = [("0 xmult", lambda v: (v or 0) == 0),
          ("1 xmult", lambda v: (v or 0) == 1),
          ("2+ xmult", lambda v: (v or 0) >= 2)]
    mb = [("margin<1", lambda v: v is not None and v < 1.0),
          ("1<=m<2", lambda v: v is not None and 1.0 <= v < 2.0),
          ("2<=m<4", lambda v: v is not None and 2.0 <= v < 4.0),
          ("m>=4", lambda v: v is not None and v >= 4.0)]
    sb = [("0 scal", lambda v: (v or 0) == 0),
          ("1 scal", lambda v: (v or 0) == 1),
          ("2+ scal", lambda v: (v or 0) >= 2)]
    # categorical features (present only in dec-066+ records) — NaN-safe: a bucket
    # with 0 rows just prints n=0. Auto-detected below.
    eb = [("0 econ", lambda v: (v or 0) == 0),
          ("1 econ", lambda v: (v or 0) == 1),
          ("2+ econ", lambda v: (v or 0) >= 2)]
    rb = [("0 retrig", lambda v: (v or 0) == 0),
          ("1+ retrig", lambda v: (v or 0) >= 1)]

    antes = (3, 4, 5, 6)
    print_curve("REACH-8 by MARGIN at ante (product/target — the causal spine)",
                runs, "margin", mb, antes, reach)
    print_curve("REACH-8 by n_xmult at ante", runs, "n_xmult", xb, antes, reach)
    print_curve("REACH-8 by n_scaling at ante (expected: NO signal)",
                runs, "n_scaling", sb, (4, 5), reach)

    have_cat = any("n_economy" in x for r in runs for x in r)
    if have_cat:
        print_curve("REACH-8 by n_economy at ante (dec-066 field)",
                    runs, "n_economy", eb, antes, reach)
        print_curve("REACH-8 by n_retrigger at ante (dec-066 field)",
                    runs, "n_retrigger", rb, antes, reach)
    else:
        print("\n[categorical fields n_economy/n_mult/n_retrigger not yet in the "
              "log — they start logging after the dec-066 deploy; re-run once "
              "records accumulate.]")

    # "never acquired an xmult engine" — deep vs shallow (clean, uncensored)
    def never_xmult(run):
        return all((x.get("n_xmult") or 0) == 0 for x in run)
    deep = [r for r in runs if _max_ante(r) >= reach]
    shallow = [r for r in runs if 4 <= _max_ante(r) <= 6]
    if deep and shallow:
        dn = 100.0 * sum(never_xmult(r) for r in deep) / len(deep)
        sn = 100.0 * sum(never_xmult(r) for r in shallow) / len(shallow)
        print(f"\n=== NEVER got an xmult engine ===")
        print(f"  deep (reached {reach}, n={len(deep)}):   {dn:4.1f}%")
        print(f"  shallow (died 4-6, n={len(shallow)}):  {sn:4.1f}%")

    # effect-size ranking — which feature discriminates hardest at ante 5/6
    print("\n=== FEATURE EFFECT SIZE (reach-rate spread across buckets) ===")
    feats = [("margin", mb), ("n_xmult", xb), ("n_scaling", sb)]
    if have_cat:
        feats += [("n_economy", eb), ("n_retrigger", rb)]
    ranked = []
    for name, b in feats:
        sp = max(spread(runs, 5, name, b, reach), spread(runs, 6, name, b, reach))
        ranked.append((sp, name))
    for sp, name in sorted(ranked, reverse=True):
        print(f"  {name:12s} max spread (ante 5/6): {sp:5.1f} pts")

    # emit the margin survival curve as a calibration table for the planner
    calib = {"reach": reach, "runs": len(runs), "margin_reach_rate": {}}
    for ante in antes:
        _n, rows = stratified_reach_rate(runs, ante, "margin", mb, reach)
        calib["margin_reach_rate"][ante] = {
            label: {"n": cnt, "rate_pct": None if rate != rate else round(rate, 2)}
            for label, cnt, rate in rows}
    os.makedirs(CALIB_OUT.parent, exist_ok=True)
    CALIB_OUT.write_text(json.dumps(calib, indent=2))
    print(f"\nwrote calibration table -> {CALIB_OUT}")


if __name__ == "__main__":
    main()
