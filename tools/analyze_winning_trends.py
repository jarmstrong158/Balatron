"""Winning-trend miner (dec-066).

Mines logs/build_progression.jsonl for the build features that CAUSALLY predict
getting DEEP. Two design choices make it trustworthy on a <1%-win agent:

  1. Continuous outcome (default): the yardstick is MEAN MAX-ANTE REACHED, not
     "did it win." Wins are ~10-15/day — far too rare to stratify. Depth-reached
     is available on every single run, so the signal is readable today and tightens
     hourly. (A binary reach-N rate is still reported alongside, bar configurable.)

  2. Survivorship control: every stat is CONDITIONED ON REACHING the ante it is
     measured at. Comparing runs that all reached ante N and asking who went
     deeper controls for luck — a feature that still separates equally-deep runs
     is causal, not a correlate of survival.

Full-history features (margin / n_xmult / n_scaling) exist on every record.
Categorical features (n_economy / n_mult / n_retrigger, added dec-066) are
field-gated so pre-dec-066 records can't pollute their buckets as "0".

Emits logs/trend_calibration.json (empirical margin -> mean-depth + reach curve).
Read-only; run anytime to validate that a change moved the causal features.

Usage:  python -m tools.analyze_winning_trends [--reach 6] [--antes 4,5]
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


def depth_stats(runs, ante, feature, buckets, reach_bar, require_field=False):
    """Conditioned on reaching `ante`: per feature bucket, (n, mean max-ante,
    reach-`reach_bar` %). require_field drops runs whose ante-N snapshot never
    logged `feature` (keeps pre-dec-066 records out of categorical buckets)."""
    base = []
    for r in runs:
        s = _snap_at(r, ante)
        if s is None or _max_ante(r) < ante:
            continue
        if require_field and feature not in s:
            continue
        base.append(r)
    rows = []
    for label, pred in buckets:
        grp = [r for r in base if pred(_snap_at(r, ante).get(feature))]
        if grp:
            mm = sum(_max_ante(r) for r in grp) / len(grp)
            rr = 100.0 * sum(1 for r in grp if _max_ante(r) >= reach_bar) / len(grp)
            rows.append((label, len(grp), mm, rr))
        else:
            rows.append((label, 0, float("nan"), float("nan")))
    return len(base), rows


def print_feature(name, runs, feature, buckets, antes, reach_bar, require_field=False):
    print(f"\n=== mean MAX-ANTE (and reach-{reach_bar}%) by {name} ===")
    for ante in antes:
        n, rows = depth_stats(runs, ante, feature, buckets, reach_bar, require_field)
        if n == 0:
            continue
        print(f"  at ante {ante} (n={n} reached it):")
        for label, cnt, mm, rr in rows:
            if cnt == 0:
                print(f"    {label:11s} n=0")
            else:
                bar = "#" * int((mm - 3.5) / 0.1) if mm == mm and mm > 3.5 else ""
                print(f"    {label:11s} n={cnt:5d}  mean-ante {mm:4.2f}  reach-{reach_bar} {rr:4.1f}%  {bar}")


EFFECT_MIN_N = 30   # ignore buckets thinner than this in the effect-size ranking
                    # — a lone lucky run (n=1) must not dominate the spread.


def depth_spread(runs, ante, feature, buckets, reach_bar, require_field=False):
    """Effect size = spread in MEAN MAX-ANTE across buckets with n>=EFFECT_MIN_N
    at an ante (thin buckets excluded so a single lucky run can't distort it)."""
    _n, rows = depth_stats(runs, ante, feature, buckets, reach_bar, require_field)
    mm = [m for _l, c, m, _r in rows if c >= EFFECT_MIN_N and m == m]
    return (max(mm) - min(mm)) if len(mm) >= 2 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=int, default=6,
                    help="secondary binary bar reported alongside mean-ante (default 6)")
    ap.add_argument("--antes", default="4,5",
                    help="comma list of measurement checkpoints (default 4,5)")
    ap.add_argument("--min-run-ante", type=int, default=2)
    ap.add_argument("--bp", default=str(BP_PATH))
    args = ap.parse_args()
    antes = tuple(int(a) for a in args.antes.split(","))
    reach = args.reach

    runs = reconstruct_runs(Path(args.bp), args.min_run_ante)
    have_cat = [r for r in runs if any("n_economy" in x for x in r)]
    print(f"reconstructed runs: {len(runs):,}  (with dec-066 categorical fields: "
          f"{len(have_cat):,})")
    print(f"outcome = mean max-ante reached  |  secondary bar = reach-{reach}")

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
    eb = [("0 econ", lambda v: (v or 0) == 0),
          ("1 econ", lambda v: (v or 0) == 1),
          ("2+ econ", lambda v: (v or 0) >= 2)]
    lb = [("0 mult", lambda v: (v or 0) == 0),
          ("1 mult", lambda v: (v or 0) == 1),
          ("2+ mult", lambda v: (v or 0) >= 2)]
    rb = [("0 retrig", lambda v: (v or 0) == 0),
          ("1+ retrig", lambda v: (v or 0) >= 1)]

    # full-history features (present on every record)
    print_feature("margin (power/target — causal spine)", runs, "margin", mb, antes, reach)
    print_feature("n_xmult", runs, "n_xmult", xb, antes, reach)
    print_feature("n_scaling (expected: no signal)", runs, "n_scaling", sb, antes, reach)

    # dec-066 categorical features — field-gated to the slice that logs them
    cat_feats = [("n_economy", eb), ("n_mult", lb), ("n_retrigger", rb)]
    if have_cat:
        for name, b in cat_feats:
            print_feature(f"{name} (dec-066 slice)", runs, name, b, antes, reach,
                          require_field=True)
    else:
        print("\n[no dec-066 categorical records yet — re-run once they accumulate]")

    # "never acquired an xmult engine" — deep vs shallow (clean, uncensored)
    def never_xmult(run):
        return all((x.get("n_xmult") or 0) == 0 for x in run)
    deep = [r for r in runs if _max_ante(r) >= 8]
    shallow = [r for r in runs if 4 <= _max_ante(r) <= 6]
    if deep and shallow:
        dn = 100.0 * sum(never_xmult(r) for r in deep) / len(deep)
        sn = 100.0 * sum(never_xmult(r) for r in shallow) / len(shallow)
        print(f"\n=== never got an xmult engine ===")
        print(f"  deep (reached 8, n={len(deep)}):   {dn:4.1f}%")
        print(f"  shallow (died 4-6, n={len(shallow)}):  {sn:4.1f}%")

    # effect-size ranking (spread in mean max-ante across buckets)
    print("\n=== FEATURE EFFECT SIZE (spread in mean max-ante across buckets) ===")
    feats = [("margin", mb, False), ("n_xmult", xb, False), ("n_scaling", sb, False)]
    if have_cat:
        feats += [(n, b, True) for n, b in cat_feats]
    ranked = [(max(depth_spread(runs, a, n, b, reach, req) for a in antes), n)
              for n, b, req in feats]
    for sp, name in sorted(ranked, reverse=True):
        print(f"  {name:12s} max mean-ante spread: {sp:.2f} antes")

    # calibration table: empirical margin -> mean-depth + reach curve
    calib = {"reach_bar": reach, "runs": len(runs), "margin_depth": {}}
    for ante in antes:
        _n, rows = depth_stats(runs, ante, "margin", mb, reach)
        calib["margin_depth"][ante] = {
            label: {"n": cnt,
                    "mean_ante": None if mm != mm else round(mm, 3),
                    "reach_pct": None if rr != rr else round(rr, 2)}
            for label, cnt, mm, rr in rows}
    os.makedirs(CALIB_OUT.parent, exist_ok=True)
    CALIB_OUT.write_text(json.dumps(calib, indent=2))
    print(f"\nwrote calibration table -> {CALIB_OUT}")


if __name__ == "__main__":
    main()
