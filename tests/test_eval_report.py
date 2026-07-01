"""Tests for the evaluation reporting instrument (eval_report.py)."""
import json
import math
from eval_report import wilson, advance_curve, win_rate
from evaluate import _done_seeds


def test_done_seeds_resume(tmp_path):
    # dec-055: evaluate.py resumes by skipping seeds already in the results file.
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps({"seed": s, "ante": 4, "won": False})
                           for s in ("AAA", "BBB", "CCC")) + "\n")
    assert _done_seeds(str(p)) == {"AAA", "BBB", "CCC"}
    assert _done_seeds(str(tmp_path / "missing.jsonl")) == set()  # no file -> empty


def test_wilson_basic():
    # known: 0/0 -> zeros; 50/100 ~ center 0.5 with a real interval
    assert wilson(0, 0) == (0.0, 0.0, 0.0)
    p, lo, hi = wilson(50, 100)
    assert abs(p - 0.5) < 1e-9
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61      # ~[0.40, 0.60]
    # rare event: 1/1000 -> tight, low, lower bound > 0 but small
    p, lo, hi = wilson(1, 1000)
    assert lo >= 0 and hi < 0.01 and p == 0.001


def test_advance_curve_and_winrate():
    # 10 games: antes 1..10 (one each). reach>=A advancing to A+1 is deterministic.
    games = [{"ante": a, "won": a >= 9} for a in range(1, 11)]
    rows = {r["ante"]: r for r in advance_curve(games)}
    # of the 10 reaching >=1, 9 reach >=2 -> 90%
    assert rows[1]["reached"] == 10 and rows[1]["advanced"] == 9
    # curve covers antes 1..8; of 3 reaching >=8, 2 reach >=9 (the win gate)
    assert rows[8]["reached"] == 3 and rows[8]["advanced"] == 2
    p, lo, hi, w, n = win_rate(games)
    assert w == 2 and n == 10            # ante 9 and 10 count as wins
    assert lo < p < hi
