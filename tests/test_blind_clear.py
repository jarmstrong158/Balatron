"""dec-098: per-blind clear rate as the primary experiment metric.

A run clears ~24 blinds, so win rate is per-blind clear rate to the 24th power.
That single fact explains the whole shape of this investigation:

  * no single lever can move win rate much, because reaching 10% needs +5pp per
    blind at EVERY ante simultaneously;
  * every prior A/B (dec-079/081/083/093, 60-600 runs) was underpowered by
    construction for realistic effect sizes and produced NON-results, not
    measured nulls.

These tests pin the arithmetic those conclusions rest on, so a future reader can
trust the reframing rather than re-deriving it.
"""
import math

import pytest

from audit_blind_clear import n_needed, rates, wilson


def _blind(ante, beaten, step=0):
    return {"ante": ante, "beaten": beaten, "step": step}


# --------------------------------------------------------------------------
# The compounding claim
# --------------------------------------------------------------------------

def test_win_rate_is_per_blind_clear_to_the_24th():
    """The load-bearing arithmetic. Two DIFFERENT numbers were conflated on first
    writing: the uniform 0.858^24 = 2.53%, versus the product of the actual
    per-ante rates = 2.22%. The per-ante product is lower because the late antes
    are much worse than the mean and the exponent punishes them. Both are the
    right order against the 0.83-1.18% measured, with the remaining gap from
    skips and non-independence."""
    assert 0.858 ** 24 == pytest.approx(0.0253, abs=0.001)

    per_ante = {1: .978, 2: .953, 3: .942, 4: .886, 5: .804, 6: .772, 7: .749, 8: .777}
    prod = 1.0
    for a in per_ante:
        prod *= per_ante[a] ** 3
    assert prod == pytest.approx(0.0222, abs=0.001)
    assert prod < 0.858 ** 24, "the per-ante product must be the harsher number"


def test_reaching_ten_percent_needs_five_points_per_blind():
    """Why no single lever fixes the plateau: +5pp at EVERY ante at once."""
    need = 0.10 ** (1 / 24)
    assert need == pytest.approx(0.909, abs=0.001)
    assert need - 0.858 == pytest.approx(0.051, abs=0.002)


def test_a_small_per_blind_gain_is_a_large_win_rate_gain():
    """+1pp per blind is a 1.3x win rate — real, worth having, and invisible to a
    600-run eval. This is why the old nulls cannot be trusted."""
    assert (0.868 / 0.858) ** 24 == pytest.approx(1.32, abs=0.03)


# --------------------------------------------------------------------------
# Power: the reason the old A/Bs could not have worked
# --------------------------------------------------------------------------

def test_detecting_a_small_lever_needs_far_more_than_we_ever_ran():
    """dec-079/081/083/093 ran 60-600 runs. Detecting +1pp/blind as a win-rate
    difference needs ~18,600 RUNS per arm."""
    n = n_needed(0.0100, 0.0031)      # 1.00% -> 1.31%, the +1pp/blind equivalent
    assert n > 10_000


def test_per_blind_is_about_24x_more_efficient_than_win_rate():
    """~24x, and the number is not a coincidence: a run yields 24 blinds of
    evidence but only ONE bit of win/loss, so scoring per blind recovers roughly
    the information the binary outcome throws away.

    First stated as "~7x", which was wrong — that came from comparing an ante-4
    base (0.886) at delta 0.005 against a 0.858 base at delta 0.01, i.e. mixing
    both the base rate AND the effect size between the two sides."""
    as_blinds = n_needed(0.858, 0.01)
    as_runs_in_blinds = n_needed(0.0100, 0.0031) * 24
    ratio = as_runs_in_blinds / as_blinds
    assert 20 < ratio < 28, ratio


def test_a_two_point_lever_is_actually_testable():
    """The practical payoff: +2pp/blind needs a few thousand blinds per arm, which
    is about a day of training rather than months."""
    assert n_needed(0.858, 0.02) < 6_000


def test_n_needed_grows_as_the_effect_shrinks():
    assert n_needed(0.858, 0.005) > n_needed(0.858, 0.01) > n_needed(0.858, 0.03)


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------

def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson(886, 1000)
    assert lo < 0.886 < hi
    # a wider interval on less data
    lo2, hi2 = wilson(89, 100)
    assert (hi2 - lo2) > (hi - lo)


def test_wilson_handles_empty_and_extremes():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson(50, 50)
    assert hi <= 1.0 + 1e-9


def test_rates_groups_by_ante():
    by = rates([_blind(4, True), _blind(4, False), _blind(5, True)])
    assert sum(by[4]) == 1 and len(by[4]) == 2
    assert sum(by[5]) == 1 and len(by[5]) == 1


def test_clear_rate_is_computed_over_the_right_denominator():
    """Guards the obvious inversion: clear rate is beaten/total, not
    beaten/failed."""
    by = rates([_blind(4, True)] * 9 + [_blind(4, False)])
    v = by[4]
    assert sum(v) / len(v) == pytest.approx(0.9)


def test_compounding_is_three_blinds_per_ante():
    """Each ante has small, big and boss. Using one blind per ante would
    understate the decay by a factor of three in the exponent."""
    p = {4: 0.886, 5: 0.804}
    prod = 1.0
    for a in p:
        prod *= p[a] ** 3
    assert prod == pytest.approx(0.886 ** 3 * 0.804 ** 3, rel=1e-9)
    assert prod < min(p.values())


def test_math_module_is_actually_used_for_the_power_calc():
    """Sanity: n_needed must depend on the base rate, not just the delta."""
    assert not math.isclose(n_needed(0.50, 0.01), n_needed(0.95, 0.01))
