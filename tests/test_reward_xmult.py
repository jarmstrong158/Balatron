"""Tests for the xmult-differentiating reward changes (dec-032).

The deep audit found the plateau is bound by the reward not distinguishing
xmult builds from additive ones. Two changes target that:
  1. _check_scaling_growth pays xmult-engine growth REWARD_XMULT_GROWTH_PREMIUM x
     the additive-scaler rate (the dense, non-rarity-gated causal signal).
  2. _check_xmult_acquisition's phase floor lifted 0.3 -> 0.7 so the FIRST xmult
     bought early (Stabilize phase) isn't half-paid for its timing.

Run:  pytest tests/test_reward_xmult.py
"""

import math
import pytest

from environment.reward import (
    RewardCalculator, _joker_is_xmult,
    REWARD_SCALING_GROWTH, REWARD_XMULT_GROWTH_PREMIUM,
    compute_phase_weights,
)


def test_joker_is_xmult_classification():
    assert _joker_is_xmult({"key": "j_hologram"})      # xmult_scaling
    assert _joker_is_xmult({"key": "j_cavendish"})     # xmult fixed
    assert _joker_is_xmult({"key": "j_vampire"})       # xmult_scaling
    assert not _joker_is_xmult({"key": "j_ride_the_bus"})  # additive mult scaling
    assert not _joker_is_xmult({"key": "j_bull"})          # additive
    assert not _joker_is_xmult({"key": ""})                # empty


def _growth_reward(key, old, new):
    """Reward from one scaling step for a single joker of the given key."""
    rc = RewardCalculator(phase=1)
    rc._prev_scaling_values = {1: old}
    state = {"jokers": {"cards": [{"id": 1, "key": key}]}}
    return rc._check_scaling_growth({1: new}, state)


def test_xmult_growth_pays_premium_over_additive():
    # Equal value growth: the xmult engine must pay exactly the premium multiple.
    x = _growth_reward("j_hologram", 1.0, 5.0)      # xmult scaler
    a = _growth_reward("j_ride_the_bus", 1.0, 5.0)  # additive scaler
    assert a > 0
    assert x == pytest.approx(a * REWARD_XMULT_GROWTH_PREMIUM)


def test_growth_matches_expected_log_formula():
    old, new = 2.0, 20.0
    expect = (math.log10(new + 1) - math.log10(old + 1)) * REWARD_SCALING_GROWTH * REWARD_XMULT_GROWTH_PREMIUM
    assert _growth_reward("j_hologram", old, new) == pytest.approx(expect)


def test_no_new_state_falls_back_to_base_rate():
    # Backward-compatible: without new_state, no slot is known-xmult -> base rate.
    rc = RewardCalculator(phase=1)
    rc._prev_scaling_values = {1: 1.0}
    r = rc._check_scaling_growth({1: 5.0})  # new_state omitted
    expect = (math.log10(6.0) - math.log10(2.0)) * REWARD_SCALING_GROWTH
    assert r == pytest.approx(expect)


def test_acquisition_phase_floor_lifted_for_early_antes():
    """Buying an xmult at ante 1 (Stabilize) must now pay the lifted 0.7 floor,
    not the old 0.3 — so foundational early buys aren't suppressed."""
    rc = RewardCalculator(phase=1)
    prev = {"jokers": {"cards": []}}
    # acquire one fixed xmult joker at ante 1
    new = {"ante_num": 1, "jokers": {"cards": [{"id": 7, "key": "j_cavendish"}]}}
    r = rc._check_xmult_acquisition(prev, new)

    from environment.reward import REWARD_XMULT_ACQUIRE_FIXED
    _, w_scale, _ = compute_phase_weights(1)
    expect_new = REWARD_XMULT_ACQUIRE_FIXED * (0.7 + 0.3 * w_scale)
    expect_old = REWARD_XMULT_ACQUIRE_FIXED * (0.3 + 0.7 * w_scale)
    assert r == pytest.approx(expect_new)
    assert r > expect_old  # strictly more rewarded early than before


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
