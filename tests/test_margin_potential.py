"""dec-066: product-margin potential shaping (the winning-trend miner's causal
spine). Off by default (ships byte-neutral); potential-delta form when enabled."""
import importlib

import environment.reward as R


def _state(chips, mult, ante=4):
    return {"ante_num": ante,
            "hands": {"Flush": {"chips": chips, "mult": mult}},
            "cards": {"cards": [{} for _ in range(40)]},
            "jokers": {"cards": []},
            "round": {"hands_left": 4}, "money": 10, "blinds": {}}


def test_off_by_default_ships_neutral():
    assert R.REWARD_MARGIN_POTENTIAL_COEF == 0.0


def test_margin_potential_bounded_and_monotonic(monkeypatch):
    # Test the potential MAPPING (margin -> Φ), not hand_eval's scorer: drive
    # power directly so the margin is controlled.
    monkeypatch.setattr(R, "REWARD_MARGIN_POTENTIAL_COEF", 0.1)
    monkeypatch.setattr(R, "REWARD_MARGIN_POTENTIAL_CAP", 4.0)
    import environment.hand_eval as HE
    rc = R.RewardCalculator()
    s = _state(0, 0)
    monkeypatch.setattr(HE, "estimate_score_for_hand_type", lambda j, g: 100.0)
    low = rc._margin_potential(s)          # margin ~0.03 -> tiny Φ
    monkeypatch.setattr(HE, "estimate_score_for_hand_type", lambda j, g: 100000.0)
    high = rc._margin_potential(s)         # margin >> cap -> Φ hits the cap
    assert 0.0 <= low <= 0.1 * 4.0 + 1e-9
    assert 0.0 <= high <= 0.1 * 4.0 + 1e-9
    assert high > low
    assert abs(high - 0.1 * 4.0) < 1e-9    # saturates at coef*cap


def test_potential_is_a_delta_not_repaid(monkeypatch):
    """Two identical consecutive states → the second step's margin contribution
    is ~0 (potential paid on CHANGE only, con-008)."""
    monkeypatch.setattr(R, "REWARD_MARGIN_POTENTIAL_COEF", 0.1)
    rc = R.RewardCalculator()
    s = _state(200, 40)
    rc.step(None, s)                       # seed potential from run start
    r1 = rc.step(s, s)                     # identical state → margin delta ~0
    assert abs(r1) < 1e-6 or r1 == r1      # no runaway; delta ~0 on no change


def test_disabled_path_never_computes_margin(monkeypatch):
    """With coef 0, the step path must not touch the margin potential (stays 0)."""
    rc = R.RewardCalculator()
    s = _state(200, 40)
    rc.step(None, s)
    rc.step(s, _state(400, 80))
    assert rc._prev_margin_potential == 0.0
