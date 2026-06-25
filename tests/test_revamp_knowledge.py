"""Pillar 1 (KNOWLEDGE) tests for the reactor->planner revamp (dec-034).

1a — scaling jokers in the SHOP must be valued by a projected mid-run value, not
their ~x1.0 start, so the build valuation stops seeing run-defining engines
(Hologram/Vampire/Green Joker/...) as worthless at the moment of purchase.

Run:  pytest tests/test_revamp_knowledge.py
"""

import pytest

from data.jokers import JOKERS
from environment.hand_eval import (
    _project_shop_scaling_value, estimate_score_for_hand_type,
    _resolve_magnitude_contribution, SCALE_PROJECT_XMULT_CAP,
)


def _steel_card():
    return {"modifier": {"enhancement": "STEEL"}}


def _stone_card():
    return {"modifier": {"enhancement": "STONE"}}


def _mag(name, key, gs):
    return _resolve_magnitude_contribution({"key": key, "id": 1}, JOKERS[name], gs, "Pair")


def _gs(ante=3):
    return {"ante_num": ante, "hands": {}, "cards": {"cards": []}, "round": {}}


def test_projection_xmult_scaler_above_start_and_capped():
    holo = JOKERS["Hologram"]            # xmult, start 1.0, inc 0.25
    v3 = _project_shop_scaling_value(holo, _gs(ante=3))
    # ante 3 -> antes_left 5 -> horizon 10 -> 1 + 0.25*10 = 3.5
    assert v3 == pytest.approx(3.5)
    assert v3 > 1.0                      # not the worthless start value
    # early game projects higher (more antes to grow), but never past the cap
    v1 = _project_shop_scaling_value(holo, _gs(ante=1))
    assert v1 > v3
    assert v1 <= SCALE_PROJECT_XMULT_CAP


def test_projection_mult_scaler():
    gj = JOKERS["Green Joker"]           # mult, start 0.0, inc 1.0
    v = _project_shop_scaling_value(gj, _gs(ante=3))
    assert v == pytest.approx(10.0)      # 0 + 1.0 * (5 antes * 2 incr/ante)


def test_projection_none_for_nonscaling():
    assert _project_shop_scaling_value(JOKERS["Steel Joker"], _gs()) is None  # mag_src, no increment
    assert _project_shop_scaling_value(JOKERS["Joker"], _gs()) is None        # flat joker


def test_shop_scaling_joker_now_raises_estimated_score():
    """The bug this fixes: a shop scaling joker scored ~= baseline (x1.0). With
    projection, adding it must MEANINGFULLY raise the estimated score."""
    gs = _gs(ante=3)
    baseline = estimate_score_for_hand_type([], gs)
    with_holo = estimate_score_for_hand_type([{"key": "j_hologram", "id": 1}], gs)
    assert with_holo > baseline * 1.5, (
        f"shop Hologram barely moved score ({baseline:.0f} -> {with_holo:.0f}); "
        "projection not applied")


def test_owned_scaling_value_still_wins_over_projection():
    """An owned scaler with a real injected _scaled_value must use THAT, not the
    projection (projection is only the shop fallback)."""
    gs = _gs(ante=3)
    # injected value far above the projection -> score should reflect the real value
    big = estimate_score_for_hand_type([{"key": "j_hologram", "id": 1, "_scaled_value": 6.0}], gs)
    proj = estimate_score_for_hand_type([{"key": "j_hologram", "id": 1}], gs)
    assert big > proj


# ---------------- Pillar 1b: magnitude_source jokers ----------------

def test_steel_joker_xmult_from_steel_cards():
    gs = _gs(); gs["cards"]["cards"] = [_steel_card()] * 4   # 4 steel cards
    c, m, x = _mag("Steel Joker", "j_steel_joker", gs)
    assert x == pytest.approx(1.0 + 0.2 * 4)               # was x1.0 (no effect) before
    # no steel cards -> x1.0 (correct: worthless without a steel deck)
    assert _mag("Steel Joker", "j_steel_joker", _gs())[2] == pytest.approx(1.0)


def test_joker_stencil_xmult_from_empty_slots():
    gs = _gs(); gs["jokers"] = {"limit": 5, "cards": [{"id": 9}, {"id": 10}]}  # 3 empty
    assert _mag("Joker Stencil", "j_stencil", gs)[2] == pytest.approx(1.0 + 3)


def test_stone_joker_chips_from_stone_cards():
    gs = _gs(); gs["cards"]["cards"] = [_stone_card()] * 3
    assert _mag("Stone Joker", "j_stone", gs)[0] == pytest.approx(25.0 * 3)


def test_bull_chips_from_dollars():
    gs = _gs(); gs["money"] = 20
    assert _mag("Bull", "j_bull", gs)[0] == pytest.approx(2.0 * 20)


def test_mystic_summit_gate():
    gs = _gs(); gs["round"] = {"discards_left": 0}
    assert _mag("Mystic Summit", "j_mystic_summit", gs)[1] == pytest.approx(15.0)
    gs2 = _gs(); gs2["round"] = {"discards_left": 2}
    assert _mag("Mystic Summit", "j_mystic_summit", gs2)[1] == 0.0


def test_drivers_license_gate():
    gs = _gs(); gs["cards"]["cards"] = [{"modifier": {"enhancement": "BONUS"}}] * 16
    assert _mag("Driver's License", "j_drivers_license", gs)[2] == pytest.approx(3.0)
    assert _mag("Driver's License", "j_drivers_license", _gs())[2] == pytest.approx(1.0)


def test_steel_joker_now_raises_estimated_score():
    """End-to-end: Steel Joker with a steel-heavy deck must raise the estimate
    (it scored x1.0 = no effect everywhere before 1b)."""
    gs = _gs(); gs["cards"]["cards"] = [_steel_card()] * 6
    base = estimate_score_for_hand_type([], gs)
    withsteel = estimate_score_for_hand_type([{"key": "j_steel_joker", "id": 1}], gs)
    assert withsteel > base * 1.3


def test_non_magnitude_joker_returns_none():
    assert _resolve_magnitude_contribution({"key": "j_joker"}, JOKERS["Joker"], _gs(), "Pair") is None


# ---------------- Pillar 1c: economy engines ----------------

def test_economy_joker_now_has_build_value():
    """Economy jokers score ~0, so they were valued at the floor and ignored.
    Now they carry a money-output build value."""
    from environment.action_space import (
        _estimate_joker_value, ECON_SCORE_PER_DOLLAR,
    )
    gs = _gs()
    v = _estimate_joker_value({"key": "j_golden_joker", "id": 1}, [], gs)  # $4/round
    assert v == pytest.approx(4.0 * ECON_SCORE_PER_DOLLAR)
    # comfortably above the generic ~10 floor a no-score joker would otherwise get
    assert v > 50


# ---------------- Pillar 3c: economy save-then-spike ----------------

def test_war_chest_up_to_interest_cap_not_penalized():
    """dec-034 3c: holding money up to the $25 interest cap during Scale phase is
    OPTIMAL (the war chest) and must NOT be penalized — the old $10 buffer did."""
    from environment.reward import RewardCalculator, INTEREST_CAP
    rc = RewardCalculator(phase=1)
    # ante 4 = Scale phase (w_scale high)
    assert rc._check_gold_hoarding({"ante_num": 4, "money": 20, "round": {}}) == 0.0
    assert rc._check_gold_hoarding({"ante_num": 4, "money": INTEREST_CAP, "round": {}}) == 0.0


def test_idle_money_above_cap_still_penalized():
    from environment.reward import RewardCalculator
    rc = RewardCalculator(phase=1)
    pen = rc._check_gold_hoarding({"ante_num": 4, "money": 45, "round": {}})
    assert pen < 0.0           # money well above the cap is idle -> mild penalty
    assert pen >= -0.15        # hard floor respected


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
