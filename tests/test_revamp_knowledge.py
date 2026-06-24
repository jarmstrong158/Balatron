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
    SCALE_PROJECT_XMULT_CAP,
)


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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
