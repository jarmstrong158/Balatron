"""dec-070: estimate_score_for_hand_type must weight by play frequency.

The bug: the projection returned the best score among any hand type played at
least ONCE, so a single lucky Straight Flush at ante 2 pinned the estimate to a
hand the bot never repeated (realized/projected ~0.30 at every ante).

Run:  pytest tests/test_score_projection.py
"""

import pytest

from environment.hand_eval import estimate_score_for_hand_type


def _gs(played=None):
    hands = {ht: {"played": n} for ht, n in (played or {}).items()}
    return {"ante_num": 3, "hands": hands, "cards": {"cards": []}, "round": {}}


def test_no_play_history_falls_back_to_pair():
    """Early game must still work: no history means no weights to build, so the
    estimate stays the raw Pair baseline — well under any real build."""
    no_history = estimate_score_for_hand_type([], _gs())
    sf_build = estimate_score_for_hand_type([], _gs({"Straight Flush": 40}))
    assert 0 < no_history < sf_build / 10


def test_play_share_not_play_count_drives_the_weight():
    """Weights are each type's SHARE of plays, so a lone Pair and 40 Pairs
    describe the same bot and must project the same."""
    assert (estimate_score_for_hand_type([], _gs({"Pair": 1}))
            == pytest.approx(estimate_score_for_hand_type([], _gs({"Pair": 40}))))


def test_one_lucky_high_hand_does_not_pin_the_projection():
    """The dec-070 bug. 1 Straight Flush among 40 Pairs must barely move the
    estimate — pre-fix this returned the full Straight Flush score."""
    grinder = estimate_score_for_hand_type([], _gs({"Pair": 40}))
    lucky = estimate_score_for_hand_type([], _gs({"Pair": 40, "Straight Flush": 1}))
    assert lucky < grinder * 1.5, (
        f"one lucky Straight Flush moved the projection {grinder:.0f} -> "
        f"{lucky:.0f}; play-frequency weighting not applied")


def test_projection_tracks_what_the_bot_actually_plays():
    """A committed Straight Flush build must still project far above a Pair
    grinder — the fix must not just clamp everything to the low hands."""
    grinder = estimate_score_for_hand_type([], _gs({"Pair": 40}))
    sf_build = estimate_score_for_hand_type([], _gs({"Straight Flush": 40}))
    assert sf_build > grinder * 3


def test_unplayed_types_stay_reachable_but_do_not_dominate():
    """The floor is split across types, not applied per-type: unplayed hands
    keep nonzero weight, but must not re-inflate a Pair grinder's estimate.
    The no-history fallback gives us the raw unweighted Pair score to compare."""
    raw_pair = estimate_score_for_hand_type([], _gs())
    grinder = estimate_score_for_hand_type([], _gs({"Pair": 40}))
    # the floor lets unplayed types pull the estimate above a pure Pair score...
    assert grinder > raw_pair
    # ...but a per-type 0.05 floor would have put this ~7x over, not ~2x.
    assert grinder < raw_pair * 3
