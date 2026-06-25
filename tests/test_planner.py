"""Pillar 2 (PLANNING) tests for the reactor->planner revamp (dec-034).

The planner ranks shop buys by multi-ante SURVIVABILITY (how deep the resulting
build can go against the blind-target curve), not immediate score — the core
planning capability the reactor lacked.

Run:  pytest tests/test_planner.py
"""

import pytest

from environment.planner import (
    ante_target, build_survivability, build_value, rank_shop_jokers,
    HANDS_PER_BLIND,
)


def _gs(ante=2):
    # A developed (leveled) Pair so there's a real base for xmult to multiply —
    # at a near-empty board, +mult and xmult genuinely tie (correct Balatro
    # truth), which is itself a planner sanity check, not a bug.
    return {"ante_num": ante, "hands": {"Pair": {"played": 5, "chips": 60, "mult": 8}},
            "cards": {"cards": []}, "round": {}}


def test_ante_target_monotonic_and_boss_scaling():
    # boss is 2x small; targets strictly increase with ante, including past 8
    assert ante_target(1, "boss") == 600
    assert ante_target(3, "small") < ante_target(3, "big") < ante_target(3, "boss")
    prev = 0
    for a in range(1, 13):
        t = ante_target(a, "boss")
        assert t > prev
        prev = t


def test_survivability_rises_with_a_stronger_build():
    gs = _gs()
    none = build_survivability([], gs)
    weak = build_survivability([{"key": "j_joker", "id": 1}], gs)          # +4 mult flat
    strong = build_survivability([{"key": "j_cavendish", "id": 2}], gs)    # x3 xmult
    assert strong > weak >= none


def test_build_value_prefers_the_build_defining_joker():
    """An xmult engine should plan-rank above a small flat joker — the kind of
    call greedy immediate-score got wrong for fresh/under-leveled engines."""
    gs = _gs()
    x = build_value({"key": "j_cavendish", "id": 1}, [], gs)      # x3 mult
    flat = build_value({"key": "j_joker", "id": 2}, [], gs)       # +4 mult
    assert x > flat


def test_rank_orders_best_first():
    gs = _gs()
    shop = [{"key": "j_joker", "id": 1},          # weak flat
            {"key": "j_cavendish", "id": 2},      # xmult
            {"key": "j_hologram", "id": 3}]       # scaling xmult (projected by Pillar 1)
    ranked = rank_shop_jokers(shop, [], gs)
    assert [i for i, _ in ranked][0] != 0          # the weak flat joker is not ranked first
    assert ranked == sorted(ranked, key=lambda t: t[1], reverse=True)


def test_survivability_uses_hands_per_blind():
    # sanity: the metric scales per-hand power by HANDS_PER_BLIND (smoke check it
    # doesn't crash and returns a sane fractional ante)
    s = build_survivability([{"key": "j_cavendish", "id": 1}], _gs())
    assert 0.0 <= s <= 12.0
    assert HANDS_PER_BLIND > 1.0


def test_planner_pick_joker_integration():
    """Integration: ActionExecutor._planner_pick_joker must run without error
    (the local-import scope bug that made it fail-fast on every live call) and
    pick the build-best affordable shop joker over a weak one."""
    from training.action_executor import ActionExecutor
    ax = ActionExecutor(policy_authority=True)
    raw = {
        "ante_num": 2,
        "hands": {"Pair": {"played": 5, "chips": 60, "mult": 8}},
        "cards": {"cards": []},
        "round": {},
        "shop": {"cards": [
            {"key": "j_joker", "id": 1, "cost": {"buy": 4}},        # weak flat (slot 0)
            {"key": "j_cavendish", "id": 2, "cost": {"buy": 4}},    # xmult     (slot 1)
        ]},
    }
    # policy wanted slot 0 (the weak one); planner should override to slot 1
    pick = ax._planner_pick_joker([], raw, money=10, default_idx=0)
    assert pick == 1, f"planner picked {pick}, expected the xmult joker (slot 1)"


def test_planner_pick_falls_back_on_unaffordable():
    from training.action_executor import ActionExecutor
    ax = ActionExecutor(policy_authority=True)
    raw = {"ante_num": 2, "hands": {"Pair": {"played": 5, "chips": 60, "mult": 8}},
           "cards": {"cards": []}, "round": {},
           "shop": {"cards": [{"key": "j_cavendish", "id": 2, "cost": {"buy": 99}}]}}
    # nothing affordable -> keep the policy's pick
    assert ax._planner_pick_joker([], raw, money=5, default_idx=3) == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
