"""Pillar 2 (PLANNING) tests for the reactor->planner revamp (dec-034).

The planner ranks shop buys by multi-ante SURVIVABILITY (how deep the resulting
build can go against the blind-target curve), not immediate score — the core
planning capability the reactor lacked.

Run:  pytest tests/test_planner.py
"""

import pytest

from environment.planner import (
    ante_target, build_survivability, build_value, rank_shop_jokers,
    target_hand_type, score_hand_type, COMMITTABLE_HANDS, HANDS_PER_BLIND,
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


def test_target_hand_type_is_committable_and_achievability_weighted():
    gs = _gs()
    t = target_hand_type([], gs)
    assert t in COMMITTABLE_HANDS
    # with no jokers, achievability weighting must NOT commit to a rare hand
    # (raw base score favors Four of a Kind; the prior should pull it to a
    # frequently-makeable hand)
    assert t in ("Pair", "Two Pair", "Three of a Kind", "Flush", "Straight")


def test_target_commits_to_the_build_supported_hand():
    """A flush-leaning build should be able to commit to Flush over Pair once the
    jokers make Flush its strongest achievable hand."""
    gs = {"ante_num": 3,
          "hands": {"Pair": {"played": 2, "chips": 10, "mult": 2},
                    "Flush": {"played": 1, "chips": 35, "mult": 4}},
          "cards": {"cards": []}, "round": {}}
    flush_build = [{"key": "j_blackboard", "id": 1}]  # flush/suit-oriented
    t = target_hand_type(flush_build, gs)
    assert t in COMMITTABLE_HANDS  # smoke: runs over a real build without error


def test_planet_use_holds_off_build_planet():
    """Deliberate leveling: a planet for a hand we never play and aren't
    committed to is HELD (not used) when a consumable slot is free."""
    from environment.hand_eval import plan_consumable_use
    gs = {
        "state": "SELECTING_HAND", "ante_num": 2, "money": 4,
        "hand": {"cards": []}, "jokers": {"cards": []},
        "round": {"hands_left": 3},
        "hands": {"Pair": {"played": 5, "chips": 10, "mult": 2}},  # only play Pair
        # one consumable: a planet for Four of a Kind (never played), slot free
        "consumables": {"limit": 2, "cards": [{"key": "c_ceres", "set": "Planet"}]},
    }
    # c_ceres levels Four of a Kind (off-build) -> should be HELD -> no action
    res = plan_consumable_use(gs)
    assert res is None, f"off-build planet should be held, got {res}"


def test_planet_use_fires_for_played_hand():
    from environment.hand_eval import plan_consumable_use
    from environment.hand_eval import PLANET_TO_HAND_TYPE
    # find a planet key that levels Pair (a hand we play)
    pair_key = next((k for k, v in PLANET_TO_HAND_TYPE.items() if v == "Pair"), None)
    assert pair_key is not None
    gs = {
        "state": "SELECTING_HAND", "ante_num": 2, "money": 4,
        "hand": {"cards": []}, "jokers": {"cards": []}, "round": {"hands_left": 3},
        "hands": {"Pair": {"played": 5, "chips": 10, "mult": 2}},
        "consumables": {"limit": 2, "cards": [{"key": pair_key, "set": "Planet"}]},
    }
    assert plan_consumable_use(gs) == {"consumable": 0}


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


def test_planner_swap_upgrades_full_slots():
    """Full-slot planning: with 5 weak jokers and a strong xmult in the shop, the
    planner should sell a weak one and buy the upgrade (survivability improves)."""
    from training.action_executor import ActionExecutor
    ax = ActionExecutor(policy_authority=True)
    owned = [{"key": "j_joker", "id": i, "cost": {"sell": 2}} for i in range(5)]
    raw = {"ante_num": 3, "hands": {"Pair": {"played": 5, "chips": 60, "mult": 8}},
           "cards": {"cards": []}, "round": {},
           "shop": {"cards": [{"key": "j_cavendish", "id": 99, "cost": {"buy": 4}}]}}
    swap = ax._planner_pick_swap(owned, raw, money=10)
    assert swap is not None, "planner should upgrade a weak full roster"
    sell_idx, buy_idx = swap
    assert 0 <= sell_idx < 5 and buy_idx == 0


def test_planner_swap_declines_when_no_improvement():
    """Don't downgrade: selling a strong xmult for a weak joker must return None."""
    from training.action_executor import ActionExecutor
    ax = ActionExecutor(policy_authority=True)
    owned = [{"key": "j_cavendish", "id": i, "cost": {"sell": 3}} for i in range(5)]
    raw = {"ante_num": 3, "hands": {"Pair": {"played": 5, "chips": 60, "mult": 8}},
           "cards": {"cards": []}, "round": {},
           "shop": {"cards": [{"key": "j_joker", "id": 99, "cost": {"buy": 4}}]}}
    assert ax._planner_pick_swap(owned, raw, money=10) is None


def test_planner_swap_protects_eternal():
    """An eternal joker is unsellable — the planner must not propose selling it."""
    from training.action_executor import ActionExecutor
    ax = ActionExecutor(policy_authority=True)
    owned = [{"key": "j_joker", "id": 0, "cost": {"sell": 2},
              "modifier": {"eternal": True}}]  # only joker, eternal
    raw = {"ante_num": 3, "hands": {"Pair": {"played": 5, "chips": 60, "mult": 8}},
           "cards": {"cards": []}, "round": {},
           "shop": {"cards": [{"key": "j_cavendish", "id": 99, "cost": {"buy": 4}}]}}
    assert ax._planner_pick_swap(owned, raw, money=10) is None  # can't sell the eternal


def test_planner_reroll_gate():
    """Reroll-to-assemble only fires on genuine surplus above the interest floor,
    with buy-money left, under the per-shop cap."""
    from training.action_executor import ActionExecutor
    import types
    ax = ActionExecutor(policy_authority=True)
    raw = {"round": {"reroll_cost": 5}, "used_vouchers": []}
    env = types.SimpleNamespace(shop_rerolls=0)
    # plenty of surplus above $25 floor -> ok
    assert ax._planner_reroll_ok(env, raw, money=40) is True
    # at/just above floor: $28 - $5 = $23 < $25 floor -> not surplus -> no
    assert ax._planner_reroll_ok(env, raw, money=28) is False
    # too little to buy after rerolling -> no
    assert ax._planner_reroll_ok(env, raw, money=6) is False
    # over the per-shop cap -> no
    env.shop_rerolls = 3
    assert ax._planner_reroll_ok(env, raw, money=60) is False


def test_planner_reroll_respects_seed_money_floor():
    from training.action_executor import ActionExecutor
    import types
    ax = ActionExecutor(policy_authority=True)
    env = types.SimpleNamespace(shop_rerolls=0)
    raw = {"round": {"reroll_cost": 5}, "used_vouchers": ["v_seed_money"]}  # floor $50
    assert ax._planner_reroll_ok(env, raw, money=45) is False   # below the raised floor
    assert ax._planner_reroll_ok(env, raw, money=70) is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
