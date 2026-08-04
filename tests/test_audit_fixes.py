"""dec-100: fixes for the 08-03 code audit — features that were silently inert.

The audit found several places where the code did not do what its comments said.
These tests pin each fix against the SOURCE OF TRUTH that proved the bug, so a
future edit cannot quietly restore the old behaviour.
"""
import pytest

from environment.hand_eval import (
    _estimate_joker_scoring_for_type as est,
)


def _jk(key):
    return {"joker_key": key, "key": key, "modifier": {}, "id": 1}


GS = {"hands": {"Flush": {"chips": 35, "mult": 4, "level": 1}},
      "cards": {"cards": [{} for _ in range(40)]},
      "round": {"discards_left": 3}, "jokers": {"cards": []}}


# --------------------------------------------------------------------------
# 1. round_played -> played_this_round  (the one that ACTIVELY harmed)
# --------------------------------------------------------------------------

def test_no_code_reads_the_nonexistent_round_played_key():
    """`round_played` is not a field the API returns. NOTES.md:372 documents the
    poker-hand object as (order, level, chips, mult, played, played_this_round,
    example), and game_state.py's live EventDetector reads `played_this_round`.

    Three guards in hand_eval read `round_played`, so none of them ever fired.
    Worst case was mouth_should_dig: its "already locked?" early-return never
    triggered, so it returned True on EVERY hand, and action_executor calls it as
    a HARD OVERRIDE of the policy's PLAY — on The Mouth (annotated there as the
    "highest single deep-death source (74%)") the agent discarded its whole
    budget instead of playing. Inverted behaviour, not just a dead guard.
    """
    import inspect

    from environment import hand_eval
    src = inspect.getsource(hand_eval)
    assert "round_played" not in src, "the dead key is back in hand_eval"
    assert "played_this_round" in src


def test_mouth_guard_locks_on_a_played_hand_type():
    from environment.hand_eval import mouth_should_dig

    def st(played):
        return {"blinds": {"boss": {"status": "CURRENT", "name": "The Mouth"}},
                "round": {"discards_left": 3}, "hands": played}

    assert mouth_should_dig([], [], st({"Pair": {"played_this_round": 1}})) is False


# --------------------------------------------------------------------------
# 2. Probability and rotating conditions folded into an expected value
# --------------------------------------------------------------------------

def test_bloodstone_is_valued_at_ev_not_full_payoff():
    """`effect_probability` is a schema FIELD, but the branch reading it tested
    it as a TRIGGER NAME — impossible, since validate_joker rejects triggers
    outside TRIGGER_VOCABULARY. So it was unreachable and Bloodstone (p=0.5,
    x1.5, per-card) scored the full 1.5^n. It is in HIGH_VALUE_JOKERS, so the
    over-valuation drove a buy bias."""
    x = est("Flush", [_jk("j_bloodstone")], GS)[2]
    assert x == pytest.approx(1.25 ** 2, abs=0.01), x
    assert x < 2.0, "still scoring the full payoff rather than EV"


def test_rotating_condition_jokers_are_not_valued_at_full_payoff():
    """The Idol and Ancient Joker carry rotating_condition=True with EMPTY
    trigger_ranks/trigger_suits — the real target rotates each round and is not
    in the schema. compute_joker_scoring's `any(... & set())` is therefore always
    False (x1.0), while this estimator's fallback used trigger_count=2 (x4.0).
    Neither is right; the condition is unknown, so the value is P(holds) x payoff.
    """
    idol = est("Flush", [_jk("j_idol")], GS)[2]
    ancient = est("Flush", [_jk("j_ancient_joker")], GS)[2]
    assert 1.0 < idol < 1.5, idol
    assert 1.0 < ancient < 2.1, ancient


def test_the_two_estimators_no_longer_disagree_wildly():
    """The shop/planner path and the play-selection path must not value the same
    joker by multiples of each other — that is what let the shop buy a card the
    hand-chooser then ignored."""
    from environment.hand_eval import classify_hand, compute_joker_scoring

    def card(r, s, v):
        return {"value": {"rank": r, "suit": s, "value": v}, "modifier": {}}
    hand = [card("A", "Hearts", 14), card("K", "Hearts", 13),
            card("Q", "Hearts", 12), card("J", "Hearts", 11),
            card("9", "Hearts", 9)]
    ht, si = classify_hand(hand)
    for key in ("j_idol", "j_bloodstone", "j_ancient_joker"):
        shop = est("Flush", [_jk(key)], GS)[2]
        play = compute_joker_scoring(ht, hand, list(si or []), [_jk(key)], GS,
                                     base_mult=4.0)[2]
        ratio = max(shop, play) / max(min(shop, play), 1e-9)
        assert ratio < 4.0, f"{key}: shop {shop:.2f} vs play {play:.2f}"


# --------------------------------------------------------------------------
# 3. An all-zero magnitude result must not consume the joker
# --------------------------------------------------------------------------

def test_blue_joker_is_not_worth_zero():
    """_magnitude_count handles 11 detail strings then falls through to
    `return 0`. `cards_remaining` (Blue Joker) and `blinds_skipped` (Throwback)
    are missing, but the resolver still returned a non-None (0,0,1.0) — so the
    caller's `continue` skipped the normal effect path, where Blue Joker's
    per_card_remaining_in_deck branch valued it correctly. The resolver was
    deleting value the trigger path had already computed."""
    chips = est("Flush", [_jk("j_blue_joker")], GS)[0]
    assert chips > 0, "Blue Joker still worth nothing in the shop estimator"


# --------------------------------------------------------------------------
# 4. Feasibility must use the BEST sale, not the cheapest
# --------------------------------------------------------------------------

def test_feasibility_uses_max_sell_value():
    """The mask body asks `cost <= money + max(sell_prices)` — legality means
    SOME sellable joker frees the slot and funds the buy. The feasibility gate
    used min(), so whenever min(sell) < cost-money <= max(sell) it masked
    ACTION_BUY_JOKER off before the legality branch could run, silently
    defeating the dec-081 fix it was meant to admit."""
    import inspect

    from environment import action_space
    src = inspect.getsource(action_space._is_action_feasible)
    # Strip comments — the fix's own explanation mentions min(), and the first
    # version of this test matched that prose instead of the code.
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    assert "best_sell = max(" in code
    assert "min(" not in code, "a min() still guards the sell-then-buy test"
    assert "money + best_sell" in code
