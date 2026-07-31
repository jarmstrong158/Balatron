"""dec-096: reconstructing a HUMAN's actions from consecutive game states.

The agent knows what it did because it chose it. A human does not announce
actions, so they must be inferred from state deltas — and a mis-inferred action
teaches the agent the WRONG lesson from a good run, which is worse than having no
data at all. That makes `infer_action` the piece worth testing hard.

The governing rule these tests enforce: when a transition is ambiguous, return
None so the caller DROPS it. Never guess.
"""
from environment.action_space import (
    ACTION_BUY_JOKER,
    ACTION_BUY_PACK,
    ACTION_DISCARD,
    ACTION_END_SHOP,
    ACTION_PLAY,
    ACTION_REROLL,
    ACTION_SELL_JOKER,
    ACTION_USE_CONSUMABLE,
)
from human_record import Recorder, infer_action


def S(state="SELECTING_HAND", money=10, hands=3, discards=3, jokers=0,
      cons=0, shop=2, packs=2, chips=0, ante=4):
    return {
        "state": state, "ante_num": ante, "money": money,
        "round": {"hands_left": hands, "discards_left": discards, "chips": chips},
        "jokers": {"cards": [{"label": f"j{i}"} for i in range(jokers)]},
        "consumables": {"cards": [{} for _ in range(cons)]},
        "shop": {"cards": [{} for _ in range(shop)]},
        "packs": {"cards": [{} for _ in range(packs)]},
    }


def _type(prev, cur):
    got = infer_action(prev, cur)
    return got[0] if got else None


# --------------------------------------------------------------------------
# In-blind
# --------------------------------------------------------------------------

def test_play_and_discard_are_not_confused():
    """Both remove cards from hand; only a PLAY consumes a hand. If these swap,
    the agent learns to discard where a human played."""
    assert _type(S(hands=3, discards=3), S(hands=2, discards=3)) == ACTION_PLAY
    assert _type(S(hands=3, discards=3), S(hands=3, discards=2)) == ACTION_DISCARD


def test_a_play_that_also_changed_chips_is_still_a_play():
    assert _type(S(hands=3, chips=0), S(hands=2, chips=4200)) == ACTION_PLAY


def test_chips_gained_is_captured_for_the_play():
    got = infer_action(S(hands=3, chips=100), S(hands=2, chips=5100))
    assert got[0] == ACTION_PLAY and got[1]["chips_gained"] == 5000


# --------------------------------------------------------------------------
# Shop
# --------------------------------------------------------------------------

def test_buying_a_joker():
    got = infer_action(S(state="SHOP", money=12, jokers=1),
                       S(state="SHOP", money=6, jokers=2))
    assert got[0] == ACTION_BUY_JOKER and got[1]["cost"] == 6


def test_selling_a_joker_is_not_read_as_a_buy():
    """Direction matters: joker count DOWN and money UP. Reading a sale as a
    purchase would teach the agent to buy at the moment a human divested."""
    got = infer_action(S(state="SHOP", money=6, jokers=3),
                       S(state="SHOP", money=9, jokers=2))
    assert got[0] == ACTION_SELL_JOKER and got[1]["gain"] == 3


def test_reroll_is_money_down_with_nothing_acquired():
    got = infer_action(S(state="SHOP", money=15, jokers=2, cons=1),
                       S(state="SHOP", money=10, jokers=2, cons=1))
    assert got[0] == ACTION_REROLL and got[1]["cost"] == 5


def test_buying_a_pack():
    assert _type(S(state="SHOP", money=10, packs=2),
                 S(state="SHOP", money=6, packs=1)) == ACTION_BUY_PACK


def test_using_a_consumable_costs_nothing():
    """A consumable leaving inventory with money UNCHANGED is a use, not a sale."""
    assert _type(S(state="SHOP", money=10, cons=2),
                 S(state="SHOP", money=10, cons=1)) == ACTION_USE_CONSUMABLE


def test_leaving_the_shop():
    assert _type(S(state="SHOP"), S(state="BLIND_SELECT")) == ACTION_END_SHOP


# --------------------------------------------------------------------------
# THE governing rule: never guess
# --------------------------------------------------------------------------

def test_no_change_yields_no_action():
    """Most polls land mid-animation with nothing happening. Those must not
    become actions, or the demo buffer fills with phantom no-ops."""
    assert infer_action(S(), S()) is None


def test_missing_states_yield_no_action():
    assert infer_action(None, S()) is None
    assert infer_action(S(), None) is None
    assert infer_action({}, {}) is None


def test_an_unrecognised_change_is_dropped_not_guessed():
    """A change the encoding cannot express must return None. Silently guessing
    would look like coverage while labelling transitions wrongly."""
    assert infer_action(S(shop=2), S(shop=5)) is None


# --------------------------------------------------------------------------
# Coverage accounting — the number that decides whether the spike is viable
# --------------------------------------------------------------------------

def test_coverage_counts_only_real_changes():
    """Idle polls must not inflate coverage. If they counted as covered, a
    recorder that inferred nothing would still report ~100%."""
    r = Recorder(out_dir="unused")
    r.observe(S(hands=3))
    r.observe(S(hands=3))          # idle
    r.observe(S(hands=3))          # idle
    assert r.changes == 0 and r.coverage == 0.0

    r.observe(S(hands=2))          # a play
    assert r.changes == 1 and r.inferred == 1 and r.coverage == 1.0

    r.observe(S(hands=2, shop=5))  # a change it cannot label
    assert r.changes == 2 and r.inferred == 1 and r.coverage == 0.5


def test_recorder_never_emits_an_unlabelled_event():
    r = Recorder(out_dir="unused")
    for st in (S(hands=3), S(hands=3, shop=5), S(hands=2, shop=5)):
        r.observe(st)
    assert all(e["action"] is not None for e in r.events)
    assert len(r.events) == r.inferred


# --------------------------------------------------------------------------
# dec-096: the transition filter — coverage must measure DECISIONS
# --------------------------------------------------------------------------

def test_engine_transitions_are_not_decisions():
    """The first coverage figure (47%) divided by every state change, putting the
    game's own state machine in the denominator. A diagnostic showed the misses
    were dominated by scoring animations, round eval, draws and counter refills —
    none of them decisions."""
    from human_record import is_decision

    # scoring animation: chips move, nothing else
    assert not is_decision(S(chips=0), S(chips=5000))
    # engine states the game enters by itself
    for st in ("HAND_PLAYED", "DRAW_TO_HAND", "ROUND_EVAL", "GAME_OVER"):
        assert not is_decision(S(state="SELECTING_HAND"), S(state=st))
    # round boundary REFILLS counters — no player action increases them
    assert not is_decision(S(hands=0, discards=0), S(hands=4, discards=3, money=15))


def test_real_choices_are_still_decisions():
    """The filter must not excuse genuine actions — that is how a bad recorder
    would score well. Leaving the shop is a choice even though only `state`
    changes."""
    from human_record import is_decision

    assert is_decision(S(state="SHOP"), S(state="BLIND_SELECT"))
    assert is_decision(S(hands=3), S(hands=2))                      # play
    assert is_decision(S(state="SHOP", money=12, jokers=1),
                       S(state="SHOP", money=6, jokers=2))          # buy


def test_coverage_excludes_engine_noise_but_keeps_real_misses():
    """Coverage = labelled / decisions. Engine transitions must not dilute it,
    and an unlabelled DECISION must still count against it."""
    r = Recorder(out_dir="unused")
    r.observe(S(hands=3))
    r.observe(S(hands=3, chips=4000))            # engine: scoring animation
    r.observe(S(hands=2, chips=4000))            # decision: a play
    assert r.decisions == 1 and r.inferred == 1 and r.coverage == 1.0

    r.observe(S(hands=2, chips=4000, shop=5))    # decision we cannot label
    assert r.decisions == 2 and r.inferred == 1 and r.coverage == 0.5
    assert r.missed and "n_shop" in r.missed[-1]["fields"]
