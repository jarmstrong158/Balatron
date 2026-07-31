"""dec-095: score jokers by what they REALLY contribute, not the schema flag.

dec-093's forced-engine arms could not distinguish a working engine from a dead
one, because `_tier` treated any joker with `xmult=True` as tier 5 — which is
true of Blackboard (needs an all-black hand), Loyalty Card (every 6th hand) and
Cavendish (a 1-in-1000 lottery) exactly as it is of a real engine. Forcing that
nominal share from 12% to 19% made outcomes WORSE, and the engine hypothesis
stayed untestable as a result.

These tests pin the two properties the measurement depends on: that asking for a
breakdown NEVER changes the score (or the instrument would corrupt the thing it
measures), and that the breakdown actually reconciles to the totals.
"""
import pytest

from environment.hand_eval import classify_hand, compute_joker_scoring


def _card(rank, suit, val):
    return {"value": {"rank": rank, "suit": suit, "value": val}, "modifier": {}}


def _jk(key):
    return {"joker_key": key, "key": key, "modifier": {}}


BLACK_FLUSH = [_card("A", "Spades", 14), _card("K", "Spades", 13),
               _card("Q", "Spades", 12), _card("J", "Spades", 11),
               _card("9", "Spades", 9)]
RED_FLUSH = [_card("A", "Hearts", 14), _card("K", "Hearts", 13),
             _card("Q", "Hearts", 12), _card("J", "Hearts", 11),
             _card("9", "Hearts", 9)]

GS = {"hands": {"Flush": {"chips": 35, "mult": 4},
                "High Card": {"chips": 5, "mult": 1}},
      "round": {"discards_left": 3, "hands_left": 3}}


def _score(cards, jokers, breakdown=None):
    ht, si = classify_hand(cards)
    return compute_joker_scoring(ht, cards, list(si or []), jokers, GS,
                                 base_mult=4.0, breakdown=breakdown)


# --------------------------------------------------------------------------
# The instrument must not perturb what it measures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("keys", [
    [], ["j_joker"], ["j_banner"], ["j_blackboard"],
    ["j_joker", "j_blackboard", "j_cavendish", "j_banner"],
])
def test_breakdown_never_changes_the_score(keys):
    jokers = [_jk(k) for k in keys]
    assert _score(BLACK_FLUSH, jokers) == _score(BLACK_FLUSH, jokers, [])


def test_default_path_still_works_without_the_parameter():
    """REGRESSION. The first version took its snapshot from `bonus_mult`, which
    does not exist yet at that point in compute_joker_scoring (it is derived from
    during_add_mult + after_add_mult AFTER the loop). Because the snapshot runs
    unconditionally, that raised UnboundLocalError on the DEFAULT path and broke
    7 tests — an instrument that breaks scoring for every caller that never asked
    for it. The real accumulators are used now."""
    chips, mult, xmult = _score(BLACK_FLUSH, [_jk("j_joker")])
    assert mult > 0 and xmult >= 1.0


# --------------------------------------------------------------------------
# The breakdown must reconcile to the totals
# --------------------------------------------------------------------------

def test_breakdown_reconciles_to_the_totals():
    jokers = [_jk(k) for k in ("j_joker", "j_blackboard", "j_cavendish", "j_banner")]
    bd = []
    chips, mult, xmult = _score(BLACK_FLUSH, jokers, bd)
    assert sum(r["chips"] for r in bd) == pytest.approx(chips, abs=1e-6)
    assert sum(r["mult"] for r in bd) == pytest.approx(mult, abs=1e-6)
    prod = 1.0
    for r in bd:
        prod *= r["xmult"]
    assert prod == pytest.approx(xmult, rel=1e-6)


def test_every_held_joker_gets_a_row_even_when_it_does_nothing():
    """'Held and contributed nothing' IS the measurement — a joker that fires for
    zero must appear as a zero row, never be omitted, or the denominator is wrong
    and every fire-rate is overstated."""
    jokers = [_jk("j_joker"), _jk("j_banner")]
    bd = []
    _score(BLACK_FLUSH, jokers, bd)
    assert {r["joker"] for r in bd} == {"Joker", "Banner"}


# --------------------------------------------------------------------------
# THE point: conditional jokers must be distinguishable from real engines
# --------------------------------------------------------------------------

def test_blackboard_is_dead_when_its_condition_fails():
    """The exact discrimination dec-093 could not make.

    Blackboard is "X3 Mult if all cards HELD IN HAND are black", so the condition
    depends on the cards still held, NOT on the cards played. The first version of
    this test played an all-red flush with nothing held and expected x1; it got
    x3, and the code was right — with an empty hand the condition is vacuously
    true, which is how real Balatro behaves too. The discriminating case is
    holding a RED card.

    The schema flag is identical in all three situations, which is precisely why
    the flag-based _tier could not tell a live engine from a dead one.
    """
    def _bb(held):
        gs = dict(GS)
        gs["hand"] = {"cards": held}
        ht, si = classify_hand(BLACK_FLUSH)
        bd = []
        compute_joker_scoring(ht, BLACK_FLUSH, list(si or []),
                              [_jk("j_blackboard")], gs, base_mult=4.0,
                              breakdown=bd)
        return next(r for r in bd if r["joker"] == "Blackboard")["xmult"]

    assert _bb([]) == pytest.approx(3.0), "empty hand: vacuously true, fires"
    assert _bb([_card("2", "Clubs", 2)]) == pytest.approx(3.0), "all black: fires"
    assert _bb([_card("2", "Hearts", 2)]) == pytest.approx(1.0), \
        "a held RED card must make Blackboard read as DEAD — the whole point"

    # and confirm the schema flag cannot tell those apart
    import engine_forcing as ef
    assert ef._tier("Blackboard") == 5, \
        "if this changes, the motivating limitation is gone — revisit dec-095"


# --------------------------------------------------------------------------
# dec-095 follow-up: scaling jokers must not read as dead
# --------------------------------------------------------------------------

def _play_state():
    """Local state builder — `_state()` lives in test_play_quality.py and is not
    importable here."""
    return {
        "ante_num": 4,
        "hand": {"cards": list(BLACK_FLUSH)},
        "jokers": {"cards": []},
        "hands": {"Flush": {"chips": 35, "mult": 4},
                  "High Card": {"chips": 5, "mult": 1}},
        "round": {"hands_left": 3, "discards_left": 2, "chips": 0},
        "blinds": {},
    }


def test_scaling_values_are_injected_and_do_not_mutate_raw_state(tmp_path, monkeypatch):
    """The instrument's own blind spot, fixed.

    Scaling jokers keep their accumulated multiplier in `_scaled_value`, held by
    the ScalingTracker rather than in raw_state. Measuring un-injected jokers made
    every scaler read x1.0 however large it had grown — which was briefly mistaken
    for "the evaluator cannot see the scaling archetype" and nearly produced a
    duplicate implementation inside hand_eval that would have double-applied the
    multiplier on the agent's live scoring path.

    Also pins that injection happens on COPIES: raw_state is shared with the
    trainer and a logging path must never mutate it.
    """
    import json

    import play_quality

    monkeypatch.chdir(tmp_path)

    st = _play_state()
    st["jokers"] = {"cards": [{"joker_key": "j_vampire", "key": "j_vampire",
                               "id": 7, "modifier": {}}]}

    class FakeGame:
        def inject_scaling_values(self, joker_cards):
            for c in joker_cards:
                c["_scaled_value"] = 2.5

    play_quality.log_play(st, [0, 1, 2, 3, 4], game=FakeGame())
    rows = [json.loads(x) for x in
            (tmp_path / "logs" / "play_quality.jsonl").read_text(
                encoding="utf-8").splitlines() if x.strip()]
    vamp = next(r for r in rows[-1]["realized"] if r["joker"] == "Vampire")
    assert vamp["xmult"] == 2.5, "scaling joker still reading as dead"

    # raw_state must be untouched
    assert "_scaled_value" not in st["jokers"]["cards"][0]


def test_missing_game_still_logs_just_under_counted(tmp_path, monkeypatch):
    """`game` is optional — without it the row must still be written (with the
    scaler under-counted), never dropped or raised."""
    import json

    import play_quality

    monkeypatch.chdir(tmp_path)
    st = _play_state()
    st["jokers"] = {"cards": [{"joker_key": "j_vampire", "key": "j_vampire",
                               "id": 7, "modifier": {}}]}
    play_quality.log_play(st, [0, 1, 2, 3, 4])
    rows = [json.loads(x) for x in
            (tmp_path / "logs" / "play_quality.jsonl").read_text(
                encoding="utf-8").splitlines() if x.strip()]
    assert rows and rows[-1]["realized"] is not None
