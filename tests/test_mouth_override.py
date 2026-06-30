"""dec-052: The Mouth setup-override guard (mouth_should_dig).

Covers the safety short-circuits — the override must NEVER fire when the boss
isn't The Mouth, when a hand type is already locked, or when no discards remain.
(The positive dig case depends on find_best_hands/target_hand_type over real
cards and is exercised live.)"""
from environment.hand_eval import mouth_should_dig, needle_should_dig


def _needle_state(boss="The Needle", discards=3, target=1000.0, chips=0.0):
    return {
        "blinds": {"boss": {"status": "CURRENT", "name": boss, "score": target}},
        "round": {"discards_left": discards, "chips": chips},
        "hands": {},
    }


def test_needle_not_boss_never_digs():
    assert needle_should_dig([], [], _needle_state(boss="The Wall")) is False


def test_needle_no_discards_does_not_dig():
    assert needle_should_dig([], [], _needle_state(discards=0)) is False


def test_needle_already_cleared_does_not_dig():
    # round chips already >= target -> nothing to dig for
    assert needle_should_dig([], [], _needle_state(target=1000.0, chips=1000.0)) is False


def _state(boss, discards=3, played=None):
    return {
        "blinds": {"boss": {"status": "CURRENT", "name": boss}},
        "round": {"discards_left": discards},
        "hands": played or {},
    }


def test_not_mouth_never_digs():
    assert mouth_should_dig([], [], _state("The Wall")) is False
    assert mouth_should_dig([], [], _state("The Eye")) is False


def test_mouth_already_locked_does_not_dig():
    # a hand type already played this round -> type is locked -> play within it
    st = _state("The Mouth", played={"Pair": {"round_played": 1}})
    assert mouth_should_dig([], [], st) is False


def test_mouth_no_discards_does_not_dig():
    st = _state("The Mouth", discards=0)
    assert mouth_should_dig([], [], st) is False
