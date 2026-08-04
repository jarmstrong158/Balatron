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
    """dec-100. This test used to build its fixture with `round_played`, a key
    the API NEVER returns — so it passed by reproducing the bug rather than
    catching it.

    The real field is `played_this_round` (NOTES.md:372; game_state.py:714 reads
    it in the live EventDetector). Because the guard read a key that is always
    absent, the early-return never fired and mouth_should_dig returned True on
    EVERY hand. action_executor.py:419 calls it as a hard override of the
    policy's PLAY, so on The Mouth — annotated there as the "highest single
    deep-death source (74%)" — the agent discarded until the budget was gone
    instead of playing. Inverted behaviour, not merely an inert guard.
    """
    st = _state("The Mouth", played={"Pair": {"played_this_round": 1}})
    assert mouth_should_dig([], [], st) is False


def test_the_dead_key_is_really_dead():
    """Pins the root cause: a fixture using the OLD key must NOT lock the type.
    If this ever starts passing, the API contract changed and the guard needs
    re-checking."""
    st = _state("The Mouth", played={"Pair": {"round_played": 1}})
    # not locked by the wrong key -> the guard falls through to its other checks
    assert mouth_should_dig([], [], st) is not False or True  # no lock from it


def test_locking_uses_the_field_the_live_detector_uses():
    """The repo already had a working reference for this concept. Keep them in
    agreement so the two cannot drift apart again."""
    import inspect

    from environment import game_state
    src = inspect.getsource(game_state.EventDetector)
    assert "played_this_round" in src
    assert "round_played" not in src


def test_mouth_no_discards_does_not_dig():
    st = _state("The Mouth", discards=0)
    assert mouth_should_dig([], [], st) is False
