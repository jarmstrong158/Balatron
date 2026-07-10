"""Fix (INVALID_STATE desync): the execute_action retry loop must not burn
retries firing a stale action into a state the game has already left.

The trainer decodes an action from the state snapshot taken at the top of the
rollout iteration, but the game can leave that state before the send lands
(animation / blind / run transition, or an auto-action inside the poll loop).
On INVALID_STATE the loop now re-reads the LIVE state and aborts immediately
when the game is no longer in a state the method accepts, instead of retrying
into a guaranteed reject. This pins the error-message parser that drives that
decision (a parse miss must fall back to the plain retry, never a wrong abort).
"""
from training.train import _parse_required_states

# The exact string the game client raises (RuntimeError(f"BalatroBot error: {data['error']}")).
PLAY_ERR = (
    "BalatroBot error: {'data': {'name': 'INVALID_STATE'}, 'code': -32002, "
    "'message': \"Method 'play' requires one of these states: SELECTING_HAND\"}"
)
NEXT_ROUND_ERR = (
    "BalatroBot error: {'data': {'name': 'INVALID_STATE'}, 'code': -32002, "
    "'message': \"Method 'next_round' requires one of these states: SHOP\"}"
)


def test_parses_single_required_state():
    assert _parse_required_states(PLAY_ERR) == {"SELECTING_HAND"}
    assert _parse_required_states(NEXT_ROUND_ERR) == {"SHOP"}


def test_parse_miss_returns_empty_so_caller_falls_back_to_retry():
    # An unparseable message must yield an empty set — the guard treats that as
    # "don't know, keep the existing retry behavior" (never a spurious abort).
    assert _parse_required_states("some unrelated error") == set()
    assert _parse_required_states("buttons not ready yet") == set()
    assert _parse_required_states("") == set()


def test_abort_decision_matches_live_state():
    """The guard aborts iff the parsed required set is known AND the live state
    is outside it (the exact predicate used in the retry loop)."""
    required = _parse_required_states(PLAY_ERR)  # {"SELECTING_HAND"}

    def should_abort(live):
        return bool(required) and bool(live) and live not in required

    assert should_abort("SHOP") is True          # stale play into shop -> abort
    assert should_abort("BLIND_SELECT") is True   # stale play into blind -> abort
    assert should_abort("SELECTING_HAND") is False  # transient blip -> keep retrying
    assert should_abort("") is False              # unknown live state -> keep retrying
