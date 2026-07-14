"""dec-068: don't overbuild. When the score-only build already clears the
upcoming ante comfortably, marginal jokers are redundant — hold and bank
interest instead of buying near-term power."""
from training.action_executor import ActionExecutor


def _jk(k):
    return {"key": k}


def _state(chips, mult, ante=3):
    return {"ante_num": ante,
            "hands": {"Flush": {"chips": chips, "mult": mult}},
            "cards": {"cards": [{} for _ in range(40)]},
            "round": {"hands_left": 4}, "money": 20, "blinds": {}}


def test_ahead_build_is_already_clearing():
    ae = ActionExecutor()
    strong = _state(400, 80)          # projects well past ante 3
    assert ae._already_clearing([_jk("j_joker")], strong) is True


def test_behind_build_is_not_clearing():
    ae = ActionExecutor()
    weak = _state(10, 2)              # dies ~this ante
    assert ae._already_clearing([_jk("j_joker")], weak) is False


def test_buffer_boundary():
    """A build that just barely clears the current ante (no headroom) is NOT
    'already clearing' — it should keep building, not start hoarding."""
    ae = ActionExecutor()
    from environment.planner import _score_survivability
    # find a build whose survivability sits just above the current ante
    marginal = _state(60, 8, ante=3)
    surv = _score_survivability([_jk("j_joker")], marginal)
    ahead = ae._already_clearing([_jk("j_joker")], marginal)
    # consistency: flagged ahead iff surv is >= AHEAD_BUFFER past the ante
    assert ahead == ((surv - 3) >= ae.AHEAD_BUFFER)
