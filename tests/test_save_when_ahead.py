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


# ── dec-079: ante-scaled headroom (the ante-4/5 plateau fix) ────────────────
# Measured on 50k real blind snapshots, the flat AHEAD_BUFFER=1.0 fired the
# SAVE gate in 65% of ante-2 shops — suppressing buying in the exact window
# where the xmult engine must be assembled, against a target curve that
# DOUBLES per ante. These pin the shape of the fix.


def test_early_bonus_off_is_flat_dec068_behaviour():
    """Default (bonus=0) must reproduce dec-068 exactly — the A/B control."""
    ae = ActionExecutor()
    ae.AHEAD_BUFFER_EARLY_BONUS = 0.0
    for ante in (1, 2, 3, 4, 8):
        assert ae._ahead_buffer_for(ante) == ae.AHEAD_BUFFER


def test_early_bonus_demands_more_headroom_earlier():
    """With the bonus on, earlier antes demand strictly MORE headroom, so the
    agent keeps building instead of banking into a wall."""
    ae = ActionExecutor()
    ae.AHEAD_BUFFER = 1.0
    ae.AHEAD_BUFFER_EARLY_BONUS = 1.0
    reqs = [ae._ahead_buffer_for(a) for a in (1, 2, 3, 4)]
    assert reqs == sorted(reqs, reverse=True)      # monotone decreasing
    assert reqs[0] > reqs[-1]                      # ante 1 strictly hardest


def test_late_antes_unchanged_by_early_bonus():
    """Antes at/after EARLY_BUILD_UNTIL_ANTE keep the dec-060 boss-spike
    banking behaviour — the fix must not touch the late game."""
    ae = ActionExecutor()
    ae.AHEAD_BUFFER = 1.0
    ae.AHEAD_BUFFER_EARLY_BONUS = 1.0
    for ante in (4, 5, 6, 7, 8):
        assert ae._ahead_buffer_for(ante) == ae.AHEAD_BUFFER
