"""dec-069's build<->play alignment signal: `most_played` must reflect real play.

The original loop started `most_n = -1`, so before ANY hand had been played
`0 > -1` made the first hand in dict order the reported "most played", and strict
`>` meant no other zero-count hand could displace it. Every run therefore opened
by naming whichever hand the game lists first.

That is not cosmetic. It polluted 31.8% of build_progression rows — 47,472 of
them claimed "most played: Flush House" when Flush House had never been played
once across 31,041 ground-truth hands in play_quality — and inflated measured
misalignment from its true 49.6% to 65.6%.
"""
from training.train import _committed_hand_signals


def H(**played):
    """`hands` table; Flush House FIRST to reproduce the game's own ordering."""
    names = ["Flush House", "Flush", "Two Pair", "Straight", "Pair", "Full House"]
    return {n: {"played": played.get(n.replace(" ", "_"), 0), "level": 1}
            for n in names}


def test_nothing_played_yields_no_winner():
    """THE bug. With no hands played there is no most-played hand, and dict order
    must not be reported as if it were data."""
    got = _committed_hand_signals("Flush", H(), [])
    assert got["most_played"] == "", \
        f"fabricated a winner from dict order: {got['most_played']!r}"


def test_the_actual_most_played_hand_wins():
    got = _committed_hand_signals("Flush", H(Flush=3, Two_Pair=7), [])
    assert got["most_played"] == "Two Pair"


def test_first_in_dict_order_does_not_beat_a_real_count():
    """Flush House is listed first; a hand genuinely played once must still win."""
    got = _committed_hand_signals("Flush", H(Pair=1), [])
    assert got["most_played"] == "Pair"


def test_play_share_is_zero_when_nothing_played():
    got = _committed_hand_signals("Flush", H(), [])
    assert got["play_share"] == 0.0


def test_play_share_tracks_the_committed_hand():
    got = _committed_hand_signals("Flush", H(Flush=3, Two_Pair=1), [])
    assert got["play_share"] == 0.75


# --------------------------------------------------------------------------
# dec-097: the alignment carriers must not go stale across runs
# --------------------------------------------------------------------------

def test_env_session_starts_with_empty_alignment_carriers():
    """They are written only when build_progression logs. If a blind ends before
    that happens, an uninitialised or stale value would attach the PREVIOUS run's
    committed/played hand to this run's blind_results row — wrong data that looks
    perfectly valid."""
    from training.env_session import EnvSession

    e = EnvSession.__new__(EnvSession)
    EnvSession.__init__(e, env_id=0, port=12346, phase=1)
    assert e.cur_committed_ht == ""
    assert e.cur_played_ht == ""
    assert e.cur_committed_level == 0
    assert e.cur_played_level == 0


def test_reset_run_state_clears_them():
    """Pins that _reset_run_state covers the new fields, so they cannot survive
    into the next run."""
    import inspect

    from training.train import Trainer

    src = inspect.getsource(Trainer._reset_run_state)
    for f in ("cur_committed_ht", "cur_played_ht",
              "cur_committed_level", "cur_played_level"):
        assert f in src, f"{f} is not cleared on run reset — it will go stale"
