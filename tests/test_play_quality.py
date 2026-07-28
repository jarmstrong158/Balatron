"""dec-091: per-hand play-quality logging — did the agent play the BEST hand?

This is the last unexamined decision surface. Every build-side lever is closed
(dec-085 evaluation, dec-079/081 access, dec-090 no build feature predicts
clearing at antes 4-6, dec-088 credit horizon), and discard usage was measured
sensible. Chips only come from played hands, and nobody has ever checked whether
the chosen hand was the best available.

These tests pin the two things the measurement depends on: that the baseline is
computed under the SAME boss debuffs the agent played under (otherwise the gap is
manufactured), and that the logger can never break a run.
"""
import json

import play_quality


def _card(rank, suit, val):
    return {"value": {"rank": rank, "suit": suit, "value": val}, "modifier": {}}


HAND = [_card("A", "Spades", 14), _card("K", "Spades", 13),
        _card("Q", "Spades", 12), _card("J", "Spades", 11),
        _card("9", "Spades", 9), _card("2", "Hearts", 2),
        _card("3", "Hearts", 3), _card("4", "Clubs", 4)]


def _state(boss=None):
    blinds = {}
    if boss:
        blinds = {"boss": {"name": boss, "status": "CURRENT"}}
    return {
        "ante_num": 4, "hand": {"cards": HAND}, "jokers": {"cards": []},
        "hands": {"Flush": {"chips": 35, "mult": 4},
                  "Pair": {"chips": 10, "mult": 2},
                  "High Card": {"chips": 5, "mult": 1},
                  "Straight": {"chips": 30, "mult": 4},
                  "Straight Flush": {"chips": 100, "mult": 8}},
        "round": {"hands_left": 3, "discards_left": 2, "chips": 0},
        "blinds": blinds,
    }


def test_debuff_map_matches_hand_eval_source():
    """play_quality mirrors a dict that is LOCAL to
    hand_eval._plan_optimal_action_inner and therefore cannot be imported. If
    the source ever changes, this fails instead of the two silently drifting and
    skewing every capture ratio.

    (The first version of this test guessed the wrong host function and passed
    vacuously on a source string that never contained the map — hence the
    explicit non-empty assertion below.)"""
    import inspect

    from environment import hand_eval
    src = inspect.getsource(hand_eval._plan_optimal_action_inner)
    assert "BOSS_SUIT_DEBUFF" in src, "wrong host function — test would be vacuous"
    for boss, suit in play_quality.BOSS_SUIT_DEBUFF.items():
        assert f'"{boss}": "{suit}"' in src, f"{boss} drifted from hand_eval"
    assert f'boss_name == "{play_quality.BOSS_FACE_DEBUFF}"' in src


def test_detects_current_boss_only():
    assert play_quality.current_boss(_state("The Goad")) == "The Goad"
    assert play_quality.current_boss(_state()) == ""
    # a boss that is merely UPCOMING is not being played
    st = _state()
    st["blinds"] = {"boss": {"name": "The Goad", "status": "UPCOMING"}}
    assert play_quality.current_boss(st) == ""


def test_debuffs_resolved_for_current_blind():
    assert play_quality.debuffs_for(_state("The Goad")) == ("Spades", False)
    assert play_quality.debuffs_for(_state("The Head")) == ("Hearts", False)
    assert play_quality.debuffs_for(_state("The Plant")) == (None, True)
    assert play_quality.debuffs_for(_state()) == (None, False)


def _run(tmp_path, state, played, monkeypatch):
    monkeypatch.chdir(tmp_path)
    play_quality.log_play(state, played, env_id=0, global_step=1)
    p = tmp_path / "logs" / "play_quality.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-1] if rows else None


def test_logs_capture_ratio_for_a_played_hand(tmp_path, monkeypatch):
    row = _run(tmp_path, _state(), [0, 1, 2, 3, 4], monkeypatch)   # the flush
    assert row is not None
    assert row["played_n"] == 5
    assert row["best_score"] > 0
    assert 0.0 <= row["capture"] <= 1.0


def test_a_deliberately_bad_play_scores_below_the_best(tmp_path, monkeypatch):
    """THE measurement: playing one low card must capture less than the flush."""
    good = _run(tmp_path, _state(), [0, 1, 2, 3, 4], monkeypatch)
    bad = _run(tmp_path, _state(), [5], monkeypatch)              # lone 2 of Hearts
    assert bad["capture"] < good["capture"], (bad["capture"], good["capture"])
    assert bad["same_cards"] is False


def test_baseline_uses_the_same_debuffs_the_agent_played_under(tmp_path, monkeypatch):
    """Fairness guard. Under The Goad every Spade is dead, so the spade flush is
    no longer the best hand — the baseline must reflect that, or the agent gets
    charged for a hand it could not have scored."""
    plain = _run(tmp_path, _state(), [0, 1, 2, 3, 4], monkeypatch)
    goaded = _run(tmp_path, _state("The Goad"), [0, 1, 2, 3, 4], monkeypatch)
    assert goaded["debuffed_suit"] == "Spades"
    assert goaded["best_score"] < plain["best_score"], \
        "debuffed baseline should be lower — otherwise the gap is manufactured"


def test_never_raises_on_garbage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ({}, {"hand": {}}, {"hand": {"cards": HAND}}, None):
        play_quality.log_play(bad or {}, [0, 1])          # must not raise
    play_quality.log_play(_state(), [99, -1])             # out-of-range indices
    play_quality.log_play(_state(), None)
    play_quality.log_play(_state(), [])


def test_writes_nothing_when_there_is_no_play(tmp_path, monkeypatch):
    assert _run(tmp_path, _state(), [], monkeypatch) is None


def test_capture_can_never_exceed_one(tmp_path, monkeypatch):
    """INVARIANT that would have caught the first version's bug.

    `capture` is the agent's score over the BEST AVAILABLE score, so it cannot
    exceed 1.0. The initial implementation discarded classify_hand's
    scoring_indices and passed range(len(cards)), crediting every played card
    rather than only the ones that score — a Pair played among 5 cards was
    scored as if all 5 counted. That produced mean capture 1.061 (1.243 at
    ante 4) on live data, which is impossible and is how the bug surfaced.
    """
    for played in ([0, 1, 2, 3, 4], [0, 1], [5], [0, 5, 6], [0, 1, 2]):
        for boss in (None, "The Goad", "The Plant"):
            row = _run(tmp_path, _state(boss), played, monkeypatch)
            if row and row["capture"] is not None:
                assert row["capture"] <= 1.0 + 1e-6, (
                    f"capture {row['capture']} > 1 for {played} vs boss {boss} "
                    f"— the baseline is not actually the best available")


def test_playing_the_best_hand_captures_exactly_one(tmp_path, monkeypatch):
    """When the agent plays the identified best hand, capture must be 1.0 —
    both sides must compute the score the SAME way."""
    monkeypatch.chdir(tmp_path)
    from environment.hand_eval import find_best_hands
    best = find_best_hands(HAND, [], _state(), top_n=1)
    row = _run(tmp_path, _state(), list(best[0]["card_indices"]), monkeypatch)
    assert row["same_cards"] is True
    assert abs(row["capture"] - 1.0) < 1e-6, row["capture"]
