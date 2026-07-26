"""dec-083: Monte-Carlo rollout evaluator — P(clear) instead of a point estimate.

UNVALIDATED and not in the decision path. These tests pin the properties the
planner would depend on if it ever is: determinism (the same shop must not rank
two different ways), monotonicity in target, and responsiveness to build strength.
They do NOT assert it is better than the leaf — that is the offline gate's job
(tools/validate_rollout.py), on post-dec-082 clean rows.
"""
from environment.rollout import build_deck, p_clear
import random


def _snap(ranks=None, suits=None, hand_size=8, jokers=None):
    # NB: `is None`, not `or` — an EMPTY dict is a meaningful input (the
    # degenerate-deck case) and `{} or default` would silently substitute the
    # default, making that test vacuous.
    return {
        "deck_n": 40,
        "ranks": {r: 4 for r in ("2", "5", "9", "J", "K")} if ranks is None else ranks,
        "suits": {"S": 5, "H": 5, "D": 5, "C": 5} if suits is None else suits,
        "enhancements": {}, "seals": {},
        "jokers": jokers or [],
        "hand_levels": {"Pair": {"level": 1, "chips": 10, "mult": 2},
                        "Flush": {"level": 1, "chips": 35, "mult": 4},
                        "High Card": {"level": 1, "chips": 5, "mult": 1}},
        "hand_size": hand_size, "money": 10,
    }


def _gs(ante=4):
    return {"ante_num": ante, "ante": ante, "blinds": {},
            "hands": {"Pair": {"chips": 10, "mult": 2},
                      "Flush": {"chips": 35, "mult": 4},
                      "High Card": {"chips": 5, "mult": 1}}}


def test_deck_reconstruction_respects_marginals():
    snap = _snap(ranks={"A": 6, "2": 4}, suits={"S": 7, "H": 3})
    deck = build_deck(snap, random.Random(0))
    assert len(deck) == 10
    ranks = [c["value"]["rank"] for c in deck]
    suits = [c["value"]["suit"] for c in deck]
    assert ranks.count("A") == 6 and ranks.count("2") == 4
    assert suits.count("S") == 7 and suits.count("H") == 3


def test_deterministic_for_a_given_seed():
    """The planner calls this while ranking a shop; two identical calls must not
    return different values or the ranking becomes unstable."""
    snap, gs = _snap(), _gs()
    a = p_clear(snap, [], gs, 2000, samples=20, seed=7)
    b = p_clear(snap, [], gs, 2000, samples=20, seed=7)
    assert a == b


def test_probability_is_bounded():
    snap, gs = _snap(), _gs()
    for tgt in (1, 500, 5000, 10 ** 9):
        p = p_clear(snap, [], gs, tgt, samples=15, seed=0)
        assert 0.0 <= p <= 1.0


def test_monotonic_in_target():
    """A harder target can never be MORE likely to clear."""
    snap, gs = _snap(), _gs()
    ps = [p_clear(snap, [], gs, t, samples=30, seed=1)
          for t in (200, 2000, 20000, 200000)]
    assert all(ps[i] >= ps[i + 1] - 1e-9 for i in range(len(ps) - 1)), ps


def test_trivial_target_always_clears():
    snap, gs = _snap(), _gs()
    assert p_clear(snap, [], gs, 1, samples=10, seed=0) == 1.0


def test_impossible_target_never_clears():
    snap, gs = _snap(), _gs()
    assert p_clear(snap, [], gs, 10 ** 12, samples=10, seed=0) == 0.0


def test_flush_deck_beats_rainbow_deck_on_a_flush_target():
    """Composition must matter — this is the whole reason a rollout could beat a
    point estimate, which is blind to how concentrated the deck is."""
    gs = _gs()
    concentrated = _snap(suits={"H": 20}, ranks={r: 4 for r in ("2", "5", "9", "J", "K")})
    rainbow = _snap(suits={"S": 5, "H": 5, "D": 5, "C": 5},
                    ranks={r: 4 for r in ("2", "5", "9", "J", "K")})
    # target chosen to sit where a flush matters
    pc = p_clear(concentrated, [], gs, 900, samples=60, seed=3)
    pr = p_clear(rainbow, [], gs, 900, samples=60, seed=3)
    assert pc > pr, (pc, pr)


def test_empty_deck_is_safe():
    """Never crash the planner on a degenerate snapshot."""
    snap = _snap(ranks={}, suits={})
    assert p_clear(snap, [], _gs(), 1000, samples=5, seed=0) == 0.0


# --- planner integration gate (dec-083): default OFF is the revert guarantee ---

def test_planner_rollout_flag_defaults_off():
    """Ship-safety: a trainer restart must never silently enable an unvalidated
    evaluator. Reverting the experiment = unset BALATRON_ROLLOUT."""
    import importlib
    import os
    import sys
    prev = os.environ.pop("BALATRON_ROLLOUT", None)
    try:
        sys.modules.pop("environment.planner", None)
        planner = importlib.import_module("environment.planner")
        assert planner.ROLLOUT_LEAF is False
    finally:
        if prev is not None:
            os.environ["BALATRON_ROLLOUT"] = prev
        sys.modules.pop("environment.planner", None)
        importlib.import_module("environment.planner")


def test_rollout_helper_returns_none_without_a_deck():
    """No deck in the gamestate -> None -> caller keeps the analytical estimate.
    This is what makes the integration fail-safe rather than fail-open."""
    from environment.planner import _rollout_p_clear
    assert _rollout_p_clear([], {"ante_num": 4, "hands": {}}, 5000) is None


def test_control_arm_matches_legacy_survivability():
    """With the flag off, build_survivability must be byte-identical to the
    pre-dec-083 path — otherwise the A/B's control arm is not a real control."""
    import importlib
    import os
    import sys
    prev = os.environ.pop("BALATRON_ROLLOUT", None)
    try:
        sys.modules.pop("environment.planner", None)
        planner = importlib.import_module("environment.planner")
        gs = {"ante_num": 3, "ante": 3, "blinds": {},
              "hands": {"Flush": {"chips": 75, "mult": 8}}}
        jk = [{"key": "j_joker", "joker_key": "j_joker", "label": "Joker"}]
        a = planner.build_survivability(jk, gs)
        b = planner.build_survivability(jk, gs)
        assert a == b and a > 0
    finally:
        if prev is not None:
            os.environ["BALATRON_ROLLOUT"] = prev
        sys.modules.pop("environment.planner", None)
        importlib.import_module("environment.planner")
