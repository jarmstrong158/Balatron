"""dec-059: boss-aware planner — the dec-057 audit's top lever.

The planner treated every boss as a generic 2x target; boss identity (visible in
state) now gates the immediate ante at its REAL difficulty."""
from environment.planner import (
    upcoming_boss, boss_difficulty, build_survivability, BOSS_DIFFICULTY,
)


def _gs(boss=None, status="UPCOMING", ante=5):
    gs = {
        "ante_num": ante,
        "hands": {"Pair": {"chips": 60, "mult": 6, "played": 10}},
        "blinds": {},
    }
    if boss:
        gs["blinds"] = {"boss": {"status": status, "name": boss}}
    return gs


def test_upcoming_boss_parsing():
    assert upcoming_boss(_gs("The Wall")) == "The Wall"
    assert upcoming_boss(_gs("The Wall", status="CURRENT")) == "The Wall"
    # defeated boss -> next ante's boss unknown -> no adjustment
    assert upcoming_boss(_gs("The Wall", status="DEFEATED")) == ""
    assert upcoming_boss(_gs(None)) == ""
    assert upcoming_boss({}) == ""


def test_boss_difficulty_table():
    assert boss_difficulty("The Wall") == 2.0        # 4x base = 2x a normal boss
    assert boss_difficulty("The Needle") == 3.0      # 1 hand vs HANDS_PER_BLIND=3
    assert boss_difficulty("The Mouth") == 1.0       # handled by the dig override
    assert boss_difficulty("") == 1.0                # unknown -> average boss
    assert all(v >= 1.0 for v in BOSS_DIFFICULTY.values())


def test_survivability_drops_for_hard_immediate_boss():
    # Identical build/state: a Wall (2x) upcoming must never score HIGHER
    # survivability than an average boss, and should generally score lower.
    jokers = []
    s_none = build_survivability(jokers, _gs(None))
    s_wall = build_survivability(jokers, _gs("The Wall"))
    s_needle = build_survivability(jokers, _gs("The Needle"))
    assert s_wall <= s_none
    assert s_needle <= s_none
