"""Pillar 2 (dec-034 PLANNING): evaluate build decisions by their MULTI-ANTE
future consequences instead of immediate score.

Balatro is fully observable with a deterministic scoring model, so we can
SIMULATE forward — the computer's superpower the agent has never used. The
reactive policy + greedy `_estimate_joker_value` rank a shop joker by how much
it raises THIS hand's score; that's a ~ante-4 strategy. This module ranks a buy
by how much deeper into the ante blind-target curve the resulting build can go
("survivability") — the question a strong player actually asks at the shop.

First slice = depth-1 planning with a multi-ante horizon evaluation, built on the
now-accurate Pillar-1 valuation (`estimate_score_for_hand_type`, which already
projects scaling-joker growth). Full multi-step expectimax (reroll/draw chance
nodes, RL value head as the leaf) is a later refinement within this pillar.
"""

from environment.hand_eval import (
    estimate_score_for_hand_type, _estimate_joker_scoring_for_type, BASE_HAND_SCORES,
)

# Hand types a build can commit to and concentrate leveling on (Pillar 3).
COMMITTABLE_HANDS = ["Pair", "Two Pair", "Three of a Kind", "Straight",
                     "Flush", "Full House", "Four of a Kind"]

# How reliably each hand can actually be MADE each round (rough prior). Commit
# weights raw power by achievability so the build doesn't "commit" to a rare hand
# (Four of a Kind) it never plays — it must be both strong AND makeable.
HAND_ACHIEVABILITY = {
    "Pair": 1.0, "Two Pair": 0.7, "Three of a Kind": 0.6, "Straight": 0.45,
    "Flush": 0.5, "Full House": 0.35, "Four of a Kind": 0.2,
}

# Balatro White-Stake SMALL-blind base chip requirement by ante (big = 1.5x,
# boss = 2x). Known game constants; extrapolate geometrically past ante 8 so the
# curve stays monotonic for deep planning.
ANTE_BASE_TARGET = {1: 300, 2: 800, 3: 2000, 4: 5000, 5: 11000,
                    6: 20000, 7: 35000, 8: 50000}
BLIND_MULT = {"small": 1.0, "big": 1.5, "boss": 2.0}

# A blind is cleared over several played hands, not one — so a build's
# blind-clearing power is its per-hand estimate times the hands typically spent.
HANDS_PER_BLIND = 3.0
MAX_PLAN_ANTE = 12


def ante_target(ante: int, blind: str = "boss") -> float:
    """Chip requirement for a given ante + blind type (default the boss, the
    hardest gate of the ante)."""
    base = ANTE_BASE_TARGET.get(ante)
    if base is None:
        base = ANTE_BASE_TARGET[8] * (1.6 ** (max(ante, 8) - 8))
    return base * BLIND_MULT.get(blind, 2.0)


def build_survivability(jokers: list[dict], gamestate: dict) -> float:
    """Fractional highest ante whose BOSS target this build can still clear.

    A proxy for 'how deep does this build go' — the planning objective. Per-hand
    power = the MAX of the build's committed-archetype potential (forward-looking,
    so a flush build is judged on its Flush ceiling even while it's still playing
    Pairs early) and the best already-played hand (so an established build isn't
    underrated). Both include projected scaling (Pillar 1). x HANDS_PER_BLIND.
    """
    ht = target_hand_type(jokers, gamestate)
    per_hand = max(score_hand_type(ht, jokers, gamestate),
                   estimate_score_for_hand_type(jokers, gamestate))
    power = per_hand * HANDS_PER_BLIND
    if power <= 0:
        return 0.0
    prev_target = 0.0
    for a in range(1, MAX_PLAN_ANTE + 1):
        tgt = ante_target(a, "boss")
        if power >= tgt:
            prev_target = tgt
            continue
        # fractional progress into ante a (log-spaced — targets grow ~geometrically)
        import math
        lo = math.log10(prev_target + 1.0)
        hi = math.log10(tgt + 1.0)
        cur = math.log10(power + 1.0)
        frac = (cur - lo) / max(hi - lo, 1e-9)
        return (a - 1) + max(0.0, min(frac, 1.0))
    return float(MAX_PLAN_ANTE)


def build_value(joker: dict, current_jokers: list[dict], gamestate: dict) -> float:
    """Planning value of adding `joker`: the gain in build survivability (how
    many more antes deep the build can go with it). This is the multi-ante
    forward-looking signal that replaces greedy immediate-score ranking."""
    before = build_survivability(current_jokers, gamestate)
    after = build_survivability(list(current_jokers) + [joker], gamestate)
    return after - before


def score_hand_type(ht: str, jokers: list[dict], gamestate: dict) -> float:
    """Per-hand-type score for the current build (the hand's current level from
    gamestate + joker effects). Forward-looking: reflects what the jokers
    SUPPORT, not just what's been played (Pillar 3 COMMITMENT)."""
    info = gamestate.get("hands", {}).get(ht, {})
    bc, bm = BASE_HAND_SCORES.get(ht, (5, 1))
    ht_chips = info.get("chips", bc)
    ht_mult = info.get("mult", bm)
    jc, jm, jx = _estimate_joker_scoring_for_type(ht, jokers, gamestate)
    return (ht_chips + 40.0 + jc) * (ht_mult + jm) * jx


def target_hand_type(jokers: list[dict], gamestate: dict) -> str:
    """The hand type this BUILD is committed to — the one it scores highest at.
    Build-based (not the lagging most-played heuristic), so leveling and buys can
    concentrate on ONE archetype instead of diluting across hand types — the
    single biggest White-Stake scaling mistake the strategy audit flagged."""
    best, best_ht = -1.0, "Pair"
    for ht in COMMITTABLE_HANDS:
        s = score_hand_type(ht, jokers, gamestate) * HAND_ACHIEVABILITY.get(ht, 0.5)
        if s > best:
            best, best_ht = s, ht
    return best_ht


def rank_shop_jokers(shop_jokers: list[dict], current_jokers: list[dict],
                     gamestate: dict) -> list[tuple[int, float]]:
    """Return (index, build_value) for each shop joker, best first — the planner's
    recommended buy order."""
    scored = [(i, build_value(j, current_jokers, gamestate))
              for i, j in enumerate(shop_jokers)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
