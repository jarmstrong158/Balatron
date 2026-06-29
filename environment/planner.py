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
    _api_key_to_name, HAND_LEVEL_INCREMENTS, SCORING_CARD_CHIPS,
)

# Solver phase 1 (dec-036): how many effective scaling increments an engine gains
# per ante, used to PROJECT an engine's value forward when evaluating whether a
# build out-scales the exponential blind curve. Mirrors the shop-projection rate.
ENGINE_INCREMENTS_PER_ANTE = 2.0

# dec-042: committed-hand planet LEVELS assumed gained per ante. build_survivability
# used to FREEZE the hand level for every future ante, so a "commit + level it"
# plan (e.g. Photograph + Hanging Chad carried by a few Flush levels) projected as
# if it plateaued and was under-valued vs flat additive. Projecting leveling lets
# the planner SEE leveling as a real path to depth. Conservative (~<1/ante).
LEVELS_PER_ANTE = 0.8

# dec-042: stickiness for the committed archetype. The build flip-flopped across
# ~3 hand types/run, scattering planets to net only ~1 level. Once a hand is the
# most-invested (most planet levels sunk in), require a clearly better alternative
# to switch away — so leveling concentrates on one archetype.
COMMIT_HYSTERESIS = 1.25

# Hand types a build can commit to and concentrate leveling on (Pillar 3).
COMMITTABLE_HANDS = ["Pair", "Two Pair", "Three of a Kind", "Straight",
                     "Flush", "Full House", "Four of a Kind"]

# SCORING_CARD_CHIPS now lives in hand_eval.py (single source of truth, dec-043)
# and is imported above — per-hand scoring-card chips replacing the old flat +40.

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

# EMPIRICAL CALIBRATION (dec-038). The raw estimate (score x HANDS_PER_BLIND) is
# a best-case point estimate: it assumes every hand is the committed full-power
# hand. Real play is worse — you don't always draw the committed hand, boss
# debuffs cut scoring, and weak hands get discarded. Validated against 5,018
# instrumented self-play games: real boss-blind advance crosses 50% at raw
# margin 2.30x, NOT 1.0x — the estimate overshoots realized clearing by ~2.3x.
# This factor (1/2.30) recalibrates power so margin>=1 means a real ~50/50 clear,
# which stops the planner greenlighting additive builds it wrongly thinks survive
# deep and raises the marginal value of multiplicative (xmult) scaling. Applied
# ONLY in the decision path; the build_progression log stays RAW so this
# calibration can be re-validated on fresh data.
REALIZATION_FACTOR = 0.43
MAX_PLAN_ANTE = 12


def ante_target(ante: int, blind: str = "boss") -> float:
    """Chip requirement for a given ante + blind type (default the boss, the
    hardest gate of the ante)."""
    base = ANTE_BASE_TARGET.get(ante)
    if base is None:
        base = ANTE_BASE_TARGET[8] * (1.6 ** (max(ante, 8) - 8))
    return base * BLIND_MULT.get(blind, 2.0)


def _project_jokers(jokers: list[dict], antes_ahead: float) -> list[dict]:
    """Copy the joker list, advancing each SCALING engine's value antes_ahead
    antes into the future (current/start value + increment * rate * antes_ahead).
    Static jokers pass through unchanged. This is how the solver models whether a
    build keeps pace with the exponential blind curve (dec-036)."""
    if antes_ahead <= 0:
        return jokers
    from data.jokers import JOKERS
    out = []
    for j in jokers:
        n = _api_key_to_name(j.get("key", "") or j.get("joker_key", ""))
        sch = JOKERS.get(n) if n else None
        if sch and sch.get("scaling_type"):
            inc = sch.get("scaling_increment") or 0.0
            if inc > 0:
                cur = j.get("_scaled_value")
                base = cur if cur is not None else (sch.get("scaling_start_value") or 0.0)
                jj = dict(j)
                jj["_scaled_value"] = base + inc * ENGINE_INCREMENTS_PER_ANTE * antes_ahead
                out.append(jj)
                continue
        out.append(j)
    return out


def _level_committed_hand(gamestate: dict, ht: str, levels: float) -> dict:
    """Return a shallow gamestate copy with the committed hand `ht` leveled
    `levels` planet-levels forward (dec-042). Used to PROJECT leveling in
    build_survivability so a commit-and-level plan isn't valued as if frozen."""
    if levels <= 0:
        return gamestate
    inc_c, inc_m = HAND_LEVEL_INCREMENTS.get(ht, (10, 1))
    bc, bm = BASE_HAND_SCORES.get(ht, (5, 1))
    hands = dict(gamestate.get("hands", {}))
    info = dict(hands.get(ht, {}))
    info["chips"] = info.get("chips", bc) + inc_c * levels
    info["mult"] = info.get("mult", bm) + inc_m * levels
    hands[ht] = info
    gs = dict(gamestate)
    gs["hands"] = hands
    return gs


def build_survivability(jokers: list[dict], gamestate: dict) -> float:
    """TRAJECTORY-AWARE fractional deepest ante this build can clear (solver
    phase 1, dec-036). At each future ante it PROJECTS the build's engines forward
    to that ante and checks the projected power vs the boss target — so a build
    whose engines out-scale the exponential curve projects deep, while a flat
    additive build plateaus. This is the evaluator the ceiling audit (dec-035)
    requires: it values builds by whether they KEEP PACE with the curve, not by
    static current power. Committed-archetype hand fixed once (the build's plan)."""
    ht = target_hand_type(jokers, gamestate)
    try:
        cur = int(gamestate.get("ante_num", gamestate.get("ante", 1)) or 1)
    except (TypeError, ValueError):
        cur = 1
    import math
    prev_target = ante_target(cur - 1, "boss") if cur > 1 else 0.0
    for a in range(cur, MAX_PLAN_ANTE + 1):
        pj = _project_jokers(jokers, a - cur)               # engines matured to ante a
        gs_a = _level_committed_hand(gamestate, ht, LEVELS_PER_ANTE * (a - cur))
        power = score_hand_type(ht, pj, gs_a) * HANDS_PER_BLIND * REALIZATION_FACTOR
        tgt = ante_target(a, "boss")
        if power >= tgt:
            prev_target = tgt
            continue
        lo = math.log10(prev_target + 1.0)
        hi = math.log10(tgt + 1.0)
        cur_p = math.log10(max(power, 0.0) + 1.0)
        frac = (cur_p - lo) / max(hi - lo, 1e-9)
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
    scoring_chips = SCORING_CARD_CHIPS.get(ht, 40.0)   # per-hand, not flat +40 (dec-040)
    return (ht_chips + scoring_chips + jc) * (ht_mult + jm) * jx


def target_hand_type(jokers: list[dict], gamestate: dict) -> str:
    """The hand type this BUILD is committed to — the one it scores highest at.
    Build-based (not the lagging most-played heuristic), so leveling and buys can
    concentrate on ONE archetype instead of diluting across hand types — the
    single biggest White-Stake scaling mistake the strategy audit flagged."""
    # dec-042: commit hysteresis. Find the hand the build has already invested the
    # most planet-levels in (current chips above its base = levels sunk); give it a
    # stickiness bonus so the commitment doesn't thrash ante-to-ante and planets
    # concentrate on one archetype.
    hands = gamestate.get("hands", {})
    def _invested(h: str) -> float:
        bc, _ = BASE_HAND_SCORES.get(h, (5, 1))
        return hands.get(h, {}).get("chips", bc) - bc
    committed = max(COMMITTABLE_HANDS, key=_invested)
    if _invested(committed) <= 0:
        committed = None                                   # nothing leveled yet -> free choice
    best, best_ht = -1.0, "Pair"
    for ht in COMMITTABLE_HANDS:
        s = score_hand_type(ht, jokers, gamestate) * HAND_ACHIEVABILITY.get(ht, 0.5)
        if ht == committed:
            s *= COMMIT_HYSTERESIS
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
