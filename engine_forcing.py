"""FORCED ENGINE BUILDING — a probe of the premise every plateau lever assumed.

WHY THIS EXISTS (dec-093)
-------------------------
Four levers (dec-075/079/081/083) tried to make the agent build BETTER, and all
came back null or worse. Then dec-090 found no build feature predicts boss
clearing (AUC 0.48-0.57), and a run-level check found nothing predicts DEPTH
either (money 0.54-0.56, xmult 0.48-0.56, hand level 0.49-0.56).

The tempting conclusion is "builds don't matter". That conclusion is NOT
supported, because of RANGE RESTRICTION: at ante 4 the agent's build space is
0-2 xmult engines (3 engines = 2.9% of the field, 4 = 0.1%). Every measurement
compares mediocre builds against other mediocre builds and correctly finds no
difference. It says nothing about mediocre-vs-ENGINE, because the agent never
builds an engine.

So the premise underneath every lever so far -- "better builds go deeper" -- has
never actually been tested. This module tests it directly, by removing the
agent's build decision and forcing an engine.

WHAT IT DOES
------------
With BALATRON_FORCE_ENGINE=1, the shop pick ignores the planner's marginal
build_value ranking (which by construction cannot value a synergy piece that is
worthless alone) and instead buys by ENGINE PRIORITY:

    xmult / xmult-scaling  >  copiers & retriggers  >  other scaling  >  economy

and rerolls aggressively when nothing on that list is affordable, spending down
to a small reserve rather than banking.

This is deliberately CRUDE. It is not a proposed policy -- it is an instrument
for answering one question. A crude forcing that still goes deeper proves the
premise; a crude forcing that does not is strong evidence the premise is false.

READING IT (pre-registered before the run)
------------------------------------------
  * forced engines reach deeper  -> engines ARE the lever. The problem is
    acquisition/affordability, not evaluation, and there is a concrete target to
    train toward. Reopens the whole investigation.
  * forced engines do NOT go deeper -> builds do not determine depth even at the
    top of the reachable range, and the ceiling is somewhere nobody has looked.

Both outcomes are worth the run, which is why it is worth running.

SAFETY: env-gated, default OFF, so the control arm is byte-identical to current
behaviour. Never raises -- any failure returns None and the caller keeps the
planner's pick.
"""
from __future__ import annotations

import os as _os

_MODE = _os.environ.get("BALATRON_FORCE_ENGINE", "0")
FORCE_ENGINE = _MODE in ("1", "2")

# Mode 2 = BANK-THEN-ENGINE. Mode 1 (buy the best affordable engine piece, any
# tier) was measured and it does not force anything: 141 forced buys came out
# 23% xmult / 34% additive / 36% economy, 0.53 xmult per run, mean ante 3.283.
#
# The reason is an affordability cliff measured in audit_engine_affordability.py:
#
#     median bankroll at shop entry     $6
#     median tier-5 engine cost         $7
#     tier-5 affordable at $5          10%
#     tier-5 affordable at $10         99%
#
# The whole distance between "can never buy an engine" and "can buy any engine"
# is about $4, and the agent's bankroll (p25 $3, p50 $6, p75 $10) straddles it.
# So "buy the best AFFORDABLE piece" degenerates into buying cheap filler, which
# spends the bankroll DOWN and pushes it further under the wall. That is why
# mode 1 underperformed banking.
#
# Mode 2 inverts it: refuse everything below MIN_TIER and hold the money until a
# real engine is reachable. This is the actual test of the engine hypothesis;
# mode 1 was a test of "buy things sooner".
BANK_THEN_ENGINE = _MODE == "2"
MIN_TIER = int(_os.environ.get("BALATRON_ENGINE_MIN_TIER", "4"))
BANK_TARGET = int(_os.environ.get("BALATRON_ENGINE_BANK_TARGET", "10"))

# Keep enough to re-roll again rather than stranding at $0 with a bad shop.
REROLL_RESERVE = 3


def _tier(name: str) -> int:
    """Engine priority for a joker. Higher is better; 0 = not an engine piece.

    The ordering encodes what actually compounds in Balatro: a multiplicative
    engine, then things that multiply the engine (copiers/retriggers), then
    additive scaling, then economy that buys more of the above.

    Every field read here is verified present in data/jokers.py (all 150 entries
    carry the full key set, values are bools except money_per_round).

    CORRECTION (dec-100). This docstring previously claimed `scaling_type` and
    `scaling_increment` "DO NOT EXIST in the schema". Both DO exist —
    data/jokers.py JOKER_DEFAULTS carries them, 29 jokers set `scaling_type` and
    24 set `scaling_increment`. That false claim came from a truncated key dump
    and was already retracted in dec-095, but the retraction never reached this
    file, so the code kept teaching the wrong fact to whoever edited it next.

    `_tier` still reads the boolean flags rather than `scaling_type`, which is
    fine — but as a CHOICE, not because the alternative is absent. The real
    limitation is that these flags are NOMINAL: `xmult=True` is equally true of a
    working engine and of Blackboard, Cavendish or Loyalty Card, which is why
    dec-093's forcing arms could not distinguish them.
    """
    from data.jokers import JOKERS

    s = JOKERS.get(name) or {}
    if not s:
        return 0
    if s.get("xmult") or s.get("xmult_scaling"):
        return 5
    if s.get("copy") or s.get("retrigger_effect") or s.get("mass_retrigger"):
        return 4
    if s.get("mult_scaling") or s.get("chip_scaling"):
        return 3
    if s.get("economy") or (s.get("money_per_round") or 0) > 0:
        return 2
    return 0


def pick_engine_joker(raw_state: dict, money: float):
    """Index of the highest-priority AFFORDABLE engine piece in the shop.

    Returns None when forcing is off, on any error, or when the shop holds no
    engine piece worth buying (the caller should then consider a reroll).
    """
    if not FORCE_ENGINE:
        return None
    try:
        from environment.action_space import (
            BAD_JOKERS, _api_key_to_name, _is_non_joker_card,
        )

        best, best_tier = None, 0
        for i, sc in enumerate(raw_state.get("shop", {}).get("cards", []) or []):
            if _is_non_joker_card(sc):
                continue
            name = _api_key_to_name(sc.get("joker_key", "") or sc.get("key", ""))
            if not name or name in BAD_JOKERS:
                continue
            cost = sc.get("cost", {})
            cost = cost.get("buy", 999) if isinstance(cost, dict) else 999
            if cost > money:
                continue
            t = _tier(name)
            if BANK_THEN_ENGINE and t < MIN_TIER:
                continue        # filler: refuse it and keep the money
            if t > best_tier:
                best, best_tier = i, t
        return best
    except Exception:
        return None


def should_force_reroll(raw_state: dict, money: float) -> bool:
    """True when forcing is on, the shop holds NO affordable engine piece, and a
    reroll is affordable while leaving a reserve.

    This is the half the normal agent will not do: it banks instead of hunting,
    which is exactly how acquisition ends up indistinguishable from random
    (dec-080: pool is 23% xmult, E=1.17 in 5 slots, observed median 1).
    """
    if not FORCE_ENGINE:
        return False
    try:
        if pick_engine_joker(raw_state, money) is not None:
            return False
        cost = (raw_state.get("round", {}) or {}).get("reroll_cost", 5)
        if BANK_THEN_ENGINE:
            # Only hunt once we can reroll AND still afford an engine afterwards.
            # Rerolling below the target burns the very bankroll we are trying to
            # accumulate past the $6->$10 cliff, which is how mode 1 spent itself
            # under the wall.
            return money - cost >= BANK_TARGET
        return money >= cost + REROLL_RESERVE
    except Exception:
        return False
