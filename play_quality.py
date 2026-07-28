"""Per-hand play-quality logging: did the agent play the BEST available hand?

WHY THIS EXISTS (dec-091)
-------------------------
Every build-side surface is now eliminated as the ante-4/5 lever:

  reward function      exonerated (dec-073: R = 11.15*ante - 14.62, corr 0.977)
  build EVALUATION     dead end  (dec-085: four measured levers, null or worse)
  shop access/timing   dead end  (dec-079, dec-081)
  build QUALITY itself no feature predicts clearing at antes 4-6 -- suit
                       concentration, enhancements, seals, deck size, hand size,
                       xmult count and hand level ALL score AUC 0.48-0.57 (dec-090)
  credit horizon       closed    (dec-088)
  discard usage        already sensible -- the agent digs when behind, and
                       deaths that LEFT discards were CLOSER to winning

Even the top 5% of builds by raw margin still fail 16% of ante-4 and 30% of
ante-5 bosses, and there is no power threshold hiding above that (0/1/2/3 xmult
clear at 76.9/77.7/76.4/70.9% at ante 4 -- flat).

That leaves exactly one decision surface never examined: WHICH CARDS THE AGENT
PLAYS. Every chip it scores comes from a played hand, and nobody has ever checked
whether the hand it chose was the best one available. This module measures that
and nothing else.

WHAT IT RECORDS
---------------
Per played hand: the agent's chosen cards and their score, the best available
hand and ITS score, and the ratio between them. One query then answers the
question -- what fraction of the achievable score does the agent capture?

Pre-registered reading (fixed BEFORE any data, so it cannot be rationalised):
  * capture ratio well below ~0.90 -> a large, previously invisible lever, and
    the first non-build one in this investigation.
  * capture ratio ~0.95+          -> play is fine. Combined with everything
    above, antes 4-5 are variance-dominated in the strict sense and this
    architecture is at its ceiling. That is a real finding, not a failure.

FAIRNESS
--------
The best-available baseline is computed UNDER THE SAME BOSS DEBUFFS the agent
played under. Comparing against an undebuffed optimum would manufacture a gap
that is not the agent's fault.

SAFETY
------
Logging only -- it never influences an action. Every call is wrapped so a failure
here can never break a run, and the file self-rotates (dec-043: an unbounded
joker_order_log once filled the disk and silently broke checkpointing).
"""
from __future__ import annotations

import json
import os

MAX_ROWS = 40_000          # rotate below this; dec-043 disk-exhaustion lesson
_TRIM_TO = 20_000
_LOG = os.path.join("logs", "play_quality.jsonl")
_count = 0

# Mirrors the LOCAL BOSS_SUIT_DEBUFF/The Plant handling inside
# hand_eval.find_best_discard (~line 2727). It is local to that function so it
# cannot be imported; these are fixed game constants, and a test pins this copy
# against the source so the two cannot drift silently.
BOSS_SUIT_DEBUFF = {
    "The Club": "Clubs",
    "The Goad": "Spades",
    "The Head": "Hearts",
    "The Window": "Diamonds",
}
BOSS_FACE_DEBUFF = "The Plant"


def current_boss(raw_state: dict) -> str:
    """Name of the boss blind currently being PLAYED, else ''."""
    blinds = raw_state.get("blinds", {})
    if not isinstance(blinds, dict):
        return ""
    b = blinds.get("boss")
    if isinstance(b, dict) and b.get("status") == "CURRENT":
        return b.get("name", "") or ""
    return ""


def debuffs_for(raw_state: dict):
    """(debuffed_suit, boss_debuff_face) for the CURRENT blind."""
    boss = current_boss(raw_state)
    return BOSS_SUIT_DEBUFF.get(boss), (boss == BOSS_FACE_DEBUFF)


def log_play(raw_state: dict, played_indices, env_id: int = 0,
             global_step: int = 0) -> None:
    """Record one played hand alongside the best hand that was available.

    `played_indices` are indices into the CURRENT hand. Never raises.
    """
    global _count
    try:
        from environment.hand_eval import (
            classify_hand, estimate_score, find_best_hands,
        )

        hand = (raw_state.get("hand", {}) or {}).get("cards", []) or []
        jokers = (raw_state.get("jokers", {}) or {}).get("cards", []) or []
        idxs = [i for i in (played_indices or []) if 0 <= i < len(hand)]
        if not hand or not idxs:
            return

        suit, face = debuffs_for(raw_state)

        # What the agent actually played.
        # classify_hand returns (hand_type, SCORING_INDICES) — the subset that
        # actually scores. Passing range(len(cards)) instead credits every played
        # card, so a Pair played among 5 cards is scored as if all 5 counted.
        # That inflated capture ABOVE 1.0 (mean 1.061, ante-4 1.243) on the first
        # live run, which is impossible against a true best-available baseline
        # and is what exposed the bug.
        cards = [hand[i] for i in idxs]
        ht, scoring = classify_hand(cards)
        got = float(estimate_score(ht, cards, list(scoring or []), jokers,
                                   raw_state, debuffed_suit=suit,
                                   boss_debuff_face=face))

        # the best it COULD have played, under the same debuffs
        best = find_best_hands(hand, jokers, raw_state, top_n=1,
                               debuffed_suit=suit, boss_debuff_face=face)
        if not best:
            return
        best_score = float(best[0].get("estimated_score") or 0.0)
        best_type = best[0].get("hand_type")
        best_idx = sorted(best[0].get("card_indices") or [])

        rnd = raw_state.get("round", {}) or {}
        row = {
            "ante": raw_state.get("ante_num", 0),
            "blind": (raw_state.get("blinds", {}) or {}).get("boss", {}).get("name", "")
                     if current_boss(raw_state) else "",
            "is_boss": bool(current_boss(raw_state)),
            "played_type": ht,
            "played_n": len(idxs),
            "played_score": round(got, 1),
            "best_type": best_type,
            "best_score": round(best_score, 1),
            # THE statistic: fraction of the achievable score captured. Clamped
            # at 1.0 because it is a ratio to the MAXIMUM — a value above 1 is by
            # definition a measurement fault, not the agent outperforming.
            "capture": (round(min(got / best_score, 1.0), 4)
                        if best_score > 0 else None),
            "capture_raw": round(got / best_score, 4) if best_score > 0 else None,
            # Scoring-path disagreement: the agent played the SAME cards the
            # baseline picked, yet the two paths returned materially different
            # scores. Observed on 2/235 live plays (ratios of exactly 3.00x and
            # 1.81x, both boss blinds), i.e. estimate_score called directly and
            # find_best_hands' internal scoring do not always agree for identical
            # cards. Flagged rather than silently clamped, because it is a real
            # inconsistency worth chasing on its own — and because letting a 3.0x
            # row into a 33-row bucket dragged that ante's mean to 1.085.
            "path_disagreement": bool(
                sorted(idxs) == best_idx and best_score > 0
                and abs(got - best_score) / best_score > 0.01),
            "same_cards": sorted(idxs) == best_idx,
            "hand_size": len(hand),
            "hands_left": rnd.get("hands_left", -1),
            "discards_left": rnd.get("discards_left", -1),
            "chips": rnd.get("chips", 0),
            "debuffed_suit": suit,
            "debuff_face": face,
            "env": env_id,
            "step": global_step,
        }
        os.makedirs("logs", exist_ok=True)
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        _count += 1
        if _count % 2000 == 0:
            _rotate()
    except Exception as e:      # never let instrumentation break a run
        try:
            from diagnostics import warn_once
            warn_once("play_quality.log_play", e)
        except Exception:
            pass


def _rotate() -> None:
    """Keep the file bounded (dec-043)."""
    try:
        if not os.path.exists(_LOG):
            return
        with open(_LOG, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        if len(lines) <= MAX_ROWS:
            return
        with open(_LOG, "w", encoding="utf-8") as fh:
            fh.writelines(lines[-_TRIM_TO:])
    except Exception:
        pass
