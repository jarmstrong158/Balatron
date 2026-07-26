"""Monte-Carlo rollout evaluator: P(clear this blind) instead of a point estimate.

WHY THIS EXISTS
---------------
The planner's leaf (`planner.build_survivability`) answers "how much power does
this build project?" with a single deterministic number:

    power = score_hand_type(committed_ht, jokers, gs) * HANDS_PER_BLIND * RF

That assumes the build ALWAYS draws its committed hand type. Real blinds are
decided by the draw. Measured against the `beaten` label on real logged blinds
the leaf scores AUC 0.56-0.63 per ante — barely better than a coin flip — and a
learned model over aggregate build features could not beat it (0.58-0.64 once a
`deck_n` leak was removed, dec-082). Three A/Bs that improved ACCESS to shop
decisions (dec-079 buy timing, dec-081 buy legality) came back null or worse,
which is what you expect when the thing ranking the options is near-chance: acting
more on a coin-flip evaluator at a real economic cost is negative EV.

This module attacks the evaluator itself. Instead of projecting a point estimate
it SIMULATES the blind: reconstruct a plausible deck from the logged marginals,
deal a hand, pick the best play with the real scoring engine, repeat for the
blind's hands, and check the cumulative score against the EXACT target. Run that
N times and the output is a probability, not a guess:

    P(clear) = (# simulated runs whose cumulative score >= target) / N

That captures what the point estimate structurally cannot — CONSISTENCY. Two
builds with identical projected power but different deck concentration have very
different clear probabilities, and consistency is precisely what decides a boss.

STATUS: UNVALIDATED. Nothing imports this into the decision path yet. It must
first beat the current leaf's AUC offline, on post-dec-082 (uncontaminated)
snapshots. If it does not beat it, it does not ship — that gate exists because
this project has shipped three unvalidated ideas and measured null/worse.

APPROXIMATIONS (each one makes the estimate CONSERVATIVE or neutral, never
flattering, and each is a candidate refinement if the idea proves out):
  * The logged snapshot stores rank and suit counts as separate MARGINALS, not
    the joint. We sample a maximum-entropy joint consistent with both.
  * Enhancements/seals are assigned to random cards in the right quantity.
  * Discards use a simple "throw the non-contributing cards" policy rather than
    the full `find_best_discard` search (that call is far too slow for MC).
  * Per-hand joker scaling within a blind is not advanced.
"""
from __future__ import annotations

import os as _os
import random
from typing import Optional

from environment.hand_eval import classify_hand, estimate_score

# Rollouts per evaluation. 40 keeps a single call near ~10ms; the offline
# validator overrides it upward since it is not latency-bound.
ROLLOUT_SAMPLES = int(_os.environ.get("BALATRON_ROLLOUT_SAMPLES", "40"))

# Hands/discards a blind allows when the snapshot does not say.
DEFAULT_HANDS = 4
DEFAULT_DISCARDS = 3

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "H", "D", "C"]


def _rank_value(r: str) -> int:
    if r == "A":
        return 14
    if r in ("K", "Q", "J"):
        return {"J": 11, "Q": 12, "K": 13}[r]
    try:
        return int(r)
    except ValueError:
        return 2


def build_deck(snap: dict, rng: random.Random) -> list[dict]:
    """Reconstruct a plausible deck from the snapshot's marginals.

    `ranks` and `suits` are independent counts, so the true joint is unknown; we
    draw the maximum-entropy joint consistent with both by shuffling each
    marginal and zipping. Enhancements/seals are then sprinkled at their logged
    multiplicities. Returns card dicts in the API's shape so the real scoring
    engine can consume them unchanged.
    """
    ranks: list[str] = []
    for r, n in (snap.get("ranks") or {}).items():
        if r and r != "?":
            ranks.extend([r] * int(n))
    suits: list[str] = []
    for s, n in (snap.get("suits") or {}).items():
        if s and s != "?":
            suits.extend([s] * int(n))
    if not ranks or not suits:
        return []
    # Equalize lengths (marginals can disagree if some cards had unknown fields).
    n = min(len(ranks), len(suits))
    ranks, suits = ranks[:n], suits[:n]
    rng.shuffle(ranks)
    rng.shuffle(suits)

    deck = [{"value": {"rank": r, "suit": s, "value": _rank_value(r)},
             "modifier": {}}
            for r, s in zip(ranks, suits)]

    idx = list(range(len(deck)))
    rng.shuffle(idx)
    cursor = 0
    for enh, cnt in (snap.get("enhancements") or {}).items():
        for _ in range(int(cnt)):
            if cursor < len(idx):
                deck[idx[cursor]]["modifier"]["enhancement"] = enh
                cursor += 1
    for seal, cnt in (snap.get("seals") or {}).items():
        for _ in range(int(cnt)):
            if cursor < len(idx):
                deck[idx[cursor]]["modifier"]["seal"] = seal
                cursor += 1
    return deck


def _best_play(hand: list[dict], jokers: list[dict], gs: dict) -> tuple[float, list[int]]:
    """Best-scoring subset of `hand` and the indices it uses.

    `find_best_hands` enumerates every C(n,k) combination and is far too slow to
    call thousands of times, so we evaluate a small set of candidate subsets that
    covers what actually gets played: the classified best hand, and the top-k
    single-rank/suit groupings.
    """
    if not hand:
        return 0.0, []
    best_score, best_idx = 0.0, []
    n = len(hand)

    cands: list[list[int]] = []
    # whole hand + the classifier's own read of it
    cands.append(list(range(n)))
    # rank groups (pairs/trips/quads) and suit groups (flushes)
    by_rank: dict[str, list[int]] = {}
    by_suit: dict[str, list[int]] = {}
    for i, c in enumerate(hand):
        v = c.get("value", {}) or {}
        by_rank.setdefault(v.get("rank", "?"), []).append(i)
        by_suit.setdefault(v.get("suit", "?"), []).append(i)
    for g in by_rank.values():
        if len(g) >= 2:
            cands.append(g[:5])
    for g in by_suit.values():
        if len(g) >= 5:
            cands.append(g[:5])
    # highest cards (High Card / straight-ish fallback)
    order = sorted(range(n), key=lambda i: -(hand[i].get("value", {}) or {}).get("value", 0))
    cands.append(order[:5])
    cands.append(order[:1])

    for idxs in cands:
        if not idxs:
            continue
        cards = [hand[i] for i in idxs]
        try:
            ht, _ = classify_hand(cards)
            sc = estimate_score(ht, cards, list(range(len(cards))), jokers, gs)
        except Exception:
            continue
        if sc > best_score:
            best_score, best_idx = sc, idxs
    return best_score, best_idx


def _simulate_blind(snap: dict, jokers: list[dict], gs: dict, target: float,
                    hands: int, discards: int, rng: random.Random) -> bool:
    """Play one simulated blind. True if cumulative score reaches `target`."""
    deck = build_deck(snap, rng)
    if not deck:
        return False
    rng.shuffle(deck)
    hand_size = int(snap.get("hand_size", 8) or 8)
    pos = 0
    held: list[dict] = deck[pos:pos + hand_size]
    pos += len(held)

    total = 0.0
    discards_left = discards
    for _ in range(hands):
        if not held:
            break
        score, idxs = _best_play(held, jokers, gs)

        # Cheap discard policy: if this play is weak and we still have discards
        # and cards to redraw, throw what the play does not use and redraw once.
        if discards_left > 0 and pos < len(deck) and score * hands < target:
            keep = set(idxs)
            tossed = [i for i in range(len(held)) if i not in keep]
            if tossed:
                held = [held[i] for i in sorted(keep)]
                need = hand_size - len(held)
                draw = deck[pos:pos + need]
                pos += len(draw)
                held.extend(draw)
                discards_left -= 1
                score, idxs = _best_play(held, jokers, gs)

        total += score
        if total >= target:
            return True
        # played cards leave the hand; refill
        keep = [c for i, c in enumerate(held) if i not in set(idxs)]
        need = hand_size - len(keep)
        draw = deck[pos:pos + need]
        pos += len(draw)
        held = keep + draw
    return total >= target


def p_clear(snap: dict, jokers: list[dict], gs: dict, target: float,
            samples: int = ROLLOUT_SAMPLES, hands: int = DEFAULT_HANDS,
            discards: int = DEFAULT_DISCARDS,
            seed: Optional[int] = None) -> float:
    """Probability this build clears `target`, by Monte-Carlo simulation.

    Deterministic for a given `seed`, which matters because the planner must not
    rank the same shop differently on two identical calls.
    """
    if target <= 0:
        return 1.0
    rng = random.Random(seed if seed is not None else 0)
    wins = 0
    for _ in range(max(1, samples)):
        if _simulate_blind(snap, jokers, gs, target, hands, discards, rng):
            wins += 1
    return wins / max(1, samples)
