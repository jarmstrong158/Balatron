"""
Balatron — Hand Evaluator

Poker hand classification, score estimation, combo enumeration,
draw probability computation, and strategic assessment for the RL agent.

Produces 40 float features (indices 520-559) for the state vector.
Only active during SELECTING_HAND game state.

See NOTES.md for feature layout.
"""

import math
from collections import Counter
from itertools import combinations
from typing import Any, Optional

import numpy as np

from data.jokers import JOKERS


# ============================================================
# Constants
# ============================================================

# Ordered weakest → strongest (index = strength rank)
HAND_TYPES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
    "Five of a Kind", "Flush House", "Flush Five",
]
HAND_TYPE_INDEX = {name: i for i, name in enumerate(HAND_TYPES)}
NUM_HAND_TYPES = len(HAND_TYPES)

# Base chips and mult for each hand type at level 1
BASE_HAND_SCORES: dict[str, tuple[int, int]] = {
    "High Card":        (5, 1),
    "Pair":             (10, 2),
    "Two Pair":         (20, 2),
    "Three of a Kind":  (30, 3),
    "Straight":         (30, 4),
    "Flush":            (35, 4),
    "Full House":       (40, 4),
    "Four of a Kind":   (60, 7),
    "Straight Flush":   (100, 8),
    "Five of a Kind":   (120, 12),
    "Flush House":      (140, 14),
    "Flush Five":       (160, 16),
}

# Chip value each card contributes when scored
CARD_CHIP_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 10, "Q": 10, "K": 10, "A": 11,
}

# Rank ordering for straights (Ace can be low or high)
RANK_ORDER = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

FACE_RANKS = {"J", "Q", "K"}
HACK_RANKS = {"2", "3", "4", "5"}  # Hack joker retriggers these ranks

# Map schema rank names → API rank codes (for trigger_ranks matching)
_SCHEMA_RANK_TO_API: dict[str, str] = {
    "Ace": "A", "King": "K", "Queen": "Q", "Jack": "J",
    "10": "T", "9": "9", "8": "8", "7": "7", "6": "6",
    "5": "5", "4": "4", "3": "3", "2": "2",
    # Already-correct codes (in case schema uses short form)
    "A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T",
}


def _normalize_trigger_ranks(schema_ranks: list[str]) -> set[str]:
    """Convert trigger_ranks from schema format to API rank codes."""
    return {_SCHEMA_RANK_TO_API.get(r, r) for r in schema_ranks}


# Number of features this module produces
HAND_EVAL_FEATURES = 40


# ============================================================
# Card Parsing Helpers
# ============================================================

def _as_dict(val: Any) -> dict:
    """Normalize API fields that may be [] or {} to always be a dict."""
    return val if isinstance(val, dict) else {}


def parse_card(card: dict) -> tuple[str, str]:
    """Extract (rank, suit) from a BalatroBot card dict."""
    value = _as_dict(card.get("value", {}))
    raw_suit = value.get("suit", "")
    return value.get("rank", ""), _SUIT_EXPAND.get(raw_suit, raw_suit)


def card_rank(card: dict) -> str:
    return _as_dict(card.get("value", {})).get("rank", "")


def card_enhancement(card: dict) -> str:
    """Get card enhancement type (BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, LUCKY)."""
    mod = card.get("modifier", {})
    if not isinstance(mod, dict):
        return ""
    return mod.get("enhancement", "")


def card_edition(card: dict) -> str:
    """Get card edition (FOIL, HOLO, POLYCHROME)."""
    mod = card.get("modifier", {})
    if not isinstance(mod, dict):
        return ""
    return mod.get("edition", "")


def card_seal(card: dict) -> str:
    """Get card seal (RED, BLUE, GOLD, PURPLE) or empty string."""
    mod = card.get("modifier", {})
    if not isinstance(mod, dict):
        return ""
    return (mod.get("seal") or "").upper()


_SUIT_EXPAND = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}


def card_suit(card: dict) -> str:
    raw = _as_dict(card.get("value", {})).get("suit", "")
    return _SUIT_EXPAND.get(raw, raw)


def card_is_wild(card: dict) -> bool:
    """Check if card has WILD enhancement (counts as all suits)."""
    return card_enhancement(card) == "WILD"


ALL_SUITS = {"Hearts", "Diamonds", "Clubs", "Spades"}


def card_effective_suits(card: dict) -> set[str]:
    """Return the set of suits a card counts as.

    Wild cards count as ALL suits. Normal cards return {their suit}.
    """
    if card_is_wild(card):
        return ALL_SUITS
    s = card_suit(card)
    return {s} if s else set()


def card_chips(card: dict) -> int:
    """Chip value a card contributes when scored."""
    rank = card_rank(card)
    return CARD_CHIP_VALUES.get(rank, 0)


# ============================================================
# Hand Classification
# ============================================================

def classify_hand(cards: list[dict]) -> tuple[str, list[int]]:
    """Classify a set of 1-5 cards into a Balatro hand type.

    Args:
        cards: list of card dicts from the API

    Returns:
        (hand_type_name, scoring_card_indices) where indices are
        positions within the input cards list
    """
    n = len(cards)
    if n == 0:
        return "High Card", []

    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Wild cards count as all suits — find best suit count including wilds
    wild_count = sum(1 for c in cards if card_is_wild(c))
    non_wild_suit_counts = Counter(card_suit(c) for c in cards if not card_is_wild(c))
    best_suit_count = (max(non_wild_suit_counts.values()) if non_wild_suit_counts else 0) + wild_count

    # Count of each frequency
    freq = Counter(rank_counts.values())
    is_flush = n >= 5 and best_suit_count >= 5
    is_straight = _is_straight(ranks) if n >= 5 else False

    # Five of a Kind (5 same rank)
    if freq.get(5, 0) >= 1:
        majority_rank = rank_counts.most_common(1)[0][0]
        scoring = [i for i, r in enumerate(ranks) if r == majority_rank]
        if is_flush:
            return "Flush Five", scoring
        return "Five of a Kind", scoring

    # Four of a Kind
    if freq.get(4, 0) >= 1:
        quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
        scoring = [i for i, r in enumerate(ranks) if r == quad_rank]
        return "Four of a Kind", scoring

    # Full House (3+2)
    if freq.get(3, 0) >= 1 and freq.get(2, 0) >= 1:
        scoring = list(range(n))  # all 5 cards score
        if is_flush:
            return "Flush House", scoring
        return "Full House", scoring

    # Flush
    if is_flush:
        if is_straight:
            scoring = list(range(n))
            return "Straight Flush", scoring
        scoring = list(range(n))
        return "Flush", scoring

    # Straight
    if is_straight:
        scoring = list(range(n))
        return "Straight", scoring

    # Three of a Kind
    if freq.get(3, 0) >= 1:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        scoring = [i for i, r in enumerate(ranks) if r == trip_rank]
        return "Three of a Kind", scoring

    # Two Pair
    if freq.get(2, 0) >= 2:
        pair_ranks = [r for r, c in rank_counts.items() if c == 2]
        scoring = [i for i, r in enumerate(ranks) if r in pair_ranks]
        return "Two Pair", scoring

    # Pair
    if freq.get(2, 0) == 1:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        scoring = [i for i, r in enumerate(ranks) if r == pair_rank]
        return "Pair", scoring

    # High Card — only the highest card scores
    if n > 0:
        best_idx = max(range(n), key=lambda i: RANK_ORDER.get(ranks[i], 0))
        return "High Card", [best_idx]

    return "High Card", []


def _is_straight(ranks: list[str]) -> bool:
    """Check if 5 ranks form a straight (including A-low)."""
    if len(ranks) < 5:
        return False
    values = sorted(set(RANK_ORDER.get(r, 0) for r in ranks))
    if len(values) < 5:
        return False
    # Normal straight check
    if values[-1] - values[0] == 4:
        return True
    # Ace-low straight: A,2,3,4,5
    if set(values) == {2, 3, 4, 5, 14}:
        return True
    return False


# ============================================================
# Score Estimation
# ============================================================

def estimate_score(hand_type: str, cards: list[dict],
                   scoring_indices: list[int],
                   jokers: list[dict], gamestate: dict,
                   debuffed_suit: str | None = None,
                   boss_debuff_face: bool = False) -> float:
    """Estimate the score for a hand, including joker effects.

    Uses the Balatro formula:
        score = (base_chips + card_chips + joker_chips) × (base_mult + joker_mult) × Π(joker_xmult)

    Args:
        hand_type: classified hand type name
        cards: the cards being played
        scoring_indices: which cards in the hand actually score
        jokers: list of joker dicts from the API
        gamestate: full gamestate dict
        debuffed_suit: if set, cards of this suit contribute 0 chips
        boss_debuff_face: if True, face cards contribute 0 chips

    Returns:
        estimated score as float
    """
    # Get chips/mult from the API — these are the ACTUAL values at current planet level
    hands_data = gamestate.get("hands", {})
    hand_info = hands_data.get(hand_type, {})

    # API provides exact chips/mult for the hand type at its current level
    # Fall back to base values if API data unavailable
    base_chips_default, base_mult_default = BASE_HAND_SCORES.get(hand_type, (5, 1))
    level_chips = hand_info.get("chips", base_chips_default)
    level_mult = hand_info.get("mult", base_mult_default)

    # Card chip contributions — only SCORING cards contribute chips in Balatro
    # Debuffed cards (boss blind effect) contribute 0
    def _is_debuffed(c: dict) -> bool:
        if debuffed_suit and card_suit(c) == debuffed_suit:
            return True
        if boss_debuff_face and card_rank(c) in FACE_RANKS:
            return True
        return False

    # ── Retrigger detection for card chips/enhancements ──
    # In Balatro, card retriggers repeat: chips + enhancement + edition + per-card jokers
    # Sources: Red Seal (+1), Hanging Chad (+2 on first scored card),
    #          Sock and Buskin (+1 on face cards), Hack (+1 on 2,3,4,5),
    #          Seltzer (+1 on all cards), Dusk (+1 on all cards on last hand)
    chip_first_retriggers = 0
    chip_face_retriggers = 0
    chip_low_retriggers = 0   # Hack: ranks 2,3,4,5
    chip_all_retriggers = 0
    for j in jokers:
        jk = j.get("joker_key", "") or j.get("key", "")
        jn = _api_key_to_name(jk)
        if jn == "Hanging Chad":
            chip_first_retriggers += 2
        elif jn == "Sock and Buskin":
            chip_face_retriggers += 1
        elif jn == "Hack":
            chip_low_retriggers += 1
        elif jn == "Seltzer":
            chip_all_retriggers += 1
        elif jn == "Dusk":
            # Dusk: +1 retrigger on ALL cards on last hand of round
            hands_left_val = gamestate.get("round", {}).get("hands_left",
                              gamestate.get("current_round", {}).get("hands_left", 4))
            if hands_left_val <= 1:
                chip_all_retriggers += 1

    scoring_chips = 0.0
    for card_pos_in_scoring, i in enumerate(scoring_indices):
        if i >= len(cards):
            continue
        c = cards[i]
        if _is_debuffed(c):
            continue
        extra = chip_all_retriggers
        if card_pos_in_scoring == 0:
            extra += chip_first_retriggers
        rank = card_rank(c)
        if rank in FACE_RANKS:
            extra += chip_face_retriggers
        if rank in HACK_RANKS:
            extra += chip_low_retriggers
        if card_seal(c) == "RED":
            extra += 1
        total_triggers = 1 + extra
        # Stone cards contribute 0 rank chips (only +50 from enhancement)
        if card_enhancement(c) != "STONE":
            scoring_chips += card_chips(c) * total_triggers

    total_chips = level_chips + scoring_chips
    total_mult = level_mult

    # Enhancement effects on SCORING cards only (kickers don't trigger)
    scoring_set = set(scoring_indices)
    enhance_xmult = 1.0
    for card_pos_in_scoring, i in enumerate(scoring_indices):
        if i >= len(cards):
            continue
        c = cards[i]
        if _is_debuffed(c):
            continue

        # Compute total retriggers for this card (same logic as chips)
        extra = chip_all_retriggers
        if card_pos_in_scoring == 0:
            extra += chip_first_retriggers
        rank = card_rank(c)
        if rank in FACE_RANKS:
            extra += chip_face_retriggers
        if rank in HACK_RANKS:
            extra += chip_low_retriggers
        if card_seal(c) == "RED":
            extra += 1
        seal_retriggers = 1 + extra

        enh = card_enhancement(c)
        if enh == "BONUS":
            total_chips += 30 * seal_retriggers
        elif enh == "MULT":
            total_mult += 4 * seal_retriggers
        elif enh == "GLASS":
            # x2 per Glass card scored (shatter risk is future-hand, not current)
            enhance_xmult *= 2.0 ** seal_retriggers
        elif enh == "STONE":
            total_chips += 50 * seal_retriggers
        elif enh == "LUCKY":
            # 20% chance of +20 mult per scored Lucky card
            total_mult += 20 * 0.2 * seal_retriggers

    # Edition effects on SCORING cards only (Foil/Holo/Polychrome)
    # Uses same retrigger count as enhancements
    for card_pos_in_scoring, i in enumerate(scoring_indices):
        if i >= len(cards):
            continue
        c = cards[i]
        if _is_debuffed(c):
            continue
        extra = chip_all_retriggers
        if card_pos_in_scoring == 0:
            extra += chip_first_retriggers
        rank = card_rank(c)
        if rank in FACE_RANKS:
            extra += chip_face_retriggers
        if rank in HACK_RANKS:
            extra += chip_low_retriggers
        if card_seal(c) == "RED":
            extra += 1
        retriggers = 1 + extra
        ed = card_edition(c)
        if ed == "FOIL":
            total_chips += 50 * retriggers
        elif ed == "HOLO":
            total_mult += 10 * retriggers
        elif ed == "POLYCHROME":
            enhance_xmult *= 1.5 ** retriggers

    # Enhancement effects on HELD cards (not played)
    all_hand_cards = gamestate.get("hand", {}).get("cards", [])
    played_ids = {c.get("id", id(c)) for c in cards}
    for c in all_hand_cards:
        if c.get("id", id(c)) in played_ids:
            continue  # skip played cards
        enh = card_enhancement(c)
        if enh == "STEEL":
            enhance_xmult *= 1.5  # x1.5 per Steel card held

    # Joker contributions — pass level_mult so ordering correction kicks in
    joker_chips, joker_mult, joker_xmult = compute_joker_scoring(
        hand_type, cards, scoring_indices, jokers, gamestate,
        base_mult=level_mult,
    )

    total_chips += joker_chips
    total_mult += joker_mult

    score = total_chips * total_mult * joker_xmult * enhance_xmult
    return max(score, 0.0)


def _resolve_copy_source(jokers: list[dict], idx: int, direction: str) -> Optional[dict]:
    """Resolve a copy joker (Blueprint/Brainstorm) to the actual joker it copies.

    Follows copy chains: if Blueprint copies another Blueprint/Brainstorm,
    keeps resolving until a real joker is found.

    Args:
        jokers: ordered list of joker dicts
        idx: index of the copy joker in the list
        direction: "right" for Blueprint, "left" for Brainstorm

    Returns:
        The resolved target joker dict, or None if no valid target.
    """
    COPY_NAMES = {"Blueprint", "Brainstorm"}
    visited: set[int] = {idx}
    max_depth = len(jokers)  # prevent infinite loops

    if direction == "left":
        # Brainstorm: start scanning from leftmost
        scan_start = 0
    else:
        # Blueprint: start scanning from idx+1
        scan_start = idx + 1

    current_scan = scan_start
    for _ in range(max_depth):
        if direction == "left":
            # Scan left-to-right from position 0
            target_idx = None
            for i in range(current_scan, len(jokers)):
                if i not in visited:
                    target_idx = i
                    break
        else:
            # Scan rightward from current position
            target_idx = None
            for i in range(current_scan, len(jokers)):
                if i not in visited:
                    target_idx = i
                    break

        if target_idx is None:
            return None

        visited.add(target_idx)
        target = jokers[target_idx]
        tk = target.get("joker_key", "") or target.get("key", "")
        tn = _api_key_to_name(tk)

        if tn not in COPY_NAMES:
            return target  # Found a real joker

        # It's another copy joker — follow its chain
        ts = JOKERS.get(tn, {})
        next_dir = ts.get("copy_target", "")
        if next_dir == "right":
            current_scan = target_idx + 1
            direction = "right"
        else:
            current_scan = 0
            direction = "left"

    return None


def compute_joker_scoring(hand_type: str, cards: list[dict],
                          scoring_indices: list[int],
                          jokers: list[dict], gamestate: dict,
                          base_mult: float = 0.0) -> tuple[float, float, float]:
    """Compute joker chip/mult/xmult contributions.

    Handles all major scoring triggers:
    - any_hand_played, specific_hand_type, specific_suit, specific_rank
    - face_card, scoring_card
    - scoring_hand_size (Half Joker)
    - card_held_in_hand (Baron, Blackboard, Shoot the Moon)
    - per_joker_owned (Abstract Joker)
    - per_card_remaining_in_deck (Blue Joker)
    - per_dollar_held (Bull, Bootstraps)
    - final_hand_of_round (Acrobat)
    - effect_probability (Bloodstone, etc.)
    - periodic (Loyalty Card)
    - per_specific_joker_present (Baseball Card)

    Properly models Balatro's left-to-right scoring order:
    - during_card xmult fires early (on qualifying cards), multiplying only
      the mult accumulated so far (base_mult), NOT additive mult from later
      cards or after-card jokers.
    - after_card xmult fires last, multiplying everything.

    When base_mult > 0, applies ordering correction so the caller formula
    ``(base_mult + bonus_mult) * xmult_product`` gives the correct result:
    ``((base_mult * during_xmult) + during_add + after_add) * after_xmult``

    Returns:
        (bonus_chips, bonus_mult, xmult_product)
    """
    bonus_chips = 0.0
    # Track during-card and after-card additive mult separately
    during_add_mult = 0.0
    after_add_mult = 0.0
    # Track during-card and after-card xmult separately
    during_xmult = 1.0
    after_xmult = 1.0

    scoring_cards = [cards[i] for i in scoring_indices if i < len(cards)]
    scoring_ranks = [card_rank(c) for c in scoring_cards]
    scoring_suits = [card_suit(c) for c in scoring_cards]
    # Effective suit sets per scoring card (Wild = all suits)
    scoring_suit_sets = [card_effective_suits(c) for c in scoring_cards]

    # Per-card-instance jokers (Greedy, Fibonacci, etc.) fire only on
    # SCORING cards, not kickers.

    # ── Retrigger detection ──
    # Hanging Chad: first scored card retriggers 2 extra times (3 total)
    # Sock and Buskin: all face cards retrigger 1 extra time
    # Hack: 2,3,4,5 retrigger 1 extra time
    # Seltzer: all cards retrigger 1 extra time (temporary)
    # Red Seal: card retriggers 1 extra time (per-card, from seal)
    first_card_retriggers = 0  # extra triggers on first scored card
    face_retriggers = 0        # extra triggers on each face card
    low_retriggers = 0         # extra triggers on 2,3,4,5 (Hack)
    all_retriggers = 0         # extra triggers on all cards
    red_seal_indices: set[int] = set()  # scoring card positions with Red Seal
    for j in jokers:
        jk = j.get("joker_key", "") or j.get("key", "")
        jn = _api_key_to_name(jk)
        if jn == "Hanging Chad":
            first_card_retriggers += 2
        elif jn == "Sock and Buskin":
            face_retriggers += 1
        elif jn == "Hack":
            low_retriggers += 1
        elif jn == "Seltzer":
            all_retriggers += 1
        elif jn == "Dusk":
            # Dusk: +1 retrigger on ALL cards on last hand of round
            hands_left_val = gamestate.get("round", {}).get("hands_left",
                              gamestate.get("current_round", {}).get("hands_left", 4))
            if hands_left_val <= 1:
                all_retriggers += 1

    # Detect Red Seal on scoring cards (retrigger source on the card itself)
    for card_pos, c in enumerate(scoring_cards):
        if card_seal(c) == "RED":
            red_seal_indices.add(card_pos)

    # Pre-compute held cards (cards NOT in the played hand) for held-in-hand triggers
    all_hand_cards = gamestate.get("hand", {}).get("cards", [])
    played_indices = set()
    # Cards passed to this function are the played cards — figure out which
    # hand cards are being held (not played)
    played_card_ids = set()
    for c in cards:
        played_card_ids.add(c.get("id", id(c)))
    held_cards = [c for c in all_hand_cards if c.get("id", id(c)) not in played_card_ids]
    held_ranks = [card_rank(c) for c in held_cards]
    held_suits = [card_suit(c) for c in held_cards]

    # Pre-compute context values for always-on triggers
    num_jokers = len(jokers)
    deck_cards = gamestate.get("cards", {}).get("cards", [])
    deck_remaining = len(deck_cards)
    money = gamestate.get("money", 0)
    hands_left = gamestate.get("round", {}).get("hands_left",
                  gamestate.get("current_round", {}).get("hands_left", 4))

    # Count uncommon jokers for Baseball Card
    uncommon_count = 0
    for j in jokers:
        jk = j.get("joker_key", "") or j.get("key", "")
        jname = _api_key_to_name(jk)
        if jname and jname in JOKERS:
            rarity = JOKERS[jname].get("rarity", "")
            if rarity == "uncommon":
                uncommon_count += 1

    # Resolve copy jokers (Blueprint/Brainstorm) — duplicate copied joker's scoring
    # Follows copy chains: Blueprint→Brainstorm→real joker resolves to real joker
    resolved_jokers = list(jokers)
    for idx, joker in enumerate(jokers):
        jk = joker.get("joker_key", "") or joker.get("key", "")
        jname = _api_key_to_name(jk)
        if not jname or jname not in JOKERS:
            continue
        schema = JOKERS[jname]
        if not schema.get("copy"):
            continue
        target_dir = schema.get("copy_target", "")
        copy_src = _resolve_copy_source(jokers, idx, target_dir)
        if copy_src is not None:
            resolved_jokers.append(copy_src)

    for joker in resolved_jokers:
        joker_key = joker.get("joker_key", "") or joker.get("key", "")
        # Convert API key to schema name
        name = _api_key_to_name(joker_key)
        if not name or name not in JOKERS:
            if joker_key and joker_key.startswith("j_"):
                _UNRESOLVED_KEYS.add(joker_key)
            continue

        schema = JOKERS[name]
        triggers = schema.get("triggers") or []
        score_effect = schema.get("score_effect") or []

        if not score_effect:
            continue

        # Check if this joker triggers for this hand
        triggered = False
        trigger_count = 1  # how many times it triggers (for per-card jokers)

        for trigger in triggers:
            if trigger == "any_hand_played":
                triggered = True

            elif trigger == "specific_hand_type":
                target_hand = schema.get("trigger_hand_type", "")
                if hand_type == target_hand:
                    triggered = True
                # "contains" type jokers — hand must contain the sub-hand
                if _hand_contains(hand_type, target_hand):
                    triggered = True

            elif trigger == "specific_suit":
                target_suits_set = set(schema.get("trigger_suits") or [])
                detail_logic = schema.get("trigger_detail_logic", "any")
                if detail_logic == "all":
                    # Must have ALL target suits in scoring cards (e.g. Flower Pot)
                    # Wild cards count as all suits
                    all_effective = set()
                    for ss in scoring_suit_sets:
                        all_effective |= ss
                    if target_suits_set.issubset(all_effective):
                        triggered = True
                elif schema.get("per_card_instance"):
                    # Fires on every SCORING card with matching suit (e.g. Greedy Joker)
                    # Wild cards match all suits
                    count = sum(1 for ss in scoring_suit_sets if ss & target_suits_set)
                    if count > 0:
                        triggered = True
                        trigger_count = count
                else:
                    if any(ss & target_suits_set for ss in scoring_suit_sets):
                        triggered = True

            elif trigger == "specific_rank":
                # Skip if card_held_in_hand is the primary trigger (Baron, Shoot the Moon,
                # Reserved Parking) — those are handled by card_held_in_hand above.
                # Also skip jokers with no scoring_timing (8 Ball = economy effect, not mult).
                if "card_held_in_hand" in triggers:
                    pass  # handled by card_held_in_hand trigger
                elif schema.get("scoring_timing") != "during_card":
                    pass  # not a scoring trigger (e.g. 8 Ball creates planets on economy)
                else:
                    target_ranks = _normalize_trigger_ranks(schema.get("trigger_ranks") or [])
                    if schema.get("per_card_instance"):
                        # Fires on every SCORING card with matching rank (e.g. Fibonacci)
                        count = sum(1 for r in scoring_ranks if r in target_ranks)
                        if count > 0:
                            triggered = True
                            trigger_count = count
                    else:
                        if any(r in target_ranks for r in scoring_ranks):
                            triggered = True

            elif trigger == "face_card":
                # Skip if card_held_in_hand is the primary trigger (Reserved Parking)
                if "card_held_in_hand" in triggers:
                    pass  # handled by card_held_in_hand trigger
                elif schema.get("per_card_instance"):
                    # Fires on every SCORING face card (e.g. Scary Face, Sock and Buskin)
                    count = sum(1 for r in scoring_ranks if r in FACE_RANKS)
                    if count > 0:
                        triggered = True
                        trigger_count = count
                else:
                    if any(r in FACE_RANKS for r in scoring_ranks):
                        triggered = True

            elif trigger == "scoring_card":
                if schema.get("per_card_instance"):
                    triggered = True
                    trigger_count = len(scoring_cards)
                elif scoring_cards:
                    triggered = True

            # ── NEW TRIGGERS ──

            elif trigger == "scoring_hand_size":
                # Half Joker: +20 mult if played hand ≤ 3 cards
                # Square Joker: scaling chips per card played (exactly 4)
                hand_size = len(cards)
                threshold = schema.get("event_count_threshold", 3)
                comparison = schema.get("event_count_comparison", "maximum")
                if comparison == "maximum" and hand_size <= threshold:
                    triggered = True
                elif comparison == "minimum" and hand_size >= threshold:
                    triggered = True
                elif comparison == "exact" and hand_size == threshold:
                    triggered = True

            elif trigger == "card_held_in_hand":
                # Baron: x1.5 per King held in hand
                # Shoot the Moon: +13 mult per Queen held
                # Blackboard: x3 if ALL held cards are spades or clubs
                if name == "Blackboard":
                    # x3 mult if all held cards are spades or clubs
                    if held_cards and all(s in ("Spades", "Clubs") for s in held_suits):
                        triggered = True
                    elif not held_cards:
                        triggered = True  # no held cards = condition trivially met
                elif name == "Raised Fist":
                    # +2× the rank of the lowest held card as mult
                    if held_cards:
                        triggered = True
                        # Special handling: use lowest rank value as mult
                        min_rank_val = min(CARD_CHIP_VALUES.get(r, 0) for r in held_ranks) if held_ranks else 0
                        # Override: we'll add 2 * min_rank as mult directly
                        after_add_mult += 2 * min_rank_val
                        continue  # skip normal effect application
                else:
                    # Per-card held triggers (Baron, Shoot the Moon)
                    target_ranks_held = _normalize_trigger_ranks(schema.get("trigger_ranks") or [])
                    if target_ranks_held:
                        count = sum(1 for r in held_ranks if r in target_ranks_held)
                        if count > 0:
                            triggered = True
                            trigger_count = count

            elif trigger == "per_joker_owned":
                # Abstract Joker: +3 mult per joker owned
                triggered = True
                trigger_count = num_jokers

            elif trigger == "per_card_remaining_in_deck":
                # Blue Joker: +2 chips per remaining deck card
                triggered = True
                trigger_count = deck_remaining

            elif trigger == "per_dollar_held":
                # Bull: +2 chips per $ held
                # Bootstraps: +2 mult per $5 held
                triggered = True
                if name == "Bootstraps":
                    trigger_count = money // 5
                else:
                    trigger_count = money

            elif trigger == "final_hand_of_round":
                # Acrobat: x3 mult on final hand
                if hands_left <= 1:
                    triggered = True

            elif trigger == "effect_probability":
                # Bloodstone: 50% x1.5 on Hearts scored
                # Lucky Cat, etc.
                prob = schema.get("effect_probability", 0.0)
                target_suits_prob = schema.get("trigger_suits") or []
                if target_suits_prob:
                    count = sum(1 for s in scoring_suits if s in target_suits_prob)
                    if count > 0:
                        triggered = True
                        # Weight by probability — EV calculation
                        trigger_count = count
                        # We'll apply prob-weighted effect below
                        timing = schema.get("scoring_timing", "after_cards")
                        for effect in score_effect:
                            if effect == "xmult":
                                xmult_val = schema.get("xmult_value") or 1.0
                                # EV of probability-gated xmult: (1-p) + p*xmult per card
                                for _ in range(count):
                                    ev_x = (1 - prob) + prob * xmult_val
                                    if timing == "during_card":
                                        during_xmult *= ev_x
                                    else:
                                        after_xmult *= ev_x
                            elif effect == "mult":
                                val = (schema.get("mult_value") or 0.0) * count * prob
                                if timing == "during_card":
                                    during_add_mult += val
                                else:
                                    after_add_mult += val
                            elif effect == "chips":
                                bonus_chips += (schema.get("chip_value") or 0.0) * count * prob
                        triggered = False  # already applied, skip normal path
                        continue

            elif trigger == "periodic":
                # Loyalty Card: x4 mult every 4th hand played
                # We can't know exact counter, so use EV: triggers 25% of the time
                threshold = schema.get("scaling_threshold") or 4
                triggered = True
                # Apply as probability-weighted (always after_cards timing)
                for effect in score_effect:
                    if effect == "xmult":
                        xmult_val = schema.get("xmult_value") or 1.0
                        p_trigger = 1.0 / max(threshold, 1)
                        after_xmult *= (1 - p_trigger) + p_trigger * xmult_val
                triggered = False  # already applied
                continue

            elif trigger == "per_specific_joker_present":
                # Baseball Card: x1.5 per uncommon joker
                if uncommon_count > 0:
                    triggered = True
                    trigger_count = uncommon_count

        if not triggered:
            # Scaling jokers with non-scoring triggers (Flash Card, Constellation,
            # Hologram, Campfire, Glass Joker, etc.) fire on EVERY hand in Balatro
            # but their trigger field describes what makes them grow, not when they
            # score.  If we have a tracked _scaled_value, apply it unconditionally.
            _sv = joker.get("_scaled_value")
            _st = schema.get("scaling_type")
            if _sv is not None and _st:
                triggered = True
                trigger_count = 1
            else:
                continue

        # Determine timing bucket for this joker's effects
        timing = schema.get("scoring_timing", "after_cards")

        # ── Apply retrigger bonuses to during_card effects ──
        # Retriggers cause a card to score again, re-firing all during_card jokers.
        # Hanging Chad: first scored card retriggers 2 extra times.
        # Sock and Buskin: each face card retriggers 1 extra time.
        # Seltzer: all cards retrigger 1 extra time.
        retrigger_bonus = 0  # extra trigger count from retriggers
        if timing == "during_card" and schema.get("per_card_instance"):
            # Per-card jokers: compute total extra triggers across SCORING cards
            # Retriggers only affect during_card scoring
            total_retrigger_extra = 0
            for card_pos, c in enumerate(scoring_cards):
                r = card_rank(c)
                s = card_suit(c)
                is_face = r in FACE_RANKS
                card_extra = all_retriggers
                if card_pos == 0:
                    card_extra += first_card_retriggers
                if is_face:
                    card_extra += face_retriggers
                if r in HACK_RANKS:
                    card_extra += low_retriggers
                if card_pos in red_seal_indices:
                    card_extra += 1  # Red Seal retrigger
                if card_extra > 0:
                    # Check if this card would fire this joker
                    fires = False
                    if "specific_suit" in triggers:
                        target_suits = set(schema.get("trigger_suits") or [])
                        # Wild cards match all suits
                        if card_effective_suits(c) & target_suits:
                            fires = True
                    if "specific_rank" in triggers:
                        target_ranks = _normalize_trigger_ranks(schema.get("trigger_ranks") or [])
                        if r in target_ranks:
                            fires = True
                    if "face_card" in triggers:
                        if is_face:
                            fires = True
                    if "scoring_card" in triggers:
                        fires = True
                    if fires:
                        total_retrigger_extra += card_extra
            retrigger_bonus = total_retrigger_extra
        elif timing == "during_card" and not schema.get("per_card_instance"):
            # Once-per-hand during_card jokers (e.g. Photograph):
            # If first SCORING card qualifies, retriggers compound the effect.
            # Hanging Chad retriggers first scoring card → Photograph fires 3x total = x2^3 = x8
            if scoring_cards:
                first_card = scoring_cards[0]
                if first_card:
                    first_rank = card_rank(first_card)
                    first_is_face = first_rank in FACE_RANKS
                    first_extra = all_retriggers + first_card_retriggers
                    if first_is_face:
                        first_extra += face_retriggers
                    if first_rank in HACK_RANKS:
                        first_extra += low_retriggers
                    if 0 in red_seal_indices:
                        first_extra += 1  # Red Seal on first card
                    if first_extra > 0:
                        qualifies = False
                        if "face_card" in triggers and first_is_face:
                            qualifies = True
                        if "specific_suit" in triggers:
                            target_suits = set(schema.get("trigger_suits") or [])
                            if card_effective_suits(first_card) & target_suits:
                                qualifies = True
                        if "scoring_card" in triggers:
                            qualifies = True
                        if qualifies:
                            retrigger_bonus = first_extra

        # For scaling jokers, use runtime accumulated value instead of static schema
        # The _scaled_value is injected by inject_scaling_values() from the ScalingTracker
        scaled_value = joker.get("_scaled_value")
        scaling_type = schema.get("scaling_type")

        # Apply retrigger bonus to trigger count for additive effects
        effective_trigger_count = trigger_count + retrigger_bonus

        # Apply effects (probability-weighted when effect_probability is set)
        prob = schema.get("effect_probability")
        for effect in score_effect:
            if effect == "chips":
                if scaling_type == "chips" and scaled_value is not None:
                    val = scaled_value  # already accumulated, don't multiply by trigger_count
                else:
                    val = (schema.get("chip_value") or 0.0) * effective_trigger_count
                bonus_chips += val * prob if prob else val
            elif effect == "mult":
                if scaling_type == "mult" and scaled_value is not None:
                    val = scaled_value
                else:
                    val = (schema.get("mult_value") or 0.0) * effective_trigger_count
                val = val * prob if prob else val
                if timing == "during_card":
                    during_add_mult += val
                else:
                    after_add_mult += val
            elif effect == "chips_and_mult":
                chip_val = (schema.get("chip_value") or 0.0) * effective_trigger_count
                mult_val = (schema.get("mult_value") or 0.0) * effective_trigger_count
                bonus_chips += chip_val * prob if prob else chip_val
                mult_val = mult_val * prob if prob else mult_val
                if timing == "during_card":
                    during_add_mult += mult_val
                else:
                    after_add_mult += mult_val
            elif effect == "xmult":
                if scaling_type == "xmult" and scaled_value is not None:
                    xmult_val = max(scaled_value, 1.0)
                else:
                    xmult_val = schema.get("xmult_value") or 1.0
                if prob:
                    # EV of probability-gated xmult: (1-p) + p*xmult per card
                    ev_per_card = (1 - prob) + prob * xmult_val
                    if schema.get("per_card_instance") and effective_trigger_count > 1:
                        x = ev_per_card ** effective_trigger_count
                    else:
                        # Once-per-hand xmult with retriggers: compound
                        x = ev_per_card ** max(1 + retrigger_bonus, 1)
                else:
                    if schema.get("per_card_instance") and effective_trigger_count > 1:
                        x = xmult_val ** effective_trigger_count
                    elif trigger_count > 1:
                        # Multi-trigger xmult (e.g. Baseball Card x1.5 per
                        # uncommon joker). Each trigger compounds.
                        x = xmult_val ** trigger_count
                    else:
                        # Once-per-hand xmult with retriggers: compound
                        # e.g. Photograph x2 retriggered 2x = x2^3 = x8
                        x = xmult_val ** max(1 + retrigger_bonus, 1)
                if timing == "during_card":
                    during_xmult *= x
                else:
                    after_xmult *= x

    # Apply joker EDITION bonuses (Foil +50 chips, Holo +10 mult, Polychrome x1.5)
    # Edition bonuses fire AFTER the joker's own scoring effect, effectively
    # after all card processing — treat as after_cards timing.
    for joker in jokers:
        modifier = joker.get("modifier", {})
        edition = modifier.get("edition", "") if isinstance(modifier, dict) else ""
        if edition == "FOIL":
            bonus_chips += 50
        elif edition == "HOLO":
            after_add_mult += 10
        elif edition == "POLYCHROME":
            after_xmult *= 1.5

    # ── Compute properly ordered result ──
    # In Balatro, scoring processes cards left-to-right:
    #   1. during_card xmult fires FIRST (on qualifying cards), multiplying
    #      only base_mult (accumulated so far)
    #   2. during_card additive mult fires on subsequent cards, stacking ON TOP
    #   3. after_card additive mult fires after all cards
    #   4. after_card xmult fires last, multiplying everything
    #
    # Correct formula:
    #   final = ((base_mult * during_xmult) + during_add + after_add) * after_xmult
    #
    # Caller formula: (base_mult + bonus_mult) * xmult_product
    #
    # To make caller formula match, we set:
    #   xmult_product = after_xmult
    #   bonus_mult = during_add + after_add + base_mult * (during_xmult - 1)
    #
    # This way: (base_mult + bonus_mult) * xmult_product
    #         = (base_mult + during_add + after_add + base_mult*(during_xmult-1)) * after_xmult
    #         = (base_mult*during_xmult + during_add + after_add) * after_xmult  ✓

    bonus_mult = during_add_mult + after_add_mult
    if base_mult > 0 and during_xmult != 1.0:
        # Apply ordering correction: during_xmult only amplifies base_mult
        bonus_mult += base_mult * (during_xmult - 1.0)
        xmult_product = after_xmult
    else:
        # Legacy mode (base_mult not provided) or no during_xmult:
        # fall back to old formula to avoid breaking callers that don't pass base_mult
        xmult_product = during_xmult * after_xmult

    return bonus_chips, bonus_mult, xmult_product


def _hand_contains(played_hand: str, target_hand: str) -> bool:
    """Check if a played hand type contains a target sub-hand.

    e.g. Full House contains Pair and Three of a Kind.
    """
    # Containment relationships in Balatro
    # Key = the sub-hand a joker triggers on.
    # Value = set of played hand types that CONTAIN that sub-hand.
    containment = {
        "Pair": {"Two Pair", "Three of a Kind", "Full House", "Flush House",
                 "Four of a Kind", "Five of a Kind", "Flush Five"},
        "Two Pair": {"Full House", "Flush House"},
        "Three of a Kind": {"Full House", "Flush House",
                            "Four of a Kind", "Five of a Kind", "Flush Five"},
        "Full House": {"Flush House"},
        "Four of a Kind": {"Five of a Kind", "Flush Five"},
        "Straight": {"Straight Flush"},
        "Flush": {"Straight Flush", "Flush House", "Flush Five"},
    }
    return played_hand in containment.get(target_hand, set())


def _api_key_to_name(key: str) -> Optional[str]:
    """Convert BalatroBot joker key to our schema name.

    Handles various API key formats and naming quirks.
    Uses a reverse lookup cache for O(1) matching.
    """
    if not key.startswith("j_"):
        return None

    # Use cached reverse lookup if available
    if key in _API_KEY_CACHE:
        return _API_KEY_CACHE[key]

    # Try direct conversion
    name = key[2:].replace("_", " ").title()

    # Fix common title-case issues
    name = name.replace("Dna", "DNA")
    name = name.replace("Mr ", "Mr. ")
    name = name.replace("Oops All 6S", "Oops! All 6s")
    # Fix articles/conjunctions that .title() over-capitalizes
    for word in [" The ", " And ", " A ", " An ", " Of ", " In ", " On ", " To "]:
        name = name.replace(word, word.lower())
    # But "The" at start of name stays capitalized
    if name.startswith("the "):
        name = "T" + name[1:]

    name = name.replace("Driver's license", "Driver's License")

    if name in JOKERS:
        _API_KEY_CACHE[key] = name
        return name

    # Try appending " Joker" — API sometimes omits it (e.g. j_jolly → Jolly Joker)
    with_joker = name + " Joker"
    if with_joker in JOKERS:
        _API_KEY_CACHE[key] = with_joker
        return with_joker

    # No match found
    _API_KEY_CACHE[key] = None
    return None


# Reverse lookup cache: API key → schema name
_API_KEY_CACHE: dict[str, Optional[str]] = {}
# Track unresolved API keys so we can catch missing misspellings
_UNRESOLVED_KEYS: set[str] = set()


def _build_api_key_cache() -> None:
    """Pre-build cache mapping API-style keys to schema names."""
    for name in JOKERS:
        # Generate the expected API key from the schema name
        key = "j_" + name.lower().replace(" ", "_").replace(".", "").replace("!", "").replace("'", "").replace("-", "_")
        _API_KEY_CACHE[key] = name

        # Also cache without " joker" suffix
        if name.endswith(" Joker"):
            short_name = name[:-6]  # Remove " Joker"
            short_key = "j_" + short_name.lower().replace(" ", "_").replace(".", "").replace("!", "").replace("'", "")
            _API_KEY_CACHE[short_key] = name

    # Balatro API key overrides — complete mapping from enums.lua for every
    # key that doesn't match our auto-generated "j_" + name pattern.
    # Source: balatrobot/src/lua/utils/enums.lua (card.config.center.key)
    _API_KEY_OVERRIDES = {
        # Misspellings in Balatro's internal keys
        "j_gluttenous_joker": "Gluttonous Joker",
        "j_selzer": "Seltzer",
        "j_caino": "Canio",
        # Abbreviated keys (API drops words from name)
        "j_chaos": "Chaos the Clown",
        "j_oops": "Oops! All 6s",
        "j_mail": "Mail-In Rebate",
        "j_delayed_grat": "Delayed Gratification",
        "j_trading": "Trading Card",
        "j_gift": "Gift Card",
        "j_business": "Business Card",
        "j_baseball": "Baseball Card",
        "j_flash": "Flash Card",
        "j_ceremonial": "Ceremonial Dagger",
        "j_smiley": "Smiley Face",
        "j_ticket": "Golden Ticket",
        "j_trousers": "Spare Trousers",
        "j_todo_list": "To Do List",
        "j_stencil": "Joker Stencil",
        # "The X" jokers — API drops "The"
        "j_duo": "The Duo",
        "j_trio": "The Trio",
        "j_family": "The Family",
        "j_order": "The Order",
        "j_tribe": "The Tribe",
        "j_idol": "The Idol",
        # Completely different key from name
        "j_ring_master": "Showman",
    }
    _API_KEY_CACHE.update(_API_KEY_OVERRIDES)


# Build cache on import
_build_api_key_cache()


# ============================================================
# Combo Enumeration
# ============================================================

def find_best_hands(hand_cards: list[dict], jokers: list[dict],
                    gamestate: dict, top_n: int = 3,
                    debuffed_suit: str | None = None,
                    boss_debuff_face: bool = False) -> list[dict]:
    """Find the top N scoring hands from available cards.

    Enumerates all C(n, k) combinations for k=1..5 and scores each.
    Respects boss blind debuffs — debuffed cards score 0 chips and don't
    trigger joker effects.

    Args:
        hand_cards: cards currently in hand
        jokers: owned jokers
        gamestate: full gamestate dict
        top_n: how many top hands to return
        debuffed_suit: if set, cards of this suit are debuffed (0 chips, no triggers)
        boss_debuff_face: if True, face cards are debuffed (The Plant)

    Returns:
        list of dicts sorted by score descending:
        [{hand_type, estimated_score, card_indices, scoring_indices, num_cards}]
    """
    n = len(hand_cards)
    if n == 0:
        return []

    results = []
    max_k = min(n, 5)

    for k in range(1, max_k + 1):
        for combo_indices in combinations(range(n), k):
            combo_cards = [hand_cards[i] for i in combo_indices]
            hand_type, local_scoring = classify_hand(combo_cards)

            # Filter out debuffed cards from scoring indices —
            # they still count for hand classification but score 0 and
            # don't trigger joker effects
            if debuffed_suit or boss_debuff_face:
                filtered_scoring = []
                for si in local_scoring:
                    c = combo_cards[si]
                    if debuffed_suit and card_suit(c) == debuffed_suit:
                        continue
                    if boss_debuff_face and card_rank(c) in FACE_RANKS:
                        continue
                    filtered_scoring.append(si)
            else:
                filtered_scoring = local_scoring

            # Map local scoring indices back to hand indices
            scoring_in_hand = [combo_indices[i] for i in filtered_scoring]

            score = estimate_score(
                hand_type, combo_cards, filtered_scoring,
                jokers, gamestate,
                debuffed_suit=debuffed_suit,
                boss_debuff_face=boss_debuff_face,
            )

            results.append({
                "hand_type": hand_type,
                "estimated_score": score,
                "card_indices": list(combo_indices),
                "scoring_indices": scoring_in_hand,
                "num_cards": k,
            })

    # Sort by score descending
    results.sort(key=lambda x: x["estimated_score"], reverse=True)
    return results[:top_n]


# ============================================================
# Played Card Ordering (Position-Aware Jokers)
# ============================================================

# Jokers that care about card position in the played hand
POSITION_JOKERS = {
    "Hanging Chad": "first",      # Retriggers first played card
    "Photograph": "first_face",   # x2 mult on first face card played
    "Sock and Buskin": "face",    # Retriggers all face cards (order less important)
}


def optimize_play_order(card_indices: list[int], hand_cards: list[dict],
                        jokers: list[dict]) -> list[int]:
    """Reorder played card indices to maximize scoring from position-aware jokers.

    Key jokers:
    - Hanging Chad: retriggers first played card → put highest-value card first
    - Photograph: x2 mult on first face card → put best face card first
    - If both: best face card first satisfies both

    Args:
        card_indices: 0-based indices into hand_cards of cards to play
        hand_cards: all cards currently in hand
        jokers: owned jokers list

    Returns:
        Reordered list of card_indices
    """
    if len(card_indices) <= 1:
        return card_indices

    # Check which position-aware jokers we have
    has_hanging_chad = False
    has_photograph = False
    for joker in jokers:
        jk = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(jk)
        if name == "Hanging Chad":
            has_hanging_chad = True
        elif name == "Photograph":
            has_photograph = True

    if not has_hanging_chad and not has_photograph:
        return card_indices  # No position-aware jokers, order doesn't matter

    # Identify which of the played cards are scoring vs kickers
    played_cards = [hand_cards[i] for i in card_indices]
    hand_type, local_scoring = classify_hand(played_cards)
    scoring_idx_set = set(local_scoring)  # indices into played_cards

    def _card_score(pos: int, idx: int) -> float:
        """Score a card for first-position value. Higher = better to put first.
        pos = position in the played card list, idx = index in hand_cards."""
        is_scoring = pos in scoring_idx_set
        card = hand_cards[idx]
        rank = card_rank(card)
        is_face = rank in FACE_RANKS
        chip_val = CARD_CHIP_VALUES.get(rank, 0)
        score = float(chip_val)

        if not is_scoring:
            # Kickers should never be first — they don't trigger anything
            return -1000.0

        # Enhancement bonuses for retrigger value
        enh = card_enhancement(card)
        if enh == "GLASS":
            score *= 2.0
        elif enh == "MULT":
            score += 4.0
        elif enh == "BONUS":
            score += 30.0
        elif enh == "LUCKY":
            score += 4.0

        # Edition bonuses
        ed = card_edition(card)
        if ed == "POLYCHROME":
            score *= 1.5
        elif ed == "HOLO":
            score += 10.0
        elif ed == "FOIL":
            score += 50.0

        # Red Seal: retriggers this card — being first with Hanging Chad
        # means 4 total triggers (1 base + 2 Chad + 1 Red Seal)
        seal = card_seal(card)
        if seal == "RED":
            score *= 1.5  # Red Seal cards are worth more in first position

        # Photograph: massive bonus for face cards being first
        if has_photograph and is_face:
            score += 200.0

        # Hanging Chad: retrigger bonus — highest value card first
        if has_hanging_chad:
            if is_face:
                score += 50.0
            if enh == "GLASS":
                score += 100.0
            if seal == "RED":
                score += 80.0  # Red Seal + Hanging Chad = 4 triggers

        return score

    # Sort: highest score first — scoring cards before kickers
    scored = [(idx, _card_score(pos, idx)) for pos, idx in enumerate(card_indices)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored]


# ============================================================
# Draw Probability Computation
# ============================================================

def compute_draw_outs(hand_cards: list[dict], deck_cards: list[dict],
                      gamestate: dict) -> dict[str, float]:
    """Compute approximate probability of drawing into each hand type
    after discarding, given known deck composition.

    Uses a simplified model: for each hand type, estimate the probability
    of completing it by drawing from the remaining deck.

    Returns:
        dict of {hand_type: probability} for hand types Pair and above
    """
    hand_ranks = Counter(card_rank(c) for c in hand_cards)
    hand_suits = Counter(card_suit(c) for c in hand_cards)
    deck_ranks = Counter(card_rank(c) for c in deck_cards)
    deck_suits = Counter(card_suit(c) for c in deck_cards)
    deck_size = len(deck_cards)

    if deck_size == 0:
        return {ht: 0.0 for ht in HAND_TYPES[1:]}  # skip High Card

    probs = {}

    # P(Pair+): probability of drawing a card matching any hand rank
    matching_in_deck = sum(deck_ranks.get(r, 0) for r in hand_ranks if hand_ranks[r] >= 1)
    probs["Pair"] = min(matching_in_deck / deck_size, 1.0) if hand_ranks else 0.0

    # P(Three of a Kind+): need 2 more of a rank we have 1 of, or 1 more of a pair
    pair_ranks = [r for r, c in hand_ranks.items() if c == 2]
    if pair_ranks:
        # Already have pair, need 1 more match
        best_count = max(deck_ranks.get(r, 0) for r in pair_ranks)
        probs["Three of a Kind"] = min(best_count / deck_size, 1.0)
    else:
        probs["Three of a Kind"] = 0.0

    # P(Two Pair+): if we have 1 pair, need another pair
    if pair_ranks:
        non_pair_ranks = [r for r in hand_ranks if hand_ranks[r] == 1]
        matching = sum(deck_ranks.get(r, 0) for r in non_pair_ranks)
        probs["Two Pair"] = min(matching / deck_size, 1.0)
    else:
        probs["Two Pair"] = probs["Pair"]  # need any match

    # P(Straight): count cards that would complete a straight
    straight_outs = _count_straight_outs(hand_ranks, deck_ranks)
    probs["Straight"] = min(straight_outs / max(deck_size, 1), 1.0)

    # P(Flush): count cards of our most common suit in deck
    if hand_suits:
        best_suit = hand_suits.most_common(1)[0]
        suit_name, suit_count = best_suit
        needed = 5 - suit_count
        available = deck_suits.get(suit_name, 0)
        if needed <= 0:
            probs["Flush"] = 1.0
        elif needed == 1:
            probs["Flush"] = min(available / deck_size, 1.0)
        else:
            # Rough estimate: P(getting `needed` of suit from draws)
            probs["Flush"] = _hypergeometric_approx(available, deck_size, needed)
    else:
        probs["Flush"] = 0.0

    # P(Full House): need trip + pair
    trip_ranks = [r for r, c in hand_ranks.items() if c >= 3]
    if trip_ranks and pair_ranks:
        probs["Full House"] = 0.8  # already have it or close
    elif trip_ranks:
        matching = sum(deck_ranks.get(r, 0) for r in hand_ranks if hand_ranks[r] < 3)
        probs["Full House"] = min(matching / deck_size, 1.0) * 0.5
    elif len(pair_ranks) >= 2:
        best_count = max(deck_ranks.get(r, 0) for r in pair_ranks)
        probs["Full House"] = min(best_count / deck_size, 1.0)
    else:
        probs["Full House"] = 0.0

    # P(Four of a Kind)
    if trip_ranks:
        best_count = max(deck_ranks.get(r, 0) for r in trip_ranks)
        probs["Four of a Kind"] = min(best_count / deck_size, 1.0)
    else:
        probs["Four of a Kind"] = 0.0

    # P(Straight Flush): intersection of straight and flush potential
    probs["Straight Flush"] = probs["Straight"] * probs["Flush"] * 0.3

    # Rare hands — very low baseline
    probs["Five of a Kind"] = 0.0
    probs["Flush House"] = probs["Full House"] * probs["Flush"] * 0.2
    probs["Flush Five"] = 0.0

    return probs


def _count_straight_outs(hand_ranks: Counter, deck_ranks: Counter) -> int:
    """Count cards in deck that could contribute to a straight."""
    hand_values = set()
    for r in hand_ranks:
        v = RANK_ORDER.get(r, 0)
        if v:
            hand_values.add(v)
            if v == 14:  # Ace can be low
                hand_values.add(1)

    outs = 0
    # Check each possible 5-card straight window
    for low in range(1, 11):  # 1-5 through 10-14
        window = set(range(low, low + 5))
        have = window & hand_values
        need = window - hand_values
        if len(need) <= 2 and len(have) >= 3:
            # Count cards in deck that fill the gaps
            for v in need:
                rank = _value_to_rank(v)
                if rank:
                    outs += deck_ranks.get(rank, 0)

    return outs


_RANK_ORDER_INV = {v: k for k, v in RANK_ORDER.items()}


def _value_to_rank(value: int) -> Optional[str]:
    """Convert numeric value back to rank string."""
    if value == 1:
        return "A"  # Ace low
    return _RANK_ORDER_INV.get(value)


def _hypergeometric_approx(successes_in_pop: int, pop_size: int,
                            needed: int) -> float:
    """Rough approximation of P(drawing at least `needed` successes)."""
    if needed <= 0:
        return 1.0
    if successes_in_pop < needed:
        return 0.0
    # Simple approximation: (s/n)^needed * combinations factor
    p = successes_in_pop / max(pop_size, 1)
    return min(p ** needed * 2.0, 1.0)  # *2 for combinatorial correction


# ============================================================
# Deck Awareness
# ============================================================

def compute_deck_features(deck_cards: list[dict]) -> dict:
    """Compute deck composition features.

    Returns:
        dict with suit_ratios, rank_stats, etc.
    """
    n = len(deck_cards)
    if n == 0:
        return {
            "suit_ratios": [0.25, 0.25, 0.25],
            "top_rank_ratio": 0.0,
            "face_ratio": 0.0,
            "ace_ratio": 0.0,
            "high_ratio": 0.0,
            "entropy": 1.0,
        }

    suit_counts = Counter(card_suit(c) for c in deck_cards)
    rank_counts = Counter(card_rank(c) for c in deck_cards)

    # Sort suits by count descending, take top 3 ratios
    sorted_suits = sorted(suit_counts.values(), reverse=True)
    while len(sorted_suits) < 4:
        sorted_suits.append(0)
    suit_ratios = [sorted_suits[i] / n for i in range(3)]

    # Most common rank
    top_rank_ratio = max(rank_counts.values()) / n if rank_counts else 0.0

    # Face cards (J, Q, K)
    face_count = sum(rank_counts.get(r, 0) for r in FACE_RANKS)
    face_ratio = face_count / n

    # Aces
    ace_ratio = rank_counts.get("A", 0) / n

    # High cards (T, J, Q, K, A)
    high_count = sum(rank_counts.get(r, 0) for r in ["T", "J", "Q", "K", "A"])
    high_ratio = high_count / n

    # Shannon entropy of suit distribution (normalized to 0-1)
    entropy = 0.0
    for count in suit_counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    max_entropy = math.log2(4)  # uniform across 4 suits
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "suit_ratios": suit_ratios,
        "top_rank_ratio": top_rank_ratio,
        "face_ratio": face_ratio,
        "ace_ratio": ace_ratio,
        "high_ratio": high_ratio,
        "entropy": normalized_entropy,
    }


# ============================================================
# Strategic Assessment — Main Entry Point
# ============================================================

def assess_strategy(gamestate: dict) -> np.ndarray:
    """Compute all 40 hand evaluation features for the state vector.

    Only meaningful during SELECTING_HAND state.

    Args:
        gamestate: full gamestate dict from the API

    Returns:
        numpy array of shape (40,) with features for indices 520-559
    """
    features = np.zeros(HAND_EVAL_FEATURES, dtype=np.float32)

    # Extract key data from gamestate
    # API structure: hand cards at gs["hand"]["cards"], deck at gs["cards"]["cards"]
    hand_cards = gamestate.get("hand", {}).get("cards", [])
    deck_cards = gamestate.get("cards", {}).get("cards", [])
    jokers_raw = gamestate.get("jokers", {}).get("cards", [])
    round_data = gamestate.get("round", {})
    hands_data = gamestate.get("hands", {})

    if not hand_cards:
        return features

    # Get blind target
    blind_target = _get_blind_target(gamestate)
    chips_scored = round_data.get("chips", 0)
    remaining_target = max(blind_target - chips_scored, 0)
    hands_left = round_data.get("hands_left", 0)
    discards_left = round_data.get("discards_left", 0)

    # ── Best Hand Info (8 features, 0-7) ──
    top_hands = find_best_hands(hand_cards, jokers_raw, gamestate, top_n=3)

    if top_hands:
        best = top_hands[0]
        best_score = best["estimated_score"]
        best_type = best["hand_type"]
        best_type_idx = HAND_TYPE_INDEX.get(best_type, 0)

        features[0] = best_type_idx / (NUM_HAND_TYPES - 1)  # normalized type index
        features[1] = _log_scale(best_score, 6.0)
        features[2] = min(best_score / max(remaining_target, 1), 3.0) / 3.0
        features[3] = 1.0 if best_score >= remaining_target else 0.0
        features[6] = best["num_cards"] / 5.0

        # 2nd and 3rd best ratios
        if len(top_hands) >= 2 and best_score > 0:
            features[4] = top_hands[1]["estimated_score"] / best_score
        if len(top_hands) >= 3 and best_score > 0:
            features[5] = top_hands[2]["estimated_score"] / best_score

        # Joker trigger match
        features[7] = 1.0 if _any_joker_triggers(best_type, jokers_raw) else 0.0

    # ── Strategic Assessment (6 features, 8-13) ──
    max_hands = round_data.get("hands_left", 4)  # rough max from game start
    max_discards = round_data.get("discards_left", 3)

    features[8] = min(hands_left / max(max_hands, 1), 1.0)
    features[9] = min(discards_left / max(max_discards, 1), 1.0)
    features[10] = min(chips_scored / max(blind_target, 1), 1.0)

    # Should discard heuristic
    if top_hands and discards_left > 0:
        draw_probs = compute_draw_outs(hand_cards, deck_cards, gamestate)
        # Check if any draw prob for better hand type is significant
        best_type_idx_val = HAND_TYPE_INDEX.get(top_hands[0]["hand_type"], 0)
        can_improve = any(
            draw_probs.get(ht, 0) > 0.3
            for ht in HAND_TYPES[best_type_idx_val + 1:]
        )
        features[11] = 1.0 if can_improve else 0.0
    else:
        features[11] = 0.0

    # Risk level: high if few hands left and not close to target
    if hands_left > 0 and top_hands:
        best_score = top_hands[0]["estimated_score"]
        hands_needed = math.ceil(remaining_target / max(best_score, 1))
        risk = min(hands_needed / max(hands_left, 1), 1.0)
        features[12] = risk
        features[13] = min(hands_needed / max(hands_left, 1), 1.0)
    else:
        features[12] = 1.0
        features[13] = 1.0

    # ── Draw/Discard Potential (13 features, 14-26) ──
    draw_probs = compute_draw_outs(hand_cards, deck_cards, gamestate)
    draw_hand_types = [
        "Pair", "Two Pair", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush",
    ]
    for i, ht in enumerate(draw_hand_types):
        features[14 + i] = draw_probs.get(ht, 0.0)

    # Best discard count (heuristic: discard non-scoring cards)
    if top_hands:
        best_cards_used = set(top_hands[0]["card_indices"])
        non_scoring = [i for i in range(len(hand_cards)) if i not in best_cards_used]
        features[22] = min(len(non_scoring), 5) / 5.0

        # Expected score after discard (approximate as current best)
        features[23] = _log_scale(top_hands[0]["estimated_score"], 6.0)
        features[24] = 1.0  # improvement ratio placeholder
    else:
        features[22] = 0.0
        features[23] = 0.0
        features[24] = 0.0

    features[25] = len(deck_cards) / 52.0
    avg_chip = sum(card_chips(c) for c in deck_cards) / max(len(deck_cards), 1)
    features[26] = min(avg_chip / 11.0, 1.0)

    # ── Score Context (5 features, 27-31) ──
    if top_hands:
        best = top_hands[0]
        _, j_chips, j_mult, j_xmult = _get_joker_totals(
            best["hand_type"], hand_cards, best.get("scoring_indices", []),
            jokers_raw, gamestate
        )
        features[27] = _log_scale(j_chips, 3.0)
        features[28] = _log_scale(j_mult, 2.0)
        features[29] = _log_scale(j_xmult, 1.5)

        # Planet level for best hand type
        hand_info = hands_data.get(best["hand_type"], {})
        features[30] = min(hand_info.get("level", 1) / 10.0, 1.0)
        features[31] = len(best.get("scoring_indices", [])) / 5.0

    # ── Deck Awareness (8 features, 32-39) ──
    deck_feats = compute_deck_features(deck_cards)
    features[32] = deck_feats["suit_ratios"][0]
    features[33] = deck_feats["suit_ratios"][1]
    features[34] = deck_feats["suit_ratios"][2]
    features[35] = deck_feats["top_rank_ratio"]
    features[36] = deck_feats["face_ratio"]
    features[37] = deck_feats["ace_ratio"]
    features[38] = deck_feats["high_ratio"]
    features[39] = deck_feats["entropy"]

    return features


# ============================================================
# Internal Helpers
# ============================================================

# ============================================================
# Discard Advisor — Expected Value Calculator
# ============================================================

def find_best_discard(hand_cards: list[dict], deck_cards: list[dict],
                      jokers: list[dict], gamestate: dict) -> dict:
    """Find the best discard strategy by computing expected value.

    Evaluates key discard strategies:
    - Keep best current hand (discard the rest)
    - Keep flush draw cards (discard non-suited)
    - Keep straight draw cards (discard non-connected)
    - Keep pairs/trips (discard singletons)

    For each strategy, estimates EV = P(improved hand) * score(improved)
    + P(no improvement) * score(fallback).

    Returns:
        dict with:
            keep_indices: set of card indices to KEEP (discard the rest)
            strategy: name of chosen strategy
            expected_score: estimated EV
            discard_indices: set of card indices to DISCARD
    """
    n = len(hand_cards)
    if n == 0:
        return {"keep_indices": set(), "discard_indices": set(),
                "strategy": "none", "expected_score": 0.0}

    deck_size = len(deck_cards)
    deck_suits = Counter(card_suit(c) for c in deck_cards)
    deck_ranks = Counter(card_rank(c) for c in deck_cards)

    # Current best hand (no discard)
    best_current = find_best_hands(hand_cards, jokers, gamestate, top_n=1)
    current_score = best_current[0]["estimated_score"] if best_current else 0.0
    current_keep = set(best_current[0]["card_indices"]) if best_current else set(range(n))

    candidates = []

    # Strategy 1: Keep current best hand, discard rest
    candidates.append({
        "keep_indices": current_keep,
        "strategy": "keep_best",
        "expected_score": current_score,
    })

    # Strategy 2: Flush draw — keep cards of the most common suit
    suit_groups: dict[str, list[int]] = {}
    for i in range(n):
        s = card_suit(hand_cards[i])
        suit_groups.setdefault(s, []).append(i)

    for suit, indices in suit_groups.items():
        if len(indices) < 3:
            continue  # need at least 3 of a suit to consider
        needed = 5 - len(indices)
        available = deck_suits.get(suit, 0)
        num_draws = min(n - len(indices), 5)  # how many cards we'd discard/redraw

        if needed <= 0:
            # Already have flush
            flush_cards = [hand_cards[i] for i in indices[:5]]
            ht, si = classify_hand(flush_cards)
            ev = estimate_score(ht, flush_cards, si, jokers, gamestate)
            candidates.append({
                "keep_indices": set(indices[:5]),
                "strategy": f"flush_{suit}",
                "expected_score": ev,
            })
        elif needed <= num_draws and deck_size > 0:
            # Calculate P(completing flush)
            p_flush = _draw_probability(available, deck_size, needed, num_draws)

            # Score if flush hits — use API chips/mult + joker effects
            flush_info = gamestate.get("hands", {}).get("Flush", {})
            flush_base_chips_default, flush_base_mult_default = BASE_HAND_SCORES["Flush"]
            flush_chips = flush_info.get("chips", flush_base_chips_default)
            flush_mult = flush_info.get("mult", flush_base_mult_default)
            kept_chips = sum(card_chips(hand_cards[i]) for i in indices)
            avg_deck_chip = sum(card_chips(c) for c in deck_cards) / max(deck_size, 1)
            drawn_chips = avg_deck_chip * needed
            # Estimate joker effects on a COMPLETED 5-card flush of this suit.
            # Build a synthetic 5-card hand: kept cards + filler cards of the same suit
            # so per-card-instance jokers (Onyx Agate, Bloodstone) score all 5 cards.
            kept = [hand_cards[i] for i in indices]
            filler = [c for c in deck_cards if card_suit(c) == suit][:needed]
            if len(filler) < needed:
                # Not enough same-suit deck cards — duplicate last kept card as stand-in
                filler.extend([kept[-1]] * (needed - len(filler)))
            projected_flush = (kept + filler)[:5]
            j_chips, j_mult, j_xmult = compute_joker_scoring(
                "Flush", projected_flush, list(range(len(projected_flush))),
                jokers, gamestate, base_mult=flush_mult,
            )
            flush_score = (flush_chips + kept_chips + drawn_chips + j_chips) * (flush_mult + j_mult) * j_xmult

            # Score if flush misses — fall back to best hand from kept cards
            kept_cards = [hand_cards[i] for i in indices]
            ht, si = classify_hand(kept_cards)
            fallback_score = estimate_score(ht, kept_cards, si, jokers, gamestate)

            ev = p_flush * flush_score + (1 - p_flush) * fallback_score
            candidates.append({
                "keep_indices": set(indices),
                "strategy": f"flush_draw_{suit}",
                "expected_score": ev,
            })

    # Strategy 3: Straight draw — keep connected cards
    rank_positions: dict[int, list[int]] = {}
    for i in range(n):
        r = card_rank(hand_cards[i])
        v = RANK_ORDER.get(r, 0)
        if v:
            rank_positions.setdefault(v, []).append(i)
            if v == 14:  # Ace can be low
                rank_positions.setdefault(1, []).append(i)

    # Check each straight window
    best_straight_ev = 0.0
    best_straight_keep: set[int] = set()
    for low in range(1, 11):
        window = set(range(low, low + 5))
        keep = []
        for v in window:
            if v in rank_positions:
                keep.append(rank_positions[v][0])  # take first card of that rank
        keep = list(set(keep))  # deduplicate (ace appearing twice)

        if len(keep) < 4:
            continue  # need at least 4 connected for a worthwhile draw

        needed = 5 - len(keep)
        num_draws = min(n - len(keep), 5)
        missing_values = window - set(v for v in rank_positions if v in window)

        if needed <= num_draws and needed <= 2 and deck_size > 0:
            # Count outs in deck
            outs = 0
            for v in missing_values:
                rank = _value_to_rank(v)
                if rank:
                    outs += deck_ranks.get(rank, 0)

            p_straight = _draw_probability(outs, deck_size, needed, num_draws)

            straight_info = gamestate.get("hands", {}).get("Straight", {})
            straight_chips_default, straight_mult_default = BASE_HAND_SCORES["Straight"]
            straight_chips = straight_info.get("chips", straight_chips_default)
            straight_mult = straight_info.get("mult", straight_mult_default)
            # Estimate joker effects on straight
            kept_cards_for_score = [hand_cards[i] for i in keep]
            j_chips, j_mult, j_xmult = compute_joker_scoring(
                "Straight", kept_cards_for_score, list(range(len(kept_cards_for_score))),
                jokers, gamestate, base_mult=straight_mult,
            )
            straight_score = (straight_chips + 40 + j_chips) * (straight_mult + j_mult) * j_xmult

            # Fallback
            kept_cards = [hand_cards[i] for i in keep]
            ht, si = classify_hand(kept_cards)
            fallback_score = estimate_score(ht, kept_cards, si, jokers, gamestate)

            ev = p_straight * straight_score + (1 - p_straight) * fallback_score
            if ev > best_straight_ev:
                best_straight_ev = ev
                best_straight_keep = set(keep)

    if best_straight_keep:
        candidates.append({
            "keep_indices": best_straight_keep,
            "strategy": "straight_draw",
            "expected_score": best_straight_ev,
        })

    # Strategy 4: Keep pairs/trips, discard singletons
    rank_count = Counter(card_rank(hand_cards[i]) for i in range(n))
    paired_indices = set()
    for i in range(n):
        r = card_rank(hand_cards[i])
        if rank_count[r] >= 2:
            paired_indices.add(i)

    if paired_indices and len(paired_indices) < n:
        # Keep paired cards, draw replacements for singletons
        num_draws = min(n - len(paired_indices), 5)
        # EV: current pair score + chance of improving
        kept_cards = [hand_cards[i] for i in sorted(paired_indices)]
        ht, si = classify_hand(kept_cards)
        base_ev = estimate_score(ht, kept_cards, si, jokers, gamestate)

        # Chance of drawing into trips/full house
        for r, cnt in rank_count.items():
            if cnt == 2:
                in_deck = deck_ranks.get(r, 0)
                p_trip = _draw_probability(in_deck, deck_size, 1, num_draws) if deck_size > 0 else 0
                trip_bonus = estimate_score("Three of a Kind", kept_cards, list(range(len(kept_cards))), jokers, gamestate)
                base_ev += p_trip * (trip_bonus - base_ev) * 0.5  # weighted improvement

        candidates.append({
            "keep_indices": paired_indices,
            "strategy": "keep_pairs",
            "expected_score": base_ev,
        })

    # Pick the best strategy
    best = max(candidates, key=lambda c: c["expected_score"])
    all_indices = set(range(n))
    best["discard_indices"] = all_indices - best["keep_indices"]

    return best


# ============================================================
# Strategic Play/Discard Advisor
# ============================================================

def _card_joker_attractiveness(card: dict, jokers: list[dict]) -> float:
    """Estimate a card's bonus value from joker triggers.

    Returns a score bonus reflecting how much extra value this card generates
    when scored, considering all owned jokers' rank/suit/face triggers.
    Higher = better card to include in projected hands.

    Handles BOTH per_card_instance jokers (fire on every qualifying card)
    AND once-per-hand jokers (fire on first qualifying card, e.g. Photograph).
    Once-per-hand jokers give a smaller bonus (÷3) since only one card needs
    to qualify, but the bonus is still large enough to ensure at least one
    qualifying card is included in the projected hand.
    """
    bonus = 0.0
    rank = card_rank(card)
    suit = card_suit(card)
    is_face = rank in FACE_RANKS

    for joker in jokers:
        jk = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(jk)
        if not name or name not in JOKERS:
            continue
        schema = JOKERS[name]
        triggers = schema.get("triggers") or []
        score_effect = schema.get("score_effect") or []

        if not score_effect:
            continue

        # Skip triggers that aren't card-quality-dependent
        # (any_hand_played, scoring_hand_size, card_held_in_hand, etc.
        #  fire regardless of which cards are in the hand)
        card_triggers = {"specific_rank", "specific_suit", "face_card", "scoring_card"}
        if not any(t in card_triggers for t in triggers):
            continue

        # Check if this card would trigger this joker
        fires = False
        if "specific_rank" in triggers:
            target_ranks = _normalize_trigger_ranks(schema.get("trigger_ranks") or [])
            if rank in target_ranks:
                fires = True
        if "specific_suit" in triggers:
            target_suits = schema.get("trigger_suits") or []
            if suit in target_suits:
                fires = True
        if "face_card" in triggers:
            if is_face:
                fires = True
        if "scoring_card" in triggers:
            fires = True

        if not fires:
            continue

        is_per_card = schema.get("per_card_instance", False)
        # Per-card jokers fire on EVERY qualifying card → full bonus each
        # Once-per-hand jokers fire on first qualifying card → amortized bonus
        # (÷3 so that at least one qualifying card beats non-qualifying ones,
        #  but doesn't over-inflate when multiple qualifying cards exist)
        scale = 1.0 if is_per_card else 0.33

        # Estimate the value this trigger adds
        prob = schema.get("effect_probability") or 1.0
        for effect in score_effect:
            if effect == "xmult":
                xmult_val = schema.get("xmult_value") or 1.0
                # xmult is multiplicative — even a single x2 is huge
                # Use a large bonus to ensure these cards are prioritized
                bonus += (xmult_val - 1.0) * 1000.0 * prob * scale
            elif effect == "mult":
                bonus += (schema.get("mult_value") or 0.0) * prob * scale
            elif effect == "chips":
                bonus += (schema.get("chip_value") or 0.0) * prob * scale
            elif effect == "chips_and_mult":
                bonus += (schema.get("chip_value") or 0.0) * prob * scale
                bonus += (schema.get("mult_value") or 0.0) * prob * scale

    return bonus


def _project_hand_score(hand_type: str, num_scoring_cards: int,
                        kept_cards: list[dict], jokers: list[dict],
                        gamestate: dict, extra_chips: float = 0.0,
                        target_suit: str | None = None,
                        deck_cards: list[dict] | None = None,
                        target_ranks: list[str] | None = None) -> float:
    """Project the score for a hand type by building a realistic projected hand.

    Instead of scaling joker effects from partial cards, this fills in
    missing cards from the deck and scores the full projected hand through
    compute_joker_scoring for accurate joker interaction.

    Filler cards are ranked by joker attractiveness + chip value, so cards
    that trigger jokers (e.g. Kings/Queens for Triboulet) are preferred.
    """
    need = num_scoring_cards - len(kept_cards)

    # Pre-compute joker attractiveness for filler sorting
    def _filler_sort_key(c: dict) -> float:
        return _card_joker_attractiveness(c, jokers) + card_chips(c)

    if need > 0 and deck_cards:
        # Build projected hand: kept cards + best matching cards from deck
        filler: list[dict] = []
        if target_suit and target_ranks:
            # Straight Flush: grab cards matching BOTH suit AND rank
            for rank in target_ranks:
                for c in deck_cards:
                    if card_rank(c) == rank and card_suit(c) == target_suit and c not in filler:
                        filler.append(c)
                        break
            filler = filler[:need]
        elif target_suit:
            # Flush: grab best cards of the target suit from deck
            # Ranked by joker trigger value + chip value (not just chips)
            suit_cards = [c for c in deck_cards if card_suit(c) == target_suit]
            suit_cards.sort(key=_filler_sort_key, reverse=True)
            filler = suit_cards[:need]
        elif target_ranks:
            # Straight: grab one card per missing rank from deck
            # When multiple cards share a rank, pick the one with best joker value
            for rank in target_ranks:
                candidates = [c for c in deck_cards if card_rank(c) == rank and c not in filler]
                if candidates:
                    candidates.sort(key=_filler_sort_key, reverse=True)
                    filler.append(candidates[0])
            filler = filler[:need]

        # If we couldn't find enough filler, pad with generic estimates
        if len(filler) < need:
            extra_chips += 7.0 * (need - len(filler))

        projected_cards = list(kept_cards) + filler
    else:
        projected_cards = list(kept_cards)

    hand_info = gamestate.get("hands", {}).get(hand_type, {})
    base_c, base_m = BASE_HAND_SCORES.get(hand_type, (5, 1))
    h_chips = hand_info.get("chips", base_c)
    h_mult = hand_info.get("mult", base_m)

    if len(projected_cards) == 0:
        # No cards at all — fall back to generic estimator
        # Compute deck suit fractions if deck is available
        _ds: Optional[dict[str, float]] = None
        if deck_cards:
            _suit_expand = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
            _scounts: dict[str, int] = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
            for _c in deck_cards:
                _sv = _c.get("value", {})
                _ss = _suit_expand.get(_sv.get("suit", ""), "")
                if _ss in _scounts:
                    _scounts[_ss] += 1
            _stotal = max(len(deck_cards), 1)
            _ds = {s: n / _stotal for s, n in _scounts.items()}
        j_chips, j_mult, j_xmult = _estimate_joker_scoring_for_type(
            hand_type, jokers, gamestate, dominant_suit=target_suit,
            deck_suits=_ds,
        )
    else:
        # Score the full projected hand with all jokers — no scaling needed
        scoring_indices = list(range(len(projected_cards)))
        j_chips, j_mult, j_xmult = compute_joker_scoring(
            hand_type, projected_cards, scoring_indices, jokers, gamestate,
            base_mult=h_mult,
        )
    card_chip_sum = sum(card_chips(c) for c in projected_cards) + extra_chips

    total_score = (h_chips + card_chip_sum + j_chips) * (h_mult + j_mult) * j_xmult

    # Log projection breakdown for Flush/Straight Flush to diagnose low scores
    if hand_type in ("Flush", "Straight Flush") and projected_cards:
        proj_ranks = [card_rank(c) for c in projected_cards]
        proj_suits = [card_suit(c) for c in projected_cards]
        face_count = sum(1 for r in proj_ranks if r in FACE_RANKS)
    return total_score


# ============================================================
# Round-Level Strategy Planner
# ============================================================

# Module-level strategy cache — persists across calls within a round
_round_strategy: dict = {
    "fingerprint": None,
    "committed_type": None,  # e.g. "Flush:Clubs", "Straight", "Four of a Kind:K"
}


def _multi_discard_probability(outs: int, deck_size: int, needed: int,
                                cards_per_discard: int, num_discards: int) -> float:
    """P(accumulating at least `needed` outs across `num_discards` sequential discards).

    Each discard draws `cards_per_discard` cards from a deck of `deck_size` containing
    `outs` good cards. Uses cumulative hypergeometric across multiple tries.

    For needing 1 card: P = 1 - product(P(miss on each discard))
    For needing 2+: tracks probability distribution across discard rounds.
    """
    if needed <= 0:
        return 1.0
    if outs < needed or deck_size <= 0 or num_discards <= 0:
        return 0.0

    if needed == 1:
        # Fast path: P(at least 1 across D tries) = 1 - product(P(0 each time))
        p_fail_all = 1.0
        remaining_deck = deck_size
        remaining_outs = outs
        for _ in range(num_discards):
            if remaining_deck <= 0 or remaining_outs <= 0:
                break
            p_miss = _draw_probability(remaining_outs, remaining_deck, 1, cards_per_discard)
            p_fail_all *= (1.0 - p_miss)
            # Drawn cards leave the deck — outs shrink proportionally
            if remaining_deck > 0:
                expected_outs_drawn = remaining_outs * cards_per_discard / remaining_deck
                remaining_outs = max(0, round(remaining_outs - expected_outs_drawn))
            remaining_deck -= cards_per_discard
        return 1.0 - p_fail_all

    # General case: track distribution of "outs collected so far"
    # dist[k] = P(have collected exactly k outs so far), capped at needed
    dist = [0.0] * (needed + 1)
    dist[0] = 1.0

    remaining_deck = deck_size
    remaining_outs = outs
    for _ in range(num_discards):
        if remaining_deck <= 0 or remaining_outs <= 0:
            break
        new_dist = [0.0] * (needed + 1)
        for have in range(needed):
            if dist[have] < 1e-12:
                continue
            max_draw = min(cards_per_discard, remaining_outs, needed - have)
            for k in range(0, max_draw + 1):
                p_k = _draw_probability(remaining_outs, remaining_deck, k, cards_per_discard)
                p_k_plus = _draw_probability(remaining_outs, remaining_deck, k + 1, cards_per_discard) if k < max_draw else 0.0
                p_exact = p_k - p_k_plus
                got = min(have + k, needed)
                new_dist[got] += dist[have] * max(p_exact, 0.0)
        new_dist[needed] += dist[needed]
        dist = new_dist
        # Drawn cards leave the deck — outs shrink proportionally
        if remaining_deck > 0:
            expected_outs_drawn = remaining_outs * cards_per_discard / remaining_deck
            remaining_outs = max(0, round(remaining_outs - expected_outs_drawn))
        remaining_deck -= cards_per_discard

    return min(max(dist[needed], 0.0), 1.0)


def _enumerate_targets(hand_cards: list[dict], deck_cards: list[dict],
                       jokers: list[dict], gamestate: dict,
                       debuffed_suit: str | None = None) -> list[dict]:
    """Enumerate all achievable hand type targets with projected scores.

    Returns list of targets, each with:
        hand_type, detail, keep_indices, needed, outs, cards_per_discard, projected_score
    """
    n = len(hand_cards)
    deck_size = len(deck_cards)
    deck_suits = Counter(card_suit(c) for c in deck_cards)
    deck_ranks = Counter(card_rank(c) for c in deck_cards)

    targets: list[dict] = []

    # Pre-compute hand composition
    suit_groups: dict[str, list[int]] = {}
    rank_groups: dict[str, list[int]] = {}
    for i in range(n):
        s = card_suit(hand_cards[i])
        r = card_rank(hand_cards[i])
        suit_groups.setdefault(s, []).append(i)
        rank_groups.setdefault(r, []).append(i)

    # --- FLUSH per suit ---
    # Check ALL four suits, not just suits in hand — a suit with 0 cards
    # in hand can still be chased if there are enough in the deck.
    ALL_SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
    for suit in ALL_SUITS:
        if debuffed_suit and suit == debuffed_suit:
            continue  # Boss blind debuffs this suit — skip
        indices = suit_groups.get(suit, [])
        have = len(indices)
        needed = 5 - have
        available = deck_suits.get(suit, 0)
        discard_count = min(n - have, 5)

        if needed <= 0:
            # Already have flush
            flush_cards = [hand_cards[i] for i in indices[:5]]
            ht, si = classify_hand(flush_cards)
            score = estimate_score(ht, flush_cards, si, jokers, gamestate)
            targets.append({
                "hand_type": "Flush", "detail": f"Flush:{suit}",
                "keep_indices": indices[:5], "needed": 0, "outs": 0,
                "cards_per_discard": 0, "projected_score": score,
            })
        elif available >= needed:
            kept = [hand_cards[i] for i in indices]
            projected = _project_hand_score(
                "Flush", 5, kept, jokers, gamestate, 0.0,
                target_suit=suit, deck_cards=deck_cards
            )
            targets.append({
                "hand_type": "Flush", "detail": f"Flush:{suit}",
                "keep_indices": list(indices), "needed": needed, "outs": available,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

    # --- PAIR / TRIPS / QUADS per rank ---
    for rank, indices in rank_groups.items():
        have = len(indices)
        in_deck = deck_ranks.get(rank, 0)

        # Chase PAIR from a single card (most common chase in Balatro!)
        if have == 1 and in_deck >= 1:
            kept = [hand_cards[i] for i in indices]
            projected = _project_hand_score(
                "Pair", 2, kept, jokers, gamestate, 0.0,
                deck_cards=deck_cards, target_ranks=[rank]
            )
            discard_count = min(n - have, 5)
            targets.append({
                "hand_type": "Pair", "detail": f"Pair:{rank}",
                "keep_indices": list(indices), "needed": 1, "outs": in_deck,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

        if have == 2 and in_deck >= 1:
            kept = [hand_cards[i] for i in indices]
            projected = _project_hand_score(
                "Three of a Kind", 3, kept, jokers, gamestate, 0.0,
                deck_cards=deck_cards, target_ranks=[rank]
            )
            discard_count = min(n - have, 5)
            targets.append({
                "hand_type": "Three of a Kind", "detail": f"Trips:{rank}",
                "keep_indices": list(indices), "needed": 1, "outs": in_deck,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

        if have >= 3 and in_deck >= 1:
            kept = [hand_cards[i] for i in indices]
            projected = _project_hand_score(
                "Four of a Kind", 4, kept, jokers, gamestate, 0.0,
                deck_cards=deck_cards, target_ranks=[rank]
            )
            discard_count = min(n - have, 5)
            targets.append({
                "hand_type": "Four of a Kind", "detail": f"Quads:{rank}",
                "keep_indices": list(indices[:3]), "needed": 1, "outs": in_deck,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

    # --- TWO PAIR from single pair + singles ---
    # If we have exactly 1 pair and some singles, chase a second pair
    rank_counts = Counter(card_rank(hand_cards[i]) for i in range(n))
    single_ranks_for_2p = [r for r, c in rank_counts.items() if c == 1]
    pair_ranks_for_2p = [r for r, c in rank_counts.items() if c == 2]

    if len(pair_ranks_for_2p) == 1 and single_ranks_for_2p:
        existing_pair_rank = pair_ranks_for_2p[0]
        existing_pair_idx = rank_groups[existing_pair_rank]
        # Sum outs for all single ranks in hand that have matches in deck
        second_pair_outs = sum(deck_ranks.get(r, 0) for r in single_ranks_for_2p)
        if second_pair_outs > 0:
            # Keep the existing pair + best single (highest chip value)
            best_single_rank = max(single_ranks_for_2p,
                                    key=lambda r: CARD_CHIP_VALUES.get(r, 0))
            keep_idx = list(existing_pair_idx) + rank_groups.get(best_single_rank, [])[:1]
            non_keep = [i for i in range(n) if i not in keep_idx]
            kept = [hand_cards[i] for i in keep_idx]
            projected = _project_hand_score(
                "Two Pair", 4, kept, jokers, gamestate, 0.0,
                deck_cards=deck_cards, target_ranks=single_ranks_for_2p[:3]
            )
            discard_count = min(len(non_keep), 5)
            targets.append({
                "hand_type": "Two Pair", "detail": f"TwoPair:{existing_pair_rank}+?",
                "keep_indices": keep_idx, "needed": 1, "outs": second_pair_outs,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

    # --- FULL HOUSE ---
    trip_ranks = [r for r, c in rank_counts.items() if c >= 3]
    pair_ranks = [r for r, c in rank_counts.items() if c == 2]

    if trip_ranks and not pair_ranks:
        trip_r = trip_ranks[0]
        trip_idx = rank_groups[trip_r]
        non_trip = [i for i in range(n) if i not in trip_idx]
        non_trip_ranks_in_hand = set(card_rank(hand_cards[i]) for i in non_trip)
        pair_outs = sum(deck_ranks.get(r, 0) for r in non_trip_ranks_in_hand)
        if pair_outs > 0:
            kept = [hand_cards[i] for i in trip_idx]
            # Full house filler: grab a card that pairs with a non-trip rank
            filler_ranks = list(non_trip_ranks_in_hand)
            projected = _project_hand_score(
                "Full House", 5, kept, jokers, gamestate, 0.0,
                deck_cards=deck_cards, target_ranks=filler_ranks[:2]
            )
            discard_count = min(len(non_trip), 5)
            targets.append({
                "hand_type": "Full House", "detail": "FullHouse",
                "keep_indices": list(trip_idx), "needed": 1, "outs": pair_outs,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

    if len(pair_ranks) >= 2:
        for pr in pair_ranks:
            in_deck = deck_ranks.get(pr, 0)
            if in_deck >= 1:
                keep_idx = rank_groups[pr] + [i for r2 in pair_ranks if r2 != pr for i in rank_groups[r2]]
                non_keep = [i for i in range(n) if i not in keep_idx]
                kept = [hand_cards[i] for i in keep_idx]
                projected = _project_hand_score(
                    "Full House", 5, kept, jokers, gamestate, 0.0,
                    deck_cards=deck_cards, target_ranks=[pr]
                )
                discard_count = min(len(non_keep), 5)
                targets.append({
                    "hand_type": "Full House", "detail": f"FullHouse:{pr}",
                    "keep_indices": list(keep_idx), "needed": 1, "outs": in_deck,
                    "cards_per_discard": discard_count, "projected_score": projected,
                })
                break

    # --- STRAIGHT FLUSH per suit ---
    # Check before plain straights — highest scoring regular hand type (100×8)
    # For each suit, find sequential runs and check if a 5-card straight flush
    # is achievable from hand + deck cards of that suit.
    for suit in ALL_SUITS:
        if debuffed_suit and suit == debuffed_suit:
            continue
        # Get all values for this suit in hand
        suit_hand_vals: dict[int, int] = {}  # value -> hand index
        for i in suit_groups.get(suit, []):
            v = RANK_ORDER.get(card_rank(hand_cards[i]), 0)
            if v and v not in suit_hand_vals:
                suit_hand_vals[v] = i
                if v == 14:
                    suit_hand_vals.setdefault(1, i)

        # Get all values for this suit in deck
        suit_deck_vals: Counter = Counter()
        for c in deck_cards:
            if card_suit(c) == suit:
                v = RANK_ORDER.get(card_rank(c), 0)
                if v:
                    suit_deck_vals[v] += 1
                    if v == 14:
                        suit_deck_vals[1] += 1

        for low in range(1, 11):
            window = set(range(low, low + 5))
            have_vals = window & set(suit_hand_vals.keys())
            need_vals = window - have_vals
            if len(have_vals) < 2 or len(need_vals) > 3:
                continue

            # Check if ALL missing values exist in deck for this suit
            outs_per_val = {v: suit_deck_vals.get(v, 0) for v in need_vals}
            if any(ct == 0 for ct in outs_per_val.values()):
                continue  # Can't complete — missing a rank in this suit

            # For multi-discard probability, outs = min individual out count
            # (conservative: need exactly 1 of each missing rank in this suit)
            needed = len(need_vals)
            min_outs = min(outs_per_val.values()) if outs_per_val else 0
            # Total suit outs in deck for discard cycling
            total_suit_outs = sum(outs_per_val.values())

            keep_idx = list(set(suit_hand_vals[v] for v in have_vals))
            non_keep = [i for i in range(n) if i not in keep_idx]
            discard_count = min(len(non_keep), 5)

            kept = [hand_cards[i] for i in keep_idx]
            missing_ranks = [_value_to_rank(v) for v in need_vals if _value_to_rank(v)]
            projected = _project_hand_score(
                "Straight Flush", 5, kept, jokers, gamestate, 0.0,
                target_suit=suit, deck_cards=deck_cards, target_ranks=missing_ranks
            )
            targets.append({
                "hand_type": "Straight Flush",
                "detail": f"StraightFlush:{suit}:{low}-{low+4}",
                "keep_indices": keep_idx, "needed": needed, "outs": total_suit_outs,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

    # --- STRAIGHT ---
    hand_values: dict[int, int] = {}
    for i in range(n):
        v = RANK_ORDER.get(card_rank(hand_cards[i]), 0)
        if v and v not in hand_values:
            hand_values[v] = i
            if v == 14:
                hand_values.setdefault(1, i)

    for low in range(1, 11):
        window = set(range(low, low + 5))
        have_vals = window & set(hand_values.keys())
        need_vals = window - have_vals
        if len(have_vals) < 3 or len(need_vals) > 2:
            continue

        keep_idx = list(set(hand_values[v] for v in have_vals))
        non_keep = [i for i in range(n) if i not in keep_idx]

        outs = 0
        for v in need_vals:
            rank = _value_to_rank(v)
            if rank:
                outs += deck_ranks.get(rank, 0)

        needed = len(need_vals)
        if outs >= needed:
            kept = [hand_cards[i] for i in keep_idx]
            missing_ranks = [_value_to_rank(v) for v in need_vals if _value_to_rank(v)]
            projected = _project_hand_score(
                "Straight", 5, kept, jokers, gamestate, 0.0,
                deck_cards=deck_cards, target_ranks=missing_ranks
            )
            discard_count = min(len(non_keep), 5)
            targets.append({
                "hand_type": "Straight", "detail": f"Straight:{low}-{low+4}",
                "keep_indices": keep_idx, "needed": needed, "outs": outs,
                "cards_per_discard": discard_count, "projected_score": projected,
            })

    return targets


def plan_optimal_action(hand_cards: list[dict], deck_cards: list[dict],
                        jokers: list[dict], gamestate: dict) -> dict:
    """Round-level strategy planner.

    Instead of greedy one-discard-at-a-time decisions, this:
    1. Enumerates ALL achievable hand types and scores them with current jokers
    2. Computes multi-discard probability (cumulative across all available discards)
    3. Picks the highest EV target and commits to it
    4. Keeps cards that contribute to the target, discards everything else

    Returns:
        {"action": "play"|"discard", "cards": [indices], "reason": str,
         "play_score": float, "discard_ev": float, "target": float}
    """
    global _round_strategy

    n = len(hand_cards)
    if n == 0:
        return {"action": "play", "cards": [], "reason": "no cards",
                "play_score": 0, "discard_ev": 0, "target": 0}

    # Extract round info
    round_info = gamestate.get("round", {})
    blind_target = round_info.get("blind_target",
                   _get_blind_target(gamestate))
    chips_scored = round_info.get("chips", 0)
    remaining = max(blind_target - chips_scored, 0)
    hands_left = round_info.get("hands_left", 1)
    discards_left = round_info.get("discards_left", 0)
    per_hand_needed = remaining / max(hands_left, 1)

    deck_size = len(deck_cards)

    # ================================================================
    # BOSS BLIND DETECTION — adjust strategy for boss effects
    # ================================================================
    boss_name = ""
    debuffed_suit = None    # Suit debuffed by boss (cards of this suit score 0)
    boss_debuff_face = False  # The Plant: face cards debuffed
    boss_one_hand_type = False  # The Mouth: can only play 1 hand type all round
    boss_no_repeat_type = False  # The Eye: can't repeat hand types
    boss_must_play_5 = False  # The Psychic: must play exactly 5 cards
    boss_halve_base = False  # The Flint: halves base chips AND mult
    boss_one_hand_only = False  # The Needle: only 1 hand allowed
    boss_no_discards = False  # The Water: start with 0 discards
    boss_lose_money = False  # The Tooth: lose $1 per card played
    boss_hook = False  # The Hook: discards 2 random cards after each hand
    boss_ox_hand = ""  # The Ox: playing this hand type sets money to $0
    boss_verdant = False  # Verdant Leaf: all cards debuffed until joker sold
    boss_crimson = False  # Crimson Heart: random joker disabled each hand
    boss_amber = False  # Amber Acorn: jokers flipped and shuffled
    boss_cerulean = False  # Cerulean Bell: 1 card forced selected
    boss_serpent = False  # The Serpent: always draw 3 cards after play/discard
    boss_manacle = False  # The Manacle: -1 hand size
    boss_pillar = False  # The Pillar: previously played cards debuffed

    blinds = gamestate.get("blinds", {})
    if isinstance(blinds, dict):
        for b in blinds.values():
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                boss_name = b.get("name", "")
                break

    # Map boss names to their strategic effects
    BOSS_SUIT_DEBUFF = {
        "The Club": "Clubs",     # Debuffs all Clubs
        "The Goad": "Spades",    # Debuffs all Spades
        "The Head": "Hearts",    # Debuffs all Hearts
        "The Window": "Diamonds",  # Debuffs all Diamonds
    }
    if boss_name in BOSS_SUIT_DEBUFF:
        debuffed_suit = BOSS_SUIT_DEBUFF[boss_name]
    if boss_name == "The Plant":
        boss_debuff_face = True
    if boss_name == "The Psychic":
        boss_must_play_5 = True
    if boss_name == "The Mouth":
        boss_one_hand_type = True
    if boss_name == "The Eye":
        boss_no_repeat_type = True
    if boss_name == "The Flint":
        boss_halve_base = True
    if boss_name == "The Needle":
        boss_one_hand_only = True
    if boss_name == "The Water":
        boss_no_discards = True
    if boss_name == "The Tooth":
        boss_lose_money = True
    if boss_name == "The Hook":
        boss_hook = True
    if boss_name == "The Ox":
        # Find most played hand type
        hand_usage = gamestate.get("hands", {})
        most_played = ""
        most_count = 0
        for ht, info in hand_usage.items():
            if isinstance(info, dict):
                played = info.get("played", 0)
                if played > most_count:
                    most_count = played
                    most_played = ht
        boss_ox_hand = most_played
    if boss_name == "Verdant Leaf":
        boss_verdant = True
    if boss_name == "Crimson Heart":
        boss_crimson = True
    if boss_name == "Amber Acorn":
        boss_amber = True
    if boss_name == "Cerulean Bell":
        boss_cerulean = True
    if boss_name == "The Serpent":
        boss_serpent = True
    if boss_name == "The Manacle":
        boss_manacle = True
    if boss_name == "The Pillar":
        boss_pillar = True

    if boss_name:
        print(f"[BOSS] {boss_name} detected", flush=True)

    # ================================================================
    # SCALING JOKER AWARENESS — avoid resetting accumulated value
    # ================================================================
    # Check for jokers that reset on face cards (Ride the Bus) or
    # on non-matching hand types (Obelisk, Green Joker).
    avoid_face_cards = False
    face_card_reset_value = 0.0  # accumulated mult we'd lose
    for joker in jokers:
        jk = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(jk)
        if not name or name not in JOKERS:
            continue
        schema = JOKERS[name]
        if (schema.get("scaling_resets") and
                schema.get("scaling_reset_trigger") == "face_card"):
            # This joker resets when face cards are played
            scaled_val = joker.get("_scaled_value", 0)
            if scaled_val and scaled_val > 1:
                # We've accumulated meaningful value — avoid face cards
                avoid_face_cards = True
                face_card_reset_value = max(face_card_reset_value, scaled_val)

    # ================================================================
    # SEAL AWARENESS — Purple Seal prefer discard, Blue Seal prefer hold
    # ================================================================
    # Purple Seal: creates a Tarot card when discarded — actively WANT to discard
    # Blue Seal: creates a Planet card when held at end of round — DON'T discard
    # Gold Seal: +$3 when scored — prefer to include in played hands
    # Red Seal: retrigger (handled in scoring) — prefer to play high-value Red cards
    purple_seal_indices: set[int] = set()
    blue_seal_indices: set[int] = set()
    gold_seal_indices: set[int] = set()
    red_seal_indices_hand: set[int] = set()
    for i, c in enumerate(hand_cards):
        s = card_seal(c)
        if s == "PURPLE":
            purple_seal_indices.add(i)
        elif s == "BLUE":
            blue_seal_indices.add(i)
        elif s == "GOLD":
            gold_seal_indices.add(i)
        elif s == "RED":
            red_seal_indices_hand.add(i)

    # ================================================================
    # STEP 1: Best hand playable RIGHT NOW
    # ================================================================
    top_hands = find_best_hands(hand_cards, jokers, gamestate, top_n=10,
                                debuffed_suit=debuffed_suit,
                                boss_debuff_face=boss_debuff_face)

    # If we need to avoid face cards, re-rank hands to penalize those
    # that would reset a scaling joker
    if avoid_face_cards and len(top_hands) > 1:
        for h in top_hands:
            scored_cards = [hand_cards[i] for i in h["card_indices"]
                           if i < len(hand_cards)]
            has_face = any(card_rank(c) in FACE_RANKS for c in scored_cards)
            if has_face:
                # Penalize: subtract the accumulated scaling value we'd lose
                # This makes face-card hands compete against the mult they'd destroy
                h["estimated_score"] -= face_card_reset_value * h["estimated_score"] * 0.3
        # Re-sort
        top_hands.sort(key=lambda h: -h["estimated_score"])

    # ── BOSS-SPECIFIC SCORE ADJUSTMENTS ──

    # The Flint: base chips AND mult are halved → scores drop ~75%
    if boss_halve_base:
        for h in top_hands:
            h["estimated_score"] *= 0.25  # halved chips × halved mult = 25%

    # Verdant Leaf: ALL cards debuffed (0 chips, no triggers) unless joker sold
    # Only joker-native effects contribute. Scores drop to near-zero.
    if boss_verdant:
        for h in top_hands:
            h["estimated_score"] *= 0.05  # almost nothing scores

    # Crimson Heart: 1 random joker disabled per hand
    # Estimate ~20% score loss per hand on average (1 of 5 jokers gone)
    if boss_crimson and len(jokers) > 0:
        joker_penalty = 1.0 - (1.0 / max(len(jokers), 1))
        for h in top_hands:
            h["estimated_score"] *= joker_penalty

    # The Ox: playing most-played hand sets money to $0 — avoid that hand type
    if boss_ox_hand:
        for h in top_hands:
            if h["hand_type"] == boss_ox_hand:
                h["estimated_score"] *= 0.1  # heavily penalize, don't play it

    # The Pillar: previously played cards are debuffed
    # Later hands score less as more cards get debuffed. Estimate ~15% penalty.
    if boss_pillar:
        for h in top_hands:
            h["estimated_score"] *= 0.85

    # Re-sort after adjustments
    if boss_halve_base or boss_verdant or boss_crimson or boss_ox_hand or boss_pillar:
        top_hands.sort(key=lambda h: -h["estimated_score"])

    best_play = top_hands[0] if top_hands else None
    best_play_score = best_play["estimated_score"] if best_play else 0.0
    best_hand_type = best_play["hand_type"] if best_play else "?"
    play_cards = list(best_play["card_indices"]) if best_play else [0]

    # Debug: log top 3 hands for diagnostics
    if top_hands:
        hand_summary = []
        for h in top_hands[:3]:
            cards_desc = []
            for ci in h["card_indices"]:
                if ci < len(hand_cards):
                    c = hand_cards[ci]
                    r = card_rank(c)
                    s = card_suit(c)[:1]
                    cards_desc.append(f"{r}{s}")
            hand_summary.append(f"{h['hand_type']}({','.join(cards_desc)})={h['estimated_score']:.0f}")
        print(f"[HAND] Top: {' | '.join(hand_summary)} | "
              f"target={remaining:.0f} hands={hands_left} disc={discards_left}",
              flush=True)

    # Effective draws: real discards + spare hands (play weak to cycle cards)
    # Each hand played draws ~5 new cards, just like a discard does.
    # Play-cycling is weaker than real discards: costs a hand, less flexible
    # card selection, might not dump exactly what you want. Discount by 0.5.
    spare_hands = max(hands_left - 1, 0)  # save last hand for the big play
    effective_draws = discards_left + int(spare_hands * 0.5)

    # ================================================================
    # STEP 2: Hard constraints — must play
    # ================================================================
    if hands_left <= 1 and discards_left <= 0:
        # Truly last hand, no options — play best available
        return {"action": "play", "cards": play_cards,
                "reason": f"must play ({best_play_score:.0f})",
                "play_score": best_play_score, "discard_ev": 0, "target": per_hand_needed}
    # When hands > 1, even with 0 discards, fall through to strategy planner.
    # Extra hands act as "discards that also score" — play weak cards to cycle
    # through the deck and chase the best possible final hand.

    # ================================================================
    # STEP 3: Round-level strategy — enumerate and score ALL targets
    # ================================================================
    # Check if we're in the same round (reset strategy on new round)
    fingerprint = blind_target  # stable across the whole round
    if _round_strategy["fingerprint"] != fingerprint:
        _round_strategy["fingerprint"] = fingerprint
        _round_strategy["committed_type"] = None

    committed_type = _round_strategy["committed_type"]

    targets = _enumerate_targets(hand_cards, deck_cards, jokers, gamestate,
                                    debuffed_suit=debuffed_suit)

    # ── BOSS FILTERS: restrict targets based on boss effects ──
    # The Eye: can't repeat hand types this round
    if boss_no_repeat_type:
        played_types = set()
        hand_usage = gamestate.get("hands", {})
        for ht, info in hand_usage.items():
            if isinstance(info, dict) and info.get("round_played", 0) > 0:
                played_types.add(ht)
        if played_types:
            targets = [t for t in targets if t.get("detail", "") not in played_types]
            top_hands = [h for h in top_hands if h["hand_type"] not in played_types]
            if top_hands:
                best_play = top_hands[0]
                best_play_score = best_play["estimated_score"]
                best_hand_type = best_play["hand_type"]
                play_cards = list(best_play["card_indices"])

    # The Mouth: only one hand type can be played this round
    if boss_one_hand_type and committed_type:
        targets = [t for t in targets if t.get("detail", "") == committed_type]
        top_hands = [h for h in top_hands if h["hand_type"] == committed_type]
        if top_hands:
            best_play = top_hands[0]
            best_play_score = best_play["estimated_score"]
            best_hand_type = best_play["hand_type"]
            play_cards = list(best_play["card_indices"])

    # The Needle: only 1 hand — make it count, never discard
    if boss_one_hand_only:
        discards_left = 0  # treat as no discards available

    # Score each target with multi-discard probability
    scored_targets: list[dict] = []
    for t in targets:
        needed = t["needed"]
        projected = t["projected_score"]

        if needed == 0:
            # Already achievable — play it
            ev = projected
            p = 1.0
        else:
            cpd = t["cards_per_discard"]
            if cpd <= 0:
                continue
            p = _multi_discard_probability(
                t["outs"], deck_size, needed, cpd, effective_draws
            )
            if p < 0.03:
                continue
            # Fallback: if we miss entirely, we play best available
            fallback = best_play_score * 0.7
            ev = p * projected + (1 - p) * fallback

        # Commitment bias: small boost for target we're already chasing,
        # but NOT so large that it blocks switching to clearly better options
        display_ev = ev
        if committed_type and t["detail"] == committed_type:
            ev *= 1.1  # mild bias — enough to break ties, not enough to block switches

        scored_targets.append({
            **t, "ev": ev, "display_ev": display_ev, "p": p,
        })

    # Sort by EV
    scored_targets.sort(key=lambda t: -t["ev"])

    # Log unresolved joker keys (rare — only on new mismatches)
    if _UNRESOLVED_KEYS:
        print(f"[WARN] Unresolved joker keys (not in schema): {_UNRESOLVED_KEYS}", flush=True)

    # ================================================================
    # STEP 5: Decision — play now or chase best target
    # ================================================================
    best_target = scored_targets[0] if scored_targets else None
    best_target_ev = best_target["ev"] if best_target else 0.0

    # ── INSTANT WIN CHECK ──
    # If best play RIGHT NOW beats the entire remaining score, play it.
    # No chasing, no cycling — just win. Extra score = extra money, but
    # wasting hands/discards on marginal improvements risks misplays.
    if best_play_score >= remaining and remaining > 0:
        _round_strategy["committed_type"] = None
        return {"action": "play", "cards": play_cards,
                "reason": f"wins round ({best_play_score:.0f} >= {remaining:.0f})",
                "play_score": best_play_score, "discard_ev": 0, "target": remaining}

    # ── DISCARD-FIRST PRINCIPLE ──
    # In Balatro, the winning strategy is: use ALL discards to build ONE
    # massive hand, then play it. Never burn hands on weak plays while
    # discards are still available.
    #
    # Only play when:
    #   a) The hand wins outright (>= remaining) — handled above
    #   b) No discards left AND no better target to cycle toward
    #
    # The 70% threshold only applies when deciding whether to play-cycle
    # (burn a hand to draw cards) after discards are exhausted.

    # Check if any achievable target (need=0) scores higher than best play
    best_ready = None
    for t in scored_targets:
        if t["needed"] == 0:
            if best_ready is None or t["projected_score"] > best_ready["projected_score"]:
                best_ready = t

    # Use the best of find_best_hands vs best_ready target
    effective_play_score = best_play_score
    effective_play_cards = play_cards
    effective_play_label = best_hand_type
    if best_ready and best_ready["projected_score"] > effective_play_score:
        effective_play_score = best_ready["projected_score"]
        effective_play_cards = best_ready["keep_indices"]
        effective_play_label = best_ready["detail"]

    # Helper: order discard candidates with seal + face card awareness
    # Priority: Purple Seal first (want tarot), face cards (if avoid_face),
    # then normal, Blue Seal last (want to hold for planet)
    def _order_discards(indices: list[int]) -> list[int]:
        purple = []
        face = []
        normal = []
        blue = []
        for i in indices:
            if i in blue_seal_indices:
                blue.append(i)  # last — keep these
            elif i in purple_seal_indices:
                purple.append(i)  # first — want to discard for tarot
            elif avoid_face_cards and card_rank(hand_cards[i]) in FACE_RANKS:
                face.append(i)  # second — avoid resetting scaling joker
            else:
                normal.append(i)
        return purple + face + normal + blue

    # Also: Blue Seal cards should be in keep_set when possible
    # (they generate Planet cards when held at end of round)

    # ── DISCARDS AVAILABLE: discard toward better hands, or play strong ones ──
    if discards_left > 0:
        # Find best chase target (any target that needs cards drawn)
        best_chase = None
        for t in scored_targets:
            if t["needed"] > 0:
                best_chase = t
                break  # already sorted by EV

        # Estimate total scoring across remaining hands.
        # Current hand: known exact score. Future hands: use realistic baseline
        # from estimate_score_for_hand_type which computes expected value using
        # actual planet levels, joker effects, and hand-type probabilities.
        future_hands = max(hands_left - 1, 0)
        baseline_score = estimate_score_for_hand_type(jokers, gamestate)
        total_projected = effective_play_score + (baseline_score * future_hands)
        comfort_ratio = total_projected / max(remaining, 1)

        # ── VIABILITY CHECK: can ANY chase target actually help beat the blind?
        # If the best chase EV across all remaining hands can't beat the target,
        # discarding is pointless — just play immediately to score what we can.
        if best_chase and remaining > 0:
            # Best possible outcome: chase EV for one big hand + remaining hands
            # of current-level play
            chase_ev = best_chase["ev"]
            best_outcome = chase_ev + baseline_score * max(hands_left - 1, 0)
            if best_outcome < remaining * 0.5:
                # Even the best chase can't get us close — don't waste discards
                _round_strategy["committed_type"] = None
                return {"action": "play", "cards": effective_play_cards,
                        "reason": f"hopeless chase (best={best_outcome:.0f} < "
                                  f"50% of {remaining:.0f}), play now",
                        "play_score": effective_play_score, "discard_ev": 0,
                        "target": remaining}

        # SMART MULTI-HAND CHECK: if the current hand is strong enough
        # to win across remaining hands AND no chase target's EV meaningfully
        # beats it, just play it.
        if comfort_ratio >= 3.0:
            chase_threshold = 2.0
        elif comfort_ratio >= 2.0:
            chase_threshold = 1.5
        else:
            chase_threshold = 1.2

        chase_ev = best_chase["ev"] if best_chase else 0
        chase_beats_current = (best_chase and
                               chase_ev > effective_play_score * chase_threshold)

        if total_projected >= remaining and remaining > 0 and not chase_beats_current:
            _round_strategy["committed_type"] = None
            return {"action": "play", "cards": effective_play_cards,
                    "reason": f"strong hand ({effective_play_score:.0f} × "
                              f"{hands_left} = {total_projected:.0f} >= {remaining:.0f}, "
                              f"comfort={comfort_ratio:.1f}×)",
                    "play_score": effective_play_score, "discard_ev": 0,
                    "target": remaining}

        if chase_beats_current:
            # Additional check: chase must also be viable against the blind
            # Don't discard toward a Three of a Kind (3000) when target is 16000
            chase_best_outcome = chase_ev + baseline_score * max(hands_left - 1, 0)
            if chase_best_outcome < remaining * 0.7:
                # Chase target is better than current but still can't win —
                # play current hand to bank the score, don't waste discards
                _round_strategy["committed_type"] = None
                return {"action": "play", "cards": effective_play_cards,
                        "reason": f"chase unviable ({best_chase['detail']} "
                                  f"EV={chase_ev:.0f}, outcome={chase_best_outcome:.0f} "
                                  f"< 70% of {remaining:.0f})",
                        "play_score": effective_play_score, "discard_ev": 0,
                        "target": remaining}

            _round_strategy["committed_type"] = best_chase["detail"]
            keep_set = set(best_chase["keep_indices"])
            keep_set.update(i for i in blue_seal_indices if i not in keep_set)
            non_target = [i for i in range(n) if i not in keep_set]
            non_target = _order_discards(non_target)
            discard_set = non_target[:5]
            if not discard_set:
                _round_strategy["committed_type"] = None
                return {"action": "play", "cards": effective_play_cards,
                        "reason": f"no safe discards for chase "
                                  f"({effective_play_score:.0f})",
                        "play_score": effective_play_score, "discard_ev": 0,
                        "target": remaining}
            return {"action": "discard", "cards": discard_set,
                    "reason": f"chase {best_chase['detail']} "
                              f"(P={best_chase['p']:.0%}, EV={best_chase['ev']:.0f}, "
                              f"proj={best_chase['projected_score']:.0f})",
                    "play_score": effective_play_score, "discard_ev": best_chase["ev"],
                    "target": remaining}

        # Can't one-shot and no great chase target.  Only discard if
        # there's a realistic chance of improvement toward the target.
        can_multihand = total_projected >= remaining
        if not can_multihand or effective_play_score < remaining * 0.5:
            # Check if fishing can plausibly help
            best_possible_ev = best_chase["ev"] if best_chase else effective_play_score
            fish_outcome = best_possible_ev + baseline_score * max(hands_left - 1, 0)
            if fish_outcome >= remaining * 0.3:
                # Fishing has some hope — discard non-scoring cards
                scoring_set = set(best_play["scoring_indices"]) if best_play else set()
                keep_set = set(scoring_set)
                if best_ready:
                    keep_set.update(best_ready["keep_indices"])
                keep_set.update(i for i in blue_seal_indices if i not in keep_set)
                non_scoring = [i for i in range(n) if i not in keep_set]
                non_scoring = _order_discards(non_scoring)
                if non_scoring:
                    return {"action": "discard", "cards": non_scoring[:5],
                            "reason": f"fish (play={effective_play_score:.0f}, "
                                      f"fish_outcome={fish_outcome:.0f} vs {remaining:.0f})",
                            "play_score": effective_play_score, "discard_ev": 0,
                            "target": remaining}

        # Play current hand — it's either strong enough or there's nothing
        # better to chase
        _round_strategy["committed_type"] = None
        return {"action": "play", "cards": effective_play_cards,
                "reason": f"best available ({effective_play_score:.0f})",
                "play_score": effective_play_score, "discard_ev": 0,
                "target": remaining}

    # ── NO DISCARDS LEFT ──
    # Now decide: play current hand or burn a hand to cycle cards?

    # Key insight: if best_play * hands_left < remaining, we CANNOT win by
    # playing the current hand repeatedly. We MUST find a stronger hand type
    # or we lose. This makes cycling more aggressive when behind.
    total_projected = effective_play_score * hands_left
    can_win_by_repeating = total_projected >= remaining and remaining > 0
    desperately_behind = (remaining > 0 and hands_left > 1 and
                          total_projected < remaining * 0.7)

    # Play-cycle: burn a hand to draw new cards, chasing a better hand.
    if spare_hands >= 1:
        best_chase = None
        for t in scored_targets:
            if t["needed"] > 0:
                best_chase = t
                break

        should_cycle = False
        if best_chase:
            if desperately_behind:
                # We're going to lose anyway — chase anything better
                should_cycle = best_chase["ev"] > effective_play_score
            elif spare_hands >= 2 and best_chase["ev"] > effective_play_score * 1.5:
                # Normal cycle: need significant improvement
                should_cycle = True
            elif not can_win_by_repeating and best_chase["ev"] > effective_play_score:
                # Can't win by repeating current hand — chase if EV is better
                should_cycle = True

        if should_cycle and best_chase:
            cycle_reason = "DESPERATE " if desperately_behind else ""
            _round_strategy["committed_type"] = best_chase["detail"]
            keep_set = set(best_chase["keep_indices"])
            non_target = [i for i in range(n) if i not in keep_set]
            play_cycle = non_target[:5]
            if len(play_cycle) < 1:
                play_cycle = list(range(min(n, 5)))
            return {"action": "play", "cards": play_cycle,
                    "reason": f"{cycle_reason}cycle for {best_chase['detail']} "
                              f"(P={best_chase['p']:.0%}, proj={best_chase['projected_score']:.0f}, "
                              f"total_proj={total_projected:.0f} vs remain={remaining:.0f})",
                    "play_score": effective_play_score, "discard_ev": best_chase["ev"],
                    "target": remaining}

    # Default: play best available
    _round_strategy["committed_type"] = None
    return {"action": "play", "cards": effective_play_cards,
            "reason": f"best available ({effective_play_score:.0f}, "
                      f"total_proj={total_projected:.0f} vs remain={remaining:.0f})",
            "play_score": effective_play_score,
            "discard_ev": best_target_ev if best_target else 0.0,
            "target": remaining}


def _draw_probability(outs: int, deck_size: int, needed: int,
                      num_draws: int) -> float:
    """P(drawing at least `needed` outs in `num_draws` draws from deck).

    Exact hypergeometric CDF: P(X >= needed) where X ~ Hypergeometric(deck_size, outs, num_draws).
    Formula: sum over k from needed..min(num_draws, outs) of C(outs,k)*C(deck-outs, draws-k) / C(deck, draws)
    """
    if needed <= 0:
        return 1.0
    if outs < needed or deck_size <= 0 or num_draws < needed:
        return 0.0
    if num_draws > deck_size:
        num_draws = deck_size

    non_outs = deck_size - outs
    total_ways = math.comb(deck_size, num_draws)
    if total_ways == 0:
        return 0.0

    p = 0.0
    for k in range(needed, min(num_draws, outs) + 1):
        miss = num_draws - k
        if miss > non_outs:
            continue
        p += math.comb(outs, k) * math.comb(non_outs, miss) / total_ways

    return min(max(p, 0.0), 1.0)


def _log_scale(value: float, scale: float) -> float:
    """Log10 normalization."""
    if value <= 0:
        return 0.0
    return math.log10(value + 1.0) / scale


def _get_blind_target(gamestate: dict) -> float:
    """Extract the current blind's target score."""
    blinds = gamestate.get("blinds", {})
    if isinstance(blinds, dict):
        for blind in blinds.values():
            if isinstance(blind, dict) and blind.get("status") == "CURRENT":
                return blind.get("score", 300)
    return 300.0


def _any_joker_triggers(hand_type: str, jokers: list[dict]) -> bool:
    """Check if any joker specifically triggers on this hand type."""
    for joker in jokers:
        name = _api_key_to_name(joker.get("joker_key", "") or joker.get("key", ""))
        if not name or name not in JOKERS:
            continue
        schema = JOKERS[name]
        triggers = schema.get("triggers") or []
        if "specific_hand_type" in triggers:
            target = schema.get("trigger_hand_type", "")
            if hand_type == target or _hand_contains(hand_type, target):
                return True
    return False


def _get_joker_totals(hand_type: str, cards: list[dict],
                      scoring_indices: list[int],
                      jokers: list[dict],
                      gamestate: dict) -> tuple[str, float, float, float]:
    """Get joker scoring totals plus hand type for convenience."""
    chips, mult, xmult = compute_joker_scoring(
        hand_type, cards, scoring_indices, jokers, gamestate
    )
    return hand_type, chips, mult, xmult


def _get_dominant_suit(jokers: list[dict]) -> Optional[str]:
    """Determine which suit the current joker lineup favors most.

    Analyzes suit-specific jokers and weights by their scoring power.
    Returns the best suit, or None if no suit jokers are present.
    """
    suit_scores: dict[str, float] = {"Hearts": 0.0, "Diamonds": 0.0,
                                      "Spades": 0.0, "Clubs": 0.0}

    for joker in jokers:
        joker_key = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(joker_key)
        if not name or name not in JOKERS:
            continue

        schema = JOKERS[name]
        triggers = schema.get("triggers") or []

        has_suit_trigger = ("specific_suit" in triggers
                           or "suit_specific_discard" in triggers)
        if not has_suit_trigger:
            continue

        target_suits = schema.get("trigger_suits") or schema.get("discard_trigger_suits") or []

        # Weight by scoring power — xmult is king, then mult, then chips
        mult_val = schema.get("mult_value") or 0.0
        xmult_val = schema.get("xmult_value") or 1.0
        chip_val = schema.get("chip_value") or 0.0
        prob = schema.get("effect_probability") or 1.0

        power = chip_val * 0.1 + mult_val * 2.0
        if xmult_val > 1.0:
            power += (xmult_val - 1.0) * 50.0
        power = max(power, 1.0) * prob

        for suit in target_suits:
            if suit in suit_scores:
                suit_scores[suit] += power

    best_suit = max(suit_scores, key=suit_scores.get)
    if suit_scores[best_suit] > 0:
        return best_suit
    return None


def estimate_score_for_hand_type(jokers: list[dict], gamestate: dict) -> float:
    """Estimate realistic scoring power based on most-played hand types.

    Uses actual API chips/mult (planet levels included) + joker effects.
    Estimates based on hand types the bot ACTUALLY plays (from play counts),
    not theoretical best. Falls back to common types for early game.

    Args:
        jokers: owned joker dicts from the API
        gamestate: full gamestate dict

    Returns:
        estimated realistic per-hand score
    """
    hands_data = gamestate.get("hands", {})

    # Figure out which hand types the bot actually plays
    # Use play counts to weight the estimate toward realistic hands
    COMMON_HAND_TYPES = [
        "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush",
        "Five of a Kind", "Flush House", "Flush Five",
    ]

    # Determine which suit our jokers favor for smarter Flush evaluation
    dominant_suit = _get_dominant_suit(jokers)

    # Compute deck suit fractions for accurate suit trigger estimation
    deck_cards = gamestate.get("cards", {}).get("cards", [])
    ds: Optional[dict[str, float]] = None
    if deck_cards:
        suit_expand = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
        counts: dict[str, int] = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
        for c in deck_cards:
            v = c.get("value", {})
            s = suit_expand.get(v.get("suit", ""), "")
            if s in counts:
                counts[s] += 1
        total = max(len(deck_cards), 1)
        ds = {s: n / total for s, n in counts.items()}

    # Detect retrigger jokers once for chip bonus estimation
    has_hanging_chad = False
    has_sock_buskin = False
    has_hack = False
    has_dusk = False
    has_seltzer = False
    for j in jokers:
        jk = j.get("joker_key", "") or j.get("key", "")
        jn = _api_key_to_name(jk)
        if jn == "Hanging Chad":
            has_hanging_chad = True
        elif jn == "Sock and Buskin":
            has_sock_buskin = True
        elif jn == "Hack":
            has_hack = True
        elif jn == "Dusk":
            has_dusk = True
        elif jn == "Seltzer":
            has_seltzer = True

    # Retrigger chip bonus is handled INSIDE _estimate_joker_scoring_for_type
    # via retrigger_extra on per-card jokers. We only need the base card chip
    # contribution from retriggers on the cards themselves (not joker effects).
    retrig_card_chip_bonus = 0.0
    # Retriggers re-fire card chips: avg card ~8 chips
    if has_hanging_chad:
        retrig_card_chip_bonus += 8.0 * 2   # first card × 8 chips × 2 retriggers
    if has_sock_buskin:
        retrig_card_chip_bonus += 10.0 * 2  # ~2 face cards × 10 chips × 1 retrigger
    if has_hack:
        retrig_card_chip_bonus += 4.0 * 2   # ~2 low cards × avg 3.5 chips × 1 retrigger
    if has_seltzer:
        retrig_card_chip_bonus += 8.0 * 5   # all 5 cards × 8 chips × 1 retrigger
    if has_dusk:
        # Dusk fires on last hand only — weight by 1/hands_left as estimate
        est_hands = gamestate.get("round", {}).get("hands_left",
                     gamestate.get("current_round", {}).get("hands_left", 4))
        retrig_card_chip_bonus += 8.0 * 5 / max(est_hands, 1)

    # Collect hand types with their play frequency
    scored_types = []
    for ht_name in COMMON_HAND_TYPES:
        ht_info = hands_data.get(ht_name, {})
        times_played = ht_info.get("played", 0)

        base_chips_default, base_mult_default = BASE_HAND_SCORES.get(ht_name, (5, 1))
        ht_chips = ht_info.get("chips", base_chips_default)
        ht_mult = ht_info.get("mult", base_mult_default)

        scoring_card_chips = 40.0 + retrig_card_chip_bonus

        j_chips, j_mult, j_xmult = _estimate_joker_scoring_for_type(
            ht_name, jokers, gamestate, dominant_suit=dominant_suit,
            deck_suits=ds,
        )

        total_chips = ht_chips + scoring_card_chips + j_chips
        total_mult = ht_mult + j_mult
        score = total_chips * total_mult * j_xmult

        scored_types.append((ht_name, score, times_played))

    # Sort by score
    scored_types.sort(key=lambda x: x[1], reverse=True)

    # If we have play history, use the best score among frequently-played types
    # (at least 1 play = the bot has achieved this hand type before)
    played_scores = [s for _, s, p in scored_types if p >= 1]
    if played_scores:
        return played_scores[0]  # best score among types we've actually played

    # Early game / no history — use Pair as the realistic baseline
    # (the most common hand type for any deck)
    for ht_name, score, _ in scored_types:
        if ht_name == "Pair":
            return score

    # Fallback
    return scored_types[0][1] if scored_types else 100.0


# Per-level chip/mult increments for each hand type (from game source)
HAND_LEVEL_INCREMENTS: dict[str, tuple[int, int]] = {
    # hand_type: (l_chips, l_mult)
    "Flush Five":       (50, 3),
    "Flush House":      (40, 4),
    "Five of a Kind":   (35, 3),
    "Straight Flush":   (40, 4),
    "Four of a Kind":   (30, 3),
    "Full House":       (25, 2),
    "Flush":            (15, 2),
    "Straight":         (30, 3),
    "Three of a Kind":  (20, 2),
    "Two Pair":         (20, 1),
    "Pair":             (15, 1),
    "High Card":        (10, 1),
}


def pick_best_planet(pack_cards: list[dict], jokers: list[dict],
                     gamestate: dict) -> int:
    """Pick the best planet card from a celestial pack.

    Scores each planet by simulating +1 level on its hand type and computing
    the marginal score increase with current jokers. This makes the choice
    joker-aware: if you have flush jokers, Jupiter (Flush) is worth more
    than Mercury (Pair) even if you've played more Pairs.

    Args:
        pack_cards: cards available in the pack
        gamestate: full gamestate dict
        jokers: owned joker dicts

    Returns:
        index of the best card to pick
    """
    from environment.action_space import PLANET_TO_HAND_TYPE

    hands_data = gamestate.get("hands", {})
    dominant_suit = _get_dominant_suit(jokers)

    # Compute deck suit fractions for accurate estimation
    deck_cards = gamestate.get("cards", {}).get("cards", [])
    ds: Optional[dict[str, float]] = None
    if deck_cards:
        suit_expand = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
        counts: dict[str, int] = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
        for c in deck_cards:
            v = c.get("value", {})
            s = suit_expand.get(v.get("suit", ""), "")
            if s in counts:
                counts[s] += 1
        total = max(len(deck_cards), 1)
        ds = {s: n / total for s, n in counts.items()}

    best_idx = 0
    best_gain = -1.0

    for i, pc in enumerate(pack_cards):
        key = pc.get("key", "")

        # Black Hole always wins (levels ALL hand types)
        if key == "c_black_hole":
            return i

        ht_name = PLANET_TO_HAND_TYPE.get(key)
        if not ht_name:
            continue

        ht_info = hands_data.get(ht_name, {})
        base_chips_default, base_mult_default = BASE_HAND_SCORES.get(ht_name, (5, 1))
        current_chips = ht_info.get("chips", base_chips_default)
        current_mult = ht_info.get("mult", base_mult_default)

        # Per-level increment
        l_chips, l_mult = HAND_LEVEL_INCREMENTS.get(ht_name, (10, 1))

        # Estimate card chips contributed by scoring cards (rough average)
        scoring_card_chips = 40.0

        # Current score with jokers
        j_chips, j_mult, j_xmult = _estimate_joker_scoring_for_type(
            ht_name, jokers, gamestate, dominant_suit=dominant_suit,
            deck_suits=ds,
        )
        current_score = (current_chips + scoring_card_chips + j_chips) * \
                        (current_mult + j_mult) * j_xmult

        # Score after +1 level
        new_chips = current_chips + l_chips
        new_mult = current_mult + l_mult
        new_score = (new_chips + scoring_card_chips + j_chips) * \
                    (new_mult + j_mult) * j_xmult

        gain = new_score - current_score

        # Bonus: if this is a hand type the bot has actually played,
        # it's more valuable (we know we can achieve it)
        times_played = ht_info.get("played", 0)
        if times_played > 0:
            gain *= 1.2  # 20% bonus for proven hand types

        if gain > best_gain:
            best_gain = gain
            best_idx = i

    return best_idx


def _estimate_joker_scoring_for_type(hand_type: str, jokers: list[dict],
                                      gamestate: dict,
                                      dominant_suit: Optional[str] = None,
                                      deck_suits: Optional[dict[str, float]] = None,
                                      ) -> tuple[float, float, float]:
    """Estimate joker chip/mult/xmult for a hand type without specific cards.

    Used for shop/strategy decisions where we don't have exact card info.
    Estimates per-card triggers based on typical scoring card counts.

    Args:
        dominant_suit: The suit favored by current joker lineup (from _get_dominant_suit).
                       Used to accurately estimate suit trigger counts for Flush hands.
        deck_suits: Optional dict mapping suit name -> fraction of deck (0-1).
                    When provided, suit trigger counts scale by actual deck composition.
    """
    bonus_chips = 0.0
    bonus_mult = 0.0
    xmult_product = 1.0

    is_flush = hand_type in ("Flush", "Straight Flush", "Flush House", "Flush Five")

    num_jokers = len(jokers)
    money = gamestate.get("money", 0)
    deck_remaining = len(gamestate.get("cards", {}).get("cards", []))
    hands_left = gamestate.get("round", {}).get("hands_left",
                  gamestate.get("current_round", {}).get("hands_left", 4))

    # Detect retrigger jokers for rough estimation
    has_hanging_chad = False
    has_sock_buskin = False
    has_hack = False
    has_dusk = False
    has_seltzer = False
    for j in jokers:
        jk = j.get("joker_key", "") or j.get("key", "")
        jn = _api_key_to_name(jk)
        if jn == "Hanging Chad":
            has_hanging_chad = True
        elif jn == "Sock and Buskin":
            has_sock_buskin = True
        elif jn == "Hack":
            has_hack = True
        elif jn == "Dusk":
            has_dusk = True
        elif jn == "Seltzer":
            has_seltzer = True

    # Count uncommon jokers for Baseball Card
    uncommon_count = 0
    for j in jokers:
        jk = j.get("joker_key", "") or j.get("key", "")
        jname = _api_key_to_name(jk)
        if jname and jname in JOKERS:
            if JOKERS[jname].get("rarity", "") == "uncommon":
                uncommon_count += 1

    # Resolve copy jokers (Blueprint/Brainstorm) by finding what they copy
    # and adding the copied joker's schema as an extra scoring pass.
    # Follows copy chains: Blueprint→Brainstorm→real joker resolves to real joker
    resolved_jokers = list(jokers)
    for idx, joker in enumerate(jokers):
        jk = joker.get("joker_key", "") or joker.get("key", "")
        jname = _api_key_to_name(jk)
        if not jname or jname not in JOKERS:
            continue
        schema = JOKERS[jname]
        if not schema.get("copy"):
            continue
        target_dir = schema.get("copy_target", "")
        copy_src = _resolve_copy_source(jokers, idx, target_dir)
        if copy_src is not None:
            resolved_jokers.append(copy_src)  # score it again as a duplicate

    for joker in resolved_jokers:
        joker_key = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(joker_key)
        if not name or name not in JOKERS:
            continue

        schema = JOKERS[name]
        triggers = schema.get("triggers") or []
        score_effect = schema.get("score_effect") or []

        if not score_effect:
            continue

        triggered = False
        trigger_count = 1

        for trigger in triggers:
            if trigger == "any_hand_played":
                triggered = True
            elif trigger == "specific_hand_type":
                target_hand = schema.get("trigger_hand_type", "")
                if hand_type == target_hand or _hand_contains(hand_type, target_hand):
                    triggered = True
            elif trigger == "scoring_card":
                triggered = True
                if schema.get("per_card_instance"):
                    trigger_count = 4  # estimate ~4 scoring cards
            elif trigger == "face_card":
                triggered = True
                if schema.get("per_card_instance"):
                    trigger_count = 2  # estimate ~2 face cards
            elif trigger == "specific_suit":
                target_suits = schema.get("trigger_suits") or []
                if is_flush and dominant_suit:
                    # Flush = all 5 cards are the SAME suit.
                    # If this joker's suit matches the dominant suit, all 5 trigger.
                    # If it doesn't match, 0 trigger — a Spades flush has 0 Hearts.
                    if dominant_suit in target_suits:
                        triggered = True
                        trigger_count = 5 if schema.get("per_card_instance") else 1
                    else:
                        triggered = False
                else:
                    # Non-flush hands: estimate matching suit cards based on
                    # actual deck composition instead of hardcoded 2
                    triggered = True
                    if schema.get("per_card_instance"):
                        if deck_suits and target_suits:
                            # Use actual deck fractions: 5 scoring cards * suit fraction
                            suit_frac = max(deck_suits.get(s, 0.25) for s in target_suits)
                            trigger_count = max(1, round(5 * suit_frac))
                        else:
                            trigger_count = 2  # fallback
            elif trigger == "specific_rank":
                # Estimate ~1-2 matching rank cards
                triggered = True
                if schema.get("per_card_instance"):
                    trigger_count = 1
            elif trigger == "scoring_hand_size":
                # Half Joker: ≤3 cards, Square Joker: exactly 4 cards
                threshold = schema.get("event_count_threshold", 3)
                comparison = schema.get("event_count_comparison", "maximum")
                if comparison == "maximum" and threshold >= 3:
                    # Half Joker: ~30% of hands qualify (many are 5 cards)
                    for effect in score_effect:
                        if effect == "mult":
                            bonus_mult += (schema.get("mult_value") or 0.0) * 0.3
                    continue
                elif comparison == "exact" and threshold == 4:
                    # Square Joker: ~40% of hands are naturally 4 cards
                    triggered = True
            elif trigger == "per_joker_owned":
                triggered = True
                trigger_count = num_jokers
            elif trigger == "per_card_remaining_in_deck":
                triggered = True
                trigger_count = max(deck_remaining, 30)  # estimate ~30 if unknown
            elif trigger == "per_dollar_held":
                triggered = True
                if name == "Bootstraps":
                    trigger_count = money // 5
                else:
                    trigger_count = money
            elif trigger == "final_hand_of_round":
                # Estimate as probability-weighted (1 in 4 hands)
                triggered = True
                for effect in score_effect:
                    if effect == "xmult":
                        xmult_val = schema.get("xmult_value") or 1.0
                        xmult_product *= (0.75) + 0.25 * xmult_val
                triggered = False
                continue
            elif trigger == "card_held_in_hand":
                # Conservative estimate: held cards contribute something
                if name == "Blackboard":
                    # ~20% chance all held are spades/clubs
                    triggered = True
                    for effect in score_effect:
                        if effect == "xmult":
                            xmult_val = schema.get("xmult_value") or 1.0
                            xmult_product *= 0.8 + 0.2 * xmult_val
                    triggered = False
                    continue
                elif name == "Raised Fist":
                    bonus_mult += 6.0  # estimate avg low rank * 2
                    continue
                else:
                    # Baron, Shoot the Moon — estimate ~1 matching held card
                    triggered = True
                    trigger_count = 1
            elif trigger == "per_specific_joker_present":
                if uncommon_count > 0:
                    triggered = True
                    trigger_count = uncommon_count
            elif trigger == "effect_probability":
                prob = schema.get("effect_probability", 0.0)
                triggered = True
                # Use suit-aware count for probability jokers (e.g., Bloodstone)
                target_suits = schema.get("trigger_suits") or []
                if is_flush and dominant_suit and target_suits:
                    prob_count = 5 if dominant_suit in target_suits else 0
                else:
                    prob_count = 2
                for effect in score_effect:
                    if effect == "xmult":
                        xmult_val = schema.get("xmult_value") or 1.0
                        for _ in range(prob_count):
                            xmult_product *= (1 - prob) + prob * xmult_val
                    elif effect == "mult":
                        bonus_mult += (schema.get("mult_value") or 0.0) * prob_count * prob
                triggered = False
                continue
            elif trigger == "periodic":
                threshold = schema.get("scaling_threshold") or 4
                for effect in score_effect:
                    if effect == "xmult":
                        xmult_val = schema.get("xmult_value") or 1.0
                        p = 1.0 / max(threshold, 1)
                        xmult_product *= (1 - p) + p * xmult_val
                continue

        if not triggered:
            _sv = joker.get("_scaled_value")
            _st = schema.get("scaling_type")
            if _sv is not None and _st:
                triggered = True
                trigger_count = 1
            else:
                continue

        # Retrigger boost for rough estimation:
        # Hanging Chad: first scored card triggers 2 extra times.
        # For per-card-instance jokers, add 2 extra for the first card.
        # For once-per-hand jokers (Photograph), compound xmult 3x.
        retrigger_extra = 0
        timing = schema.get("scoring_timing", "after_cards")
        if timing == "during_card" and has_hanging_chad:
            if schema.get("per_card_instance"):
                retrigger_extra = 2  # first card triggers 2 extra times
            else:
                retrigger_extra = 2  # once-per-hand compounds on retrigger
        if timing == "during_card" and has_sock_buskin and "face_card" in triggers:
            if schema.get("per_card_instance"):
                retrigger_extra += 2  # ~2 face cards each retrigger 1x
            else:
                retrigger_extra += 1  # face-triggered once-per-hand gets 1 extra
        if timing == "during_card" and has_hack and "scoring_card" in triggers:
            if schema.get("per_card_instance"):
                retrigger_extra += 2  # ~2 low cards (2-5) each retrigger 1x
        if timing == "during_card" and has_seltzer:
            if schema.get("per_card_instance"):
                retrigger_extra += trigger_count  # all cards retrigger 1x
            else:
                retrigger_extra += 1
        if timing == "during_card" and has_dusk:
            # Dusk fires on last hand — weighted estimate
            est_hands = gamestate.get("round", {}).get("hands_left",
                         gamestate.get("current_round", {}).get("hands_left", 4))
            dusk_weight = 1.0 / max(est_hands, 1)
            if schema.get("per_card_instance"):
                retrigger_extra += max(1, round(trigger_count * dusk_weight))
            else:
                retrigger_extra += max(0, round(dusk_weight))

        effective_count = trigger_count + (retrigger_extra if schema.get("per_card_instance") else 0)

        # For scaling jokers, use runtime value when available
        scaled_value = joker.get("_scaled_value")
        scaling_type = schema.get("scaling_type")

        for effect in score_effect:
            if effect == "chips":
                if scaling_type == "chips" and scaled_value is not None:
                    bonus_chips += scaled_value
                else:
                    bonus_chips += (schema.get("chip_value") or 0.0) * effective_count
            elif effect == "mult":
                if scaling_type == "mult" and scaled_value is not None:
                    bonus_mult += scaled_value
                else:
                    bonus_mult += (schema.get("mult_value") or 0.0) * effective_count
            elif effect == "xmult":
                if scaling_type == "xmult" and scaled_value is not None:
                    xmult_val = max(scaled_value, 1.0)
                else:
                    xmult_val = schema.get("xmult_value") or 1.0
                if schema.get("per_card_instance") and effective_count > 1:
                    xmult_product *= xmult_val ** effective_count
                elif trigger_count > 1:
                    # Multi-trigger xmult (e.g. Baseball Card per uncommon)
                    xmult_product *= xmult_val ** trigger_count
                else:
                    # Once-per-hand xmult with retrigger: compound
                    # e.g. Photograph x2 + Hanging Chad = x2^3 = x8
                    xmult_product *= xmult_val ** max(1 + retrigger_extra, 1)

    # Apply joker EDITION bonuses (same as compute_joker_scoring)
    for joker in jokers:
        modifier = joker.get("modifier", {})
        edition = modifier.get("edition", "") if isinstance(modifier, dict) else ""
        if edition == "FOIL":
            bonus_chips += 50
        elif edition == "HOLO":
            bonus_mult += 10
        elif edition == "POLYCHROME":
            xmult_product *= 1.5

    return bonus_chips, bonus_mult, xmult_product


# ============================================================
# Joker Reordering — Optimal Positioning
# ============================================================

# Balatro evaluates jokers left-to-right. The order matters because:
# - Chips are added first (flat), so position doesn't matter much
# - Flat mult is added (also flat), position doesn't matter much
# - xMult is MULTIPLIED, so it should come LAST for maximum effect
#   (all chips and flat mult are already accumulated before xmult applies)
# - Copy jokers (Blueprint=copies RIGHT, Brainstorm=copies LEFTMOST)
#   need specific positioning to copy the best target


def _find_dagger_sacrifice(
    jokers: list[dict],
    current_order: list[int],
    gamestate: Optional[dict],
) -> Optional[int]:
    """Decide whether to sacrifice a joker to Ceremonial Dagger.

    Dagger destroys the joker to its RIGHT when a blind is selected,
    gaining 2× that joker's sell_value as permanent mult.

    Returns the 0-based index of the joker to sacrifice, or None.

    Decision logic:
    - Never sacrifice Blueprint, Brainstorm, or strong xmult jokers
    - Compare 2× sell_value (permanent mult gain) vs ongoing scoring contribution
    - For growth jokers like Egg (sell_value scales +$3/round), delay sacrifice
      to maximize value — only sacrifice when scoring pressure is high
    - Earlier antes = keep growing; later antes = more urgency to cash in
    """
    if not gamestate:
        return None

    # Parse game context
    ante = gamestate.get("ante_num", 1)
    rnd = gamestate.get("round", {})
    hands_left = rnd.get("hands_left", 4)
    current_chips = rnd.get("chips", 0)

    # Get blind target
    blinds = gamestate.get("blinds", {})
    blind_target = 0.0
    if isinstance(blinds, dict):
        for b in blinds.values():
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                blind_target = b.get("score", 0)
                break

    # Protected joker names — never sacrifice these
    PROTECTED = {"Blueprint", "Brainstorm", "Ceremonial Dagger"}

    # Evaluate each joker as a sacrifice candidate
    best_candidate = None
    best_value_ratio = 0.0  # ratio of sacrifice_gain / ongoing_value

    for idx in current_order:
        if idx < 0 or idx >= len(jokers):
            continue

        joker = jokers[idx]
        joker_key = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(joker_key)

        # Skip protected jokers
        if name in PROTECTED:
            continue

        # Skip unknown jokers (can't evaluate)
        if not name or name not in JOKERS:
            continue

        schema = JOKERS[name]

        # Never sacrifice strong xmult jokers (value compounds multiplicatively)
        xv = schema.get("xmult_value") or 1.0
        if xv >= 1.5:
            continue

        # Get sell value
        cost = joker.get("cost", {})
        if isinstance(cost, dict):
            sell_value = cost.get("sell", 0)
        else:
            sell_value = 0
        if sell_value <= 0:
            continue

        # Permanent mult gain from sacrifice
        dagger_mult_gain = 2.0 * sell_value

        # Estimate ongoing scoring contribution per hand
        scoring_power = _joker_scoring_power(joker)

        # Growth jokers (Egg etc.) — sell_value increases over time
        # Delay sacrifice: the longer we wait, the bigger the payoff
        # But don't wait forever — by ante 5+ the game gets hard
        is_growth = schema.get("scaling_type") == "sell_value"
        if is_growth:
            # Egg gains $3/round → each round of waiting = +6 permanent mult
            # Only sacrifice growth jokers when ante pressure is high
            if ante < 4:
                continue  # Too early, let it grow
            # At ante 4+, sacrifice if the mult gain is substantial
            # (Egg at ante 4 has had ~9 rounds = ~$27 sell → 54 mult)

        # Economy-only jokers (no score_effect) are prime candidates
        score_effect = schema.get("score_effect") or []
        is_economy_only = not score_effect and schema.get("economy", False)

        # Calculate value ratio: how much better is sacrificing vs keeping?
        if scoring_power <= 0.1:
            # Non-scoring joker — sacrifice value is essentially free
            value_ratio = dagger_mult_gain * 10.0  # Very favorable
        else:
            # Compare: permanent mult gain vs per-hand scoring contribution
            # Permanent mult applies every hand for rest of run
            # Estimate remaining hands: ~4 hands/round × 3 rounds/ante × remaining antes
            remaining_antes = max(8 - ante, 1)
            estimated_remaining_hands = remaining_antes * 3 * 4
            # But the mult gain also benefits all those hands
            # Simple heuristic: sacrifice if gain > 3× per-hand contribution
            value_ratio = dagger_mult_gain / max(scoring_power, 0.01)

        # Economy-only jokers get a bonus (they don't score)
        if is_economy_only:
            value_ratio *= 5.0

        # Threshold: only sacrifice if the ratio is clearly favorable
        # Higher threshold early game (be conservative), lower late game
        threshold = max(5.0 - (ante - 1) * 0.5, 1.5)

        if value_ratio > threshold and value_ratio > best_value_ratio:
            best_value_ratio = value_ratio
            best_candidate = idx

    return best_candidate


def _joker_scoring_power(joker: dict) -> float:
    """Rate a joker's scoring power for ordering decisions."""
    joker_key = joker.get("joker_key", "") or joker.get("key", "")
    name = _api_key_to_name(joker_key)
    if not name or name not in JOKERS:
        return 5.0
    s = JOKERS[name]
    xv = s.get("xmult_value") or 1.0
    mv = s.get("mult_value") or 0.0
    cv = s.get("chip_value") or 0.0
    return (xv - 1.0) * 50.0 + mv * 2.0 + cv * 0.1


def _resolve_copy_target(jokers: list[dict], order: list[int],
                         pos: int, direction: str) -> Optional[int]:
    """Resolve Blueprint/Brainstorm copy chains to the actual target joker.

    Blueprint copies the first non-copy joker to its RIGHT.
    Brainstorm copies the first non-copy joker starting from the LEFT.
    Both skip over other Blueprint/Brainstorm jokers in the chain.

    Returns the joker index in the original jokers list, or None if no
    valid target exists.
    """
    COPY_NAMES = {"Blueprint", "Brainstorm"}
    visited = set()

    if direction == "right":
        # Blueprint: scan rightward from pos+1
        scan_range = range(pos + 1, len(order))
        for scan_pos in scan_range:
            idx = order[scan_pos]
            if idx in visited or idx < 0 or idx >= len(jokers):
                continue
            visited.add(idx)
            jk = jokers[idx].get("joker_key", "") or jokers[idx].get("key", "")
            jn = _api_key_to_name(jk)
            if jn not in COPY_NAMES:
                return idx
            # Follow Blueprint chain rightward
            if jn == "Blueprint":
                continue  # keep scanning right
            # Brainstorm in a Blueprint chain: Brainstorm copies leftmost,
            # which is already being resolved — skip to avoid infinite loop
        return None
    else:
        # Brainstorm: always targets the LEFTMOST joker (order[0]).
        # If Brainstorm IS the leftmost, it copies itself → no effect.
        if len(order) == 0:
            return None
        leftmost_idx = order[0]
        if leftmost_idx < 0 or leftmost_idx >= len(jokers):
            return None

        # Check if the leftmost joker is Brainstorm itself
        lk = jokers[leftmost_idx].get("joker_key", "") or jokers[leftmost_idx].get("key", "")
        ln = _api_key_to_name(lk)
        if ln == "Brainstorm":
            return None  # Brainstorm copies itself — no effect

        if ln not in COPY_NAMES:
            return leftmost_idx  # Normal joker — copy it directly

        # Leftmost is Blueprint — follow Blueprint's copy chain (rightward)
        if ln == "Blueprint":
            for scan_pos in range(1, len(order)):
                idx = order[scan_pos]
                if idx in visited or idx < 0 or idx >= len(jokers):
                    continue
                visited.add(idx)
                jk = jokers[idx].get("joker_key", "") or jokers[idx].get("key", "")
                jn = _api_key_to_name(jk)
                if jn not in COPY_NAMES:
                    return idx
                if jn == "Blueprint":
                    continue  # keep following chain right
            return None

        return None


def _score_joker_order_with_cards(jokers: list[dict], order: list[int],
                                   hand_type: str, cards: list[dict],
                                   scoring_indices: list[int],
                                   gamestate: Optional[dict] = None) -> float:
    """Score a joker ordering using actual cards, simulating left-to-right.

    For each joker in order, computes its contribution using the real
    cards (suits, ranks, enhancements) and applies chips/mult/xmult
    sequentially just like Balatro does.
    """
    hand_info = gamestate.get("hands", {}).get(hand_type, {}) if gamestate else {}
    base_c, base_m = BASE_HAND_SCORES.get(hand_type, (5, 1))
    chips = float(hand_info.get("chips", base_c)) + sum(card_chips(cards[i]) for i in scoring_indices)
    mult = float(hand_info.get("mult", base_m))

    # Pre-compute card properties for trigger checking
    played_suits = [card_suit(cards[i]) for i in scoring_indices]
    played_ranks = [card_rank(cards[i]) for i in scoring_indices]
    has_face = any(r in ("J", "Q", "K") for r in played_ranks)

    for pos, idx in enumerate(order):
        if idx < 0 or idx >= len(jokers):
            continue
        joker = jokers[idx]
        joker_key = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(joker_key)
        schema = JOKERS.get(name) if name else None

        j_chips, j_mult, j_xmult = _compute_single_joker_effect(
            joker, name, schema, hand_type, played_suits, played_ranks, has_face
        )

        # Blueprint: copies the joker to its RIGHT (skipping other Blueprints)
        if name == "Blueprint":
            copy_target = _resolve_copy_target(jokers, order, pos, "right")
            if copy_target is not None:
                rj = jokers[copy_target]
                rk = rj.get("joker_key", "") or rj.get("key", "")
                rn = _api_key_to_name(rk)
                rs = JOKERS.get(rn) if rn else None
                rc, rm, rx = _compute_single_joker_effect(
                    rj, rn, rs, hand_type, played_suits, played_ranks, has_face
                )
                j_chips += rc; j_mult += rm; j_xmult *= rx

        # Brainstorm: copies the LEFTMOST joker (resolving through copy chains)
        if name == "Brainstorm":
            copy_target = _resolve_copy_target(jokers, order, pos, "left")
            if copy_target is not None:
                lj = jokers[copy_target]
                lk = lj.get("joker_key", "") or lj.get("key", "")
                ln = _api_key_to_name(lk)
                ls = JOKERS.get(ln) if ln else None
                lc, lm, lx = _compute_single_joker_effect(
                    lj, ln, ls, hand_type, played_suits, played_ranks, has_face
                )
                j_chips += lc; j_mult += lm; j_xmult *= lx

        # Apply LEFT TO RIGHT
        chips += j_chips
        mult += j_mult
        mult *= j_xmult

    return chips * mult


def _compute_single_joker_effect(
    joker: dict, name: str | None, schema: Optional[dict],
    hand_type: str, played_suits: list[str], played_ranks: list[str],
    has_face: bool
) -> tuple[float, float, float]:
    """Compute one joker's chips/mult/xmult using actual played card data."""
    j_chips = 0.0
    j_mult = 0.0
    j_xmult = 1.0

    if schema:
        effects = schema.get("score_effect") or []
        triggers = schema.get("triggers") or []

        # Determine if this joker triggers and how many times
        trigger_count = 1
        triggered = False

        if not triggers or "any_hand_played" in triggers:
            triggered = True
        if "specific_hand_type" in triggers:
            target_ht = schema.get("trigger_hand_type") or ""
            if target_ht and (hand_type == target_ht or _hand_contains(hand_type, target_ht)):
                triggered = True
        if "scoring_card" in triggers:
            triggered = True
            trigger_count = len(played_suits)
        if "face_card" in triggers:
            if has_face:
                triggered = True
                if schema.get("per_card_instance"):
                    trigger_count = sum(1 for r in played_ranks if r in ("J", "Q", "K"))
        if "specific_suit" in triggers:
            target_suits = schema.get("trigger_suits") or []
            matching = sum(1 for s in played_suits if s in target_suits)
            if matching > 0:
                triggered = True
                if schema.get("per_card_instance"):
                    trigger_count = matching
        if "specific_rank" in triggers:
            target_ranks = _normalize_trigger_ranks(schema.get("trigger_ranks") or [])
            matching = sum(1 for r in played_ranks if r in target_ranks)
            if matching > 0:
                triggered = True
                if schema.get("per_card_instance"):
                    trigger_count = matching

        if not triggered:
            # Scaling jokers with non-scoring triggers fire every hand
            _sv = joker.get("_scaled_value")
            _st = schema.get("scaling_type")
            if _sv is not None and _st:
                triggered = True
                trigger_count = 1

        if triggered:
            scaled_value = joker.get("_scaled_value")
            scaling_type = schema.get("scaling_type")
            if "chips" in effects or "chips_and_mult" in effects:
                if scaling_type == "chips" and scaled_value is not None:
                    j_chips = scaled_value
                else:
                    j_chips = (schema.get("chip_value") or 0.0) * trigger_count
            if "mult" in effects or "chips_and_mult" in effects:
                if scaling_type == "mult" and scaled_value is not None:
                    j_mult = scaled_value
                else:
                    j_mult = (schema.get("mult_value") or 0.0) * trigger_count
            if "xmult" in effects:
                if scaling_type == "xmult" and scaled_value is not None:
                    xv = max(scaled_value, 1.0)
                else:
                    xv = schema.get("xmult_value") or 1.0
                if trigger_count > 1 and schema.get("per_card_instance"):
                    j_xmult = xv ** trigger_count
                else:
                    j_xmult = xv
    else:
        # Unknown joker — check ability_extra
        extra = joker.get("ability_extra")
        if isinstance(extra, dict):
            if "Xmult" in extra or "x_mult" in extra:
                j_xmult = float(extra.get("Xmult") or extra.get("x_mult") or 1.0)
            if "mult" in extra or "mult_mod" in extra:
                j_mult = float(extra.get("mult") or extra.get("mult_mod") or 0.0)
            if "chips" in extra or "chip_mod" in extra:
                j_chips = float(extra.get("chips") or extra.get("chip_mod") or 0.0)

    # Edition bonuses
    mod = joker.get("modifier", {})
    if isinstance(mod, dict):
        edition = mod.get("edition", "")
        if edition == "FOIL":
            j_chips += 50
        elif edition == "HOLO":
            j_mult += 10
        elif edition == "POLYCHROME":
            j_xmult *= 1.5

    return j_chips, j_mult, j_xmult


def compute_optimal_joker_order(jokers: list[dict], gamestate: Optional[dict] = None,
                                hand_cards: list[dict] | None = None,
                                deck_cards: list[dict] | None = None) -> Optional[list[int]]:
    """Compute the optimal left-to-right ordering for jokers.

    Brute-forces all permutations (max 5! = 120) and scores each one
    using left-to-right simulation with actual cards.

    When hand_cards are available (during play), scores against the actual
    best hands. When not available (shop phase), builds representative
    hands from deck composition.

    Returns a list of 0-based indices representing the new order,
    or None if no reorder is needed (already optimal or ≤1 joker).
    """
    if len(jokers) <= 1:
        return None

    from itertools import permutations

    # Build test hands from actual cards
    test_hands: list[tuple[str, list[dict], list[int], float]] = []
    # Each entry: (hand_type, cards, scoring_indices, weight)

    if hand_cards and len(hand_cards) > 0:
        # During play: use actual best hands from current cards
        top = find_best_hands(hand_cards, jokers, gamestate or {}, top_n=5)
        for h in top:
            ht = h["hand_type"]
            ci = list(h["card_indices"])
            cards_for_hand = [hand_cards[i] for i in ci]
            si = list(range(len(cards_for_hand)))
            # Weight by score (better hands matter more)
            test_hands.append((ht, cards_for_hand, si, h["estimated_score"]))

        # Also test flush draws per suit if we have enough cards
        for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]:
            suit_cards = [c for c in hand_cards if card_suit(c) == suit]
            if len(suit_cards) < 3 and deck_cards:
                # Add deck cards of this suit to simulate the flush
                extra = [c for c in deck_cards if card_suit(c) == suit]
                extra.sort(key=lambda c: card_chips(c), reverse=True)
                suit_cards = suit_cards + extra[:5 - len(suit_cards)]
            if len(suit_cards) >= 5:
                flush_cards = suit_cards[:5]
                si = list(range(5))
                test_hands.append(("Flush", flush_cards, si, 1.0))

    if not test_hands:
        # Shop phase or no hand cards: build representative hands from deck
        all_cards = list(deck_cards or [])
        if not all_cards:
            return None

        # Best pair: find most common rank
        rank_groups: dict[str, list[dict]] = {}
        for c in all_cards:
            rank_groups.setdefault(card_rank(c), []).append(c)

        # Test a pair of the most common rank
        for rank, cards in sorted(rank_groups.items(), key=lambda x: -len(x[1])):
            if len(cards) >= 2:
                pair_cards = cards[:2] + all_cards[:3]  # pair + 3 kickers
                pair_cards = pair_cards[:5]
                test_hands.append(("Pair", pair_cards, list(range(len(pair_cards))), 1.0))
                break

        # Test a flush per suit
        suit_groups: dict[str, list[dict]] = {}
        for c in all_cards:
            suit_groups.setdefault(card_suit(c), []).append(c)
        for suit, cards in suit_groups.items():
            if len(cards) >= 5:
                flush_cards = sorted(cards, key=lambda c: card_chips(c), reverse=True)[:5]
                test_hands.append(("Flush", flush_cards, list(range(5)), 1.0))

        # Test a straight if possible
        vals: dict[int, dict] = {}
        for c in all_cards:
            v = RANK_ORDER.get(card_rank(c), 0)
            if v and v not in vals:
                vals[v] = c
        for low in range(1, 11):
            window = list(range(low, low + 5))
            if all(v in vals for v in window):
                straight_cards = [vals[v] for v in window]
                test_hands.append(("Straight", straight_cards, list(range(5)), 1.0))
                break

    if not test_hands:
        return None

    indices = list(range(len(jokers)))
    best_order = None
    best_score = -1.0

    for perm in permutations(indices):
        order = list(perm)
        total = 0.0
        for ht, cards, si, weight in test_hands:
            s = _score_joker_order_with_cards(jokers, order, ht, cards, si, gamestate)
            total += s * weight
        if total > best_score:
            best_score = total
            best_order = order

    if best_order is None or best_order == indices:
        return None

    return best_order


# ============================================================
# Consumable Usage Planner
# ============================================================

# Planet card key → hand type it levels up
PLANET_TO_HAND_TYPE = {
    "c_pluto": "High Card", "c_mercury": "Pair", "c_uranus": "Two Pair",
    "c_venus": "Three of a Kind", "c_saturn": "Straight", "c_jupiter": "Flush",
    "c_earth": "Full House", "c_mars": "Four of a Kind",
    "c_neptune": "Straight Flush", "c_planet_x": "Five of a Kind",
    "c_ceres": "Flush House", "c_eris": "Flush Five",
}

# Suit name → API suit code
SUIT_NAME_TO_CODE = {
    "Diamonds": "D", "Clubs": "C", "Hearts": "H", "Spades": "S",
}


def _get_joker_suit_synergies(jokers: list[dict]) -> set[str]:
    """Find which suits our jokers care about.

    Returns set of suit codes (D, C, H, S) that jokers reward.
    """
    wanted_suits: set[str] = set()
    for joker in jokers:
        joker_key = joker.get("joker_key", "") or joker.get("key", "")
        name = _api_key_to_name(joker_key)
        if not name or name not in JOKERS:
            continue
        schema = JOKERS[name]
        # Check both trigger_suit (singular) and trigger_suits (plural list)
        trigger_suits = schema.get("trigger_suits") or []
        trigger_suit = schema.get("trigger_suit")
        if trigger_suit:
            trigger_suits = list(trigger_suits) + [trigger_suit]
        for suit_name in trigger_suits:
            code = SUIT_NAME_TO_CODE.get(suit_name)
            if code:
                wanted_suits.add(code)
    return wanted_suits


def _get_best_hand_type(gamestate: dict) -> str:
    """Find the hand type the player uses most / scores best with."""
    hands_data = gamestate.get("hands", {})
    best_ht = "Pair"
    best_score = 0
    for ht_name in ("Pair", "Two Pair", "Three of a Kind", "Straight",
                     "Flush", "Full House", "Four of a Kind"):
        ht_info = hands_data.get(ht_name, {})
        if not isinstance(ht_info, dict):
            continue
        played = ht_info.get("played", 0)
        if played > 0:
            base_c, base_m = BASE_HAND_SCORES.get(ht_name, (5, 1))
            score = ht_info.get("chips", base_c) * ht_info.get("mult", base_m)
            if score > best_score:
                best_score = score
                best_ht = ht_name
    return best_ht


def _find_weakest_hand_cards(hand_cards: list[dict], count: int,
                              exclude_ranks: Optional[set[str]] = None,
                              exclude_suits: Optional[set[str]] = None,
                              ) -> list[int]:
    """Find the weakest cards in hand (lowest chip value, non-enhanced).

    Returns 0-based indices of the weakest cards.
    """
    scored: list[tuple[float, int]] = []
    for i, card in enumerate(hand_cards):
        rank, suit = parse_card(card)
        if not rank:
            continue
        chip_val = CARD_CHIP_VALUES.get(rank, 0)
        # Penalize cards we want to keep
        penalty = 0
        if exclude_ranks and rank in exclude_ranks:
            penalty += 100
        if exclude_suits and (suit in exclude_suits
                              or SUIT_NAME_TO_CODE.get(suit, "") in exclude_suits):
            penalty += 50
        # Enhanced cards are more valuable
        modifier = card.get("modifier", {})
        if isinstance(modifier, dict) and modifier.get("enhancement"):
            penalty += 30
        scored.append((chip_val + penalty, i))

    scored.sort(key=lambda x: x[0])
    return [idx for _, idx in scored[:count]]


def _find_highest_value_cards(hand_cards: list[dict], count: int,
                                preferred_ranks: Optional[set[str]] = None,
                                preferred_suits: Optional[set[str]] = None,
                                ) -> list[int]:
    """Find the highest-value cards in hand for enhancement.

    Returns 0-based indices of the best targets (most-played, highest chip value).
    """
    scored: list[tuple[float, int]] = []
    for i, card in enumerate(hand_cards):
        rank, suit = parse_card(card)
        if not rank:
            continue
        chip_val = CARD_CHIP_VALUES.get(rank, 0)
        bonus = 0
        if preferred_ranks and rank in preferred_ranks:
            bonus += 50
        if preferred_suits and (suit in preferred_suits
                                or SUIT_NAME_TO_CODE.get(suit, "") in preferred_suits):
            bonus += 25
        # Already enhanced → don't double-enhance
        modifier = card.get("modifier", {})
        if isinstance(modifier, dict) and modifier.get("enhancement"):
            bonus -= 100
        scored.append((chip_val + bonus, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scored[:count]]


def _find_suit_change_targets(hand_cards: list[dict], target_suit_code: str,
                                count: int = 3) -> list[int]:
    """Find cards that would benefit most from a suit change.

    Returns 0-based indices of cards NOT already in the target suit,
    prioritizing high-value cards.
    """
    target_suit_name = _SUIT_EXPAND.get(target_suit_code, target_suit_code)
    candidates: list[tuple[float, int]] = []
    for i, card in enumerate(hand_cards):
        rank, suit = parse_card(card)
        if not rank:
            continue
        if suit == target_suit_name:
            continue  # Already the right suit
        chip_val = CARD_CHIP_VALUES.get(rank, 0)
        candidates.append((chip_val, i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in candidates[:count]]


def _find_death_targets(hand_cards: list[dict],
                         jokers: list[dict],
                         gamestate: dict) -> Optional[tuple[int, int]]:
    """Find the best pair of cards for Death tarot.

    Death copies the RIGHT card's rank/suit onto the LEFT card.
    Strategy: suit-aware — prioritize copying high-value cards in the
    suit our jokers want. Makes more face cards in the dominant suit
    for flush builds with face-card jokers (Photograph, etc.).

    Returns (left_idx, right_idx) where right overwrites left, or None.
    """
    if len(hand_cards) < 2:
        return None

    wanted_suits = _get_joker_suit_synergies(jokers)
    dominant_suit = _get_dominant_suit(jokers)
    # Convert dominant suit to code if it's a full name
    dominant_code = SUIT_NAME_TO_CODE.get(dominant_suit, dominant_suit) if dominant_suit else ""

    # Score each card as a copy SOURCE (right card = what we duplicate)
    source_scores: list[tuple[float, int]] = []
    for i, card in enumerate(hand_cards):
        rank = card_rank(card)
        _, suit = parse_card(card)
        suit_code = SUIT_NAME_TO_CODE.get(suit, suit)
        if not rank:
            continue

        score = 0.0
        # Rank value: Kings > Queens > Jacks > Aces > numbers
        if rank == "K":
            score += 15.0
        elif rank == "Q":
            score += 12.0
        elif rank == "J":
            score += 10.0
        elif rank == "A":
            score += 8.0
        else:
            score += float(CARD_CHIP_VALUES.get(rank, 0)) / 10.0

        # HUGE bonus if card is in a wanted suit — duplicating it
        # means more cards in the suit our jokers need
        if suit_code in wanted_suits:
            score += 20.0
        elif dominant_code and suit_code == dominant_code:
            score += 15.0

        # Bonus for face cards when we have face-card jokers
        has_face_joker = False
        for j in jokers:
            jk = j.get("joker_key", "") or j.get("key", "")
            jn = _api_key_to_name(jk)
            if jn in ("Photograph", "Sock and Buskin", "Pareidolia",
                       "Scary Face", "Smiley Face"):
                has_face_joker = True
                break
        if has_face_joker and rank in FACE_RANKS:
            score += 10.0

        # Bonus for enhancements/seals (Death preserves rank+suit, not mods)
        # Actually Death copies everything, so enhanced source = enhanced copy
        source_scores.append((score, i))

    if not source_scores:
        return None

    source_scores.sort(key=lambda x: x[0], reverse=True)
    copy_source = source_scores[0][1]
    source_rank = card_rank(hand_cards[copy_source])
    _, source_suit = parse_card(hand_cards[copy_source])
    source_suit_code = SUIT_NAME_TO_CODE.get(source_suit, source_suit)

    # Score each card as an OVERWRITE target (left card = what we replace)
    overwrite_candidates: list[tuple[float, int]] = []
    for i, card in enumerate(hand_cards):
        if i == copy_source:
            continue
        rank = card_rank(card)
        _, suit = parse_card(card)
        suit_code = SUIT_NAME_TO_CODE.get(suit, suit)
        if not rank:
            continue
        # Don't overwrite a card that's already the same rank+suit
        if rank == source_rank and suit_code == source_suit_code:
            continue

        # Lower score = better overwrite target (we WANT to lose this card)
        score = float(CARD_CHIP_VALUES.get(rank, 0))
        # Penalty for overwriting face cards (they're valuable)
        if rank in FACE_RANKS:
            score += 50.0
        # Penalty for overwriting cards in wanted suit (don't destroy good suits)
        if suit_code in wanted_suits:
            score += 30.0
        elif dominant_code and suit_code == dominant_code:
            score += 25.0
        # Penalty for enhanced cards
        modifier = card.get("modifier", {})
        if isinstance(modifier, dict) and modifier.get("enhancement"):
            score += 20.0
        overwrite_candidates.append((score, i))

    if not overwrite_candidates:
        return None

    overwrite_candidates.sort(key=lambda x: x[0])
    left_idx = overwrite_candidates[0][1]  # Weakest card → gets overwritten

    return (left_idx, copy_source)


def plan_consumable_use(gamestate: dict) -> Optional[dict]:
    """Decide whether and how to use a held consumable.

    Called during SELECTING_HAND state. Returns an action dict for the
    API call, or None if no consumable should be used right now.

    Returns: {"consumable": int, "cards": list[int]} or {"consumable": int}
    """
    consumable_cards = gamestate.get("consumables", {}).get("cards", [])
    if not consumable_cards:
        return None

    hand_cards = gamestate.get("hand", {}).get("cards", [])
    jokers = gamestate.get("jokers", {}).get("cards", [])
    money = gamestate.get("money", 0)
    _state = gamestate.get("state", "?")
    ante = gamestate.get("ante_num", 1)
    rnd = gamestate.get("round", {})
    hands_left = rnd.get("hands_left", 4)

    best_ht = _get_best_hand_type(gamestate)
    wanted_suits = _get_joker_suit_synergies(jokers)

    # Evaluate each consumable, pick the best one to use
    for cons_idx, cons in enumerate(consumable_cards):
        key = cons.get("key", "")
        card_set = cons.get("set", "")

        # ── PLANET CARDS ──
        # Always use planet cards — they level up hand types permanently
        if "PLANET" in card_set.upper() or key in PLANET_TO_HAND_TYPE:
            return {"consumable": cons_idx}

        # ── THE HERMIT ──
        # Doubles money (cap $20). Use at $20+ for guaranteed max payout,
        # or $10-19 for good value. Also use at $5+ on last hand.
        if key == "c_hermit":
            if money >= 20:
                return {"consumable": cons_idx}
            if 10 <= money <= 19:
                return {"consumable": cons_idx}
            if money >= 5 and hands_left <= 1:
                return {"consumable": cons_idx}
            continue  # Wait for better timing

        # ── TEMPERANCE ──
        # Gives total sell value of all jokers (cap $50). Always use.
        if key == "c_temperance":
            return {"consumable": cons_idx}

        # ── WHEEL OF FORTUNE ──
        # 1-in-4 chance of adding edition to random joker. Always use if we have jokers.
        if key == "c_wheel_of_fortune":
            if jokers:
                return {"consumable": cons_idx}
            continue

        # ── JUDGEMENT ──
        # Creates random joker. Use if we have joker slots.
        if key == "c_judgement":
            joker_count = gamestate.get("jokers", {}).get("count", 0)
            joker_limit = gamestate.get("jokers", {}).get("limit", 5)
            if joker_count < joker_limit:
                return {"consumable": cons_idx}
            continue

        # ── THE FOOL ──
        # Creates last Tarot/Planet used. Use if consumable slot available.
        if key == "c_fool":
            cons_count = len(consumable_cards)
            cons_limit = gamestate.get("consumables", {}).get("limit", 2)
            if cons_count < cons_limit:
                return {"consumable": cons_idx}
            continue

        # ── HIGH PRIESTESS ──
        # Creates up to 2 random Planet cards. Use if slots available.
        if key == "c_high_priestess":
            cons_count = len(consumable_cards)
            cons_limit = gamestate.get("consumables", {}).get("limit", 2)
            if cons_count < cons_limit:
                return {"consumable": cons_idx}
            continue

        # ── EMPEROR ──
        # Creates up to 2 random Tarot cards. Use if slots available.
        if key == "c_emperor":
            cons_count = len(consumable_cards)
            cons_limit = gamestate.get("consumables", {}).get("limit", 2)
            if cons_count < cons_limit:
                return {"consumable": cons_idx}
            continue

        # ── Cards below require hand cards ──
        if not hand_cards:
            continue

        # ── DEATH ──
        # Copy right card onto left card. Use strategically.
        if key == "c_death":
            targets = _find_death_targets(hand_cards, jokers, gamestate)
            if targets:
                left_idx, right_idx = targets
                return {"consumable": cons_idx, "cards": [left_idx, right_idx]}
            continue

        # ── STRENGTH ──
        # Increase rank of 1-2 cards by 1. Priority: make Kings (Q→K),
        # then make face cards (10→J), then upgrade other non-face cards
        # toward face card territory. Never target Aces or Kings.
        if key == "c_strength":
            queens = []     # Q → K (highest priority)
            tens = []       # 10 → J (second priority)
            jacks = []      # J → Q (third priority)
            others = []     # everything else below 10
            for i, card in enumerate(hand_cards):
                rank, _ = parse_card(card)
                if not rank or rank in ("A", "K"):
                    continue
                if rank == "Q":
                    queens.append(i)
                elif rank == "T":
                    tens.append(i)
                elif rank == "J":
                    jacks.append(i)
                elif RANK_ORDER.get(rank, 0) < 10:
                    others.append((RANK_ORDER.get(rank, 0), i))

            # Build target list in priority order
            targets: list[int] = []
            for pool in [queens, tens, jacks]:
                for idx in pool:
                    if len(targets) >= 2:
                        break
                    targets.append(idx)
                if len(targets) >= 2:
                    break
            # Fill remaining slots with highest non-face cards
            if len(targets) < 2 and others:
                others.sort(key=lambda x: x[0], reverse=True)  # Highest first (closer to face)
                for _, idx in others:
                    if len(targets) >= 2:
                        break
                    targets.append(idx)

            if targets:
                return {"consumable": cons_idx, "cards": targets}
            continue

        # ── SUIT CHANGE TAROTS (Star/Moon/Sun/World) ──
        # Use suit conversion strategically:
        # - If target suit matches joker synergies: ALWAYS use (high priority)
        # - If no suit jokers: use for deck consistency (moderate priority)
        # - If target suit CONFLICTS with joker synergies: SKIP
        if key in SUIT_CHANGE_TAROTS:
            target_code, max_targets = SUIT_CHANGE_TAROTS[key]
            target_suit_name = SUIT_CODE_TO_NAME.get(target_code, "")

            if wanted_suits:
                if target_code in wanted_suits:
                    # Target matches our joker strategy — high value conversion
                    targets = _find_suit_change_targets(
                        hand_cards, target_code, count=max_targets)
                    if targets:
                        return {"consumable": cons_idx, "cards": targets}
                # else: target suit doesn't match our jokers — skip
            else:
                # No suit jokers — still useful for flush consistency
                targets = _find_suit_change_targets(
                    hand_cards, target_code, count=max_targets)
                if targets:
                    return {"consumable": cons_idx, "cards": targets}
            continue

        # ── ENHANCEMENT TAROTS ──
        if key in ENHANCEMENT_TAROTS:
            effect_name, target_count = ENHANCEMENT_TAROTS[key]
            # Steel and Glass are the most valuable enhancements
            # Chariot (Steel, x1.5) and Justice (Glass, x2) → enhance best scoring cards
            # Devil (Gold, $3) → enhance cards we hold often (economy)
            # Tower (Stone, +50 chips) → makes card lose rank/suit, use on weak cards
            # Empress (Mult, +4) → enhance scoring cards
            # Hierophant (Bonus, +30 chips) → enhance scoring cards
            # Magician (Lucky) → enhance scoring cards
            # Lovers (Wild) → any suit, good for flush builds

            if effect_name == "Stone":
                # Stone removes rank/suit — use on weakest cards
                targets = _find_weakest_hand_cards(hand_cards, target_count)
            elif effect_name == "Gold":
                # Gold gives $3 if held at end of round — use on cards we DON'T play
                targets = _find_weakest_hand_cards(hand_cards, target_count)
            elif effect_name == "Wild":
                # Wild = any suit — useful for flush builds, enhance a non-matching card
                if wanted_suits:
                    # Find cards not in any wanted suit
                    non_matching = []
                    for i, card in enumerate(hand_cards):
                        _, suit = parse_card(card)
                        suit_code = SUIT_NAME_TO_CODE.get(suit, suit)
                        if suit_code not in wanted_suits:
                            non_matching.append(i)
                    targets = non_matching[:target_count] if non_matching else \
                        _find_highest_value_cards(hand_cards, target_count)
                else:
                    targets = _find_highest_value_cards(hand_cards, target_count)
            else:
                # Steel, Glass, Mult, Bonus, Lucky → enhance best scoring cards
                # Prefer cards with ranks our jokers care about
                preferred_ranks: set[str] = set()
                for joker in jokers:
                    jk = joker.get("joker_key", "") or joker.get("key", "")
                    name = _api_key_to_name(jk)
                    if name and name in JOKERS:
                        tr = JOKERS[name].get("trigger_ranks")
                        if tr:
                            preferred_ranks.update(_normalize_trigger_ranks(tr))

                targets = _find_highest_value_cards(
                    hand_cards, target_count,
                    preferred_ranks=preferred_ranks or None,
                    preferred_suits=wanted_suits or None,
                )

            if targets:
                return {"consumable": cons_idx, "cards": targets}
            continue

        # ── THE HANGED MAN ──
        # Destroy 1-2 cards. Use to thin deck of weak cards.
        # Only use if we have a clear strategy (late ante, deck thinning helps).
        if key == "c_hanged_man":
            if len(hand_cards) >= 4:
                # Find the weakest cards that don't match our build
                preferred_ranks: set[str] = set()
                for joker in jokers:
                    jk = joker.get("joker_key", "") or joker.get("key", "")
                    name = _api_key_to_name(jk)
                    if name and name in JOKERS:
                        tr = JOKERS[name].get("trigger_ranks")
                        if tr:
                            preferred_ranks.update(_normalize_trigger_ranks(tr))

                targets = _find_weakest_hand_cards(
                    hand_cards, 2,
                    exclude_ranks=preferred_ranks or {"K", "Q", "J", "A"},
                    exclude_suits=wanted_suits or None,
                )
                if targets:
                    return {"consumable": cons_idx, "cards": targets}
            continue

    return None  # No consumable worth using right now


# ============================================================
# Tarot Pack Evaluation
# ============================================================

# Suit change tarots: key → (target_suit_code, max_targets)
SUIT_CHANGE_TAROTS: dict[str, tuple[str, int]] = {
    "c_star": ("D", 3),     # The Star → Diamonds
    "c_moon": ("C", 3),     # The Moon → Clubs
    "c_sun": ("H", 3),      # The Sun → Hearts
    "c_world": ("S", 3),    # The World → Spades
}

# Suit code → full name
SUIT_CODE_TO_NAME: dict[str, str] = {
    "H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades",
}

# Enhancement tarots: key → (enhancement_name, max_targets)
ENHANCEMENT_TAROTS: dict[str, tuple[str, int]] = {
    "c_magician": ("Lucky", 2),
    "c_empress": ("Mult", 2),
    "c_heirophant": ("Bonus", 2),
    "c_lovers": ("Wild", 1),
    "c_chariot": ("Steel", 1),
    "c_justice": ("Glass", 1),
    "c_devil": ("Gold", 1),
    "c_tower": ("Stone", 1),
}

# Non-targeting tarots (use without hand targets)
NO_TARGET_TAROTS: set[str] = {
    "c_hermit", "c_temperance", "c_wheel_of_fortune",
    "c_judgement", "c_fool", "c_high_priestess", "c_emperor",
}


def evaluate_pack_tarot(pack_cards: list[dict], hand_cards: list[dict],
                        jokers: list[dict], gamestate: dict
                        ) -> Optional[tuple[int, list[int]]]:
    """Evaluate tarot cards in a pack and select the best one with targets.

    Returns (pick_index, target_indices) or None if all tarots are worthless.
    target_indices is a list of 0-based hand card indices, empty for non-targeting.
    """
    if not pack_cards or not hand_cards:
        return None

    wanted_suits = _get_joker_suit_synergies(jokers)
    best_pick: Optional[tuple[float, int, list[int]]] = None  # (score, card_idx, targets)

    for pc_idx, pc in enumerate(pack_cards):
        key = pc.get("key", "")
        score = 0.0
        targets: list[int] = []

        # ── Suit change tarots — high value if we have suit-specific jokers ──
        if key in SUIT_CHANGE_TAROTS:
            target_suit_code, max_targets = SUIT_CHANGE_TAROTS[key]
            target_suit_name = SUIT_CODE_TO_NAME.get(target_suit_code, "")

            if target_suit_code in wanted_suits:
                # Our jokers want this suit — high value
                score = 8.0
                targets = _find_suit_change_targets(
                    hand_cards, target_suit_code, max_targets
                )
                if not targets:
                    score = 0.5  # All cards already this suit — minimal value
                else:
                    # Bonus for more convertible cards
                    score += len(targets) * 1.0
            else:
                # No joker synergy, but suit conversion still has some value
                score = 1.5
                targets = _find_suit_change_targets(
                    hand_cards, target_suit_code, max_targets
                )
                if not targets:
                    score = 0.0

        # ── Enhancement tarots ──
        elif key in ENHANCEMENT_TAROTS:
            enh_name, max_targets = ENHANCEMENT_TAROTS[key]

            if enh_name == "Steel":
                # Steel (x1.5 mult when held) — very strong
                score = 7.0
                targets = _find_highest_value_cards(
                    hand_cards, max_targets,
                    preferred_suits=wanted_suits or None,
                )
            elif enh_name == "Glass":
                # Glass (x2 mult but may shatter) — strong but risky
                score = 6.0
                targets = _find_highest_value_cards(
                    hand_cards, max_targets,
                    preferred_suits=wanted_suits or None,
                )
            elif enh_name == "Mult":
                # +4 mult per card — decent
                score = 5.0
                targets = _find_highest_value_cards(
                    hand_cards, max_targets,
                    preferred_suits=wanted_suits or None,
                )
            elif enh_name == "Lucky":
                # Lucky card (+20 mult 1/5, +$20 1/15) — decent
                score = 4.0
                targets = _find_highest_value_cards(
                    hand_cards, max_targets,
                    preferred_suits=wanted_suits or None,
                )
            elif enh_name == "Bonus":
                # +30 chips — moderate
                score = 3.0
                targets = _find_highest_value_cards(
                    hand_cards, max_targets,
                    preferred_suits=wanted_suits or None,
                )
            elif enh_name == "Wild":
                # Wild card (counts as all suits) — great for suit jokers
                score = 6.0 if wanted_suits else 2.0
                # Target cards that DON'T match any wanted suit
                non_matching: list[tuple[float, int]] = []
                for i, card in enumerate(hand_cards):
                    _, suit = parse_card(card)
                    suit_code = SUIT_NAME_TO_CODE.get(suit, "")
                    if suit and suit_code not in wanted_suits:
                        chip_val = CARD_CHIP_VALUES.get(card_rank(card), 0)
                        non_matching.append((chip_val, i))
                non_matching.sort(key=lambda x: x[0], reverse=True)
                targets = [idx for _, idx in non_matching[:max_targets]]
                if not targets:
                    targets = _find_highest_value_cards(hand_cards, max_targets)
            elif enh_name in ("Gold", "Stone"):
                # Gold ($3 if held) / Stone (+50 chips, no rank/suit)
                score = 2.0
                targets = _find_weakest_hand_cards(hand_cards, max_targets)
            else:
                score = 1.0
                targets = _find_highest_value_cards(hand_cards, max_targets)

            if not targets:
                score = 0.0

        # ── Strength — rank upgrade (+1) ──
        elif key == "c_strength":
            score = 5.0
            # Prioritize Q→K, 10→J, J→Q upgrades
            strength_targets: list[tuple[float, int]] = []
            for i, card in enumerate(hand_cards):
                rank = card_rank(card)
                if not rank or rank in ("A", "K"):
                    continue  # Can't upgrade Aces or Kings meaningfully
                priority = 0.0
                if rank == "Q":
                    priority = 10.0  # Q→K (face card + highest chip value)
                elif rank == "T":
                    priority = 8.0   # 10→J (makes a face card)
                elif rank == "J":
                    priority = 7.0   # J→Q
                else:
                    priority = float(CARD_CHIP_VALUES.get(rank, 0))
                strength_targets.append((priority, i))
            strength_targets.sort(key=lambda x: x[0], reverse=True)
            targets = [idx for _, idx in strength_targets[:2]]
            if not targets:
                score = 0.0

        # ── Death — copy card ──
        elif key == "c_death":
            # Death is extremely powerful: duplicate a King of Hearts into
            # a weak off-suit card = instant deck improvement. Even better
            # with suit synergies (more cards in the right suit).
            score = 8.0
            if wanted_suits:
                score += 3.0  # suit-focused Death is game-winning
            death_pair = _find_death_targets(hand_cards, jokers, gamestate)
            if death_pair is not None:
                targets = list(death_pair)
            else:
                score = 0.0

        # ── The Hanged Man — destroy cards ──
        elif key == "c_hanged_man":
            # Deck thinning is one of the strongest strategies in Balatro.
            # Removing off-suit/low-value cards increases flush draw odds,
            # concentrates the deck toward synergy cards, and makes every
            # draw more likely to hit. Value scales with ante and suit focus.
            ante = gamestate.get("ante_num", 1)
            deck_size = gamestate.get("cards", {}).get("count", 52)
            # Base value: always good, scales with ante
            score = 4.0 + min(ante, 6)  # 5.0 at ante 1, up to 10.0 at ante 6+
            # Extra value if we have suit synergies (thinning off-suit = more flushes)
            if wanted_suits:
                score += 3.0
            # Extra value on smaller decks (each removal is proportionally bigger)
            if deck_size <= 40:
                score += 2.0
            preferred_ranks: set[str] = set()
            for joker in jokers:
                jk = joker.get("joker_key", "") or joker.get("key", "")
                name = _api_key_to_name(jk)
                if name and name in JOKERS:
                    tr = JOKERS[name].get("trigger_ranks")
                    if tr:
                        preferred_ranks.update(_normalize_trigger_ranks(tr))
            targets = _find_weakest_hand_cards(
                hand_cards, 2,
                exclude_ranks=preferred_ranks or {"K", "Q", "J", "A"},
                exclude_suits=wanted_suits or None,
            )
            if not targets:
                score = 0.0

        # ── Non-targeting tarots ──
        elif key in NO_TARGET_TAROTS:
            if key == "c_judgement":
                # Creates random joker — great if slots available
                joker_count = gamestate.get("jokers", {}).get("count", 0)
                joker_limit = gamestate.get("jokers", {}).get("limit", 5)
                if joker_count < joker_limit:
                    score = 7.0
                else:
                    score = 1.0  # No slots — low value
            elif key == "c_hermit":
                money = gamestate.get("money", 0)
                score = 4.0 if money >= 10 else 2.0
            elif key == "c_temperance":
                score = 3.0
            elif key == "c_wheel_of_fortune":
                score = 4.0
            elif key == "c_high_priestess":
                score = 2.0
            elif key == "c_emperor":
                score = 1.5
            elif key == "c_fool":
                score = 1.0
            targets = []  # No targeting needed

        else:
            # Unknown tarot — pick with low priority
            score = 0.5
            targets = []

        if score > 0 and (best_pick is None or score > best_pick[0]):
            best_pick = (score, pc_idx, targets)

    if best_pick is None:
        return None

    return (best_pick[1], best_pick[2])


def compute_tarot_value(tarot_key: str, jokers: list[dict],
                        deck_cards: list[dict]) -> float:
    """Compute expected value of a tarot card given joker lineup and deck.

    Returns a normalized value [0, 1] where 1.0 = extremely valuable.
    Used for state vector encoding to signal pack purchase value.
    """
    wanted_suits = _get_joker_suit_synergies(jokers)

    # Suit change tarots
    if tarot_key in SUIT_CHANGE_TAROTS:
        target_suit_code, _ = SUIT_CHANGE_TAROTS[tarot_key]
        if target_suit_code in wanted_suits:
            # Count how many deck cards would benefit from conversion
            target_suit_name = SUIT_CODE_TO_NAME.get(target_suit_code, "")
            non_matching = sum(
                1 for c in deck_cards
                if card_suit(c) != target_suit_name
            )
            deck_size = max(len(deck_cards), 1)
            # High value if many cards can be converted
            return min(0.5 + (non_matching / deck_size) * 0.5, 1.0)
        return 0.15  # Some value even without synergy

    # Enhancement tarots
    if tarot_key in ENHANCEMENT_TAROTS:
        enh_name, _ = ENHANCEMENT_TAROTS[tarot_key]
        if enh_name in ("Steel", "Glass"):
            return 0.7
        if enh_name in ("Mult", "Lucky"):
            return 0.5
        if enh_name == "Wild" and wanted_suits:
            return 0.6
        return 0.3

    # Strength / Death
    if tarot_key == "c_strength":
        return 0.5
    if tarot_key == "c_death":
        return 0.6

    # Non-targeting tarots
    if tarot_key == "c_judgement":
        return 0.7
    if tarot_key == "c_wheel_of_fortune":
        return 0.4
    if tarot_key == "c_hermit":
        return 0.4
    if tarot_key == "c_temperance":
        return 0.3
    if tarot_key == "c_hanged_man":
        return 0.3

    return 0.1  # Unknown tarot
