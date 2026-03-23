"""
Balatron — Game State

API client for BalatroBot + state vector translation + scaling tracker.

Flow:
1. Call gamestate API → raw JSON
2. Diff against previous snapshot + known action → detect events
3. Update scaling tracker for all owned jokers
4. Translate everything into fixed-size state vector for the network

See NOTES.md for full state vector layout.
"""

import asyncio
import json
import math
from typing import Any, Optional

import aiohttp
import numpy as np

from data.jokers import JOKERS
from environment.hand_eval import assess_strategy, HAND_EVAL_FEATURES


# ============================================================
# Constants
# ============================================================

API_URL = "http://127.0.0.1:12346"

# State vector section sizes
GAME_META_SIZE = 42  # 16 original + 22 boss blind + 4 reroll value
HAND_LEVELS_SIZE = 79       # 13 × 3 (level, chips, mult) + 13 play freq + 26 ROI deltas + 1 best target
DECK_COMP_SIZE = 61          # 52 rank×suit + 9 enhancement counts
VOUCHER_SIZE = 32
JOKER_FINGERPRINT_SIZE = 51  # raw fingerprint from encode_joker_fingerprint
JOKER_SLOT_SIZE = 54  # fingerprint (51) + synergy contribution (1) + sell guard (1) + sell score delta (1)
JOKER_SLOTS = 5
HAND_CARD_SIZE = 8
HAND_CARD_SLOTS = 12
CONSUMABLE_SIZE = 6
CONSUMABLE_SLOTS = 2
SHOP_JOKER_SIZE = 54  # fingerprint (51) + cost (1) + affordable (1) + upgrade delta (1)
SHOP_JOKER_SLOTS = 3
SHOP_VOUCHER_SIZE = 5
SHOP_VOUCHER_SLOTS = 2
SHOP_PACK_SIZE = 5
SHOP_PACK_SLOTS = 2

STATE_VECTOR_SIZE = (
    GAME_META_SIZE
    + HAND_LEVELS_SIZE
    + DECK_COMP_SIZE
    + VOUCHER_SIZE
    + JOKER_SLOTS * JOKER_SLOT_SIZE
    + HAND_CARD_SLOTS * HAND_CARD_SIZE
    + CONSUMABLE_SLOTS * CONSUMABLE_SIZE
    + SHOP_JOKER_SLOTS * SHOP_JOKER_SIZE
    + SHOP_VOUCHER_SLOTS * SHOP_VOUCHER_SIZE
    + SHOP_PACK_SLOTS * SHOP_PACK_SIZE
    + HAND_EVAL_FEATURES  # 40 hand evaluation features
)

# Rank/suit mappings for dense encoding
RANK_MAP = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}
RANK_NORM = {k: v / 12.0 for k, v in RANK_MAP.items()}

SUIT_MAP = {"H": 0, "D": 1, "C": 2, "S": 3}
SUIT_NORM = {k: v / 3.0 for k, v in SUIT_MAP.items()}

ENHANCEMENT_MAP = {
    "BONUS": 0, "MULT": 1, "WILD": 2, "GLASS": 3,
    "STEEL": 4, "STONE": 5, "GOLD": 6, "LUCKY": 7,
}

EDITION_MAP = {"FOIL": 0, "HOLO": 1, "POLYCHROME": 2, "NEGATIVE": 3}

SEAL_MAP = {"RED": 0, "BLUE": 1, "GOLD": 2, "PURPLE": 3}

# Poker hand types — order matches Balatro's internal ordering
HAND_TYPE_ORDER = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
    "Royal Flush", "Five of a Kind", "Flush House", "Flush Five",
]

# Celestial card level-up deltas: (chips_per_level, mult_per_level)
# Fixed amounts added each time a planet card is used for that hand type.
_CELESTIAL_DELTAS: dict[str, tuple[int, int]] = {
    "High Card":        (10, 1),
    "Pair":             (15, 1),
    "Two Pair":         (20, 1),
    "Three of a Kind":  (20, 2),
    "Straight":         (30, 3),
    "Flush":            (15, 2),
    "Full House":       (25, 2),
    "Four of a Kind":   (30, 3),
    "Straight Flush":   (40, 4),
    "Royal Flush":      (40, 4),
    "Five of a Kind":   (35, 3),
    "Flush House":      (40, 4),
    "Flush Five":       (40, 4),
}

# Joker key → our schema name mapping
# BalatroBot uses keys like "j_green_joker", our schema uses "Green Joker"
def _api_key_to_name(key: str) -> Optional[str]:
    """Convert BalatroBot joker key to our schema name.

    Delegates to hand_eval's cached version which handles all API key
    overrides, misspellings, and abbreviations.
    """
    from environment.hand_eval import _api_key_to_name as _heval_key_to_name
    return _heval_key_to_name(key)


# Voucher key → index mapping
VOUCHER_KEYS = [
    "v_overstock_norm", "v_overstock_plus", "v_clearance_sale",
    "v_liquidation", "v_hone", "v_glow_up", "v_reroll_surplus",
    "v_reroll_glut", "v_crystal_ball", "v_omen_globe",
    "v_telescope", "v_observatory", "v_grabber", "v_nacho_tong",
    "v_wasteful", "v_recyclomancy", "v_tarot_merchant",
    "v_tarot_tycoon", "v_planet_merchant", "v_planet_tycoon",
    "v_seed_money", "v_money_tree", "v_blank", "v_antimatter",
    "v_magic_trick", "v_illusion", "v_hieroglyph", "v_petroglyph",
    "v_directors_cut", "v_retcon", "v_paint_brush", "v_palette",
]
VOUCHER_MAP = {k: i for i, k in enumerate(VOUCHER_KEYS)}

FACE_RANKS = {"J", "Q", "K"}

# ============================================================
# Boss blind debuff classification
# ============================================================
# Categories (binary flags):
#   0: suit_debuff    — debuffs all cards of one suit
#   1: face_debuff    — debuffs face cards (J/Q/K)
#   2: hand_restrict  — restricts which hands can be played
#   3: discard_remove — removes or limits discards
#   4: hand_type_debuff — debuffs/weakens a specific hand type
#   5: other          — everything else (HP modifiers, draw limits, etc.)

# Boss name → (debuff_category_index, debuffed_suit_or_None)
# debuffed_suit uses SUIT_MAP keys: "H", "D", "C", "S"
BOSS_BLIND_INFO: dict[str, tuple[int, Optional[str]]] = {
    # Suit debuffs (category 0)
    "The Club":    (0, "C"),   # Debuffs all Clubs
    "The Goad":    (0, "S"),   # Debuffs all Spades
    "The Head":    (0, "H"),   # Debuffs all Hearts
    "The Window":  (0, "D"),   # Debuffs all Diamonds
    # Face debuff (category 1)
    "The Plant":   (1, None),  # Face cards are debuffed
    # Hand restrictions (category 2)
    "The Psychic": (2, None),  # Must play 5 cards
    "The Mouth":   (2, None),  # Only play 1 hand type per round
    "The Eye":     (2, None),  # Can't repeat hand types
    "The Needle":  (2, None),  # Only 1 hand per round
    # Discard removal (category 3)
    "The Hook":    (3, None),  # Discards 2 random cards per hand
    "The Ox":      (3, None),  # Playing most played hand sets money to $0
    "The Wheel":   (3, None),  # 1 in 7 chance cards are drawn face down
    # Hand type debuff (category 4)
    "The Flint":   (4, None),  # Halves base chips and mult
    "The Water":   (4, None),  # Start with 0 discards
    # Other (category 5)
    "The Wall":    (5, None),  # Blind has extra large score requirement
    "The House":   (5, None),  # First hand is drawn face down
    "The Mark":    (5, None),  # All face cards are drawn face down
    "The Fish":    (5, None),  # Cards are drawn face down after each hand
    "The Pillar":  (5, None),  # Cards played previously this ante are debuffed
    "The Serpent":  (5, None), # After play or discard, always draw 3 cards
    "The Manacle": (5, None),  # -1 hand size
    "Cerulean Bell": (5, None), # Forces 1 card to always be selected
    "Crimson Heart": (5, None), # 1 random joker disabled each hand
    "Amber Acorn":  (5, None),  # Flips and shuffles all jokers
    "Verdant Leaf": (5, None),  # All cards are debuffed until 1 card is sold
    "Violet Vessel": (5, None), # Very large blind score
}

SUIT_INDEX = {"H": 0, "D": 1, "C": 2, "S": 3}


# ============================================================
# Normalization helpers
# ============================================================

def _log_norm(value: float, scale: float = 1.0) -> float:
    """Log-scale normalization for large values (chips, scores)."""
    if value <= 0:
        return 0.0
    return math.log10(value + 1.0) / scale


def _clamp_norm(value: float, max_val: float) -> float:
    """Linear normalization clamped to [0, 1]."""
    if max_val <= 0:
        return 0.0
    return min(value / max_val, 1.0)


def _as_dict(val: Any) -> dict:
    """Normalize API fields that may be [] or {} to always be a dict."""
    return val if isinstance(val, dict) else {}


def _get_current_blind(blinds: Any) -> Optional[dict]:
    """Extract the current/active blind from the blinds structure.

    API returns blinds as {boss, small, big} dict, each with status CURRENT/UPCOMING.
    """
    if not isinstance(blinds, dict):
        return None
    for blind in blinds.values():
        if isinstance(blind, dict) and blind.get("status") == "CURRENT":
            return blind
    return None


def _get_upcoming_blind(blinds: Any) -> Optional[dict]:
    """Extract the next upcoming blind from the blinds structure.

    Returns the first blind with status UPCOMING, prioritized by game order:
    small → big → boss.
    """
    if not isinstance(blinds, dict):
        return None
    # Check in game order so we get the *next* one
    for key in ("small", "big", "boss"):
        b = blinds.get(key)
        if isinstance(b, dict) and b.get("status") == "UPCOMING":
            return b
    return None


def _encode_blind_features(blind: Optional[dict], vec: np.ndarray, offset: int) -> None:
    """Encode blind type (3 flags) + debuff category (6 flags) + debuffed suit (4 flags).

    Writes 13 floats starting at offset. All zero if blind is None.
    """
    if not blind:
        return

    # Blind type: 3-way one-hot (small, big, boss)
    btype = blind.get("type", "")
    if btype == "SMALL":
        vec[offset + 0] = 1.0
    elif btype == "BIG":
        vec[offset + 1] = 1.0
    elif btype == "BOSS":
        vec[offset + 2] = 1.0

    # Boss debuff classification
    name = blind.get("name", "")
    info = BOSS_BLIND_INFO.get(name)
    if info:
        cat_idx, debuffed_suit = info
        # Debuff category flags (6): indices 3-8
        vec[offset + 3 + cat_idx] = 1.0
        # Debuffed suit one-hot (4): indices 9-12
        if debuffed_suit and debuffed_suit in SUIT_INDEX:
            vec[offset + 9 + SUIT_INDEX[debuffed_suit]] = 1.0


# ============================================================
# Scaling Tracker
# ============================================================

class ScalingTracker:
    """Tracks runtime scaling values for all owned jokers.

    Maintains current_value, event_counter, and expiry_counter
    per joker slot. Updated by calling update() with detected
    events between gamestate snapshots.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking state for a new run."""
        self._joker_values: dict[int, float] = {}  # slot_id → current_value
        self._event_counters: dict[int, int] = {}   # slot_id → event count
        self._expiry_counters: dict[int, int] = {}   # slot_id → remaining
        self._joker_keys: dict[int, str] = {}        # slot_id → joker key
        self._consecutive_no_face: int = 0           # Ride the Bus
        self._consecutive_no_most_played: int = 0    # Obelisk

    def on_jokers_changed(self, joker_cards: list[dict]):
        """Called when the joker lineup changes. Syncs tracked slots."""
        current_ids = set()
        for card in joker_cards:
            slot_id = card["id"]
            key = card.get("key", "")
            current_ids.add(slot_id)

            if slot_id not in self._joker_keys or self._joker_keys[slot_id] != key:
                # New joker in this slot — initialize
                name = _api_key_to_name(key)
                schema = JOKERS.get(name, {}) if name else {}
                start = schema.get("scaling_start_value")
                self._joker_values[slot_id] = start if start is not None else 0.0
                self._event_counters[slot_id] = 0
                self._joker_keys[slot_id] = key

                # Initialize expiry
                if schema.get("expiry"):
                    threshold = schema.get("expiry_threshold", 0)
                    self._expiry_counters[slot_id] = threshold
                else:
                    self._expiry_counters[slot_id] = -1  # no expiry

        # Remove jokers that are no longer present
        stale = set(self._joker_keys.keys()) - current_ids
        for slot_id in stale:
            self._joker_keys.pop(slot_id, None)
            self._joker_values.pop(slot_id, None)
            self._event_counters.pop(slot_id, None)
            self._expiry_counters.pop(slot_id, None)

    def update(self, events: list[str], event_counts: dict[str, int],
               action_context: Optional[dict] = None):
        """Update all joker scaling values based on detected events.

        Args:
            events: list of event names (from trigger vocabulary)
            event_counts: per-event item counts (e.g. cards discarded)
            action_context: details about the action taken
                - "cards_played": list of card dicts
                - "cards_discarded": list of card dicts
                - "face_cards_scored": bool
                - "hand_type_played": str
                - "most_played_hand_type": str
        """
        ctx = action_context or {}

        # Track Ride the Bus consecutive counter
        if "any_hand_played" in events:
            if ctx.get("face_cards_scored", False):
                self._consecutive_no_face = 0
            else:
                self._consecutive_no_face += 1

        for slot_id, key in list(self._joker_keys.items()):
            name = _api_key_to_name(key)
            if not name or name not in JOKERS:
                continue
            schema = JOKERS[name]

            # Skip non-scaling jokers
            if not schema.get("scaling_type") and not schema.get("scaling_decay"):
                continue

            self._update_growth(slot_id, schema, events, event_counts, ctx)
            self._update_decay(slot_id, schema, events, event_counts)
            self._update_reset(slot_id, schema, events, ctx)
            self._update_expiry(slot_id, schema, events)

    def _update_growth(self, slot_id: int, schema: dict,
                       events: list[str], event_counts: dict[str, int],
                       ctx: dict):
        """Apply growth increment if scaling driver event occurred."""
        driver = schema.get("scaling_driver")
        method = schema.get("scaling_method")
        if not driver or driver not in events:
            return

        # Special case: Ride the Bus (negation trigger)
        if schema.get("negation_trigger") and schema.get("name") == "Ride the Bus":
            self._joker_values[slot_id] = float(self._consecutive_no_face)
            return

        # Special case: Obelisk (dynamic condition + negation)
        if schema.get("scaling_dynamic_condition") == "most_played_hand_type":
            hand_played = ctx.get("hand_type_played", "")
            most_played = ctx.get("most_played_hand_type", "")
            if hand_played == most_played:
                # Reset handled in _update_reset
                return
            # Fall through to normal growth

        # Determine increment count
        count = 1
        if schema.get("scaling_per_item"):
            count = event_counts.get(driver, 1)

        increment = schema.get("scaling_increment", 0.0)
        source = schema.get("scaling_increment_source", "static")

        if source == "derived":
            # Ceremonial Dagger — derived from destroyed joker sell value
            multiplier = schema.get("scaling_increment_multiplier", 1.0)
            derived_val = ctx.get("derived_scaling_value", 0.0)
            increment = derived_val * multiplier

        if method == "counter_threshold":
            threshold = schema.get("scaling_threshold", 1)
            self._event_counters[slot_id] = self._event_counters.get(slot_id, 0) + count
            while self._event_counters[slot_id] >= threshold:
                self._event_counters[slot_id] -= threshold
                self._joker_values[slot_id] = self._joker_values.get(slot_id, 0.0) + increment
        else:
            # flat_addition
            self._joker_values[slot_id] = self._joker_values.get(slot_id, 0.0) + (increment * count)

    def _update_decay(self, slot_id: int, schema: dict,
                      events: list[str], event_counts: dict[str, int]):
        """Apply decay if decay driver event occurred."""
        if not schema.get("scaling_decay"):
            return

        driver = schema.get("scaling_decay_driver")
        if not driver or driver not in events:
            return

        amount = schema.get("scaling_decay_amount", 0.0)
        count = 1
        if schema.get("scaling_decay_per_item"):
            count = event_counts.get(driver, 1)

        self._joker_values[slot_id] = self._joker_values.get(slot_id, 0.0) - (amount * count)

        floor = schema.get("scaling_decay_floor", 0.0)
        if self._joker_values[slot_id] <= floor:
            self._joker_values[slot_id] = floor
            # Note: destruction at floor is handled by the game itself.
            # We'll detect it when the joker disappears from the next gamestate.

    def _update_reset(self, slot_id: int, schema: dict,
                      events: list[str], ctx: dict):
        """Reset scaling value if reset trigger fired."""
        if not schema.get("scaling_resets"):
            return

        reset_trigger = schema.get("scaling_reset_trigger")
        if not reset_trigger:
            return

        should_reset = False

        if reset_trigger in events:
            should_reset = True

        # Special: Ride the Bus resets on face card scored
        if reset_trigger == "face_card" and ctx.get("face_cards_scored", False):
            should_reset = True

        # Special: Obelisk resets on most played hand type
        if schema.get("scaling_dynamic_condition") == "most_played_hand_type":
            if ctx.get("hand_type_played") == ctx.get("most_played_hand_type"):
                should_reset = True

        if should_reset:
            self._joker_values[slot_id] = schema.get("scaling_reset_value", 0.0)

    def _update_expiry(self, slot_id: int, schema: dict, events: list[str]):
        """Update expiry counter."""
        if not schema.get("expiry") or schema.get("expiry_type") == "probabilistic":
            return  # Probabilistic expiry handled by game

        timing = schema.get("expiry_check_timing")
        if timing and timing in events:
            if self._expiry_counters.get(slot_id, -1) > 0:
                self._expiry_counters[slot_id] -= 1

    def get_value(self, slot_id: int) -> float:
        """Get current scaled value for a joker slot."""
        return self._joker_values.get(slot_id, 0.0)

    def get_expiry_remaining(self, slot_id: int) -> int:
        """Get remaining expiry count. -1 if no expiry."""
        return self._expiry_counters.get(slot_id, -1)


# ============================================================
# Event Detector
# ============================================================

class EventDetector:
    """Detects game events by diffing consecutive gamestates.

    Combined with the known action, this gives us full event
    information for the scaling tracker.
    """

    def __init__(self):
        self._prev_state: Optional[dict] = None

    def reset(self):
        self._prev_state = None

    def detect(self, new_state: dict, action: Optional[str] = None,
               action_params: Optional[dict] = None) -> tuple[list[str], dict[str, int], dict]:
        """Compare new state to previous, return detected events.

        Returns:
            events: list of trigger vocabulary event names
            event_counts: per-event item counts
            action_context: additional context for scaling tracker
        """
        events = []
        event_counts: dict[str, int] = {}
        context: dict[str, Any] = {}
        prev = self._prev_state

        if prev is None:
            self._prev_state = new_state
            return events, event_counts, context

        prev_round = prev.get("round", {})
        new_round = new_state.get("round", {})

        # Hand played
        hands_before = prev_round.get("hands_played", 0)
        hands_after = new_round.get("hands_played", 0)
        if hands_after > hands_before:
            events.append("any_hand_played")

            # Determine hand-specific events
            if hands_after == 1 and hands_before == 0:
                events.append("first_hand_of_round")

            new_hands_left = new_round.get("hands_left", 0)
            if new_hands_left == 0:
                events.append("final_hand_of_round")

        # Discard
        discards_before = prev_round.get("discards_used", 0)
        discards_after = new_round.get("discards_used", 0)
        if discards_after > discards_before:
            events.append("on_discard")
            discard_count = discards_after - discards_before
            # Count actual cards discarded from action params
            if action == "discard" and action_params and "cards" in action_params:
                discard_count = len(action_params["cards"])
            event_counts["on_discard"] = discard_count

        # Round change
        if new_state.get("round_num", 0) > prev.get("round_num", 0):
            events.append("on_round_start")
            events.append("end_of_round")

        # Ante change
        if new_state.get("ante_num", 0) > prev.get("ante_num", 0):
            events.append("on_ante_up")
            events.append("on_boss_blind_defeated")

        # Blind selected / skipped
        if action == "select":
            events.append("on_blind_selected")
        elif action == "skip":
            events.append("on_blind_skip")
            events.append("on_booster_pack_skipped")

        # Shop actions
        if action == "reroll":
            events.append("on_shop_reroll")
        if action == "sell":
            events.append("on_card_sold")
        if action == "buy":
            events.append("on_shop_enter")  # implicit

        # Consumable use detection
        prev_consumables = {c["id"] for c in prev.get("consumables", {}).get("cards", [])}
        new_consumables = {c["id"] for c in new_state.get("consumables", {}).get("cards", [])}
        used = prev_consumables - new_consumables
        if used and action == "use":
            # Determine consumable type from previous state
            for c in prev.get("consumables", {}).get("cards", []):
                if c["id"] in used:
                    card_set = c.get("set", "").upper()
                    if "TAROT" in card_set:
                        events.append("per_tarot_used")
                    elif "PLANET" in card_set:
                        events.append("per_planet_used")
                    elif "SPECTRAL" in card_set:
                        events.append("per_spectral_used")

        # Booster pack opened
        if new_state.get("state") == "SMODS_BOOSTER_OPENED" and prev.get("state") != "SMODS_BOOSTER_OPENED":
            events.append("on_booster_pack_opened")

        # Card added to deck — compare deck sizes
        prev_deck_count = prev.get("cards", {}).get("count", 0)
        new_deck_count = new_state.get("cards", {}).get("count", 0)
        if new_deck_count > prev_deck_count:
            events.append("card_added_to_deck")
            event_counts["card_added_to_deck"] = new_deck_count - prev_deck_count

        # Card destroyed — deck size decreased outside of normal play
        if new_deck_count < prev_deck_count and action not in ("play",):
            events.append("card_destroyed")
            event_counts["card_destroyed"] = prev_deck_count - new_deck_count

        # Joker destroyed — joker count decreased
        prev_joker_ids = {j["id"] for j in prev.get("jokers", {}).get("cards", [])}
        new_joker_ids = {j["id"] for j in new_state.get("jokers", {}).get("cards", [])}
        destroyed_jokers = prev_joker_ids - new_joker_ids
        if destroyed_jokers and action != "sell":
            events.append("on_joker_destroyed")

        # Build action context from what we know
        if action == "play" and action_params:
            indices = action_params.get("cards", [])
            hand_cards = prev.get("hand", {}).get("cards", [])
            played_cards = [hand_cards[i] for i in indices if i < len(hand_cards)]
            context["cards_played"] = played_cards

            face_scored = any(
                _as_dict(c.get("value", {})).get("rank", "") in FACE_RANKS
                for c in played_cards
            )
            context["face_cards_scored"] = face_scored

        if action == "discard" and action_params:
            indices = action_params.get("cards", [])
            hand_cards = prev.get("hand", {}).get("cards", [])
            discarded_cards = [hand_cards[i] for i in indices if i < len(hand_cards)]
            context["cards_discarded"] = discarded_cards

            # Count discard trigger ranks
            for c in discarded_cards:
                rank = _as_dict(c.get("value", {})).get("rank", "")
                if rank == "J":
                    event_counts["rank_specific_discard"] = event_counts.get("rank_specific_discard", 0) + 1
                    events.append("rank_specific_discard")

        # Determine hand type from poker hand levels
        # (check which hand type's played_this_round increased)
        if "any_hand_played" in events:
            prev_hands = prev.get("hands", {})
            new_hands = new_state.get("hands", {})
            for hand_name in HAND_TYPE_ORDER:
                prev_played = prev_hands.get(hand_name, {}).get("played_this_round", 0)
                new_played = new_hands.get(hand_name, {}).get("played_this_round", 0)
                if new_played > prev_played:
                    context["hand_type_played"] = hand_name
                    events.append("specific_hand_type")
                    break

            # Find most played hand type
            most_played = ""
            most_count = 0
            for hand_name in HAND_TYPE_ORDER:
                total = new_hands.get(hand_name, {}).get("played", 0)
                if total > most_count:
                    most_count = total
                    most_played = hand_name
            context["most_played_hand_type"] = most_played

        self._prev_state = new_state
        return events, event_counts, context


# ============================================================
# Joker Fingerprint Encoder
# ============================================================

def encode_joker_fingerprint(joker_key: str, edition: Optional[str],
                             modifiers: dict, scaled_value: float,
                             expiry_remaining: int,
                             sell_value: float = 0.0,
                             deck_suits: Optional[dict[str, float]] = None) -> np.ndarray:
    """Encode a joker into a fixed-size fingerprint vector.

    Args:
        deck_suits: optional dict mapping suit name → fraction of deck (0-1).
                    Used to compute synergy signal. If None, synergy = 0.
    Returns array of JOKER_FINGERPRINT_SIZE floats.
    """
    vec = np.zeros(JOKER_FINGERPRINT_SIZE, dtype=np.float32)
    name = _api_key_to_name(joker_key)
    if not name or name not in JOKERS:
        return vec

    schema = JOKERS[name]

    idx = 0

    # tier_weight (1)
    vec[idx] = _clamp_norm(schema.get("tier_weight", 0.0), 10.0); idx += 1

    # Effect flags (9)
    for flag in ["chip", "mult", "xmult", "economy", "chip_scaling",
                 "mult_scaling", "xmult_scaling", "copy", "in_hand_effect"]:
        vec[idx] = 1.0 if schema.get(flag) else 0.0; idx += 1

    # Static values (4) — log normalized
    vec[idx] = _log_norm(schema.get("chip_value") or 0.0, 3.0); idx += 1
    vec[idx] = _log_norm(schema.get("mult_value") or 0.0, 2.0); idx += 1
    vec[idx] = _log_norm(schema.get("xmult_value") or 0.0, 1.0); idx += 1
    vec[idx] = _log_norm(schema.get("money_per_round") or 0.0, 1.5); idx += 1

    # Scaling fields (2)
    vec[idx] = _log_norm(schema.get("scaling_increment") or 0.0, 2.0); idx += 1
    vec[idx] = _log_norm(schema.get("scaling_start_value") or 0.0, 3.0); idx += 1

    # Key flags (6)
    for flag in ["retrigger_effect", "rule_modification", "game_parameter_effect",
                 "consumable_creation", "survival_effect", "boss_blind_effect"]:
        vec[idx] = 1.0 if schema.get(flag) else 0.0; idx += 1

    # Effect probability (1)
    vec[idx] = schema.get("effect_probability") or 0.0; idx += 1

    # Has expiry (1)
    vec[idx] = 1.0 if schema.get("expiry") else 0.0; idx += 1

    # Edition one-hot (4)
    edition_idx = EDITION_MAP.get(edition, -1) if edition else -1
    for i in range(4):
        vec[idx] = 1.0 if i == edition_idx else 0.0; idx += 1

    # Runtime modifiers (3)
    vec[idx] = 1.0 if modifiers.get("debuff") else 0.0; idx += 1
    vec[idx] = 1.0 if modifiers.get("eternal") else 0.0; idx += 1
    vec[idx] = 1.0 if modifiers.get("perishable") else 0.0; idx += 1

    # Current scaled value (1) — log normalized
    vec[idx] = _log_norm(abs(scaled_value), 3.0); idx += 1

    # Sell value (1) — normalized to $10 max
    vec[idx] = _clamp_norm(sell_value, 10.0); idx += 1

    # ── Trigger type encoding (15 flags) ──
    triggers = schema.get("triggers") or []
    trigger_suits = schema.get("trigger_suits") or []
    trigger_hand_type = schema.get("trigger_hand_type") or ""

    # Suit trigger flags (4): Hearts, Diamonds, Clubs, Spades
    for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]:
        vec[idx] = 1.0 if suit in trigger_suits else 0.0; idx += 1

    # Face card trigger (1)
    vec[idx] = 1.0 if "face_card" in triggers else 0.0; idx += 1

    # Specific rank trigger (1)
    vec[idx] = 1.0 if "specific_rank" in triggers else 0.0; idx += 1

    # Hand type trigger flags (7): the main poker hands
    for ht in ["High Card", "Pair", "Two Pair", "Three of a Kind",
               "Straight", "Flush", "Full House"]:
        vec[idx] = 1.0 if trigger_hand_type == ht else 0.0; idx += 1

    # Any hand played (1)
    vec[idx] = 1.0 if "any_hand_played" in triggers else 0.0; idx += 1

    # Scoring card trigger (1) — fires on every scoring card
    vec[idx] = 1.0 if "scoring_card" in triggers else 0.0; idx += 1

    # ── Per-card vs once-per-hand (1) ──
    vec[idx] = 1.0 if schema.get("per_card_instance") else 0.0; idx += 1

    # ── Scoring timing (1) ── during_card=1.0, after_cards=0.0
    vec[idx] = 1.0 if schema.get("scoring_timing") == "during_card" else 0.0; idx += 1

    # ── Deck synergy signal (1) ──
    # How well the current deck matches this joker's trigger condition.
    synergy = 0.0
    if deck_suits:
        if trigger_suits:
            # Suit-based joker: fraction of deck matching trigger suits
            synergy = sum(deck_suits.get(s, 0.0) for s in trigger_suits)
        elif "face_card" in triggers:
            # Face card joker: ~23% of a standard deck is face cards (12/52)
            # Use face_fraction if provided, else rough estimate
            synergy = deck_suits.get("_face_fraction", 0.23)
        elif "scoring_card" in triggers or "any_hand_played" in triggers:
            # Always-fires jokers: perfect synergy
            synergy = 1.0
    vec[idx] = synergy; idx += 1

    assert idx == JOKER_FINGERPRINT_SIZE, f"Joker fingerprint size mismatch: {idx} != {JOKER_FINGERPRINT_SIZE}"
    return vec


# ============================================================
# Card Encoder
# ============================================================

def encode_card(card: dict) -> np.ndarray:
    """Encode a playing card into a fixed-size vector."""
    vec = np.zeros(HAND_CARD_SIZE, dtype=np.float32)
    value = _as_dict(card.get("value", {}))
    modifier = _as_dict(card.get("modifier", {}))
    state = _as_dict(card.get("state", {}))

    idx = 0

    # Rank normalized (1)
    rank = value.get("rank", "")
    vec[idx] = RANK_NORM.get(rank, 0.0); idx += 1

    # Suit normalized (1)
    suit = value.get("suit", "")
    vec[idx] = SUIT_NORM.get(suit, 0.0); idx += 1

    # Enhancement normalized (1)
    enhancement = modifier.get("enhancement", "")
    vec[idx] = (ENHANCEMENT_MAP.get(enhancement, -1) + 1) / 8.0; idx += 1

    # Seal normalized (1)
    seal = modifier.get("seal", "")
    vec[idx] = (SEAL_MAP.get(seal, -1) + 1) / 4.0; idx += 1

    # Edition normalized (1)
    edition = modifier.get("edition", "")
    vec[idx] = (EDITION_MAP.get(edition, -1) + 1) / 4.0; idx += 1

    # Debuffed (1)
    vec[idx] = 1.0 if state.get("debuff") else 0.0; idx += 1

    # Base chip value normalized (1)
    base_chips = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                  "8": 8, "9": 9, "T": 10, "J": 10, "Q": 10, "K": 10, "A": 11}
    vec[idx] = base_chips.get(rank, 0) / 11.0; idx += 1

    # Is face card (1)
    vec[idx] = 1.0 if rank in FACE_RANKS else 0.0; idx += 1

    assert idx == HAND_CARD_SIZE
    return vec


# ============================================================
# State Vector Builder
# ============================================================

class GameStateManager:
    """Manages the full pipeline from API to state vector.

    Handles:
    - Async HTTP calls to BalatroBot
    - Event detection between snapshots
    - Scaling value tracking
    - State vector assembly
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._scaling_tracker = ScalingTracker()
        self._event_detector = EventDetector()
        self._current_raw: Optional[dict] = None
        self._last_action: Optional[str] = None
        self._last_action_params: Optional[dict] = None
        self._joker_eval_cache: dict = {}

    async def connect(self):
        """Open HTTP session."""
        self._session = aiohttp.ClientSession()

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def inject_scaling_values(self, joker_cards: list[dict]) -> None:
        """Inject current scaling tracker values into joker card dicts.

        Adds a '_scaled_value' key to each joker so compute_joker_scoring
        can use the accumulated value for scaling jokers like Square Joker,
        Ride the Bus, etc.
        """
        for card in joker_cards:
            card_id = card.get("id")
            if card_id is not None:
                card["_scaled_value"] = self._scaling_tracker.get_value(card_id)

    def reset(self):
        """Reset all state for a new run."""
        self._scaling_tracker.reset()
        self._event_detector.reset()
        self._current_raw = None
        self._last_action = None
        self._last_action_params = None

    async def _rpc_call(self, method: str, params: Optional[dict] = None) -> dict:
        """Make a JSON-RPC 2.0 call to BalatroBot."""
        import aiohttp

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1,
        }
        if params:
            payload["params"] = params

        timeout = aiohttp.ClientTimeout(total=30)
        async with self._session.post(API_URL, json=payload, timeout=timeout) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"BalatroBot error: {data['error']}")
            return data.get("result", {})

    async def fetch_gamestate(self) -> dict:
        """Fetch current game state from BalatroBot API."""
        return await self._rpc_call("gamestate")

    async def execute_action(self, method: str, params: Optional[dict] = None) -> dict:
        """Execute an action and return the result."""
        self._last_action = method
        self._last_action_params = params
        return await self._rpc_call(method, params)

    async def step(self) -> np.ndarray:
        """Fetch gamestate, detect events, update trackers, return state vector.

        Call this after every action to get the updated state vector.
        """
        raw = await self.fetch_gamestate()

        # Detect events
        events, event_counts, context = self._event_detector.detect(
            raw, self._last_action, self._last_action_params
        )

        # Update scaling tracker
        joker_cards = raw.get("jokers", {}).get("cards", [])
        self._scaling_tracker.on_jokers_changed(joker_cards)
        if events:
            self._scaling_tracker.update(events, event_counts, context)

        self._current_raw = raw
        return self._build_state_vector(raw)

    def _build_state_vector(self, raw: dict) -> np.ndarray:
        """Assemble the full state vector from raw gamestate."""
        vec = np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)
        offset = 0

        # Section 1: Game Meta (10)
        offset = self._encode_game_meta(vec, offset, raw)

        # Section 2: Poker Hand Levels (39)
        offset = self._encode_hand_levels(vec, offset, raw)

        # Section 3: Deck Composition (61)
        offset = self._encode_deck_comp(vec, offset, raw)

        # Section 4: Vouchers (32)
        offset = self._encode_vouchers(vec, offset, raw)

        # Section 5: Joker Slots (5 × 32 = 160)
        offset = self._encode_jokers(vec, offset, raw)

        # Section 6: Hand Cards (12 × 8 = 96)
        offset = self._encode_hand_cards(vec, offset, raw)

        # Section 7: Consumables (2 × 6 = 12)
        offset = self._encode_consumables(vec, offset, raw)

        # Section 8: Shop (110)
        offset = self._encode_shop(vec, offset, raw)

        # Section 9: Hand Evaluation Features (40)
        offset = self._encode_hand_eval(vec, offset, raw)

        assert offset == STATE_VECTOR_SIZE, f"Vector size mismatch: {offset} != {STATE_VECTOR_SIZE}"
        return vec

    def _encode_game_meta(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode game metadata into vector."""
        rnd = raw.get("round", {})
        blinds = _as_dict(raw.get("blinds", {}))

        # Find current blind score
        blind_score = 0.0
        is_boss = 0.0
        current_blind = _get_current_blind(blinds)
        if current_blind:
            blind_score = current_blind.get("score", 0)
            is_boss = 1.0 if current_blind.get("type") == "BOSS" else 0.0

        # Count blinds skipped this run
        blinds_skipped = sum(
            1 for b in blinds.values()
            if isinstance(b, dict) and b.get("status") == "SKIPPED"
        )

        money = raw.get("money", 0)

        vec[offset + 0] = _clamp_norm(raw.get("ante_num", 1), 8.0)
        vec[offset + 1] = _clamp_norm(raw.get("round_num", 1), 24.0)
        vec[offset + 2] = _log_norm(money, 3.0)
        vec[offset + 3] = _clamp_norm(rnd.get("hands_left", 0), 8.0)
        vec[offset + 4] = _clamp_norm(rnd.get("discards_left", 0), 6.0)
        vec[offset + 5] = _clamp_norm(rnd.get("reroll_cost", 5), 20.0)
        vec[offset + 6] = _log_norm(rnd.get("chips", 0), 15.0)
        vec[offset + 7] = _log_norm(blind_score, 15.0)
        vec[offset + 8] = is_boss
        vec[offset + 9] = _clamp_norm(blinds_skipped, 8.0)

        # ── Interest threshold features (5) ──
        # Distance to each $5 interest tier, normalized to [0, 1].
        # 0.0 = at or above threshold, 1.0 = $5 away.
        # Makes the $19→$20 vs $20→$19 decision explicit.
        for i, threshold in enumerate([5, 10, 15, 20, 25]):
            gap = max(threshold - money, 0)
            vec[offset + 10 + i] = _clamp_norm(gap, 5.0)

        # ── Projected end-of-round income (1) ──
        # Estimate: current money + interest + blind payout + joker passive income.
        # Interest: $1 per $5 held, capped at $5 (or $10/$25 with vouchers).
        interest_cap = 5  # default cap
        vouchers = raw.get("vouchers", {}).get("owned", [])
        if isinstance(vouchers, list):
            v_set = set(vouchers)
        else:
            v_set = set()
        if "v_money_tree" in v_set:
            interest_cap = 25
        elif "v_seed_money" in v_set:
            interest_cap = 10
        interest = min(int(money) // 5, interest_cap)

        # Blind payout: Small=$3, Big=$4, Boss=$5 (base values)
        blind_payout = 5.0 if is_boss else 4.0

        # Joker passive income
        joker_income = 0.0
        joker_cards = raw.get("jokers", {}).get("cards", [])
        for jc in joker_cards:
            jk = jc.get("key", "")
            jname = _api_key_to_name(jk)
            if jname and jname in JOKERS:
                mpr = JOKERS[jname].get("money_per_round")
                if mpr:
                    joker_income += mpr

        projected = money + interest + blind_payout + joker_income
        vec[offset + 15] = _log_norm(projected, 3.0)

        # ── Boss blind features (22) ──
        # Current blind: type (3) + debuff category (6) + debuffed suit (4) = 13
        _encode_blind_features(current_blind, vec, offset + 16)

        # Upcoming blind: type (3) + debuff category (6) = 9
        # (No debuffed suit for upcoming — agent can't plan suit avoidance
        #  until the blind is actually active and suit is revealed.)
        upcoming_blind = _get_upcoming_blind(blinds)
        if upcoming_blind:
            utype = upcoming_blind.get("type", "")
            if utype == "SMALL":
                vec[offset + 29] = 1.0
            elif utype == "BIG":
                vec[offset + 30] = 1.0
            elif utype == "BOSS":
                vec[offset + 31] = 1.0
            uname = upcoming_blind.get("name", "")
            uinfo = BOSS_BLIND_INFO.get(uname)
            if uinfo:
                vec[offset + 32 + uinfo[0]] = 1.0

        # ── Reroll value features (4) ──
        # Help the agent learn that rerolling is +EV when it has spare cash.
        reroll_cost = rnd.get("reroll_cost", 5)
        joker_slots_used = len(joker_cards)
        joker_slots_max = raw.get("jokers", {}).get("max_slots", 5)
        open_joker_slots = max(joker_slots_max - joker_slots_used, 0)

        # Safe-to-spend money: excess above current interest tier floor.
        # E.g., at $23 the floor is $20 (for $4 interest), so $3 is safe to spend.
        # Interest caps at $5 (from $25) without vouchers, $10/$25 with vouchers.
        current_interest_floor = min((int(money) // 5) * 5, interest_cap * 5)
        # Don't spend below $0
        safe_money = max(money - current_interest_floor, 0)

        # 1. Affordable rerolls at current interest tier (0-5 scale)
        affordable_rerolls = int(safe_money // reroll_cost) if reroll_cost > 0 else 0
        vec[offset + 38] = _clamp_norm(affordable_rerolls, 5.0)

        # 2. Reroll budget ratio — continuous signal of spending headroom
        #    1.0 = can comfortably afford multiple rerolls, 0.0 = can't afford any
        vec[offset + 39] = _clamp_norm(safe_money / max(reroll_cost, 1), 5.0)

        # 3. Open joker slots — rerolling is valuable when there's room to buy
        vec[offset + 40] = _clamp_norm(open_joker_slots, 5.0)

        # 4. Reroll efficiency — combines slot availability with budget
        #    High when: open slots AND can afford rerolls. Zero when: full slots or broke.
        reroll_efficiency = min(affordable_rerolls, open_joker_slots)
        vec[offset + 41] = _clamp_norm(reroll_efficiency, 5.0)

        return offset + GAME_META_SIZE

    def _encode_hand_levels(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode poker hand type levels, play frequency, and celestial ROI."""
        hands = raw.get("hands", {})

        # ── Existing: level, chips, mult per hand type (13 × 3 = 39) ──
        for i, hand_name in enumerate(HAND_TYPE_ORDER):
            hand_data = hands.get(hand_name, {})
            base = offset + i * 3
            vec[base + 0] = _clamp_norm(hand_data.get("level", 1), 20.0)
            vec[base + 1] = _log_norm(hand_data.get("chips", 0), 4.0)
            vec[base + 2] = _log_norm(hand_data.get("mult", 0), 3.0)

        freq_offset = offset + 39  # after the 39 existing floats

        # ── Play frequency counters (13) ──
        # Total times each hand type has been played this run, normalized.
        total_plays = 0
        play_counts = []
        for hand_name in HAND_TYPE_ORDER:
            played = hands.get(hand_name, {}).get("played", 0)
            play_counts.append(played)
            total_plays += played
        for i, count in enumerate(play_counts):
            # Normalize: fraction of total plays (0-1), or 0 if no plays yet
            vec[freq_offset + i] = count / max(total_plays, 1)

        roi_offset = freq_offset + 13  # after play frequencies

        # ── Celestial ROI deltas (13 chip deltas + 13 mult deltas = 26) ──
        # Fixed per-level gains from celestial cards in Balatro.
        best_roi = 0.0
        best_roi_idx = 0
        for i, hand_name in enumerate(HAND_TYPE_ORDER):
            chip_delta, mult_delta = _CELESTIAL_DELTAS.get(hand_name, (0, 0))
            vec[roi_offset + i] = _log_norm(chip_delta, 3.0)          # chip delta
            vec[roi_offset + 13 + i] = _log_norm(mult_delta, 2.0)     # mult delta

            # ── Compute weighted ROI for best-target signal ──
            # Weight = (chip_delta + mult_delta * 10) × play_frequency
            # mult is ~10x more valuable than chips in score impact
            freq = play_counts[i] / max(total_plays, 1)
            roi = (chip_delta + mult_delta * 10.0) * freq
            if roi > best_roi:
                best_roi = roi
                best_roi_idx = i

        target_offset = roi_offset + 26  # after ROI deltas

        # ── Highest ROI celestial target (1) ──
        # Encoded as normalized index (0-1) of the hand type that benefits most.
        vec[target_offset] = best_roi_idx / max(len(HAND_TYPE_ORDER) - 1, 1)

        return offset + HAND_LEVELS_SIZE

    def _encode_deck_comp(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode deck composition — suit×rank counts + enhancement counts."""
        deck_cards = raw.get("cards", {}).get("cards", [])

        # 52 rank×suit counts
        for card in deck_cards:
            value = card.get("value", {})
            rank = value.get("rank", "")
            suit = value.get("suit", "")
            if rank in RANK_MAP and suit in SUIT_MAP:
                idx = SUIT_MAP[suit] * 13 + RANK_MAP[rank]
                vec[offset + idx] += 1.0

        # Normalize counts (typically 0-4 per slot)
        for i in range(52):
            vec[offset + i] = _clamp_norm(vec[offset + i], 4.0)

        # Enhancement counts (9)
        enh_offset = offset + 52
        for card in deck_cards:
            mod = _as_dict(card.get("modifier", {}))
            enhancement = mod.get("enhancement", "")
            if enhancement in ENHANCEMENT_MAP:
                vec[enh_offset + ENHANCEMENT_MAP[enhancement]] += 1.0
            seal = mod.get("seal", "")
            if seal:
                vec[enh_offset + 8] += 1.0  # sealed count

        # Normalize enhancement counts
        total_cards = max(len(deck_cards), 1)
        for i in range(9):
            vec[enh_offset + i] = _clamp_norm(vec[enh_offset + i], total_cards)

        return offset + DECK_COMP_SIZE

    def _encode_vouchers(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode owned vouchers as binary flags."""
        used = raw.get("used_vouchers", [])
        for v_key in used:
            if v_key in VOUCHER_MAP:
                vec[offset + VOUCHER_MAP[v_key]] = 1.0
        return offset + VOUCHER_SIZE

    def _compute_build_scores(self, joker_cards: list[dict], raw: dict,
                              deck_suits: dict[str, float]) -> tuple[
                                  float, list[float], list[float], int]:
        """Compute synergy-aware build contribution and sell guard for each joker.

        Returns:
            current_score: total score with all jokers
            contributions: per-slot synergy-aware contribution (higher = more valuable)
            sell_guards: per-slot sell guard (0.0 = safe to sell, 1.0 = must keep)
            weakest_idx: index of weakest sellable joker (-1 if none)
        """
        from environment.hand_eval import estimate_score_for_hand_type

        n = min(len(joker_cards), JOKER_SLOTS)
        contributions = [0.0] * JOKER_SLOTS
        sell_guards = [0.0] * JOKER_SLOTS

        if n == 0:
            return 0.0, contributions, sell_guards, -1

        # ── Step 1: Leave-one-out scoring (base contribution) ──
        # This already captures most synergy: removing a joker that enables
        # others causes a bigger score drop than removing an isolated one.
        current_score = estimate_score_for_hand_type(joker_cards[:n], raw)
        for j_idx in range(n):
            jokers_without = [j for i, j in enumerate(joker_cards[:n]) if i != j_idx]
            score_without = estimate_score_for_hand_type(jokers_without, raw) if jokers_without else 0.0
            contributions[j_idx] = current_score - score_without

        # ── Step 2: Copy relationship bonus ──
        # Blueprint copies joker to its right, Brainstorm copies leftmost.
        # The copy SOURCE is more valuable than it appears — removing it
        # also kills the copy joker's contribution. Boost the source.
        for j_idx in range(n):
            jk = joker_cards[j_idx].get("key", "")
            jname = _api_key_to_name(jk)
            if not jname or jname not in JOKERS:
                continue
            schema = JOKERS[jname]
            if not schema.get("copy"):
                continue

            # This IS a copy joker — find what it copies
            target_dir = schema.get("copy_target", "")
            copy_src_idx = None
            if target_dir == "right" and j_idx + 1 < n:
                copy_src_idx = j_idx + 1
            elif target_dir == "left":
                for k in range(n):
                    if k != j_idx:
                        copy_src_idx = k
                        break

            if copy_src_idx is not None:
                # Boost the source: it's being duplicated, so it's worth more
                copy_bonus = max(contributions[j_idx], 0.0)
                contributions[copy_src_idx] += copy_bonus
                # Guard both the copy joker and its source
                sell_guards[j_idx] = max(sell_guards[j_idx], 0.8)
                sell_guards[copy_src_idx] = max(sell_guards[copy_src_idx], 0.9)

        # ── Step 3: Scaling potential bonus ──
        # Jokers with high scaling_increment are investments — low value now
        # but high value later. Penalize selling them.
        for j_idx in range(n):
            jk = joker_cards[j_idx].get("key", "")
            jname = _api_key_to_name(jk)
            if not jname or jname not in JOKERS:
                continue
            schema = JOKERS[jname]
            scaling_inc = schema.get("scaling_increment") or 0.0
            if scaling_inc > 0:
                scaling_type = schema.get("scaling_type", "")
                # xmult scaling is multiplicative — much more valuable long-term
                multiplier = 3.0 if "xmult" in scaling_type else 1.0
                # Bonus proportional to increment: keeps scaling jokers off the chopping block.
                # Rough scale: increment of 1.0 adds ~50 points of "keep value"
                scaling_bonus = scaling_inc * 50.0 * multiplier
                contributions[j_idx] += scaling_bonus
                # Strong guard for high-increment scalers
                guard_strength = min(scaling_inc * multiplier / 5.0, 1.0)
                sell_guards[j_idx] = max(sell_guards[j_idx], guard_strength)

        # ── Step 4: Deck synergy bonus ──
        # A joker whose trigger condition matches the current deck is worth
        # more than one mismatched with the deck. Scale contribution up.
        for j_idx in range(n):
            jk = joker_cards[j_idx].get("key", "")
            jname = _api_key_to_name(jk)
            if not jname or jname not in JOKERS:
                continue
            schema = JOKERS[jname]
            trigger_suits = schema.get("trigger_suits") or []
            triggers = schema.get("triggers") or []

            synergy_mult = 1.0
            if trigger_suits and deck_suits:
                suit_match = sum(deck_suits.get(s, 0.0) for s in trigger_suits)
                # Deck with 40% matching suit → 1.4x contribution
                synergy_mult = 1.0 + suit_match
            elif "face_card" in triggers and deck_suits:
                face_frac = deck_suits.get("_face_fraction", 0.23)
                synergy_mult = 1.0 + face_frac
            contributions[j_idx] *= synergy_mult

        # ── Step 5: Eternal jokers — unsellable, infinite guard ──
        for j_idx in range(n):
            mod = _as_dict(joker_cards[j_idx].get("modifier", {}))
            if mod.get("eternal", False):
                sell_guards[j_idx] = 1.0

        # ── Find weakest sellable joker ──
        weakest_idx = -1
        weakest_score = float("inf")
        for j_idx in range(n):
            if sell_guards[j_idx] >= 1.0:
                continue  # unsellable
            # Effective weakness: contribution discounted by sell guard
            # High guard → artificially inflates score so it won't be picked as weakest
            effective = contributions[j_idx] * (1.0 + sell_guards[j_idx] * 2.0)
            if effective < weakest_score:
                weakest_score = effective
                weakest_idx = j_idx

        return current_score, contributions, sell_guards, weakest_idx

    def _encode_jokers(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode joker slots with fingerprints + synergy-aware contribution + sell guard."""
        joker_cards = raw.get("jokers", {}).get("cards", [])

        # Compute deck suit fractions for synergy signals
        deck_cards = raw.get("cards", {}).get("cards", [])
        deck_suits = self._compute_deck_suit_fractions(deck_cards)

        n_jokers = min(len(joker_cards), JOKER_SLOTS)

        # ── Synergy-aware build scoring ──
        current_score, contributions, sell_guards, weakest_idx = \
            self._compute_build_scores(joker_cards[:n_jokers], raw, deck_suits)

        # ── Compute per-joker sell score deltas (forward-looking sell signal) ──
        # Shows what happens to scoring power if this joker is sold right now.
        # Negative = selling hurts, positive = selling is neutral or beneficial.
        sell_deltas = [0.0] * JOKER_SLOTS
        if n_jokers > 0 and current_score > 0:
            for j_idx in range(n_jokers):
                # The leave-one-out score is already computed in contributions
                score_without = current_score - contributions[j_idx]
                # Normalize as fraction of current score: -1.0 = lose 100% of scoring
                sell_deltas[j_idx] = max(min(
                    (score_without - current_score) / current_score, 1.0
                ), -2.0)

        # Store for shop encoding and reward function
        self._joker_eval_cache = {
            "current_score": current_score,
            "weakest_idx": weakest_idx,
            "n_jokers": n_jokers,
            "joker_cards": joker_cards[:n_jokers],
            "contributions": contributions[:n_jokers],
        }

        # Normalize contributions for the state vector
        max_contribution = max(abs(c) for c in contributions[:n_jokers]) if n_jokers > 0 else 1.0
        max_contribution = max(max_contribution, 1.0)

        for slot_idx in range(JOKER_SLOTS):
            slot_offset = offset + slot_idx * JOKER_SLOT_SIZE
            if slot_idx < n_jokers:
                card = joker_cards[slot_idx]
                key = card.get("key", "")
                modifier = _as_dict(card.get("modifier", {}))
                edition = modifier.get("edition")
                state_flags = _as_dict(card.get("state", {}))

                mod_dict = {
                    "debuff": state_flags.get("debuff", False),
                    "eternal": modifier.get("eternal", False),
                    "perishable": modifier.get("perishable", False),
                }

                scaled_val = self._scaling_tracker.get_value(card["id"])
                expiry_rem = self._scaling_tracker.get_expiry_remaining(card["id"])
                sell_val = _as_dict(card.get("cost", {})).get("sell", 0)

                fingerprint = encode_joker_fingerprint(
                    key, edition, mod_dict, scaled_val, expiry_rem,
                    sell_value=float(sell_val),
                    deck_suits=deck_suits,
                )
                # Write full fingerprint (51 floats)
                vec[slot_offset:slot_offset + JOKER_FINGERPRINT_SIZE] = fingerprint
                # Synergy-aware contribution (1 float) — normalized
                vec[slot_offset + JOKER_FINGERPRINT_SIZE] = contributions[slot_idx] / max_contribution
                # Sell guard (1 float) — 0.0=safe to sell, 1.0=must keep
                vec[slot_offset + JOKER_FINGERPRINT_SIZE + 1] = sell_guards[slot_idx]
                # Sell score delta (1 float) — projected score change if sold
                # Negative = selling hurts scoring. Clamped to [-2.0, 1.0]
                vec[slot_offset + JOKER_FINGERPRINT_SIZE + 2] = sell_deltas[slot_idx]
            # else: stays zero-padded

        return offset + JOKER_SLOTS * JOKER_SLOT_SIZE

    @staticmethod
    def _compute_deck_suit_fractions(deck_cards: list[dict]) -> dict[str, float]:
        """Compute suit and face-card fractions of the remaining deck."""
        suit_expand = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
        counts: dict[str, int] = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
        face_count = 0
        total = max(len(deck_cards), 1)

        for card in deck_cards:
            value = _as_dict(card.get("value", {}))
            suit_raw = value.get("suit", "")
            suit = suit_expand.get(suit_raw, suit_raw)
            if suit in counts:
                counts[suit] += 1
            rank = value.get("rank", "")
            if rank in FACE_RANKS:
                face_count += 1

        result: dict[str, float] = {s: c / total for s, c in counts.items()}
        result["_face_fraction"] = face_count / total
        return result

    def _encode_hand_cards(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode cards currently in hand."""
        hand_cards = raw.get("hand", {}).get("cards", [])

        for slot_idx in range(HAND_CARD_SLOTS):
            slot_offset = offset + slot_idx * HAND_CARD_SIZE
            if slot_idx < len(hand_cards):
                encoded = encode_card(hand_cards[slot_idx])
                vec[slot_offset:slot_offset + HAND_CARD_SIZE] = encoded

        return offset + HAND_CARD_SLOTS * HAND_CARD_SIZE

    def _encode_consumables(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode consumable slots."""
        from environment.hand_eval import compute_tarot_value

        consumable_cards = raw.get("consumables", {}).get("cards", [])
        joker_cards = raw.get("jokers", {}).get("cards", [])
        deck_cards = raw.get("cards", {}).get("cards", [])

        for slot_idx in range(CONSUMABLE_SLOTS):
            slot_offset = offset + slot_idx * CONSUMABLE_SIZE
            if slot_idx < len(consumable_cards):
                card = consumable_cards[slot_idx]
                card_set = card.get("set", "").upper()
                # Type (1)
                if "TAROT" in card_set:
                    vec[slot_offset] = 0.33
                elif "PLANET" in card_set:
                    vec[slot_offset] = 0.67
                elif "SPECTRAL" in card_set:
                    vec[slot_offset] = 1.0
                # Key as normalized hash (1)
                key = card.get("key", "")
                vec[slot_offset + 1] = (hash(key) % 1000) / 1000.0
                # Is negative (1)
                edition = _as_dict(card.get("modifier", {})).get("edition", "")
                vec[slot_offset + 2] = 1.0 if edition == "NEGATIVE" else 0.0
                # Cost (1)
                vec[slot_offset + 3] = _clamp_norm(_as_dict(card.get("cost", {})).get("buy", 0), 10.0)
                # Tarot value signal (1) — expected value given joker lineup + deck
                # 0.0 = worthless, 1.0 = extremely valuable
                if "TAROT" in card_set:
                    vec[slot_offset + 4] = compute_tarot_value(
                        key, joker_cards, deck_cards
                    )
                elif "PLANET" in card_set:
                    vec[slot_offset + 4] = 0.8  # Planets are always valuable
                else:
                    vec[slot_offset + 4] = 0.5  # Spectral — moderate default
                # Needs targeting flag (1) — does this consumable need hand card targets?
                needs_target = 0.0
                if key in ("c_strength", "c_death", "c_hanged_man",
                           "c_magician", "c_empress", "c_hierophant",
                           "c_lovers", "c_chariot", "c_justice",
                           "c_devil", "c_tower"):
                    needs_target = 1.0
                vec[slot_offset + 5] = needs_target

        return offset + CONSUMABLE_SLOTS * CONSUMABLE_SIZE

    def _encode_shop(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode shop contents with upgrade signals."""
        from environment.hand_eval import estimate_score_for_hand_type

        shop = raw.get("shop", {})
        shop_jokers = shop.get("cards", []) if isinstance(shop, dict) else []
        shop_vouchers = raw.get("vouchers", {}).get("cards", []) if raw.get("state") == "SHOP" else []
        shop_packs = raw.get("packs", {}).get("cards", []) if raw.get("state") == "SHOP" else []

        # Retrieve joker eval cache from _encode_jokers
        cache = getattr(self, "_joker_eval_cache", {})
        current_score = cache.get("current_score", 0.0)
        weakest_idx = cache.get("weakest_idx", -1)
        n_jokers = cache.get("n_jokers", 0)
        owned_jokers = cache.get("joker_cards", [])
        slots_full = n_jokers >= JOKER_SLOTS

        # Build leave-weakest-out roster once (for upgrade delta calc when slots full)
        jokers_without_weakest: list[dict] = []
        if slots_full and weakest_idx >= 0:
            jokers_without_weakest = [j for i, j in enumerate(owned_jokers) if i != weakest_idx]

        # Compute deck fractions once for all shop jokers
        deck_cards = raw.get("cards", {}).get("cards", [])
        shop_deck_suits = self._compute_deck_suit_fractions(deck_cards)

        # Shop jokers (3 slots)
        for slot_idx in range(SHOP_JOKER_SLOTS):
            slot_offset = offset + slot_idx * SHOP_JOKER_SIZE
            if slot_idx < len(shop_jokers):
                card = shop_jokers[slot_idx]
                if card.get("set", "").upper() == "JOKER" or card.get("key", "").startswith("j_"):
                    key = card.get("key", "")
                    modifier = _as_dict(card.get("modifier", {}))
                    edition = modifier.get("edition")
                    mod_dict = {
                        "debuff": False,
                        "eternal": modifier.get("eternal", False),
                        "perishable": modifier.get("perishable", False),
                    }
                    fingerprint = encode_joker_fingerprint(
                        key, edition, mod_dict, 0.0, -1,
                        deck_suits=shop_deck_suits,
                    )
                    # Full fingerprint (51 floats)
                    vec[slot_offset:slot_offset + JOKER_FINGERPRINT_SIZE] = fingerprint
                    # Cost (1)
                    buy_cost = _as_dict(card.get("cost", {})).get("buy", 0)
                    vec[slot_offset + JOKER_FINGERPRINT_SIZE] = _clamp_norm(buy_cost, 20.0)
                    # Affordable (1)
                    money = raw.get("money", 0)
                    vec[slot_offset + JOKER_FINGERPRINT_SIZE + 1] = (
                        1.0 if money >= buy_cost else 0.0
                    )
                    # Upgrade delta (1) — synergy-aware buy signal
                    # When slots full: score delta from swapping weakest for this joker
                    # When slots open: absolute contribution of adding this joker
                    # Positive = buying improves scoring, negative = waste of money
                    upgrade_delta = 0.0
                    if slots_full and jokers_without_weakest:
                        # Swap comparison: remove weakest, add this shop joker
                        swap_roster = jokers_without_weakest + [card]
                        swap_score = estimate_score_for_hand_type(swap_roster, raw)
                        if current_score > 0:
                            upgrade_delta = (swap_score - current_score) / current_score
                        else:
                            upgrade_delta = 1.0 if swap_score > 0 else 0.0
                    elif not slots_full:
                        # Open slot: how much does adding this joker improve scoring?
                        add_roster = list(owned_jokers) + [card]
                        add_score = estimate_score_for_hand_type(add_roster, raw)
                        if current_score > 0:
                            upgrade_delta = (add_score - current_score) / current_score
                        else:
                            upgrade_delta = 1.0 if add_score > 0 else 0.0
                    vec[slot_offset + JOKER_FINGERPRINT_SIZE + 2] = max(min(upgrade_delta, 2.0), -2.0)

        shop_j_end = offset + SHOP_JOKER_SLOTS * SHOP_JOKER_SIZE

        # Shop vouchers (2 × 5)
        for slot_idx in range(SHOP_VOUCHER_SLOTS):
            slot_offset = shop_j_end + slot_idx * SHOP_VOUCHER_SIZE
            if slot_idx < len(shop_vouchers):
                card = shop_vouchers[slot_idx]
                key = card.get("key", "")
                buy_cost = _as_dict(card.get("cost", {})).get("buy", 0)
                vec[slot_offset] = (VOUCHER_MAP.get(key, 0) + 1) / 32.0
                vec[slot_offset + 1] = _clamp_norm(buy_cost, 20.0)
                vec[slot_offset + 2] = 1.0 if raw.get("money", 0) >= buy_cost else 0.0
                vec[slot_offset + 3] = 0.0  # placeholder
                vec[slot_offset + 4] = 0.0  # placeholder

        shop_v_end = shop_j_end + SHOP_VOUCHER_SLOTS * SHOP_VOUCHER_SIZE

        # Shop packs (2 × 5)
        from environment.hand_eval import compute_tarot_value
        joker_cards_for_packs = raw.get("jokers", {}).get("cards", [])
        deck_cards_for_packs = raw.get("cards", {}).get("cards", [])

        for slot_idx in range(SHOP_PACK_SLOTS):
            slot_offset = shop_v_end + slot_idx * SHOP_PACK_SIZE
            if slot_idx < len(shop_packs):
                card = shop_packs[slot_idx]
                key = card.get("key", "")
                # Pack type encoding
                if "arcana" in key:
                    vec[slot_offset] = 0.25
                elif "celestial" in key:
                    vec[slot_offset] = 0.5
                elif "spectral" in key:
                    vec[slot_offset] = 0.75
                elif "buffoon" in key:
                    vec[slot_offset] = 1.0
                pack_cost = _as_dict(card.get("cost", {})).get("buy", 0)
                vec[slot_offset + 1] = _clamp_norm(pack_cost, 12.0)
                vec[slot_offset + 2] = 1.0 if raw.get("money", 0) >= pack_cost else 0.0
                # Is mega pack
                vec[slot_offset + 3] = 1.0 if "mega" in key else 0.0
                # Pack expected value (1) — how valuable is this pack type
                # given current joker lineup and deck composition?
                pack_value = 0.0
                if "arcana" in key:
                    # Tarot pack — value depends on suit synergy with jokers
                    from environment.hand_eval import _get_joker_suit_synergies
                    wanted = _get_joker_suit_synergies(joker_cards_for_packs)
                    if wanted:
                        pack_value = 0.7  # Have suit jokers → tarots are valuable
                    else:
                        pack_value = 0.3  # No suit synergy but still useful
                elif "celestial" in key:
                    pack_value = 0.6  # Planets always decent
                elif "spectral" in key:
                    pack_value = 0.5  # Spectrals are situational
                elif "buffoon" in key:
                    joker_count = len(raw.get("jokers", {}).get("cards", []))
                    joker_limit = raw.get("jokers", {}).get("limit", 5)
                    if joker_count < joker_limit:
                        pack_value = 0.8  # Joker packs with open slots
                    else:
                        pack_value = 0.4  # Full but could swap
                vec[slot_offset + 4] = pack_value

        return shop_v_end + SHOP_PACK_SLOTS * SHOP_PACK_SIZE

    def _encode_hand_eval(self, vec: np.ndarray, offset: int, raw: dict) -> int:
        """Encode hand evaluation features (40 floats).

        Only computed during SELECTING_HAND state; zero-filled otherwise.
        """
        game_state = raw.get("state", "")
        if game_state == "SELECTING_HAND":
            try:
                features = assess_strategy(raw)
                vec[offset:offset + HAND_EVAL_FEATURES] = features
            except Exception:
                pass  # leave as zeros on error
        return offset + HAND_EVAL_FEATURES

    def get_current_state_name(self) -> Optional[str]:
        """Get current BalatroBot game state name."""
        if self._current_raw:
            return self._current_raw.get("state")
        return None

    def get_raw_state(self) -> Optional[dict]:
        """Get the raw gamestate dict from last fetch."""
        return self._current_raw
