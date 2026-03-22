"""
Balatron — Action Space

Maps neural network outputs to BalatroBot API calls.

The action space is decomposed into:
1. Action type (categorical) — what kind of action to take
2. Card selection (binary mask) — which cards to play/discard
3. Target selection (categorical) — which shop item / consumable / pack card

Each game state has different valid action types. Invalid actions are
masked before the network's softmax so the agent can only pick legal moves.

See NOTES.md for full action space layout.
"""

import math
from typing import Any, Optional

import numpy as np

from environment.hand_eval import (
    find_best_hands, find_best_discard, _api_key_to_name,
    estimate_score_for_hand_type, _get_dominant_suit, BASE_HAND_SCORES,
    plan_consumable_use,
)

# Planet card key → hand type it levels up
PLANET_TO_HAND_TYPE = {
    "c_pluto": "High Card", "c_mercury": "Pair", "c_uranus": "Two Pair",
    "c_venus": "Three of a Kind", "c_saturn": "Straight", "c_jupiter": "Flush",
    "c_earth": "Full House", "c_mars": "Four of a Kind",
    "c_neptune": "Straight Flush", "c_planet_x": "Five of a Kind",
    "c_ceres": "Flush House", "c_eris": "Flush Five",
}


def _is_joker_card(card: dict) -> bool:
    """Check if a shop card is a joker (not a planet/tarot in a shop slot).

    The API may return cards with set='Joker', or empty set with a j_ key.
    Accept BOTH patterns so we never miss a buyable joker.
    """
    card_set = card.get("set", "").upper()
    if card_set == "JOKER":
        return True
    if card_set and card_set != "JOKER":
        return False  # explicitly non-joker set
    # Set is empty/missing — check key prefix
    key = card.get("joker_key", "") or card.get("key", "")
    if key.startswith("j_"):
        return True
    # Known non-joker prefixes
    if key.startswith(("c_", "p_", "v_")):
        return False
    # Truly unknown — assume joker (don't miss buys)
    return True


def _is_non_joker_card(card: dict) -> bool:
    """Check if a shop card is definitively NOT a joker (planet/tarot/etc).

    Only returns True when we're confident it's non-joker.
    """
    return not _is_joker_card(card)


def _safe_modifier(card: dict) -> dict:
    """Get card modifier as a dict (API may return [] for empty)."""
    mod = card.get("modifier", {})
    return mod if isinstance(mod, dict) else {}


def _joker_is_scoring(joker: dict) -> bool:
    """Check if a joker contributes to hand scoring (chips/mult/xmult).

    Returns True if the joker has any score_effect in our schema.
    Economy-only jokers (money_per_round, etc.) return False.
    Unknown jokers default to True (assume valuable).
    """
    from data.jokers import JOKERS
    joker_key = joker.get("joker_key", "") or joker.get("key", "")
    name = _api_key_to_name(joker_key)
    if not name or name not in JOKERS:
        return True  # unknown joker — assume it's important, don't sell

    schema = JOKERS[name]
    score_effect = schema.get("score_effect")
    if score_effect:
        return True  # has chips, mult, xmult, etc.

    # Check for xmult/mult/chip flags directly
    if schema.get("xmult") or schema.get("mult") or schema.get("chip"):
        return True
    if schema.get("xmult_scaling") or schema.get("mult_scaling") or schema.get("chip_scaling"):
        return True

    # Copy jokers (Blueprint, Brainstorm) effectively double another joker's effect
    if schema.get("copy"):
        return True

    # Retrigger jokers (Hanging Chad, Sock and Buskin, Hack, etc.)
    # multiply other joker effects — always scoring
    if schema.get("retrigger_effect"):
        return True

    return False


# Jokers that should ALWAYS be bought if affordable — game-defining power
MUST_BUY_JOKERS = {
    "Blueprint", "Brainstorm",       # copy another joker = doubles scoring
}

# High-value jokers — not quite "must buy" but FAR too strong to reroll past.
# These get a minimum delta floor and the bot MUST buy before rerolling.
HIGH_VALUE_JOKERS = {
    # Retrigger jokers — multiply everything else
    "Hanging Chad",      # retrigger first scored card 2x (insane with xmult jokers)
    "Sock and Buskin",   # retrigger face cards (face card builds dominate)
    "Hack",              # retrigger 2-5 rank cards
    "Seltzer",           # retrigger all scored cards for 10 rounds
    # Face card synergy — strongest archetype in Balatro
    "Photograph",        # x2 mult on first played face card
    "Smiley Face",       # +5 mult per face card scored
    "Scary Face",        # +30 chips per face card scored
    # Strong unconditional xmult
    "Triboulet",         # x2 on Kings and Queens
    "The Duo",           # x2 if hand contains Pair
    "The Trio",          # x2 if hand contains Three of a Kind
    "The Family",        # x2 if hand contains Four of a Kind
    "The Order",         # x2 if hand contains Straight
    "The Tribe",         # x2 if hand contains Flush
    # Economy + scaling powerhouses
    "Vampire",           # xmult that grows by eating enhanced cards
    "Campfire",          # xmult that grows when selling jokers
    "Hologram",          # xmult that grows with added cards
    # Other high-impact
    "Perkeo",            # duplicates a consumable on shop entry (insane value)
    "Fibonacci",         # +8 mult on Ace/2/3/5/8 (common ranks)
    "Bloodstone",        # 1 in 2 chance of x1.5 on Hearts
    "Ancient Joker",     # x1.5 on a random suit (rotates)
}

# Jokers that look good on paper but are traps — penalize buying
BAD_JOKERS = {
    "Flower Pot",    # x3 but requires ALL 4 suits scored — nearly impossible
}



def _interest_penalty(money: int, cost: int) -> float:
    """Compute a penalty multiplier for purchases that lose interest tiers.

    Returns a mask bias (< 1.0 means penalty, 1.0 means neutral).
    """
    current_tiers = min(money // 5, 5)
    after_tiers = min((money - cost) // 5, 5)
    lost_tiers = current_tiers - after_tiers
    if lost_tiers <= 0:
        return 1.0  # no interest lost
    # Each lost tier = $1/round lost forever. Penalize proportionally.
    return math.exp(-HAND_BIAS_STRENGTH * 0.15 * lost_tiers)


def _estimate_joker_value(joker: dict, current_jokers: list[dict],
                          gamestate: dict) -> float:
    """Estimate a joker's scoring contribution via delta evaluation.

    Computes the score difference between having and not having this joker,
    using the full scoring pipeline. This naturally captures suit synergy,
    edition bonuses, and interactions with other owned jokers.

    For a joker already owned: measures how much score drops without it.
    For a shop joker: measures how much score rises with it added.

    Returns a score delta (higher = more valuable).
    """
    from data.jokers import JOKERS
    joker_key = joker.get("joker_key", "") or joker.get("key", "")
    name = _api_key_to_name(joker_key)

    # Must-buy jokers bypass delta — always top priority
    if name in MUST_BUY_JOKERS:
        return 999999.0

    # Bad jokers bypass delta — always deprioritize
    if name in BAD_JOKERS:
        return -1.0

    # Baseline: score with current jokers
    baseline = estimate_score_for_hand_type(current_jokers, gamestate)

    # Check if this joker is IN the current list (owned) or not (shop candidate)
    joker_id = joker.get("id", id(joker))
    is_owned = any(j.get("id", id(j)) == joker_id for j in current_jokers)

    # Edition value: Negative = +1 joker slot (huge value, essentially free joker)
    # A Negative joker should almost never be sold — even a weak effect is free.
    mod = joker.get("modifier", {})
    edition = mod.get("edition", "") if isinstance(mod, dict) else ""
    edition_bonus = 0.0
    if edition == "NEGATIVE":
        # +1 slot is worth roughly the value of your average joker
        # Use a large floor so Negative jokers are never the "weakest"
        edition_bonus = max(baseline * 0.5, 500.0)

    if is_owned:
        # Score WITHOUT this joker — delta = baseline - without
        without = [j for j in current_jokers if j.get("id", id(j)) != joker_id]
        score_without = estimate_score_for_hand_type(without, gamestate)
        return (baseline - score_without) + edition_bonus
    else:
        # Score WITH this joker added — delta = with - baseline
        with_joker = list(current_jokers) + [joker]
        score_with = estimate_score_for_hand_type(with_joker, gamestate)
        raw_delta = (score_with - baseline) + edition_bonus

        # Floor: scoring jokers should NEVER return 0 when the estimator
        # just can't model their effect. If the joker has score_effect in
        # the DB, give it a minimum positive delta so it's always buyable.
        if raw_delta <= 0 and name:
            from data.jokers import JOKERS as _JDB
            if name in _JDB:
                schema = _JDB[name]
                has_effect = (schema.get("score_effect") or
                              schema.get("xmult") or schema.get("mult") or
                              schema.get("chip") or schema.get("retrigger_effect") or
                              schema.get("copy"))
                if has_effect:
                    # Minimum delta = 1% of baseline or 10, whichever is higher
                    floor = max(baseline * 0.01, 10.0)
                    if raw_delta < floor:
                        raw_delta = floor

        return raw_delta


def _joker_sell_value(joker: dict) -> int:
    """Get the sell price of a joker."""
    cost = joker.get("cost", {})
    if isinstance(cost, dict):
        return cost.get("sell", 0)
    return 0


def _get_blind_target(raw_state: dict) -> float:
    """Extract the next blind's target score.

    During SHOP, no blind is "CURRENT" — we need the UPCOMING blind.
    Returns the highest upcoming/current blind score found.
    """
    blinds = raw_state.get("blinds", {})
    best = 0.0
    if isinstance(blinds, dict):
        for b in blinds.values():
            if isinstance(b, dict) and b.get("status") in ("CURRENT", "UPCOMING", "SELECT"):
                score = b.get("score", 0)
                if score > best:
                    best = score
    return best if best > 0 else 300.0


# Logit bias strength for hand-aware card selection.
# Positive bias on best-hand cards for PLAY, on non-best cards for DISCARD.
# 5.0 strongly biases toward best hand while still allowing network to learn.
HAND_BIAS_STRENGTH = 5.0


# ============================================================
# Action Type Definitions
# ============================================================

# Master action type enum — indices into the action type logits
ACTION_PLAY = 0         # Play selected hand cards
ACTION_DISCARD = 1      # Discard selected hand cards
ACTION_BUY_JOKER = 2    # Buy a joker from shop
ACTION_BUY_VOUCHER = 3  # Buy a voucher from shop
ACTION_BUY_PACK = 4     # Buy a booster pack from shop
ACTION_SELL_JOKER = 5   # Sell an owned joker
ACTION_SELL_CONSUMABLE = 6  # Sell a consumable
ACTION_REROLL = 7        # Reroll the shop
ACTION_USE_CONSUMABLE = 8   # Use a consumable
ACTION_SELECT_BLIND = 9     # Select (accept) the current blind
ACTION_SKIP_BLIND = 10      # Skip the current blind
ACTION_SELECT_PACK_CARD = 11  # Select a card from opened booster pack
ACTION_SKIP_PACK = 12        # Skip (close) booster pack without selecting
ACTION_END_SHOP = 13         # Leave the shop (proceed to next blind)

NUM_ACTION_TYPES = 14

# Card selection slots
HAND_CARD_SLOTS = 12     # Max cards in hand
JOKER_SLOTS = 5          # Max jokers owned
CONSUMABLE_SLOTS = 2     # Max consumables
SHOP_JOKER_SLOTS = 3     # Max jokers in shop
SHOP_VOUCHER_SLOTS = 2   # Max vouchers in shop
SHOP_PACK_SLOTS = 2      # Max packs in shop
PACK_CARD_SLOTS = 5      # Max cards shown in opened pack

# Target selection: single index across all possible targets
# Layout: [shop_jokers | shop_vouchers | shop_packs | owned_jokers |
#          consumables | pack_cards]
TARGET_SHOP_JOKER_OFFSET = 0
TARGET_SHOP_VOUCHER_OFFSET = SHOP_JOKER_SLOTS
TARGET_SHOP_PACK_OFFSET = TARGET_SHOP_VOUCHER_OFFSET + SHOP_VOUCHER_SLOTS
TARGET_OWNED_JOKER_OFFSET = TARGET_SHOP_PACK_OFFSET + SHOP_PACK_SLOTS
TARGET_CONSUMABLE_OFFSET = TARGET_OWNED_JOKER_OFFSET + JOKER_SLOTS
TARGET_PACK_CARD_OFFSET = TARGET_CONSUMABLE_OFFSET + CONSUMABLE_SLOTS
NUM_TARGETS = TARGET_PACK_CARD_OFFSET + PACK_CARD_SLOTS  # 19

# Total action space dimensions for the network heads
# Play/Shop/Blind heads output:
#   action_type_logits (14) + card_selection_logits (12) + target_logits (19)
ACTION_HEAD_SIZE = NUM_ACTION_TYPES + HAND_CARD_SLOTS + NUM_TARGETS  # 45

# Action type names for logging/debugging
ACTION_NAMES = [
    "play", "discard", "buy_joker", "buy_voucher", "buy_pack",
    "sell_joker", "sell_consumable", "reroll", "use_consumable",
    "select_blind", "skip_blind", "select_pack_card", "skip_pack",
    "end_shop",
]


# ============================================================
# Valid Action Masks Per Game State
# ============================================================

# Which action types are valid in each BalatroBot game state
VALID_ACTIONS_BY_STATE = {
    "BLIND_SELECT": {ACTION_SELECT_BLIND, ACTION_SKIP_BLIND},
    "SELECTING_HAND": {
        ACTION_PLAY, ACTION_DISCARD, ACTION_USE_CONSUMABLE,
    },
    "SHOP": {
        ACTION_BUY_JOKER, ACTION_BUY_VOUCHER, ACTION_BUY_PACK,
        ACTION_SELL_JOKER, ACTION_SELL_CONSUMABLE, ACTION_REROLL,
        ACTION_USE_CONSUMABLE, ACTION_END_SHOP,
    },
    "SMODS_BOOSTER_OPENED": {ACTION_SELECT_PACK_CARD, ACTION_SKIP_PACK},
    "ROUND_EVAL": set(),   # No actions — game is resolving
    "GAME_OVER": set(),     # No actions — run is done
}


# ============================================================
# Action Mask Builder
# ============================================================

def build_action_mask(raw_state: dict) -> np.ndarray:
    """Build a binary mask over the full action space.

    Returns array of ACTION_HEAD_SIZE floats (1.0 = valid, 0.0 = invalid).
    The network's logits are added to log(mask) before softmax, so
    invalid actions get -inf and valid actions pass through.

    Args:
        raw_state: raw gamestate dict from BalatroBot API
    """
    mask = np.zeros(ACTION_HEAD_SIZE, dtype=np.float32)

    game_state = raw_state.get("state", "")
    valid_types = VALID_ACTIONS_BY_STATE.get(game_state, set())

    if not valid_types:
        return mask  # All zeros — no valid actions

    # --- Action type mask ---
    for action_type in valid_types:
        # Check additional validity constraints per action type
        if _is_action_feasible(action_type, raw_state):
            mask[action_type] = 1.0

    # --- Card selection mask + action type bias (for play/discard) ---
    if game_state == "SELECTING_HAND":
        hand_cards = raw_state.get("hand", {}).get("cards", [])
        jokers_raw = raw_state.get("jokers", {}).get("cards", [])
        deck_cards = raw_state.get("cards", {}).get("cards", [])
        card_mask_offset = NUM_ACTION_TYPES
        discards_left = raw_state.get("round", {}).get("discards_left", 0)

        best_indices = set()
        current_score = 0.0
        try:
            best = find_best_hands(hand_cards, jokers_raw, raw_state, top_n=1)
            if best:
                best_indices = set(best[0]["card_indices"])
                current_score = best[0]["estimated_score"]
        except Exception:
            pass

        # --- Blind-target-aware play/discard decision ---
        # Figure out how much score we still need
        blind_target = _get_blind_target(raw_state)
        chips_scored = raw_state.get("round", {}).get("chips", 0)
        remaining_target = max(blind_target - chips_scored, 0)
        hands_left = raw_state.get("round", {}).get("hands_left", 0)

        # Per-hand target: how much does each remaining hand need to score?
        per_hand_needed = remaining_target / max(hands_left, 1)

        # Check discard EV
        discard_ev = 0.0
        if discards_left > 0 and hand_cards and deck_cards:
            try:
                advice = find_best_discard(hand_cards, deck_cards, jokers_raw, raw_state)
                discard_ev = advice["expected_score"]
            except Exception:
                pass

        # PLAY/DISCARD BIAS: let plan_optimal_action make the real decision,
        # but bias the mask to match common cases so the NN learns faster.
        if discards_left > 0 and mask[ACTION_DISCARD] > 0:
            if remaining_target <= 0:
                # Blind already beaten — play anything to end the round
                mask[ACTION_PLAY] = math.exp(HAND_BIAS_STRENGTH * 0.8)
                mask[ACTION_DISCARD] = math.exp(-HAND_BIAS_STRENGTH * 0.8)
            elif current_score >= remaining_target:
                # Hand wins outright — play it
                mask[ACTION_PLAY] = math.exp(HAND_BIAS_STRENGTH * 0.8)
                mask[ACTION_DISCARD] = math.exp(-HAND_BIAS_STRENGTH * 0.5)
            else:
                # Hand doesn't win — mild discard bias (not unconditional).
                # plan_optimal_action does viability checks before discarding.
                per_hand = remaining_target / max(hands_left, 1)
                if current_score >= per_hand:
                    # Hand is decent enough per-hand — slight play bias
                    mask[ACTION_PLAY] = math.exp(HAND_BIAS_STRENGTH * 0.3)
                    mask[ACTION_DISCARD] = math.exp(HAND_BIAS_STRENGTH * 0.2)
                elif discard_ev > current_score * 1.3:
                    # Discard EV is meaningfully better — bias toward discard
                    mask[ACTION_DISCARD] = math.exp(HAND_BIAS_STRENGTH * 0.5)
                    mask[ACTION_PLAY] = math.exp(-HAND_BIAS_STRENGTH * 0.3)
                else:
                    # Neutral — let plan_optimal_action decide
                    mask[ACTION_PLAY] = 1.0
                    mask[ACTION_DISCARD] = 1.0
        elif discards_left <= 0:
            # No discards — must play
            if mask[ACTION_PLAY] > 0:
                mask[ACTION_PLAY] = math.exp(HAND_BIAS_STRENGTH * 0.3)

        # Card selection bias: boost best-hand cards (helps play)
        for i in range(min(len(hand_cards), HAND_CARD_SLOTS)):
            if best_indices and i in best_indices:
                mask[card_mask_offset + i] = math.exp(HAND_BIAS_STRENGTH)
            elif best_indices:
                mask[card_mask_offset + i] = math.exp(-HAND_BIAS_STRENGTH * 0.5)
            else:
                mask[card_mask_offset + i] = 1.0

    # --- Target selection mask ---
    target_offset = NUM_ACTION_TYPES + HAND_CARD_SLOTS

    if game_state == "SHOP":
        # --- Shop buying — score-gap-aware ---
        money = raw_state.get("money", 0)
        joker_cards = raw_state.get("jokers", {}).get("cards", [])
        num_jokers = len(joker_cards)
        has_joker_slot = num_jokers < JOKER_SLOTS

        # Interest awareness: $1 per $5 held, max $5 at $25
        current_interest = min(money // 5, 5)

        # Estimate current scoring power (planet levels + joker effects)
        best_hand_score = estimate_score_for_hand_type(joker_cards, raw_state)

        # Next blind target — but think AHEAD, not just the immediate blind
        next_blind_score = _get_blind_target(raw_state)
        ante = raw_state.get("ante", 1)

        # Boss blind check — if the next blind is a boss, prepare harder
        blinds = raw_state.get("blinds", {})
        next_is_boss = False
        boss_name = ""
        if isinstance(blinds, dict):
            for bkey, b in blinds.items():
                if isinstance(b, dict) and b.get("status") in ("CURRENT", "UPCOMING"):
                    if b.get("type") == "BOSS":
                        next_is_boss = True
                        boss_name = b.get("name", "")
                        # Boss blinds are harder — use boss score as target
                        boss_score = b.get("score", next_blind_score)
                        if boss_score > next_blind_score:
                            next_blind_score = boss_score

        # Boss debuff factor: some bosses make scoring much harder
        boss_debuff = 1.0
        if boss_name in ("The Flint",):
            boss_debuff = 0.5  # halves base chips and mult
        elif boss_name in ("The Needle",):
            boss_debuff = 0.25  # only 1 hand allowed
        elif boss_name in ("The Wall",):
            boss_debuff = 1.0  # extra HP, already reflected in score
        elif boss_name in ("The Club", "The Goad", "The Head", "The Window"):
            boss_debuff = 0.7  # debuffs one suit

        # Future-looking target — conservative multiplier so needs_upgrade
        # doesn't fire too eagerly (which over-prioritises packs / planets).
        future_factor = 1.0 + (ante * 0.25)
        if next_is_boss:
            future_factor = max(future_factor, 1.5)  # prepare harder for boss
        forward_target = next_blind_score * future_factor

        # Can we beat upcoming challenges with 4 hands of our best hand type?
        projected_total = best_hand_score * 4 * boss_debuff
        needs_upgrade = projected_total < forward_target

        # Shop jokers — boost scoring jokers when needed, handle upgrades
        shop_cards = raw_state.get("shop", {}).get("cards", [])
        any_buyable_joker = False
        has_scoring_joker_in_shop = False
        has_must_buy = False

        # Find weakest owned joker via delta evaluation (removing it hurts least)
        # Skip eternal, negative, MUST_BUY, retrigger, and copy jokers
        from data.jokers import JOKERS as _JOKER_DB
        weakest_owned_value = float("inf")
        weakest_owned_idx = -1
        owned_values = []
        for i, card in enumerate(joker_cards[:JOKER_SLOTS]):
            mod = _safe_modifier(card)
            if mod.get("eternal", False):
                owned_values.append(float("inf"))
                continue
            # Negative edition jokers give +1 slot — never sell
            ed = mod.get("edition", "") if isinstance(mod, dict) else ""
            if ed == "NEGATIVE":
                owned_values.append(float("inf"))
                continue
            # Never mark MUST_BUY, retrigger, or copy jokers as weakest
            _jk = card.get("joker_key", "") or card.get("key", "")
            _jname = _api_key_to_name(_jk)
            if _jname in MUST_BUY_JOKERS:
                owned_values.append(float("inf"))
                continue
            if _jname and _jname in _JOKER_DB:
                _jschema = _JOKER_DB[_jname]
                if _jschema.get("retrigger_effect") or _jschema.get("copy"):
                    owned_values.append(float("inf"))
                    continue
            value = _estimate_joker_value(card, joker_cards, raw_state)
            owned_values.append(value)
            if value < weakest_owned_value:
                weakest_owned_value = value
                weakest_owned_idx = i

        # Track if we found a shop joker worth upgrading to
        upgrade_target_idx = -1  # index of shop joker to buy
        upgrade_sell_idx = -1    # index of owned joker to sell

        # Check consumable slots for buying planets/tarots from shop
        cons_cards = raw_state.get("consumables", {}).get("cards", [])
        cons_count = len(cons_cards)
        cons_limit = raw_state.get("consumables", {}).get("limit", 2)
        has_cons_slot = cons_count < cons_limit

        # Valuable consumables worth buying from shop
        MUST_BUY_CONSUMABLES = {
            "c_hermit",       # Doubles money — always buy
        }
        GOOD_PLANET_KEYS = {
            "c_earth", "c_mercury", "c_venus", "c_mars", "c_jupiter",
            "c_saturn", "c_uranus", "c_neptune", "c_pluto", "c_planet_x",
            "c_ceres", "c_eris",
        }
        GOOD_TAROT_KEYS = {
            "c_strength", "c_death", "c_empress", "c_justice",
            "c_temperance",  # Earns money based on jokers
        }

        for i, card in enumerate(shop_cards[:SHOP_JOKER_SLOTS]):
            # Handle non-joker shop cards (planets/tarots)
            if _is_non_joker_card(card):
                card_key = card.get("key", "")
                card_set = card.get("set", "").upper()
                cost = card.get("cost", {}).get("buy", 999)

                if cost > money:
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = 0.0
                    continue

                ip = _interest_penalty(money, cost)

                if card_key in MUST_BUY_CONSUMABLES and has_cons_slot:
                    # Hermit etc — always buy, very high priority
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.7) * ip
                    any_buyable_joker = True  # prevents rerolling past it
                    print(f"[SHOP-EVAL] {card_key}: MUST-BUY consumable cost=${cost}",
                          flush=True)
                elif card_set == "PLANET" and has_cons_slot and needs_upgrade:
                    # Planet cards level up hand types — useful when we need more power
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.2) * ip
                    print(f"[SHOP-EVAL] {card_key}: planet (needs_upgrade) cost=${cost}",
                          flush=True)
                elif card_set == "TAROT" and has_cons_slot and card_key in GOOD_TAROT_KEYS:
                    # Good tarots — moderate priority
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.15) * ip
                    print(f"[SHOP-EVAL] {card_key}: good tarot cost=${cost}",
                          flush=True)
                else:
                    # Other consumables or no slot — block
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = 0.0
                continue

            cost = card.get("cost", {}).get("buy", 999)
            joker_key = card.get("joker_key", "") or card.get("key", "")
            joker_name = _api_key_to_name(joker_key)

            # Check affordability — if slots full, account for sell price of weakest
            can_afford = cost <= money
            can_slot = has_joker_slot
            if not can_slot and weakest_owned_idx >= 0:
                sell_price = _joker_sell_value(joker_cards[weakest_owned_idx])
                can_afford = cost <= (money + sell_price)

            if not can_afford:
                continue

            if joker_name in MUST_BUY_JOKERS:
                # Always buy these — they're game-defining
                any_buyable_joker = True
                has_scoring_joker_in_shop = True
                has_must_buy = True
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.8)
                if not has_joker_slot and weakest_owned_idx >= 0:
                    can_slot = True
                    upgrade_target_idx = i
                    upgrade_sell_idx = weakest_owned_idx
                continue

            if joker_name in BAD_JOKERS:
                # Trap jokers — hard block (heuristic will always reject)
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = 0.0
                continue

            # Check for high-value jokers — these get strong buy bias
            is_high_value = joker_name in HIGH_VALUE_JOKERS
            # Also treat any Polychrome/Holographic joker as high value
            shop_mod = _safe_modifier(card)
            shop_edition = shop_mod.get("edition", "") if isinstance(shop_mod, dict) else ""
            if shop_edition in ("POLYCHROME", "HOLO"):
                is_high_value = True

            # Delta evaluation: how much does adding this joker improve scoring?
            # If slots full, simulate swapping out the weakest joker.
            # Exception: Negative edition jokers bypass the slot limit entirely.
            is_scoring = _joker_is_scoring(card)
            ip = _interest_penalty(money, cost)

            is_negative = shop_edition == "NEGATIVE"

            if not has_joker_slot and not is_negative:
                if weakest_owned_idx < 0:
                    continue
                # Simulate swap: remove weakest, add shop joker
                swapped_jokers = [j for idx, j in enumerate(joker_cards[:JOKER_SLOTS])
                                  if idx != weakest_owned_idx]
                swapped_jokers.append(card)
                swap_score = estimate_score_for_hand_type(swapped_jokers, raw_state)
                # High-value jokers use a lower swap threshold (5% instead of 10%)
                swap_threshold = 1.05 if is_high_value else 1.1
                if swap_score > best_hand_score * swap_threshold:
                    # Swap is a meaningful upgrade
                    any_buyable_joker = True
                    has_scoring_joker_in_shop = True
                    # Scale boost by improvement magnitude
                    improvement = swap_score / max(best_hand_score, 1.0)
                    boost = min(improvement - 1.0, 1.0)  # cap at 1.0
                    base_boost = 0.4 if is_high_value else 0.2
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * (base_boost + boost * 0.4))
                    upgrade_target_idx = i
                    upgrade_sell_idx = weakest_owned_idx
                continue

            # Open slot available — evaluate via delta
            shop_delta = _estimate_joker_value(card, joker_cards, raw_state)

            # Debug: log delta evaluation for every shop joker
            _jlabel = joker_name or joker_key
            print(f"[SHOP-EVAL] {_jlabel}: delta={shop_delta:.0f} scoring={is_scoring} "
                  f"high_value={is_high_value} cost=${cost} slots_open={has_joker_slot}",
                  flush=True)

            if shop_delta > 0 and is_scoring:
                any_buyable_joker = True
                has_scoring_joker_in_shop = True
                # High-value jokers ALWAYS get the strong boost regardless of relative gain
                if is_high_value:
                    mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.7) * ip
                else:
                    # Scale boost by how much this joker improves scoring
                    relative_gain = shop_delta / max(best_hand_score, 1.0)
                    if needs_upgrade or relative_gain > 0.2:
                        mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.6) * ip
                    else:
                        mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.3) * ip
            elif is_high_value:
                # High-value joker — buy regardless of delta (estimator undervalues
                # retrigger/synergy effects). Trust the tier list.
                any_buyable_joker = True
                has_scoring_joker_in_shop = True
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.4) * ip
            elif shop_delta > 0:
                # Non-scoring but still positive (economy joker that helps)
                any_buyable_joker = True
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = 1.0 * ip
            elif is_scoring and has_joker_slot:
                # Scoring joker — estimator might undervalue it (delta=0 or slightly negative).
                # If we have open slots, always allow buying. A joker we can't
                # perfectly model is still better than an empty slot or a pack gamble.
                any_buyable_joker = True
                has_scoring_joker_in_shop = True
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.3) * ip
            elif not is_scoring and shop_delta == 0 and has_joker_slot:
                # Economy jokers (Egg, Delayed Gratification, etc.) have delta=0
                # because they don't affect scoring. Allow buying with slight penalty.
                any_buyable_joker = True
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = math.exp(-HAND_BIAS_STRENGTH * 0.3) * ip
            else:
                # Delta < 0 and not high-value — hard block
                mask[target_offset + TARGET_SHOP_JOKER_OFFSET + i] = 0.0

        if not any_buyable_joker and mask[ACTION_BUY_JOKER] > 0:
            # No viable joker in shop — hard block buy action entirely
            mask[ACTION_BUY_JOKER] = 0.0
        elif has_must_buy and mask[ACTION_BUY_JOKER] > 0:
            # Must-buy joker in shop — strongest possible boost
            mask[ACTION_BUY_JOKER] = math.exp(HAND_BIAS_STRENGTH * 0.8)
        elif has_scoring_joker_in_shop and mask[ACTION_BUY_JOKER] > 0:
            # Scoring joker available — always boost buying, especially with empty slots
            empty_slots = JOKER_SLOTS - num_jokers
            if empty_slots >= 2:
                # Multiple empty slots — very strong boost, fill them up
                mask[ACTION_BUY_JOKER] = math.exp(HAND_BIAS_STRENGTH * 0.5)
            elif empty_slots == 1:
                mask[ACTION_BUY_JOKER] = math.exp(HAND_BIAS_STRENGTH * 0.3)
            elif needs_upgrade:
                mask[ACTION_BUY_JOKER] = math.exp(HAND_BIAS_STRENGTH * 0.3)
            else:
                mask[ACTION_BUY_JOKER] = 1.0
        elif needs_upgrade and any_buyable_joker and mask[ACTION_BUY_JOKER] > 0:
            mask[ACTION_BUY_JOKER] = math.exp(HAND_BIAS_STRENGTH * 0.3)

        # Shop vouchers — only buy high-value vouchers that directly improve
        # scoring power or economy.  Most vouchers are traps that drain early
        # money that should go toward jokers.
        # Critical vouchers: bypass interest protection, always worth buying
        CRITICAL_VOUCHERS = {
            "v_grabber",        # +1 hand per round
            "v_wasteful",       # +1 discard per round
            "v_paint_brush",    # +1 hand size
            "v_nacho_tong",     # +1 hand (tier 2 of Grabber)
            "v_recyclomancy",   # +1 discard (tier 2 of Wasteful)
            "v_palette",        # +1 hand size (tier 2 of Paint Brush)
            "v_seed_money",     # Interest cap $25 (pays for itself)
            "v_money_tree",     # Interest cap $50
            "v_antimatter",     # +1 joker slot
        }
        # Good vouchers: only buy with $10+ cushion after purchase
        GOOD_VOUCHERS = {
            "v_hieroglyph",     # -1 ante requirement (nice, not critical)
            "v_petroglyph",     # -1 ante (tier 2)
            "v_overstock",      # +1 shop slot
            "v_overstock_plus", # +1 more shop slot
            "v_directors_cut",  # Reroll boss blind
        }
        any_good_voucher = False
        shop_vouchers = raw_state.get("vouchers", {}).get("cards", [])
        for i, card in enumerate(shop_vouchers[:SHOP_VOUCHER_SLOTS]):
            vcost = card.get("cost", {}).get("buy", 999)
            if vcost > money:
                continue
            ip = _interest_penalty(money, vcost)
            vkey = card.get("key", "")

            if vkey in CRITICAL_VOUCHERS:
                # Critical — mild boost, but still below jokers in priority
                mask[target_offset + TARGET_SHOP_VOUCHER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.15) * ip
                any_good_voucher = True
            elif vkey in GOOD_VOUCHERS and money >= vcost + 10:
                # Good but need $10+ cushion to preserve econ
                mask[target_offset + TARGET_SHOP_VOUCHER_OFFSET + i] = math.exp(-HAND_BIAS_STRENGTH * 0.1) * ip
                any_good_voucher = True
            else:
                # Bad/unknown voucher or can't afford comfortably — block
                mask[target_offset + TARGET_SHOP_VOUCHER_OFFSET + i] = 0.0

        # Block voucher action entirely if nothing worth buying
        if not any_good_voucher and mask[ACTION_BUY_VOUCHER] > 0:
            mask[ACTION_BUY_VOUCHER] = 0.0

        # Shop packs — prioritize jokers first, then packs
        shop_packs = raw_state.get("packs", {}).get("cards", [])
        any_good_pack = False
        any_affordable_pack = False
        for i, card in enumerate(shop_packs[:SHOP_PACK_SLOTS]):
            cost = card.get("cost", {}).get("buy", 999)
            if cost > money:
                continue
            any_affordable_pack = True
            ip = _interest_penalty(money, cost)
            pack_key = card.get("key", "")

            # Standard packs add cards to deck — dilutes draw odds. Block them.
            if "standard" in pack_key:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = 0.0
                continue

            # If scoring jokers are available in shop, penalize packs heavily
            if has_scoring_joker_in_shop and has_joker_slot:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = math.exp(-HAND_BIAS_STRENGTH * 0.4) * ip
                continue

            if needs_upgrade and "celestial" in pack_key:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.15) * ip
                any_good_pack = True
            elif needs_upgrade and "buffoon" in pack_key and has_joker_slot:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.2) * ip
                any_good_pack = True
            elif needs_upgrade and "arcana" in pack_key:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.05) * ip
                any_good_pack = True
            elif not needs_upgrade:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = math.exp(-HAND_BIAS_STRENGTH * 0.3) * ip
            else:
                mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = math.exp(-HAND_BIAS_STRENGTH * 0.1) * ip
        any_pack_target_valid = any(
            mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] > 0
            for i in range(min(len(shop_packs), SHOP_PACK_SLOTS))
        )
        if not any_affordable_pack or not any_pack_target_valid:
            mask[ACTION_BUY_PACK] = 0.0
        elif any_buyable_joker and has_joker_slot:
            has_free_pack = any(
                shop_packs[i].get("cost", {}).get("buy", 999) <= 0
                for i in range(min(len(shop_packs), SHOP_PACK_SLOTS))
            )
            if has_free_pack:
                mask[ACTION_BUY_PACK] = 1.0
            else:
                mask[ACTION_BUY_PACK] = 0.0
                for i in range(min(len(shop_packs), SHOP_PACK_SLOTS)):
                    mask[target_offset + TARGET_SHOP_PACK_OFFSET + i] = 0.0
        elif has_joker_slot:
            has_free_pack = any(
                shop_packs[i].get("cost", {}).get("buy", 999) <= 0
                for i in range(min(len(shop_packs), SHOP_PACK_SLOTS))
            )
            if has_free_pack:
                mask[ACTION_BUY_PACK] = 1.5
            else:
                mask[ACTION_BUY_PACK] = 1.0
        elif needs_upgrade and any_good_pack and mask[ACTION_BUY_PACK] > 0:
            mask[ACTION_BUY_PACK] = 1.0
        elif mask[ACTION_BUY_PACK] > 0:
            mask[ACTION_BUY_PACK] = 0.5

        # Owned jokers (for selling) — HARD block on scoring/negative jokers unless upgrading
        any_sellable = False
        all_scoring = True
        for i, card in enumerate(joker_cards[:JOKER_SLOTS]):
            mod = _safe_modifier(card)
            if mod.get("eternal", False):
                continue
            # Negative edition = +1 joker slot, never sell
            ed = mod.get("edition", "") if isinstance(mod, dict) else ""
            if ed == "NEGATIVE":
                mask[target_offset + TARGET_OWNED_JOKER_OFFSET + i] = 0.0
                continue
            any_sellable = True
            is_scoring = _joker_is_scoring(card)
            if not is_scoring:
                all_scoring = False

            if i == upgrade_sell_idx and upgrade_target_idx >= 0:
                # This is the weakest joker and we have a BETTER one in shop — allow selling
                mask[target_offset + TARGET_OWNED_JOKER_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.5)
            elif is_scoring:
                # Scoring joker — HARD BLOCK selling (mask = 0)
                # Never sell a scoring joker unless it's the weakest and we have an upgrade
                mask[target_offset + TARGET_OWNED_JOKER_OFFSET + i] = 0.0
            else:
                # Non-scoring (economy/utility) joker — allow sell but slight penalty
                mask[target_offset + TARGET_OWNED_JOKER_OFFSET + i] = math.exp(-HAND_BIAS_STRENGTH * 0.2)

        # Bias the SELL_JOKER action type based on joker portfolio
        if upgrade_sell_idx >= 0 and upgrade_target_idx >= 0 and mask[ACTION_SELL_JOKER] > 0:
            # Upgrade available — boost selling the weakest
            mask[ACTION_SELL_JOKER] = math.exp(HAND_BIAS_STRENGTH * 0.4)
        else:
            # No upgrade available — HARD BLOCK selling entirely.
            # The heuristic will block any sell without a shop upgrade anyway,
            # so allowing sells here just creates no-op spam.
            mask[ACTION_SELL_JOKER] = 0.0
            for i in range(JOKER_SLOTS):
                mask[target_offset + TARGET_OWNED_JOKER_OFFSET + i] = 0.0

        # Consumables (for selling or using)
        # Block selling consumables that plan_consumable_use would want to use
        consumable_cards = raw_state.get("consumables", {}).get("cards", [])
        _useful_cons = set()
        try:
            _cons_action = plan_consumable_use(raw_state)
            if _cons_action is not None:
                _useful_cons.add(_cons_action.get("consumable", -1))
        except Exception:
            pass
        # Also protect planet cards (always useful) and Hermit (timing-dependent)
        for i, c in enumerate(consumable_cards[:CONSUMABLE_SLOTS]):
            c_set = c.get("set", "")
            c_key = c.get("key", "")
            if c_set == "PLANET" or c_key in ("c_hermit", "c_temperance") or i in _useful_cons:
                # Hard block sell — these are always valuable
                mask[target_offset + TARGET_CONSUMABLE_OFFSET + i] = 0.0
            else:
                mask[target_offset + TARGET_CONSUMABLE_OFFSET + i] = 1.0

        # Reroll — conservative. Only reroll when nothing to buy and money to spare.
        if mask[ACTION_REROLL] > 0:
            reroll_cost = raw_state.get("round", {}).get("reroll_cost", 5)
            min_joker_cost = 4
            money_after_reroll = money - reroll_cost

            if money < reroll_cost or money_after_reroll < min_joker_cost:
                mask[ACTION_REROLL] = 0.0
            elif any_buyable_joker:
                # Something worth buying exists — don't reroll past it
                mask[ACTION_REROLL] = 0.0
            else:
                # Nothing buyable — allow reroll but keep it mild
                interest_floor = 25
                surplus = max(money - interest_floor, 0)
                if surplus <= 0:
                    # No surplus — penalize hard
                    mask[ACTION_REROLL] = 0.3
                elif surplus > 20:
                    # Plenty of spare cash — light nudge to reroll
                    mask[ACTION_REROLL] = 1.5
                else:
                    # Some surplus — neutral
                    mask[ACTION_REROLL] = 1.0

        # END_SHOP bias — guide but don't override
        if mask[ACTION_END_SHOP] > 0:
            can_buy_something = (any_buyable_joker or any_good_pack)
            empty_slots = JOKER_SLOTS - num_jokers
            if has_scoring_joker_in_shop and empty_slots >= 1 and any_buyable_joker:
                # Scoring joker available with empty slots — strongly penalize leaving
                mask[ACTION_END_SHOP] = math.exp(-HAND_BIAS_STRENGTH * 0.5)
            elif needs_upgrade and can_buy_something:
                # We need power and can buy it — penalize leaving
                mask[ACTION_END_SHOP] = math.exp(-HAND_BIAS_STRENGTH * 0.4)
            elif needs_upgrade and not can_buy_something:
                # Need upgrade but nothing good — slight nudge to leave
                mask[ACTION_END_SHOP] = math.exp(HAND_BIAS_STRENGTH * 0.15)
            elif has_scoring_joker_in_shop:
                # Don't urgently need upgrade but good joker available — neutral
                mask[ACTION_END_SHOP] = 1.0
            else:
                # Don't need upgrades and nothing great — slight nudge to leave
                mask[ACTION_END_SHOP] = math.exp(HAND_BIAS_STRENGTH * 0.15)

    elif game_state == "SELECTING_HAND":
        # Owned jokers (for selling during hand selection)
        joker_cards = raw_state.get("jokers", {}).get("cards", [])
        for i, card in enumerate(joker_cards[:JOKER_SLOTS]):
            if not _safe_modifier(card).get("eternal", False):
                mask[target_offset + TARGET_OWNED_JOKER_OFFSET + i] = 1.0

        # Consumables — enable targets for all owned consumables.
        # The sell_consumable heuristic in train.py guards against selling
        # planets and other useful consumables. We can't block targets here
        # because the same target offset is shared by USE_CONSUMABLE and
        # SELL_CONSUMABLE — blocking useful consumables from selling also
        # blocks them from being USED, which is backwards.
        consumable_cards = raw_state.get("consumables", {}).get("cards", [])
        for i, c in enumerate(consumable_cards[:CONSUMABLE_SLOTS]):
            mask[target_offset + TARGET_CONSUMABLE_OFFSET + i] = 1.0

    elif game_state == "BLIND_SELECT":
        # --- Blind skip logic ---
        blinds = raw_state.get("blinds", {})
        current_blind = None
        is_boss = False
        for b in (blinds.values() if isinstance(blinds, dict) else []):
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                current_blind = b
                is_boss = b.get("type") == "BOSS"
                break

        if current_blind and not is_boss:
            # Check if the skip tag is worth it
            tag_name = current_blind.get("tag_name", "")
            has_investment_tag = tag_name == "Investment Tag"

            if has_investment_tag:
                # Investment Tag gives free money — skip if we can handle it
                joker_cards = raw_state.get("jokers", {}).get("cards", [])
                scoring_power = estimate_score_for_hand_type(joker_cards, raw_state) * 4
                blind_target = current_blind.get("score", 0)

                if blind_target > 0 and scoring_power > blind_target * 2.0:
                    # Can handle skipping — Investment Tag is worth it
                    mask[ACTION_SKIP_BLIND] = math.exp(HAND_BIAS_STRENGTH * 0.4)
                    mask[ACTION_SELECT_BLIND] = math.exp(-HAND_BIAS_STRENGTH * 0.2)
                else:
                    # Too tight to skip even for Investment Tag
                    mask[ACTION_SELECT_BLIND] = math.exp(HAND_BIAS_STRENGTH * 0.3)
                    if mask[ACTION_SKIP_BLIND] > 0:
                        mask[ACTION_SKIP_BLIND] = math.exp(-HAND_BIAS_STRENGTH * 0.2)
            else:
                # No Investment tag — never skip
                mask[ACTION_SELECT_BLIND] = math.exp(HAND_BIAS_STRENGTH * 0.3)
                mask[ACTION_SKIP_BLIND] = 0.0
        elif is_boss:
            # Boss blind — always select (skip not allowed anyway)
            mask[ACTION_SELECT_BLIND] = math.exp(HAND_BIAS_STRENGTH * 0.5)

    elif game_state == "SMODS_BOOSTER_OPENED":
        # --- Smart pack card selection ---
        pack_cards = raw_state.get("pack", {}).get("cards", [])
        hands_data = raw_state.get("hands", {})

        # Find the best hand type to upgrade (most played + highest scoring)
        best_hand_type = None
        best_hand_score = 0
        for ht_name in ("Pair", "Two Pair", "Three of a Kind", "Straight",
                         "Flush", "Full House", "Four of a Kind"):
            ht_info = hands_data.get(ht_name, {})
            played = ht_info.get("played", 0)
            if played > 0:
                base_c, base_m = BASE_HAND_SCORES.get(ht_name, (5, 1))
                score = ht_info.get("chips", base_c) * ht_info.get("mult", base_m)
                if score > best_hand_score:
                    best_hand_score = score
                    best_hand_type = ht_name
        if not best_hand_type:
            best_hand_type = "Pair"  # default

        any_good_card = False
        for i, card in enumerate(pack_cards[:PACK_CARD_SLOTS]):
            card_set = card.get("set", "")
            card_key = card.get("key", "")

            if card_set == "Planet":
                # Planet card — check if it upgrades our best hand type
                target_ht = PLANET_TO_HAND_TYPE.get(card_key)
                if target_ht == best_hand_type:
                    # Perfect match — strong boost
                    mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.6)
                    any_good_card = True
                elif target_ht:
                    # Different hand type — still useful, mild boost
                    mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.2)
                    any_good_card = True
                else:
                    mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = 1.0
            elif _is_joker_card(card):
                # Joker from buffoon pack
                is_scoring = _joker_is_scoring(card)
                joker_name = _api_key_to_name(card.get("key", ""))
                if joker_name in MUST_BUY_JOKERS:
                    mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.8)
                    any_good_card = True
                elif is_scoring:
                    mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.4)
                    any_good_card = True
                else:
                    mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = 1.0
            elif card_set == "Tarot":
                # Tarot from arcana pack — generally useful
                mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = math.exp(HAND_BIAS_STRENGTH * 0.2)
                any_good_card = True
            else:
                mask[target_offset + TARGET_PACK_CARD_OFFSET + i] = 1.0

        # Bias select vs skip
        if any_good_card and mask[ACTION_SELECT_PACK_CARD] > 0:
            mask[ACTION_SELECT_PACK_CARD] = math.exp(HAND_BIAS_STRENGTH * 0.4)
        if mask[ACTION_SKIP_PACK] > 0:
            if any_good_card:
                mask[ACTION_SKIP_PACK] = math.exp(-HAND_BIAS_STRENGTH * 0.3)
            else:
                mask[ACTION_SKIP_PACK] = math.exp(HAND_BIAS_STRENGTH * 0.2)

    return mask


def _is_action_feasible(action_type: int, raw_state: dict) -> bool:
    """Check if an action type is feasible given current game state."""
    if action_type == ACTION_PLAY:
        # Need at least 1 hand remaining and at least 1 card in hand
        hands_left = raw_state.get("round", {}).get("hands_left", 0)
        hand_cards = raw_state.get("hand", {}).get("cards", [])
        return hands_left > 0 and len(hand_cards) > 0

    elif action_type == ACTION_DISCARD:
        # Need at least 1 discard remaining and at least 1 card
        discards_left = raw_state.get("round", {}).get("discards_left", 0)
        hand_cards = raw_state.get("hand", {}).get("cards", [])
        return discards_left > 0 and len(hand_cards) > 0

    elif action_type == ACTION_BUY_JOKER:
        # Need money >= cheapest shop joker AND (joker slot available OR negative joker in shop)
        joker_cards = raw_state.get("jokers", {}).get("cards", [])
        shop_cards = raw_state.get("shop", {}).get("cards", [])
        money = raw_state.get("money", 0)
        has_slot = len(joker_cards) < JOKER_SLOTS
        if not has_slot:
            # At capacity — allow if:
            # 1. A NEGATIVE edition joker is affordable (bypasses slot limit), OR
            # 2. There's an affordable shop joker AND a sellable owned joker
            #    (upgrade-via-sell path in build_action_mask and heuristic)
            has_sellable = any(
                not (_safe_modifier(j).get("eternal", False) or
                     _safe_modifier(j).get("edition", "") == "NEGATIVE")
                for j in joker_cards[:JOKER_SLOTS]
            )
            weakest_sell = 0
            if has_sellable:
                weakest_sell = min(
                    _joker_sell_value(j) for j in joker_cards[:JOKER_SLOTS]
                    if not (_safe_modifier(j).get("eternal", False) or
                            _safe_modifier(j).get("edition", "") == "NEGATIVE")
                )
            for c in shop_cards[:SHOP_JOKER_SLOTS]:
                if _is_non_joker_card(c):
                    continue
                mod = c.get("modifier", {})
                ed = mod.get("edition", "") if isinstance(mod, dict) else ""
                cost = c.get("cost", {}).get("buy", 999)
                if ed == "NEGATIVE" and cost <= money:
                    return True  # Negative edition bypasses slot limit
                if has_sellable and cost <= money + weakest_sell:
                    return True  # Can afford via sell-then-buy
            return False
        return any(
            c.get("cost", {}).get("buy", 999) <= money
            for c in shop_cards[:SHOP_JOKER_SLOTS]
            if not _is_non_joker_card(c)
        )

    elif action_type == ACTION_BUY_VOUCHER:
        shop_vouchers = raw_state.get("vouchers", {}).get("cards", [])
        money = raw_state.get("money", 0)
        return any(c.get("cost", {}).get("buy", 999) <= money for c in shop_vouchers[:SHOP_VOUCHER_SLOTS])

    elif action_type == ACTION_BUY_PACK:
        shop_packs = raw_state.get("packs", {}).get("cards", [])
        money = raw_state.get("money", 0)
        return any(c.get("cost", {}).get("buy", 999) <= money for c in shop_packs[:SHOP_PACK_SLOTS])

    elif action_type == ACTION_SELL_JOKER:
        joker_cards = raw_state.get("jokers", {}).get("cards", [])
        # Can sell any non-eternal joker
        return any(
            not _safe_modifier(c).get("eternal", False)
            for c in joker_cards
        )

    elif action_type == ACTION_SELL_CONSUMABLE:
        consumable_cards = raw_state.get("consumables", {}).get("cards", [])
        return len(consumable_cards) > 0

    elif action_type == ACTION_REROLL:
        money = raw_state.get("money", 0)
        reroll_cost = raw_state.get("round", {}).get("reroll_cost", 5)
        return money >= reroll_cost

    elif action_type == ACTION_USE_CONSUMABLE:
        consumable_cards = raw_state.get("consumables", {}).get("cards", [])
        return len(consumable_cards) > 0

    elif action_type == ACTION_SELECT_BLIND:
        return True  # Always valid during BLIND_SELECT

    elif action_type == ACTION_SKIP_BLIND:
        # Only skip for Investment Tag + strong enough scoring
        blinds = raw_state.get("blinds", {})
        if isinstance(blinds, dict):
            for b in blinds.values():
                if isinstance(b, dict) and b.get("status") == "CURRENT":
                    if b.get("type") == "BOSS":
                        return False
                    tag_name = b.get("tag_name", "")
                    if tag_name != "Investment Tag":
                        return False
                    # Check scoring power
                    joker_cards = raw_state.get("jokers", {}).get("cards", [])
                    scoring_power = estimate_score_for_hand_type(joker_cards, raw_state) * 4
                    blind_target = b.get("score", 0)
                    return blind_target > 0 and scoring_power > blind_target * 2.0
        return False

    elif action_type == ACTION_SELECT_PACK_CARD:
        pack_cards = raw_state.get("pack", {}).get("cards", [])
        return len(pack_cards) > 0

    elif action_type == ACTION_SKIP_PACK:
        return True  # Always valid during pack opening

    elif action_type == ACTION_END_SHOP:
        return True  # Always valid during shop

    return False


# ============================================================
# Action Decoder — Network Output → API Call
# ============================================================

class ActionDecoder:
    """Converts network output logits into a concrete BalatroBot API call.

    All card references use 0-based positional indices (not IDs).
    API methods match BalatroBot endpoints exactly:
    - play/discard: cards=[indices]
    - buy: card=idx OR voucher=idx OR pack=idx
    - sell: joker=idx OR consumable=idx
    - use: consumable=idx, cards=[indices] (optional targets)
    - pack: card=idx OR skip=true
    - next_round: leave shop
    - select/skip: blind selection
    - reroll: reroll shop
    """

    def decode(self, logits: np.ndarray, mask: np.ndarray,
               raw_state: dict, deterministic: bool = False) -> tuple[str, Optional[dict], dict]:
        """Decode network logits into an API call.

        Args:
            logits: raw network output, shape (ACTION_HEAD_SIZE,)
            mask: validity mask from build_action_mask()
            raw_state: current raw gamestate
            deterministic: if True, use argmax; if False, sample

        Returns:
            (api_method, api_params, info_dict)
            - api_method: BalatroBot RPC method name
            - api_params: params dict for the RPC call (or None)
            - info_dict: metadata for logging (action_type, selected_cards, etc.)
        """
        # Split logits into sections
        type_logits = logits[:NUM_ACTION_TYPES]
        card_logits = logits[NUM_ACTION_TYPES:NUM_ACTION_TYPES + HAND_CARD_SLOTS]
        target_logits = logits[NUM_ACTION_TYPES + HAND_CARD_SLOTS:]

        # Split mask the same way
        type_mask = mask[:NUM_ACTION_TYPES]
        card_mask = mask[NUM_ACTION_TYPES:NUM_ACTION_TYPES + HAND_CARD_SLOTS]
        target_mask = mask[NUM_ACTION_TYPES + HAND_CARD_SLOTS:]

        # Select action type
        action_type = self._masked_select(type_logits, type_mask, deterministic)

        if action_type is None:
            return "gamestate", None, {"action_type": "none", "error": "no_valid_actions"}

        info = {"action_type": ACTION_NAMES[action_type], "action_type_idx": action_type}

        # Route to the appropriate handler
        if action_type == ACTION_PLAY:
            return self._decode_play(card_logits, card_mask, deterministic, info, raw_state)
        elif action_type == ACTION_DISCARD:
            return self._decode_discard(card_logits, card_mask, deterministic, info, raw_state)
        elif action_type == ACTION_BUY_JOKER:
            return self._decode_buy_indexed(target_logits, target_mask,
                                            TARGET_SHOP_JOKER_OFFSET, SHOP_JOKER_SLOTS,
                                            "card", info)
        elif action_type == ACTION_BUY_VOUCHER:
            return self._decode_buy_indexed(target_logits, target_mask,
                                            TARGET_SHOP_VOUCHER_OFFSET, SHOP_VOUCHER_SLOTS,
                                            "voucher", info)
        elif action_type == ACTION_BUY_PACK:
            return self._decode_buy_indexed(target_logits, target_mask,
                                            TARGET_SHOP_PACK_OFFSET, SHOP_PACK_SLOTS,
                                            "pack", info)
        elif action_type == ACTION_SELL_JOKER:
            return self._decode_sell_indexed(target_logits, target_mask,
                                            TARGET_OWNED_JOKER_OFFSET, JOKER_SLOTS,
                                            "joker", info)
        elif action_type == ACTION_SELL_CONSUMABLE:
            return self._decode_sell_indexed(target_logits, target_mask,
                                            TARGET_CONSUMABLE_OFFSET, CONSUMABLE_SLOTS,
                                            "consumable", info)
        elif action_type == ACTION_REROLL:
            return "reroll", None, info
        elif action_type == ACTION_USE_CONSUMABLE:
            return self._decode_use_consumable(target_logits, target_mask,
                                               card_logits, card_mask,
                                               raw_state, deterministic, info)
        elif action_type == ACTION_SELECT_BLIND:
            return "select", None, info
        elif action_type == ACTION_SKIP_BLIND:
            return "skip", None, info
        elif action_type == ACTION_SELECT_PACK_CARD:
            return self._decode_pack_select(target_logits, target_mask, info, raw_state)
        elif action_type == ACTION_SKIP_PACK:
            return "pack", {"skip": True}, info
        elif action_type == ACTION_END_SHOP:
            return "next_round", None, info

        return "gamestate", None, info

    def _decode_play(self, card_logits: np.ndarray, card_mask: np.ndarray,
                     deterministic: bool,
                     info: dict, raw_state: dict) -> tuple[str, dict, dict]:
        """Decode a play action — select 1-5 card indices from hand.

        Applies logit bias toward the best classified hand combo.
        """
        biased_logits = self._apply_hand_bias(card_logits, card_mask, raw_state, play=True)
        selected_indices = self._select_cards(biased_logits, card_mask, deterministic,
                                              min_cards=1, max_cards=5)
        info["selected_cards"] = selected_indices
        return "play", {"cards": selected_indices}, info

    def _decode_discard(self, card_logits: np.ndarray, card_mask: np.ndarray,
                        deterministic: bool,
                        info: dict, raw_state: dict) -> tuple[str, dict, dict]:
        """Decode a discard action — select 1-5 card indices from hand.

        Applies logit bias toward cards NOT in the best hand combo.
        """
        biased_logits = self._apply_hand_bias(card_logits, card_mask, raw_state, play=False)
        selected_indices = self._select_cards(biased_logits, card_mask, deterministic,
                                              min_cards=1, max_cards=5)
        info["selected_cards"] = selected_indices
        return "discard", {"cards": selected_indices}, info

    def _apply_hand_bias(self, card_logits: np.ndarray, card_mask: np.ndarray,
                         raw_state: dict, play: bool) -> np.ndarray:
        """Add logit bias based on hand evaluation.

        For PLAY: boost cards in the best hand combo.
        For DISCARD: considers flush/straight draws — if 4+ cards share a
        suit, bias toward discarding the non-suit cards instead of just
        discarding non-best-hand cards.
        """
        biased = card_logits.copy()
        try:
            hand_cards = raw_state.get("hand", {}).get("cards", [])
            jokers_raw = raw_state.get("jokers", {}).get("cards", [])
            if not hand_cards:
                return biased

            best = find_best_hands(hand_cards, jokers_raw, raw_state, top_n=1)
            if not best:
                return biased

            best_indices = set(best[0]["card_indices"])
            n = min(len(card_logits), len(hand_cards))

            if play:
                for i in range(n):
                    if card_mask[i] <= 0:
                        continue
                    if i in best_indices:
                        biased[i] += HAND_BIAS_STRENGTH
                    else:
                        biased[i] -= HAND_BIAS_STRENGTH * 0.5
            else:
                # For discard: use EV calculator to find best discard strategy
                # This considers flush draws, straight draws, keeping pairs, etc.
                deck_cards = raw_state.get("cards", {}).get("cards", [])
                advice = find_best_discard(hand_cards, deck_cards, jokers_raw, raw_state)
                discard_indices = advice["discard_indices"]

                for i in range(n):
                    if card_mask[i] <= 0:
                        continue
                    if i in discard_indices:
                        biased[i] += HAND_BIAS_STRENGTH  # select for discard
                    else:
                        biased[i] -= HAND_BIAS_STRENGTH * 0.5  # keep
        except Exception:
            pass  # fall back to unbiased logits
        return biased

    def _decode_buy_indexed(self, target_logits: np.ndarray, target_mask: np.ndarray,
                            slot_offset: int, num_slots: int,
                            param_name: str,
                            info: dict) -> tuple[str, Optional[dict], dict]:
        """Decode a buy action — select item by 0-based index.

        Args:
            param_name: "card", "voucher", or "pack" — the API parameter key
        """
        sub_logits = target_logits[slot_offset:slot_offset + num_slots]
        sub_mask = target_mask[slot_offset:slot_offset + num_slots]

        idx = self._masked_select(sub_logits, sub_mask, deterministic=True)
        if idx is None:
            info["error"] = "no_valid_target"
            return "gamestate", None, info

        info["target_idx"] = idx
        return "buy", {param_name: idx}, info

    def _decode_sell_indexed(self, target_logits: np.ndarray, target_mask: np.ndarray,
                             slot_offset: int, num_slots: int,
                             param_name: str,
                             info: dict) -> tuple[str, Optional[dict], dict]:
        """Decode a sell action — select item by 0-based index.

        Args:
            param_name: "joker" or "consumable" — the API parameter key
        """
        sub_logits = target_logits[slot_offset:slot_offset + num_slots]
        sub_mask = target_mask[slot_offset:slot_offset + num_slots]

        idx = self._masked_select(sub_logits, sub_mask, deterministic=True)
        if idx is None:
            info["error"] = "no_valid_target"
            return "gamestate", None, info

        info["target_idx"] = idx
        return "sell", {param_name: idx}, info

    def _decode_use_consumable(self, target_logits: np.ndarray, target_mask: np.ndarray,
                                card_logits: np.ndarray, card_mask: np.ndarray,
                                raw_state: dict, deterministic: bool,
                                info: dict) -> tuple[str, Optional[dict], dict]:
        """Decode using a consumable — select which one, optionally target cards."""
        sub_logits = target_logits[TARGET_CONSUMABLE_OFFSET:TARGET_CONSUMABLE_OFFSET + CONSUMABLE_SLOTS]
        sub_mask = target_mask[TARGET_CONSUMABLE_OFFSET:TARGET_CONSUMABLE_OFFSET + CONSUMABLE_SLOTS]

        idx = self._masked_select(sub_logits, sub_mask, deterministic=True)
        if idx is None:
            info["error"] = "no_valid_consumable"
            return "gamestate", None, info

        info["target_idx"] = idx

        # Some consumables target hand cards (tarots that enhance)
        game_state = raw_state.get("state", "")
        if game_state == "SELECTING_HAND" and card_mask.sum() > 0:
            target_indices = self._select_cards(card_logits, card_mask, deterministic,
                                                min_cards=0, max_cards=5)
            if target_indices:
                info["target_cards"] = target_indices
                return "use", {"consumable": idx, "cards": target_indices}, info

        return "use", {"consumable": idx}, info

    def _decode_pack_select(self, target_logits: np.ndarray, target_mask: np.ndarray,
                            info: dict, raw_state: dict) -> tuple[str, Optional[dict], dict]:
        """Select a card from an opened booster pack by 0-based index."""
        sub_logits = target_logits[TARGET_PACK_CARD_OFFSET:TARGET_PACK_CARD_OFFSET + PACK_CARD_SLOTS]
        sub_mask = target_mask[TARGET_PACK_CARD_OFFSET:TARGET_PACK_CARD_OFFSET + PACK_CARD_SLOTS]

        idx = self._masked_select(sub_logits, sub_mask, deterministic=True)
        if idx is None:
            info["error"] = "no_valid_pack_card"
            return "pack", {"skip": True}, info

        info["target_idx"] = idx
        params = {"card": idx}
        # Tarot/Spectral cards may need hand card targets
        pack_cards = raw_state.get("pack", {}).get("cards", [])
        if pack_cards and idx < len(pack_cards):
            card_set = pack_cards[idx].get("set", "")
            if card_set in ("TAROT", "SPECTRAL", "Tarot", "Spectral"):
                params["targets"] = [0]
        return "pack", params, info

    # --- Selection helpers ---

    def _masked_select(self, logits: np.ndarray, mask: np.ndarray,
                       deterministic: bool) -> Optional[int]:
        """Select one index from masked logits."""
        if mask.sum() == 0:
            return None

        if deterministic:
            # Argmax over valid entries
            masked = np.where(mask > 0, logits, -np.inf)
            return int(np.argmax(masked))
        else:
            # Softmax over valid entries, then sample
            masked = np.where(mask > 0, logits, -np.inf)
            probs = _softmax(masked)
            return int(np.random.choice(len(probs), p=probs))

    def _select_cards(self, card_logits: np.ndarray, card_mask: np.ndarray,
                      deterministic: bool,
                      min_cards: int = 1, max_cards: int = 5) -> list[int]:
        """Select multiple cards using sigmoid thresholding.

        Each card slot has an independent logit. We sigmoid them,
        then either threshold (deterministic) or sample (stochastic).
        Clamp to [min_cards, max_cards].
        """
        # Apply mask
        masked_logits = np.where(card_mask > 0, card_logits, -10.0)
        probs = _sigmoid(masked_logits)

        if deterministic:
            # Select cards with prob > 0.5, sorted by probability descending
            selected = [(i, p) for i, p in enumerate(probs) if p > 0.5 and card_mask[i] > 0]
            selected.sort(key=lambda x: -x[1])
        else:
            # Sample each card independently
            rolls = np.random.random(len(probs))
            selected = [(i, probs[i]) for i in range(len(probs))
                        if rolls[i] < probs[i] and card_mask[i] > 0]
            # Shuffle to avoid position bias in clamping
            np.random.shuffle(selected)

        indices = [i for i, _ in selected]

        # Enforce min/max cards
        if len(indices) < min_cards:
            # Add highest-probability unselected cards
            remaining = [(i, probs[i]) for i in range(len(probs))
                         if i not in indices and card_mask[i] > 0]
            remaining.sort(key=lambda x: -x[1])
            for i, _ in remaining:
                indices.append(i)
                if len(indices) >= min_cards:
                    break

        if len(indices) > max_cards:
            # Keep the max_cards highest-probability ones
            scored = [(i, probs[i]) for i in indices]
            scored.sort(key=lambda x: -x[1])
            indices = [i for i, _ in scored[:max_cards]]

        indices.sort()  # Return in positional order
        return indices


# ============================================================
# Math Helpers
# ============================================================

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    if total == 0:
        # All -inf — uniform over non-inf entries
        valid = logits > -np.inf
        if valid.sum() == 0:
            return np.ones_like(logits) / len(logits)
        return valid.astype(np.float32) / valid.sum()
    return exp / total


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


# ============================================================
# Convenience
# ============================================================

def get_action_space_size() -> int:
    """Total action head output size."""
    return ACTION_HEAD_SIZE


def get_action_type_name(idx: int) -> str:
    """Get human-readable action name."""
    if 0 <= idx < NUM_ACTION_TYPES:
        return ACTION_NAMES[idx]
    return f"unknown_{idx}"
