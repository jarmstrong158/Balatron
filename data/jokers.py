"""
Balatron — Joker Data

Complete hand-coded joker database for the Balatro RL agent.
All 150 jokers encoded against the 93-field property fingerprint schema.

tier_weight values (0.0-10.0) are hand-coded by Jonny.
All other fields derived from game mechanics.

See NOTES.md for full schema reference and design decisions.
"""

from typing import Any


# ============================================================
# Schema Constants
# ============================================================

TRIGGER_VOCABULARY = {
    # Hand-based
    "any_hand_played", "specific_hand_type", "scoring_hand_size",
    "hand_type_minimum", "first_hand_of_round", "final_hand_of_round",
    "round_history_hand",
    # Card event-based
    "specific_rank", "specific_suit", "face_card", "scoring_card",
    "card_held_in_hand", "card_added_to_deck", "card_destroyed",
    "enhancement_based", "retrigger",
    # Action-based
    "on_discard", "rank_specific_discard", "suit_specific_discard",
    "on_blind_selected", "on_blind_skip", "per_blind_skipped_cumulative",
    "on_boss_blind_defeated", "on_ante_up", "on_round_start",
    "end_of_round", "end_of_shop_phase", "on_shop_enter", "on_shop_reroll",
    "on_booster_pack_opened", "on_booster_pack_skipped", "on_card_sold",
    # Economy-based
    "per_dollar_held", "per_dollar_spent",
    # Consumable-based
    "per_tarot_used", "per_planet_used", "per_spectral_used",
    # Joker-based
    "on_joker_destroyed", "per_joker_owned", "per_specific_joker_present",
    # Meta/other
    "per_card_remaining_in_deck", "per_hand_remaining", "periodic",
}

SCALING_METHODS = {"flat_addition", "counter_threshold"}

SCALING_TYPES = {"chips", "mult", "xmult", "economy", "sell_value", "hand_size"}

HAND_TYPES = {
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
    "Royal Flush", "Five of a Kind", "Flush House", "Flush Five",
}

EDITIONS = {"none", "foil", "holographic", "polychrome", "negative"}

RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10",
         "Jack", "Queen", "King", "Ace"}

SUITS = {"Hearts", "Diamonds", "Clubs", "Spades"}

ENHANCEMENTS = {"bonus", "mult", "wild", "glass", "steel",
                "stone", "gold", "lucky"}


# ============================================================
# Defaults — all 93 fields with default values
# ============================================================

JOKER_DEFAULTS: dict[str, Any] = {
    # Top-level
    "name": "",
    "tier_weight": 0.0,
    "edition": "none",

    # Effect type flags
    "economy": False,
    "chip": False,
    "mult": False,
    "chip_scaling": False,
    "mult_scaling": False,
    "xmult": False,
    "xmult_scaling": False,
    "copy": False,
    "in_hand_effect": False,

    # Effect values (static)
    "xmult_value": None,
    "chip_value": None,
    "mult_value": None,
    "money_per_round": None,

    # Trigger system
    "triggers": None,
    "trigger_combination": None,
    "trigger_detail_logic": None,
    "trigger_scope": None,
    "trigger_hand_type": None,
    "trigger_ranks": None,
    "trigger_suits": None,
    "trigger_editions": None,
    "trigger_enhancements": None,
    "discard_trigger_ranks": None,
    "discard_trigger_suits": None,
    "negation_trigger": False,
    "rotating_condition": False,
    "round_history_condition": False,
    "periodic_interval": None,
    "event_count_threshold": None,
    "event_count_comparison": None,
    "trigger_on_probability_success": False,

    # Effect system — Score
    "score_effect": None,
    "scoring_timing": None,
    "effect_target": None,

    # Effect system — Economy
    "economy_effect": False,
    "scaling_economy": False,
    "magnitude_source": None,
    "magnitude_source_detail": None,

    # Effect system — Game state
    "game_parameter_effect": False,
    "acquisition_penalty": False,
    "rule_modification": False,
    "shop_pool_modification": False,
    "shop_cost_modification": False,
    "global_probability_modifier": False,

    # Effect system — Card
    "card_modification_effect": False,
    "destructive_card_modification": False,
    "permanent_deck_scaling": False,
    "spatial_card_condition": False,

    # Effect system — Joker
    "joker_creation_effect": False,
    "forced_joker_destruction": False,
    "copy_target": None,
    "copy_timing_inheritance": None,
    "position_aware": False,
    "position_target": None,

    # Effect system — Consumable
    "consumable_creation": False,
    "consumable_specificity": None,
    "consumable_creation_type": None,

    # Effect system — Hand
    "hand_upgrade_effect": False,
    "hand_upgrade_target": None,
    "hand_subcomposition_condition": False,

    # Effect system — Retrigger
    "retrigger_effect": False,
    "mass_retrigger": False,
    "per_card_instance": False,
    "card_contribution_modification": False,

    # Effect system — Compound/Survival/Probability
    "compound_effect": False,
    "survival_effect": False,
    "boss_blind_effect": False,
    "effect_probability": None,
    "random_range_effect": False,
    "range_min": None,
    "range_max": None,

    # Effect system — Card creation
    "card_creation_effect": False,
    "card_creation_enhancement": None,

    # Scaling block — Core
    "scaling_type": None,
    "scaling_start_value": None,

    # Scaling block — Growth
    "scaling_method": None,
    "scaling_driver": None,
    "scaling_increment": None,
    "scaling_increment_source": None,
    "scaling_increment_multiplier": None,
    "scaling_increment_detail": None,
    "scaling_per_item": False,
    "scaling_threshold": None,

    # Scaling block — Dynamic condition
    "scaling_dynamic_condition": None,

    # Scaling block — Reset
    "scaling_resets": False,
    "scaling_reset_trigger": None,
    "scaling_reset_value": None,

    # Scaling block — Decay
    "scaling_decay": False,
    "scaling_decay_driver": None,
    "scaling_decay_amount": None,
    "scaling_decay_floor": None,
    "scaling_decay_per_item": False,
    "scaling_destroys_at_floor": False,

    # Scaling block — Expiry
    "expiry": False,
    "expiry_type": None,
    "expiry_threshold": None,
    "expiry_probability": None,
    "expiry_check_timing": None,
    "expiry_outcome": None,

    # Scaling block — Target
    "scaling_target": None,
}


# ============================================================
# Helper: create a joker with defaults
# ============================================================

def make_joker(**kwargs: Any) -> dict[str, Any]:
    """Create a joker dict by merging kwargs over defaults.

    Only pass fields that differ from defaults. All 93 fields
    are guaranteed present in the returned dict.
    """
    joker = dict(JOKER_DEFAULTS)
    for key, value in kwargs.items():
        if key not in JOKER_DEFAULTS:
            raise KeyError(f"Unknown joker field: {key}")
        joker[key] = value
    return joker


# ============================================================
# Validation
# ============================================================

def validate_joker(joker: dict[str, Any]) -> list[str]:
    """Validate a single joker dict. Returns list of error strings."""
    errors = []
    name = joker.get("name", "<unnamed>")

    # Required fields
    if not joker.get("name"):
        errors.append(f"{name}: missing name")

    # Trigger vocabulary check
    if joker.get("triggers"):
        for t in joker["triggers"]:
            if t not in TRIGGER_VOCABULARY:
                errors.append(f"{name}: unknown trigger '{t}'")

    # Scaling driver check
    for field in ["scaling_driver", "scaling_decay_driver",
                   "scaling_reset_trigger", "expiry_check_timing"]:
        val = joker.get(field)
        if val and val not in TRIGGER_VOCABULARY:
            errors.append(f"{name}: unknown trigger in {field}: '{val}'")

    # Scaling method check
    if joker.get("scaling_method"):
        if joker["scaling_method"] not in SCALING_METHODS:
            errors.append(f"{name}: unknown scaling_method '{joker['scaling_method']}'")

    # Scaling type check
    if joker.get("scaling_type"):
        if joker["scaling_type"] not in SCALING_TYPES:
            errors.append(f"{name}: unknown scaling_type '{joker['scaling_type']}'")

    # Hand type check
    if joker.get("trigger_hand_type"):
        if joker["trigger_hand_type"] not in HAND_TYPES:
            errors.append(f"{name}: unknown hand type '{joker['trigger_hand_type']}'")

    # Edition check
    if joker.get("edition") and joker["edition"] not in EDITIONS:
        errors.append(f"{name}: unknown edition '{joker['edition']}'")

    # Rank checks
    for field in ["trigger_ranks", "discard_trigger_ranks"]:
        if joker.get(field):
            for r in joker[field]:
                if r not in RANKS:
                    errors.append(f"{name}: unknown rank '{r}' in {field}")

    # Suit checks
    for field in ["trigger_suits", "discard_trigger_suits"]:
        if joker.get(field):
            for s in joker[field]:
                if s not in SUITS:
                    errors.append(f"{name}: unknown suit '{s}' in {field}")

    # Enhancement checks
    if joker.get("trigger_enhancements"):
        for e in joker["trigger_enhancements"]:
            if e not in ENHANCEMENTS:
                errors.append(f"{name}: unknown enhancement '{e}'")

    # Scaling consistency
    if joker.get("scaling_method") and not joker.get("scaling_type"):
        errors.append(f"{name}: has scaling_method but no scaling_type")
    if joker.get("scaling_decay") and not joker.get("scaling_type"):
        errors.append(f"{name}: has scaling_decay but no scaling_type")

    # Expiry consistency
    if joker.get("expiry"):
        if not joker.get("expiry_type"):
            errors.append(f"{name}: expiry=True but no expiry_type")
        if joker.get("expiry_type") == "probabilistic" and not joker.get("expiry_probability"):
            errors.append(f"{name}: probabilistic expiry but no expiry_probability")
        if joker.get("expiry_type") in ("activations", "rounds") and not joker.get("expiry_threshold"):
            errors.append(f"{name}: {joker['expiry_type']} expiry but no expiry_threshold")

    # All fields present
    for key in JOKER_DEFAULTS:
        if key not in joker:
            errors.append(f"{name}: missing field '{key}'")

    return errors


def validate_all(jokers: dict[str, dict]) -> list[str]:
    """Validate all jokers. Returns list of all errors."""
    all_errors = []
    for name, joker in jokers.items():
        all_errors.extend(validate_joker(joker))
    return all_errors


# ============================================================
# JOKER DATABASE — All 150 jokers
# ============================================================

JOKERS: dict[str, dict[str, Any]] = {

    # ========================================================
    # COMMON JOKERS (1-60)
    # ========================================================

    "Joker": make_joker(
        name="Joker",
        mult=True,
        mult_value=4.0,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Greedy Joker": make_joker(
        name="Greedy Joker",
        mult=True,
        mult_value=3.0,
        triggers=["specific_suit"],
        trigger_suits=["Diamonds"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Lusty Joker": make_joker(
        name="Lusty Joker",
        mult=True,
        mult_value=3.0,
        triggers=["specific_suit"],
        trigger_suits=["Hearts"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Wrathful Joker": make_joker(
        name="Wrathful Joker",
        mult=True,
        mult_value=3.0,
        triggers=["specific_suit"],
        trigger_suits=["Spades"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Gluttonous Joker": make_joker(
        name="Gluttonous Joker",
        mult=True,
        mult_value=3.0,
        triggers=["specific_suit"],
        trigger_suits=["Clubs"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Jolly Joker": make_joker(
        name="Jolly Joker",
        mult=True,
        mult_value=8.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Pair",
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Zany Joker": make_joker(
        name="Zany Joker",
        mult=True,
        mult_value=12.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Three of a Kind",
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Mad Joker": make_joker(
        name="Mad Joker",
        mult=True,
        mult_value=10.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Two Pair",
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Crazy Joker": make_joker(
        name="Crazy Joker",
        mult=True,
        mult_value=12.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Straight",
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Droll Joker": make_joker(
        name="Droll Joker",
        mult=True,
        mult_value=10.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Flush",
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Sly Joker": make_joker(
        name="Sly Joker",
        chip=True,
        chip_value=50.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Pair",
        score_effect=["chips"],
        scoring_timing="after_cards",
    ),

    "Wily Joker": make_joker(
        name="Wily Joker",
        chip=True,
        chip_value=100.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Three of a Kind",
        score_effect=["chips"],
        scoring_timing="after_cards",
    ),

    "Clever Joker": make_joker(
        name="Clever Joker",
        chip=True,
        chip_value=80.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Two Pair",
        score_effect=["chips"],
        scoring_timing="after_cards",
    ),

    "Devious Joker": make_joker(
        name="Devious Joker",
        chip=True,
        chip_value=100.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Straight",
        score_effect=["chips"],
        scoring_timing="after_cards",
    ),

    "Crafty Joker": make_joker(
        name="Crafty Joker",
        chip=True,
        chip_value=80.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Flush",
        score_effect=["chips"],
        scoring_timing="after_cards",
    ),

    "Half Joker": make_joker(
        name="Half Joker",
        mult=True,
        mult_value=20.0,
        triggers=["scoring_hand_size"],
        event_count_threshold=3,
        event_count_comparison="maximum",
        score_effect=["mult"],
        scoring_timing="after_cards",
    ),

    "Credit Card": make_joker(
        name="Credit Card",
        economy=True,
        rule_modification=True,
        # Passive: allows up to -$20 debt. No trigger.
    ),

    "Banner": make_joker(
        name="Banner",
        chip=True,
        chip_value=30.0,
        triggers=["any_hand_played"],
        score_effect=["chips"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="discards_remaining",
    ),

    "Mystic Summit": make_joker(
        name="Mystic Summit",
        mult=True,
        mult_value=15.0,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="discards_remaining",
        # Condition: 0 discards remaining
        event_count_threshold=0,
        event_count_comparison="exact",
    ),

    "8 Ball": make_joker(
        name="8 Ball",
        triggers=["specific_rank"],
        trigger_ranks=["8"],
        per_card_instance=True,
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Tarot",
        effect_probability=0.25,
    ),

    "Misprint": make_joker(
        name="Misprint",
        mult=True,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        random_range_effect=True,
        range_min=0.0,
        range_max=23.0,
    ),

    "Raised Fist": make_joker(
        name="Raised Fist",
        mult=True,
        in_hand_effect=True,
        triggers=["card_held_in_hand"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="lowest_held_card_rank",
        # Adds double the rank of lowest held card to Mult
    ),

    "Chaos the Clown": make_joker(
        name="Chaos the Clown",
        economy=True,
        shop_cost_modification=True,
        # Passive: 1 free reroll per shop
    ),

    "Scary Face": make_joker(
        name="Scary Face",
        chip=True,
        chip_value=30.0,
        triggers=["face_card"],
        score_effect=["chips"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Abstract Joker": make_joker(
        name="Abstract Joker",
        mult=True,
        mult_value=3.0,
        triggers=["per_joker_owned"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="joker_count",
    ),

    "Delayed Gratification": make_joker(
        name="Delayed Gratification",
        economy=True,
        economy_effect=True,
        triggers=["end_of_round"],
        # $2 per unused discard if no discards used
        magnitude_source="game_state",
        magnitude_source_detail="discards_remaining",
    ),

    "Hack": make_joker(
        name="Hack",
        retrigger_effect=True,
        triggers=["specific_rank"],
        trigger_ranks=["2", "3", "4", "5"],
        trigger_detail_logic="any",
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Pareidolia": make_joker(
        name="Pareidolia",
        rule_modification=True,
        # All cards are considered face cards
    ),

    "Gros Michel": make_joker(
        name="Gros Michel",
        mult=True,
        mult_value=15.0,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        expiry=True,
        expiry_type="probabilistic",
        expiry_probability=1.0 / 6.0,
        expiry_check_timing="end_of_round",
        expiry_outcome="destroyed",
    ),

    "Even Steven": make_joker(
        name="Even Steven",
        mult=True,
        mult_value=4.0,
        triggers=["specific_rank"],
        trigger_ranks=["10", "8", "6", "4", "2"],
        trigger_detail_logic="any",
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Odd Todd": make_joker(
        name="Odd Todd",
        chip=True,
        chip_value=31.0,
        triggers=["specific_rank"],
        trigger_ranks=["Ace", "9", "7", "5", "3"],
        trigger_detail_logic="any",
        score_effect=["chips"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Scholar": make_joker(
        name="Scholar",
        chip=True,
        mult=True,
        chip_value=20.0,
        mult_value=4.0,
        triggers=["specific_rank"],
        trigger_ranks=["Ace"],
        score_effect=["chips_and_mult"],
        scoring_timing="during_card",
        per_card_instance=True,
        compound_effect=True,
    ),

    "Business Card": make_joker(
        name="Business Card",
        economy=True,
        economy_effect=True,
        triggers=["face_card"],
        scoring_timing="during_card",
        per_card_instance=True,
        effect_probability=0.5,
        # $2 per face card with 1 in 2 chance
    ),

    "Supernova": make_joker(
        name="Supernova",
        mult=True,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="run_history",
        magnitude_source_detail="times_hand_type_played",
    ),

    "Ride the Bus": make_joker(
        name="Ride the Bus",
        mult=True,
        mult_scaling=True,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        negation_trigger=True,  # grows when NO face card
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="any_hand_played",
        scaling_increment=1.0,
        scaling_increment_source="static",
        scaling_resets=True,
        scaling_reset_trigger="face_card",
        scaling_reset_value=0.0,
        scaling_target="self",
    ),

    "Space Joker": make_joker(
        name="Space Joker",
        triggers=["any_hand_played"],
        hand_upgrade_effect=True,
        hand_upgrade_target="any",
        effect_probability=0.25,
    ),

    "Egg": make_joker(
        name="Egg",
        economy=True,
        scaling_type="sell_value",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="end_of_round",
        scaling_increment=3.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Burglar": make_joker(
        name="Burglar",
        game_parameter_effect=True,
        triggers=["on_blind_selected"],
        # +3 Hands, lose all discards
    ),

    "Runner": make_joker(
        name="Runner",
        chip=True,
        chip_scaling=True,
        triggers=["specific_hand_type"],
        trigger_hand_type="Straight",
        score_effect=["chips"],
        scoring_timing="after_cards",
        scaling_type="chips",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="specific_hand_type",
        scaling_increment=15.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Ice Cream": make_joker(
        name="Ice Cream",
        chip=True,
        chip_scaling=True,
        triggers=["any_hand_played"],
        score_effect=["chips"],
        scoring_timing="after_cards",
        scaling_type="chips",
        scaling_start_value=100.0,
        scaling_decay=True,
        scaling_decay_driver="any_hand_played",
        scaling_decay_amount=5.0,
        scaling_decay_floor=0.0,
        scaling_destroys_at_floor=True,
        scaling_target="self",
    ),

    "Splash": make_joker(
        name="Splash",
        rule_modification=True,
        # All played cards count in scoring
    ),

    "Blue Joker": make_joker(
        name="Blue Joker",
        chip=True,
        chip_value=2.0,
        triggers=["per_card_remaining_in_deck"],
        score_effect=["chips"],
        scoring_timing="after_cards",
        magnitude_source="deck_property",
        magnitude_source_detail="cards_remaining",
    ),

    "Fibonacci": make_joker(
        name="Fibonacci",
        mult=True,
        mult_value=8.0,
        triggers=["specific_rank"],
        trigger_ranks=["Ace", "2", "3", "5", "8"],
        trigger_detail_logic="any",
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Hiker": make_joker(
        name="Hiker",
        chip=True,
        triggers=["scoring_card"],
        scoring_timing="during_card",
        per_card_instance=True,
        permanent_deck_scaling=True,
        card_modification_effect=True,
        # +5 Chips permanently to each scored card
    ),

    "Faceless Joker": make_joker(
        name="Faceless Joker",
        economy=True,
        economy_effect=True,
        triggers=["on_discard", "face_card"],
        trigger_combination="all",
        event_count_threshold=3,
        event_count_comparison="minimum",
        # $5 if 3+ face cards discarded at once
    ),

    "Green Joker": make_joker(
        name="Green Joker",
        mult=True,
        mult_scaling=True,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="any_hand_played",
        scaling_increment=1.0,
        scaling_increment_source="static",
        scaling_decay=True,
        scaling_decay_driver="on_discard",
        scaling_decay_amount=1.0,
        scaling_decay_floor=0.0,
        scaling_destroys_at_floor=False,
        scaling_target="self",
    ),

    "Superposition": make_joker(
        name="Superposition",
        triggers=["specific_rank", "specific_hand_type"],
        trigger_combination="all",
        trigger_ranks=["Ace"],
        trigger_hand_type="Straight",
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Tarot",
    ),

    "To Do List": make_joker(
        name="To Do List",
        economy=True,
        economy_effect=True,
        triggers=["specific_hand_type"],
        rotating_condition=True,
        # $4 if hand matches random target
    ),

    "Cavendish": make_joker(
        name="Cavendish",
        xmult=True,
        xmult_value=3.0,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        expiry=True,
        expiry_type="probabilistic",
        expiry_probability=0.001,
        expiry_check_timing="end_of_round",
        expiry_outcome="destroyed",
    ),

    "Red Card": make_joker(
        name="Red Card",
        mult=True,
        mult_scaling=True,
        triggers=["on_booster_pack_skipped"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="on_booster_pack_skipped",
        scaling_increment=3.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Square Joker": make_joker(
        name="Square Joker",
        chip=True,
        chip_scaling=True,
        triggers=["scoring_hand_size"],
        event_count_threshold=4,
        event_count_comparison="exact",
        score_effect=["chips"],
        scoring_timing="after_cards",
        scaling_type="chips",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="scoring_hand_size",
        scaling_increment=4.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Stone Joker": make_joker(
        name="Stone Joker",
        chip=True,
        chip_value=25.0,
        triggers=["any_hand_played"],
        score_effect=["chips"],
        scoring_timing="after_cards",
        magnitude_source="deck_property",
        magnitude_source_detail="stone_card_count",
    ),

    "Golden Joker": make_joker(
        name="Golden Joker",
        economy=True,
        economy_effect=True,
        money_per_round=4.0,
        triggers=["end_of_round"],
    ),

    "Bull": make_joker(
        name="Bull",
        chip=True,
        chip_value=2.0,
        triggers=["per_dollar_held"],
        score_effect=["chips"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="dollars",
    ),

    "Flash Card": make_joker(
        name="Flash Card",
        mult=True,
        mult_scaling=True,
        triggers=["on_shop_reroll"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="on_shop_reroll",
        scaling_increment=2.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Popcorn": make_joker(
        name="Popcorn",
        mult=True,
        mult_scaling=True,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        scaling_type="mult",
        scaling_start_value=20.0,
        scaling_decay=True,
        scaling_decay_driver="end_of_round",
        scaling_decay_amount=4.0,
        scaling_decay_floor=0.0,
        scaling_destroys_at_floor=True,
        scaling_target="self",
    ),

    "Walkie Talkie": make_joker(
        name="Walkie Talkie",
        chip=True,
        mult=True,
        chip_value=10.0,
        mult_value=4.0,
        triggers=["specific_rank"],
        trigger_ranks=["10", "4"],
        trigger_detail_logic="any",
        score_effect=["chips_and_mult"],
        scoring_timing="during_card",
        per_card_instance=True,
        compound_effect=True,
    ),

    "Smiley Face": make_joker(
        name="Smiley Face",
        mult=True,
        mult_value=4.0,
        triggers=["face_card"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Golden Ticket": make_joker(
        name="Golden Ticket",
        economy=True,
        economy_effect=True,
        triggers=["enhancement_based"],
        trigger_enhancements=["gold"],
        scoring_timing="during_card",
        per_card_instance=True,
        # $4 per Gold card scored
    ),

    "Swashbuckler": make_joker(
        name="Swashbuckler",
        mult=True,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="joker_sell_value",
        magnitude_source_detail="all_other_jokers",
    ),

    # ========================================================
    # UNCOMMON JOKERS (61-126)
    # ========================================================

    "Joker Stencil": make_joker(
        name="Joker Stencil",
        xmult=True,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="empty_joker_slots",
    ),

    "Four Fingers": make_joker(
        name="Four Fingers",
        rule_modification=True,
        # Flushes and Straights with 4 cards
    ),

    "Mime": make_joker(
        name="Mime",
        retrigger_effect=True,
        in_hand_effect=True,
        triggers=["card_held_in_hand"],
        mass_retrigger=True,
    ),

    "Ceremonial Dagger": make_joker(
        name="Ceremonial Dagger",
        mult=True,
        mult_scaling=True,
        triggers=["on_blind_selected"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        forced_joker_destruction=True,
        position_aware=True,
        position_target="right",
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="on_blind_selected",
        scaling_increment_source="derived",
        scaling_increment_multiplier=2.0,
        scaling_increment_detail="sell_value_of_destroyed_joker",
        scaling_target="self",
    ),

    "Marble Joker": make_joker(
        name="Marble Joker",
        triggers=["on_blind_selected"],
        card_creation_effect=True,
        card_creation_enhancement="stone",
    ),

    "Loyalty Card": make_joker(
        name="Loyalty Card",
        xmult=True,
        xmult_value=4.0,
        triggers=["periodic"],
        periodic_interval=6,
        scaling_threshold=6,
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "Dusk": make_joker(
        name="Dusk",
        retrigger_effect=True,
        mass_retrigger=True,
        triggers=["final_hand_of_round"],
        scoring_timing="during_card",
    ),

    "Steel Joker": make_joker(
        name="Steel Joker",
        xmult=True,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        magnitude_source="deck_property",
        magnitude_source_detail="steel_card_count",
        # X0.2 per Steel Card added to base X1
    ),

    "Blackboard": make_joker(
        name="Blackboard",
        xmult=True,
        xmult_value=3.0,
        in_hand_effect=True,
        triggers=["card_held_in_hand"],
        trigger_suits=["Spades", "Clubs"],
        trigger_detail_logic="any",
        trigger_scope="all_cards",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "DNA": make_joker(
        name="DNA",
        triggers=["first_hand_of_round"],
        event_count_threshold=1,
        event_count_comparison="exact",
        card_creation_effect=True,
        # If first hand is single card, copy to deck
    ),

    "Sixth Sense": make_joker(
        name="Sixth Sense",
        triggers=["first_hand_of_round", "specific_rank"],
        trigger_combination="all",
        trigger_ranks=["6"],
        event_count_threshold=1,
        event_count_comparison="exact",
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Spectral",
        destructive_card_modification=True,
    ),

    "Constellation": make_joker(
        name="Constellation",
        xmult=True,
        xmult_scaling=True,
        triggers=["per_planet_used"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="per_planet_used",
        scaling_increment=0.1,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Card Sharp": make_joker(
        name="Card Sharp",
        xmult=True,
        xmult_value=3.0,
        triggers=["specific_hand_type"],
        round_history_condition=True,
        score_effect=["xmult"],
        scoring_timing="after_cards",
        # X3 if hand type already played this round
    ),

    "Madness": make_joker(
        name="Madness",
        xmult=True,
        xmult_scaling=True,
        triggers=["on_blind_selected"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        forced_joker_destruction=True,
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="on_blind_selected",
        scaling_increment=0.5,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Vampire": make_joker(
        name="Vampire",
        xmult=True,
        xmult_scaling=True,
        triggers=["enhancement_based"],
        score_effect=["xmult"],
        scoring_timing="during_card",
        per_card_instance=True,
        destructive_card_modification=True,
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="enhancement_based",
        scaling_increment=0.1,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Shortcut": make_joker(
        name="Shortcut",
        rule_modification=True,
        # Straights with gaps of 1 rank
    ),

    "Hologram": make_joker(
        name="Hologram",
        xmult=True,
        xmult_scaling=True,
        triggers=["card_added_to_deck"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="card_added_to_deck",
        scaling_increment=0.25,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Vagabond": make_joker(
        name="Vagabond",
        triggers=["any_hand_played"],
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Tarot",
        magnitude_source="game_state",
        magnitude_source_detail="dollars",
        # Creates Tarot when hand played with $4 or less
    ),

    "Baron": make_joker(
        name="Baron",
        xmult=True,
        xmult_value=1.5,
        in_hand_effect=True,
        triggers=["card_held_in_hand", "specific_rank"],
        trigger_combination="all",
        trigger_ranks=["King"],
        score_effect=["xmult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Cloud 9": make_joker(
        name="Cloud 9",
        economy=True,
        economy_effect=True,
        triggers=["end_of_round"],
        magnitude_source="deck_property",
        magnitude_source_detail="nines_in_deck",
        # $1 per 9 in full deck
    ),

    "Rocket": make_joker(
        name="Rocket",
        economy=True,
        economy_effect=True,
        scaling_economy=True,
        money_per_round=1.0,
        triggers=["end_of_round"],
        scaling_type="economy",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="on_boss_blind_defeated",
        scaling_increment=2.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Obelisk": make_joker(
        name="Obelisk",
        xmult=True,
        xmult_scaling=True,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        negation_trigger=True,
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="any_hand_played",
        scaling_increment=0.2,
        scaling_increment_source="static",
        scaling_dynamic_condition="most_played_hand_type",
        scaling_resets=True,
        scaling_reset_trigger="specific_hand_type",
        scaling_reset_value=1.0,
        scaling_target="self",
    ),

    "Midas Mask": make_joker(
        name="Midas Mask",
        triggers=["face_card"],
        scoring_timing="during_card",
        per_card_instance=True,
        card_modification_effect=True,
        # Played face cards become Gold
    ),

    "Luchador": make_joker(
        name="Luchador",
        boss_blind_effect=True,
        # Sell to disable current Boss Blind
    ),

    "Photograph": make_joker(
        name="Photograph",
        xmult=True,
        xmult_value=2.0,
        triggers=["face_card"],
        score_effect=["xmult"],
        scoring_timing="during_card",
        # First played face card only
    ),

    "Gift Card": make_joker(
        name="Gift Card",
        economy=True,
        triggers=["end_of_round"],
        scaling_type="sell_value",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="end_of_round",
        scaling_increment=1.0,
        scaling_increment_source="static",
        scaling_target="all_jokers_and_consumables",
    ),

    "Turtle Bean": make_joker(
        name="Turtle Bean",
        game_parameter_effect=True,
        scaling_type="hand_size",
        scaling_start_value=5.0,
        scaling_decay=True,
        scaling_decay_driver="end_of_round",
        scaling_decay_amount=1.0,
        scaling_decay_floor=0.0,
        scaling_destroys_at_floor=True,
        scaling_target="self",
    ),

    "Erosion": make_joker(
        name="Erosion",
        mult=True,
        mult_value=4.0,
        triggers=["any_hand_played"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="deck_property",
        magnitude_source_detail="cards_below_starting_size",
    ),

    "Reserved Parking": make_joker(
        name="Reserved Parking",
        economy=True,
        economy_effect=True,
        in_hand_effect=True,
        triggers=["card_held_in_hand", "face_card"],
        trigger_combination="all",
        per_card_instance=True,
        effect_probability=0.5,
        # $1 per face card held with 1 in 2 chance
    ),

    "Mail-In Rebate": make_joker(
        name="Mail-In Rebate",
        economy=True,
        economy_effect=True,
        triggers=["rank_specific_discard"],
        rotating_condition=True,
        per_card_instance=True,
        # $5 per discarded card of target rank
    ),

    "To the Moon": make_joker(
        name="To the Moon",
        economy=True,
        economy_effect=True,
        triggers=["end_of_round"],
        magnitude_source="game_state",
        magnitude_source_detail="dollars",
        # +$1 interest per $5
    ),

    "Hallucination": make_joker(
        name="Hallucination",
        triggers=["on_booster_pack_opened"],
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Tarot",
        effect_probability=0.5,
    ),

    "Fortune Teller": make_joker(
        name="Fortune Teller",
        mult=True,
        mult_scaling=True,
        triggers=["per_tarot_used"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="per_tarot_used",
        scaling_increment=1.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Juggler": make_joker(
        name="Juggler",
        game_parameter_effect=True,
        # +1 hand size
    ),

    "Drunkard": make_joker(
        name="Drunkard",
        game_parameter_effect=True,
        # +1 discard
    ),

    "Lucky Cat": make_joker(
        name="Lucky Cat",
        xmult=True,
        xmult_scaling=True,
        triggers=["enhancement_based"],
        trigger_enhancements=["lucky"],
        trigger_on_probability_success=True,
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="enhancement_based",
        scaling_increment=0.25,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Baseball Card": make_joker(
        name="Baseball Card",
        xmult=True,
        xmult_value=1.5,
        triggers=["per_specific_joker_present"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        # X1.5 per Uncommon joker owned
    ),

    "Diet Cola": make_joker(
        name="Diet Cola",
        economy=True,
        # Sell to create free Double Tag
    ),

    "Trading Card": make_joker(
        name="Trading Card",
        economy=True,
        economy_effect=True,
        triggers=["on_discard", "first_hand_of_round"],
        trigger_combination="all",
        event_count_threshold=1,
        event_count_comparison="exact",
        destructive_card_modification=True,
        # Destroy first single-card discard, earn $3
    ),

    "Spare Trousers": make_joker(
        name="Spare Trousers",
        mult=True,
        mult_scaling=True,
        triggers=["specific_hand_type"],
        trigger_hand_type="Two Pair",
        score_effect=["mult"],
        scoring_timing="after_cards",
        scaling_type="mult",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="specific_hand_type",
        scaling_increment=2.0,
        scaling_increment_source="static",
        scaling_target="self",
    ),

    "Ancient Joker": make_joker(
        name="Ancient Joker",
        xmult=True,
        xmult_value=1.5,
        triggers=["specific_suit"],
        rotating_condition=True,
        score_effect=["xmult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Ramen": make_joker(
        name="Ramen",
        xmult=True,
        xmult_scaling=True,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=2.0,
        scaling_decay=True,
        scaling_decay_driver="on_discard",
        scaling_decay_amount=0.01,
        scaling_decay_floor=1.0,
        scaling_decay_per_item=True,
        scaling_destroys_at_floor=True,
        scaling_target="self",
    ),

    "Seltzer": make_joker(
        name="Seltzer",
        retrigger_effect=True,
        mass_retrigger=True,
        triggers=["any_hand_played"],
        scoring_timing="during_card",
        expiry=True,
        expiry_type="activations",
        expiry_threshold=10,
        expiry_check_timing="any_hand_played",
        expiry_outcome="destroyed",
    ),

    "Castle": make_joker(
        name="Castle",
        chip=True,
        chip_scaling=True,
        triggers=["suit_specific_discard"],
        rotating_condition=True,
        score_effect=["chips"],
        scoring_timing="after_cards",
        scaling_type="chips",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="suit_specific_discard",
        scaling_increment=3.0,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Campfire": make_joker(
        name="Campfire",
        xmult=True,
        xmult_scaling=True,
        triggers=["on_card_sold"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="on_card_sold",
        scaling_increment=0.25,
        scaling_increment_source="static",
        scaling_resets=True,
        scaling_reset_trigger="on_boss_blind_defeated",
        scaling_reset_value=1.0,
        scaling_target="self",
    ),

    "Acrobat": make_joker(
        name="Acrobat",
        xmult=True,
        xmult_value=3.0,
        triggers=["final_hand_of_round"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "Sock and Buskin": make_joker(
        name="Sock and Buskin",
        retrigger_effect=True,
        triggers=["face_card"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Troubadour": make_joker(
        name="Troubadour",
        game_parameter_effect=True,
        acquisition_penalty=True,
        # +2 hand size, -1 hand per round
    ),

    "Certificate": make_joker(
        name="Certificate",
        triggers=["on_round_start"],
        card_creation_effect=True,
        # Random card with random seal added to hand
    ),

    "Smeared Joker": make_joker(
        name="Smeared Joker",
        rule_modification=True,
        # Hearts=Diamonds, Spades=Clubs
    ),

    "Throwback": make_joker(
        name="Throwback",
        xmult=True,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        magnitude_source="run_history",
        magnitude_source_detail="blinds_skipped",
        # X0.25 per blind skipped this run
    ),

    "Hanging Chad": make_joker(
        name="Hanging Chad",
        retrigger_effect=True,
        triggers=["scoring_card"],
        scoring_timing="during_card",
        # Retrigger first scoring card 2 additional times
    ),

    "Rough Gem": make_joker(
        name="Rough Gem",
        economy=True,
        economy_effect=True,
        triggers=["specific_suit"],
        trigger_suits=["Diamonds"],
        scoring_timing="during_card",
        per_card_instance=True,
        # $1 per Diamond scored
    ),

    "Bloodstone": make_joker(
        name="Bloodstone",
        xmult=True,
        xmult_value=1.5,
        triggers=["specific_suit"],
        trigger_suits=["Hearts"],
        score_effect=["xmult"],
        scoring_timing="during_card",
        per_card_instance=True,
        effect_probability=0.333,
    ),

    "Arrowhead": make_joker(
        name="Arrowhead",
        chip=True,
        chip_value=50.0,
        triggers=["specific_suit"],
        trigger_suits=["Spades"],
        score_effect=["chips"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Onyx Agate": make_joker(
        name="Onyx Agate",
        mult=True,
        mult_value=7.0,
        triggers=["specific_suit"],
        trigger_suits=["Clubs"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Glass Joker": make_joker(
        name="Glass Joker",
        xmult=True,
        xmult_scaling=True,
        triggers=["card_destroyed"],
        trigger_enhancements=["glass"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="card_destroyed",
        scaling_increment=0.75,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Showman": make_joker(
        name="Showman",
        shop_pool_modification=True,
        # Cards may appear multiple times in shop
    ),

    "Flower Pot": make_joker(
        name="Flower Pot",
        xmult=True,
        xmult_value=3.0,
        triggers=["specific_suit"],
        trigger_suits=["Diamonds", "Clubs", "Hearts", "Spades"],
        trigger_detail_logic="all",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "Wee Joker": make_joker(
        name="Wee Joker",
        chip=True,
        chip_scaling=True,
        triggers=["specific_rank"],
        trigger_ranks=["2"],
        score_effect=["chips"],
        scoring_timing="during_card",
        per_card_instance=True,
        scaling_type="chips",
        scaling_start_value=0.0,
        scaling_method="flat_addition",
        scaling_driver="specific_rank",
        scaling_increment=8.0,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Merry Andy": make_joker(
        name="Merry Andy",
        game_parameter_effect=True,
        acquisition_penalty=True,
        # +3 discards, -1 hand size
    ),

    "Seeing Double": make_joker(
        name="Seeing Double",
        xmult=True,
        xmult_value=2.0,
        triggers=["specific_suit"],
        trigger_suits=["Clubs"],
        hand_subcomposition_condition=True,
        score_effect=["xmult"],
        scoring_timing="after_cards",
        # Club + non-Club scoring card required
    ),

    "The Idol": make_joker(
        name="The Idol",
        xmult=True,
        xmult_value=2.0,
        triggers=["specific_rank", "specific_suit"],
        trigger_combination="all",
        rotating_condition=True,
        score_effect=["xmult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Matador": make_joker(
        name="Matador",
        economy=True,
        economy_effect=True,
        boss_blind_effect=True,
        triggers=["any_hand_played"],
        # $8 if hand triggers Boss Blind ability
    ),

    "Hit the Road": make_joker(
        name="Hit the Road",
        xmult=True,
        xmult_scaling=True,
        triggers=["rank_specific_discard"],
        discard_trigger_ranks=["Jack"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="rank_specific_discard",
        scaling_increment=0.5,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_resets=True,
        scaling_reset_trigger="on_round_start",
        scaling_reset_value=1.0,
        scaling_target="self",
    ),

    "Stuntman": make_joker(
        name="Stuntman",
        chip=True,
        chip_value=250.0,
        triggers=["any_hand_played"],
        score_effect=["chips"],
        scoring_timing="after_cards",
        game_parameter_effect=True,
        acquisition_penalty=True,
        # -2 hand size
    ),

    # ========================================================
    # RARE JOKERS (127-145)
    # ========================================================

    "The Duo": make_joker(
        name="The Duo",
        xmult=True,
        xmult_value=2.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Pair",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "The Trio": make_joker(
        name="The Trio",
        xmult=True,
        xmult_value=3.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Three of a Kind",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "The Family": make_joker(
        name="The Family",
        xmult=True,
        xmult_value=4.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Four of a Kind",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "The Order": make_joker(
        name="The Order",
        xmult=True,
        xmult_value=3.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Straight",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "The Tribe": make_joker(
        name="The Tribe",
        xmult=True,
        xmult_value=2.0,
        triggers=["specific_hand_type"],
        trigger_hand_type="Flush",
        score_effect=["xmult"],
        scoring_timing="after_cards",
    ),

    "Blueprint": make_joker(
        name="Blueprint",
        copy=True,
        copy_target="right",
        position_aware=True,
        position_target="right",
    ),

    "Brainstorm": make_joker(
        name="Brainstorm",
        copy=True,
        copy_target="left",
        position_aware=True,
        position_target="left",
        # Copies leftmost joker
    ),

    "Invisible Joker": make_joker(
        name="Invisible Joker",
        expiry=True,
        expiry_type="rounds",
        expiry_threshold=2,
        expiry_check_timing="end_of_round",
        expiry_outcome="transforms",
        # After 2 rounds, sell to duplicate a random joker
    ),

    "Satellite": make_joker(
        name="Satellite",
        economy=True,
        economy_effect=True,
        triggers=["end_of_round"],
        magnitude_source="run_history",
        magnitude_source_detail="unique_planets_used",
        # $1 per unique Planet used this run
    ),

    "Shoot the Moon": make_joker(
        name="Shoot the Moon",
        mult=True,
        mult_value=13.0,
        in_hand_effect=True,
        triggers=["card_held_in_hand", "specific_rank"],
        trigger_combination="all",
        trigger_ranks=["Queen"],
        score_effect=["mult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Driver's License": make_joker(
        name="Driver's License",
        xmult=True,
        xmult_value=3.0,
        triggers=["any_hand_played"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        magnitude_source="deck_property",
        magnitude_source_detail="enhanced_card_count",
        event_count_threshold=16,
        event_count_comparison="minimum",
    ),

    "Cartomancer": make_joker(
        name="Cartomancer",
        triggers=["on_blind_selected"],
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Tarot",
    ),

    "Astronomer": make_joker(
        name="Astronomer",
        economy=True,
        shop_cost_modification=True,
        # All Planet cards and Celestial Packs free
    ),

    "Burnt Joker": make_joker(
        name="Burnt Joker",
        triggers=["on_discard"],
        hand_upgrade_effect=True,
        hand_upgrade_target="specific",
        # Upgrades level of first discarded poker hand each round
    ),

    "Bootstraps": make_joker(
        name="Bootstraps",
        mult=True,
        mult_value=2.0,
        triggers=["per_dollar_held"],
        score_effect=["mult"],
        scoring_timing="after_cards",
        magnitude_source="game_state",
        magnitude_source_detail="dollars",
        # +2 Mult per $5
    ),

    "Mr. Bones": make_joker(
        name="Mr. Bones",
        survival_effect=True,
        # Prevents death at 25% chips. Self-destructs.
    ),

    "Oops! All 6s": make_joker(
        name="Oops! All 6s",
        global_probability_modifier=True,
        # Doubles all listed probabilities
    ),

    "Riff-Raff": make_joker(
        name="Riff-Raff",
        triggers=["on_blind_selected"],
        joker_creation_effect=True,
        # Create 2 Common Jokers
    ),

    "Seance": make_joker(
        name="Seance",
        triggers=["specific_hand_type"],
        trigger_hand_type="Straight Flush",
        consumable_creation=True,
        consumable_specificity="random",
        consumable_creation_type="Spectral",
    ),

    # ========================================================
    # LEGENDARY JOKERS (146-150)
    # ========================================================

    "Canio": make_joker(
        name="Canio",
        xmult=True,
        xmult_scaling=True,
        triggers=["card_destroyed", "face_card"],
        trigger_combination="all",
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="flat_addition",
        scaling_driver="card_destroyed",
        scaling_increment=1.0,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Triboulet": make_joker(
        name="Triboulet",
        xmult=True,
        xmult_value=2.0,
        triggers=["specific_rank"],
        trigger_ranks=["King", "Queen"],
        trigger_detail_logic="any",
        score_effect=["xmult"],
        scoring_timing="during_card",
        per_card_instance=True,
    ),

    "Yorick": make_joker(
        name="Yorick",
        xmult=True,
        xmult_scaling=True,
        triggers=["on_discard"],
        score_effect=["xmult"],
        scoring_timing="after_cards",
        scaling_type="xmult",
        scaling_start_value=1.0,
        scaling_method="counter_threshold",
        scaling_driver="on_discard",
        scaling_threshold=23,
        scaling_increment=1.0,
        scaling_increment_source="static",
        scaling_per_item=True,
        scaling_target="self",
    ),

    "Chicot": make_joker(
        name="Chicot",
        boss_blind_effect=True,
        # Disables all Boss Blind effects
    ),

    "Perkeo": make_joker(
        name="Perkeo",
        triggers=["end_of_shop_phase"],
        consumable_creation=True,
        consumable_specificity="random",
        # Creates Negative copy of 1 random consumable
    ),
}


# ============================================================
# Tier Weights — Phase 1 (General Competence)
#
# Scale: 0.0-10.0. Based on Mobalytics Gold Stake tier list
# with domain adjustments. Naneinf-specific bumps deferred
# to Phase 2 fine-tuning.
#
# S+ = 10, S = 8, A = 6, B = 4, C = 2
# ============================================================

TIER_WEIGHTS: dict[str, float] = {
    # --- 10.0 (S+) --- Copy engines, legendary multiplier
    "Blueprint": 10.0,
    "Brainstorm": 10.0,
    "Triboulet": 10.0,

    # --- 8.0 (S) --- Run-defining scaling / xmult
    "Vampire": 8.0,
    "Cavendish": 8.0,
    "The Duo": 8.0,
    "The Trio": 8.0,
    "The Family": 8.0,
    "Spare Trousers": 8.0,
    "Canio": 8.0,
    "Campfire": 8.0,

    # --- 7.0 --- Legendaries not in Mobalytics S+
    "Chicot": 7.0,

    # --- 6.0 (A) --- Strong scalers, economy, solid xmult
    "Hiker": 6.0,
    "Fortune Teller": 6.0,
    "Rocket": 6.0,
    "Seltzer": 6.0,
    "Trading Card": 6.0,
    "Bloodstone": 6.0,
    "Perkeo": 6.0,
    "Fibonacci": 6.0,
    "Onyx Agate": 6.0,
    "Arrowhead": 6.0,
    "Sixth Sense": 6.0,
    "Space Joker": 6.0,
    "Burnt Joker": 6.0,
    "Hologram": 6.0,
    "Driver's License": 6.0,
    "Steel Joker": 6.0,
    "Ancient Joker": 6.0,
    "Card Sharp": 6.0,
    "Baseball Card": 6.0,
    "To Do List": 6.0,
    "Business Card": 6.0,
    "Mail-In Rebate": 6.0,
    "Cloud 9": 6.0,
    "Golden Joker": 6.0,
    "To the Moon": 6.0,
    "DNA": 6.0,
    "Green Joker": 6.0,
    "Gros Michel": 6.0,
    "Ramen": 6.0,
    "Ride the Bus": 6.0,
    "Stuntman": 6.0,
    "The Tribe": 6.0,
    "Throwback": 6.0,
    "Vagabond": 6.0,

    # --- 4.0 (B) --- Solid but situational
    "Supernova": 4.0,
    "Scholar": 4.0,
    "Walkie Talkie": 4.0,
    "Sock and Buskin": 4.0,
    "Smiley Face": 4.0,
    "Scary Face": 4.0,
    "Wee Joker": 4.0,
    "Square Joker": 4.0,
    "Riff-Raff": 4.0,
    "Half Joker": 4.0,
    "Invisible Joker": 4.0,
    "Constellation": 4.0,
    "Certificate": 4.0,
    "Ceremonial Dagger": 4.0,
    "Raised Fist": 4.0,
    "Yorick": 4.0,
    "Blackboard": 4.0,
    "Shoot the Moon": 4.0,
    "Egg": 4.0,
    "Abstract Joker": 4.0,
    "Swashbuckler": 4.0,
    "Misprint": 4.0,
    "Turtle Bean": 4.0,
    "Madness": 4.0,
    "Hack": 4.0,
    "Hit the Road": 4.0,
    "Rough Gem": 4.0,
    "Gluttonous Joker": 4.0,
    "Wrathful Joker": 4.0,
    "Lusty Joker": 4.0,
    "Greedy Joker": 4.0,
    "Diet Cola": 4.0,
    "Blue Joker": 4.0,
    "Bootstraps": 4.0,
    "Burglar": 4.0,
    "Acrobat": 4.0,
    "Baron": 4.0,
    "Seeing Double": 4.0,
    "The Order": 4.0,
    "Cartomancer": 4.0,
    "Flash Card": 4.0,
    "Delayed Gratification": 4.0,
    "Even Steven": 4.0,
    "Mime": 4.0,
    "Popcorn": 4.0,
    "Castle": 4.0,
    "Odd Todd": 4.0,
    "Ice Cream": 4.0,
    "Runner": 4.0,
    "Faceless Joker": 4.0,
    "Hanging Chad": 4.0,
    "8 Ball": 4.0,
    "Photograph": 4.0,
    "Erosion": 4.0,
    "Lucky Cat": 4.0,
    "Glass Joker": 4.0,
    "Flower Pot": 4.0,
    "Obelisk": 4.0,
    "Joker Stencil": 4.0,
    "Reserved Parking": 4.0,
    "Joker": 4.0,
    "Oops! All 6s": 4.0,
    "Midas Mask": 4.0,
    "Mystic Summit": 4.0,
    "Superposition": 4.0,
    "Satellite": 4.0,

    # --- 2.0 (C) --- Weak or highly situational
    "Matador": 2.0,
    "The Idol": 2.0,
    "Juggler": 2.0,
    "Splash": 2.0,
    "Pareidolia": 2.0,
    "Loyalty Card": 2.0,
    "Dusk": 2.0,
    "Jolly Joker": 2.0,
    "Zany Joker": 2.0,
    "Wily Joker": 2.0,
    "Mad Joker": 2.0,
    "Clever Joker": 2.0,
    "Sly Joker": 2.0,
    "Bull": 2.0,
    "Banner": 2.0,
    "Smeared Joker": 2.0,
    "Astronomer": 2.0,
    "Drunkard": 2.0,
    "Droll Joker": 2.0,
    "Crafty Joker": 2.0,
    "Crazy Joker": 2.0,
    "Devious Joker": 2.0,
    "Troubadour": 2.0,
    "Hallucination": 2.0,
    "Chaos the Clown": 2.0,
    "Mr. Bones": 2.0,
    "Merry Andy": 2.0,
    "Red Card": 2.0,
    "Seance": 2.0,
    "Shortcut": 2.0,
    "Showman": 2.0,
    "Sly Joker": 2.0,
    "Stone Joker": 2.0,
    "Marble Joker": 2.0,
    "Gift Card": 2.0,
    "Luchador": 2.0,
    "Golden Ticket": 2.0,
    "Credit Card": 2.0,
    "Four Fingers": 2.0,
}

# Apply tier weights to joker data
for _name, _weight in TIER_WEIGHTS.items():
    if _name in JOKERS:
        JOKERS[_name]["tier_weight"] = _weight
    else:
        print(f"WARNING: TIER_WEIGHTS references unknown joker '{_name}'")


# ============================================================
# Validate on import
# ============================================================

_errors = validate_all(JOKERS)
if _errors:
    print(f"[WARN] JOKER VALIDATION: {len(_errors)} error(s) found:")
    for e in _errors:
        print(f"  - {e}")
