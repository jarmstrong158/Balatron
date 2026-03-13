# Balatron — Session Notes & Schema Reference

## Project Overview

**Balatron** is a reinforcement learning agent that plays Balatro autonomously, targeting naneinf scores.

**naneinf** = score displayed when the game exceeds the max 64-bit float (~1.80e308). The game breaks and displays "naneinf." It is the effective score ceiling and the target goal.

**Approach:** Hybrid — hand-coded Balatro domain knowledge + PPO neural network for nuanced decisions.

**Two training phases:**
- Phase 1: General Balatro competence (reliably clear Ante 8 white stake)
- Phase 2: Fine-tune toward naneinf hunting (transfer learning on Phase 1 weights)

**Hardware:** AMD 9800X3D, RTX 5070 Ti, PyTorch, CUDA.

---

## Architecture Decisions (LOCKED)

- **Algorithm:** PPO (not DQN — variable action space, long horizon, need stability)
- **Network:** Shared trunk + 3 separate heads (Play, Shop, Blind)
- **Batch size:** 50-100 runs per update
- **Three game states:** Blind board (passive), Shop, Play
- **State vector:** ~200-300 numbers (joker properties, economy, deck comp, blind board data, vouchers)
- **Jokers encoded as property fingerprints, NOT joker ID numbers**

---

## File Structure (LOCKED)

```
balatron/
├── environment/
│   ├── game_state.py
│   ├── action_space.py
│   ├── reward.py
│   └── capture.py
├── agent/
│   ├── network.py
│   ├── ppo.py
│   └── memory.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── checkpoints/
├── data/
│   ├── jokers.py          ← CURRENT FOCUS
│   ├── vouchers.py
│   └── weights.py
├── config.py
└── main.py
```

---

## State Vector Notes (for game_state.py later)

- Glass card count in deck needs to be tracked as a deck composition metric
- Joker position in lineup is a relevant state variable
- Enhanced card count by type needs to be tracked (drives Driver's License magnitude)

---

## Current Status

- **Phase:** Schema design for data/jokers.py
- **Schema status:** COMPLETE AND LOCKED
- **Next step:** Write jokers.py — Python structure + hand-code all 150 jokers

---

## Complete Joker Schema

### Top-Level Fields

```python
"name": str,                              # Joker name
"tier_weight": float,                     # Hand-coded value weight (0.0-10.0)
"edition": str,                           # "none" / "foil" / "holographic" / "polychrome" / "negative"
```

### Effect Type Flags (boolean, multi-true allowed)

```python
"economy": bool,                          # Generates money
"chip": bool,                             # Adds chips
"mult": bool,                             # Adds mult
"chip_scaling": bool,                     # Chips grow over time
"mult_scaling": bool,                     # Mult grows over time
"xmult": bool,                            # Multiplies mult
"xmult_scaling": bool,                    # Xmult grows over time
"copy": bool,                             # Copies another joker's effect
"in_hand_effect": bool,                   # Effect triggers from held hand, not played
```

### Effect Values (Static)

```python
"xmult_value": float,                    # Base xmult value
"chip_value": float,                      # Base chip value
"mult_value": float,                      # Base mult value
"money_per_round": float,                 # Flat money generated per round
```

### Trigger System

#### Trigger Fields

```python
"triggers": list,                         # List of trigger types from vocabulary
"trigger_combination": str,               # "any" / "all" — how multiple TRIGGERS combine
"trigger_detail_logic": str,              # "any" / "all" — how items in rank/suit LISTS combine
"trigger_scope": str,                     # "any_card" / "all_cards" — quantifier over cards
"trigger_hand_type": str,                 # Which poker hand type (e.g. "Straight", "Two Pair")
"trigger_ranks": list,                    # Specific ranks that trigger
"trigger_suits": list,                    # Specific suits that trigger
"trigger_editions": list,                 # Specific editions
"trigger_enhancements": list,             # Specific enhancements
"discard_trigger_ranks": list,            # Rank-specific discard trigger
"discard_trigger_suits": list,            # Suit-specific discard trigger
"negation_trigger": bool,                 # Fires on NOT condition
"rotating_condition": bool,               # Active condition changes each round
"round_history_condition": bool,          # Condition based on within-round history
"periodic_interval": int,                 # N for periodic trigger type
"event_count_threshold": int,             # Qualifying count for trigger
"event_count_comparison": str,            # "exact" / "minimum" / "maximum"
"trigger_on_probability_success": bool,   # Fires only when target card's probability succeeds
```

#### Trigger Vocabulary

```
HAND-BASED:
  "any_hand_played"                       # Any hand is played
  "specific_hand_type"                    # Specific poker hand type (use trigger_hand_type)
  "scoring_hand_size"                     # Number of cards in scoring hand
  "hand_type_minimum"                     # "This hand type or better"
  "first_hand_of_round"                   # First hand played in the round
  "final_hand_of_round"                   # Last hand played in the round
  "round_history_hand"                    # Hand type previously played this round

CARD EVENT-BASED:
  "specific_rank"                         # Card of specific rank (use trigger_ranks)
  "specific_suit"                         # Card of specific suit (use trigger_suits)
  "face_card"                             # Face card (Jack, Queen, King)
  "scoring_card"                          # Any card that scores
  "card_held_in_hand"                     # Card held but not played
  "card_added_to_deck"                    # Card added to the deck
  "card_destroyed"                        # Card is destroyed
  "enhancement_based"                     # Card with specific enhancement (use trigger_enhancements)
  "retrigger"                             # Card effect is retriggered

ACTION-BASED:
  "on_discard"                            # Cards are discarded
  "rank_specific_discard"                 # Specific rank discarded (use discard_trigger_ranks)
  "suit_specific_discard"                 # Specific suit discarded (use discard_trigger_suits)
  "on_blind_selected"                     # A blind is selected
  "on_blind_skip"                         # A blind is skipped
  "per_blind_skipped_cumulative"          # Cumulative blinds skipped this run
  "on_boss_blind_defeated"                # Boss blind is defeated
  "on_ante_up"                            # Ante increases
  "on_round_start"                        # Round begins
  "end_of_round"                          # Round ends
  "end_of_shop_phase"                     # Shop phase ends
  "on_shop_enter"                         # Shop is entered
  "on_shop_reroll"                        # Shop is rerolled
  "on_booster_pack_opened"                # Booster pack is opened
  "on_booster_pack_skipped"               # Booster pack is skipped
  "on_card_sold"                          # A card is sold

ECONOMY-BASED:
  "per_dollar_held"                       # Per dollar currently held
  "per_dollar_spent"                      # Per dollar spent

CONSUMABLE-BASED:
  "per_tarot_used"                        # Tarot card is used
  "per_planet_used"                       # Planet card is used
  "per_spectral_used"                     # Spectral card is used

JOKER-BASED:
  "on_joker_destroyed"                    # A joker is destroyed
  "per_joker_owned"                       # Per joker currently owned
  "per_specific_joker_present"            # Specific joker is present

META/OTHER:
  "per_card_remaining_in_deck"            # Per card remaining in deck
  "per_hand_remaining"                    # Per hand remaining this round
  "periodic"                              # Every N occurrences (use periodic_interval)
```

### Effect System

#### Score Effects

```python
"score_effect": list,                     # ["chips", "mult", "xmult", "chips_and_mult"]
"scoring_timing": str,                    # "during_card" / "after_cards" / "end_of_scoring"
"effect_target": str,                     # "self" / "all_jokers" / "all_consumables" / "all_cards" / "specific"
```

#### Economy Effects

```python
"economy_effect": bool,                   # Generates money
"scaling_economy": bool,                  # Economy effect grows over time
"magnitude_source": str,                  # "static" / "deck_property" / "game_state" / "joker_sell_value" / "run_history"
"magnitude_source_detail": str,           # Which property/state value drives magnitude
```

#### Game State Effects

```python
"game_parameter_effect": bool,            # Modifies hand size, discard count, etc.
"acquisition_penalty": bool,              # Permanent negative effect on buy
"rule_modification": bool,                # Modifies hand evaluation rules, suit equivalency
"shop_pool_modification": bool,           # Modifies what can appear in shop
"shop_cost_modification": bool,           # Modifies cost of cards/consumables
"global_probability_modifier": bool,      # Modifies all probability rolls (Oops All 6s)
```

#### Card Effects

```python
"card_modification_effect": bool,         # Permanently alters cards in deck
"destructive_card_modification": bool,    # Removes or downgrades card properties
"permanent_deck_scaling": bool,           # Permanently improves cards over time
"spatial_card_condition": bool,           # Effect depends on positional relationship of cards
```

#### Joker Effects

```python
"joker_creation_effect": bool,            # Creates new jokers
"forced_joker_destruction": bool,         # Destroys a joker as part of effect
"copy_target": str,                       # "random" / "specific" / "left" / "right"
"copy_timing_inheritance": str,           # "original" / "copy_position"
"position_aware": bool,                   # Effect depends on neighbor joker position
"position_target": str,                   # "left" / "right" / "any"
```

#### Consumable Effects

```python
"consumable_creation": bool,              # Creates a consumable
"consumable_specificity": str,            # "specific" / "random"
"consumable_creation_type": str,          # Which specific consumable
```

#### Hand Effects

```python
"hand_upgrade_effect": bool,              # Upgrades a poker hand level
"hand_upgrade_target": str,               # "any" / "most_played" / "specific"
"hand_subcomposition_condition": bool,    # Effect depends on specific combinations within hand
```

#### Retrigger Effects

```python
"retrigger_effect": bool,                 # Retriggers card effects
"mass_retrigger": bool,                   # Retriggers multiple cards simultaneously
"per_card_instance": bool,                # Magnitude multiplied by qualifying card count
"card_contribution_modification": bool,   # Alters how a card scores mid-hand
```

#### Compound / Survival / Probability

```python
"compound_effect": bool,                  # Multiple effects fire from one trigger
"survival_effect": bool,                  # Prevents death
"boss_blind_effect": bool,               # Interacts with boss blind mechanics
"effect_probability": float,              # Chance effect fires (1.0 = always)
"random_range_effect": bool,              # Magnitude is random within bounds
"range_min": float,                       # Minimum roll
"range_max": float,                       # Maximum roll
```

#### Card Creation

```python
"card_creation_effect": bool,             # Creates a playing card
"card_creation_enhancement": str,         # Enhancement of created card
```

### Scaling Block

#### Core

```python
"scaling_type": str,                      # "chips" / "mult" / "xmult" / "economy" / "sell_value" / "hand_size"
"scaling_start_value": float,             # Initial value of scaling effect
```

#### Growth

```python
"scaling_method": str,                    # "flat_addition" / "counter_threshold" / null (decay-only)
"scaling_driver": str,                    # Trigger vocabulary entry
"scaling_increment": float,               # Fixed amount per trigger (static only)
"scaling_increment_source": str,          # "static" / "derived"
"scaling_increment_multiplier": float,    # Multiplier on derived source (derived only)
"scaling_increment_detail": str,          # What derived value reads from
"scaling_per_item": bool,                 # Growth fires per qualifying card vs once per event
"scaling_threshold": int,                 # N for counter_threshold
```

#### Dynamic Condition

```python
"scaling_dynamic_condition": str,         # Game state value for dynamic trigger comparison
```

#### Reset

```python
"scaling_resets": bool,                   # Does accumulated value ever reset
"scaling_reset_trigger": str,             # Trigger vocabulary entry
"scaling_reset_value": float,             # Value after reset
```

#### Decay

```python
"scaling_decay": bool,                    # Does this joker lose value over time
"scaling_decay_driver": str,              # Trigger vocabulary entry
"scaling_decay_amount": float,            # Amount lost per trigger
"scaling_decay_floor": float,             # Minimum value
"scaling_decay_per_item": bool,           # Decay fires per qualifying card vs once per event
"scaling_destroys_at_floor": bool,        # Destroyed when value hits floor
```

#### Expiry

```python
"expiry": bool,                           # Hard cutoff exists
"expiry_type": str,                       # "activations" / "rounds" / "probabilistic"
"expiry_threshold": int,                  # Count for activations/rounds
"expiry_probability": float,              # Chance per check for probabilistic
"expiry_check_timing": str,              # Trigger vocabulary entry for check timing
"expiry_outcome": str,                    # "destroyed" / "transforms"
```

#### Target

```python
"scaling_target": str,                    # "self" / "all_jokers_and_consumables"
```

#### Scaling Methods (2 only)

1. `flat_addition` — fixed increment per trigger
2. `counter_threshold` — accumulate N events, fire, reset counter

---

## Joker Classification (150 total)

### Scaling (self-scaling, value changes during run)

**Mult scaling (7):** Green Joker, Red Card, Flash Card, Spare Trousers, Fortune Teller, Ride the Bus, Ceremonial Dagger

**Xmult scaling (9):** Constellation, Madness, Vampire, Hologram, Lucky Cat, Glass Joker, Campfire, Canio, Yorick

**Chip scaling (4):** Runner, Square Joker, Wee Joker, Castle

**Economy scaling (1):** Rocket

**Sell value scaling (2):** Egg, Gift Card

### Decay (starts high, loses value)

**Mult decay (1):** Popcorn

**Xmult decay (1):** Ramen

**Chip decay (1):** Ice Cream

**Hand size decay (1):** Turtle Bean

**Expiring (2):** Seltzer, Invisible Joker

### Reads Game State (dynamic but not internally scaling)

**Mult from state (4):** Supernova, Swashbuckler, Bootstraps, Erosion

**Xmult from state (3):** Joker Stencil, Steel Joker, Throwback

**Chips from state (4):** Blue Joker, Bull, Stone Joker, Banner

**Xmult conditional/reset (2):** Obelisk, Hit the Road

### Not Scaling (handled by effect system, not scaling block)

**Deck modification (4):** Hiker, Marble Joker, Midas Mask, DNA

**Hand upgrade (1):** Burnt Joker

### Static Effects

**Static xmult (20):** Cavendish, Card Sharp, Blackboard, Baron, Photograph, The Duo, The Trio, The Family, The Order, The Tribe, Loyalty Card, Ancient Joker, The Idol, Seeing Double, Flower Pot, Acrobat, Driver's License, Triboulet, Bloodstone, Baseball Card

**Static mult (22):** Joker, Greedy Joker, Lusty Joker, Wrathful Joker, Gluttonous Joker, Jolly Joker, Zany Joker, Mad Joker, Crazy Joker, Droll Joker, Half Joker, Mystic Summit, Misprint, Raised Fist, Abstract Joker, Even Steven, Fibonacci, Scholar, Walkie Talkie, Smiley Face, Shoot the Moon, Onyx Agate, Gros Michel

**Static chips (9):** Sly Joker, Wily Joker, Clever Joker, Devious Joker, Crafty Joker, Scary Face, Odd Todd, Arrowhead, Stuntman

**Static economy (14):** Golden Joker, Golden Ticket, Cloud 9, To the Moon, Satellite, Delayed Gratification, Business Card, Faceless Joker, Reserved Parking, To Do List, Mail-In Rebate, Rough Gem, Matador, Trading Card

**Retrigger (6):** Hack, Dusk, Mime, Sock and Buskin, Hanging Chad, Seltzer

**Rule modification (7):** Four Fingers, Shortcut, Smeared Joker, Pareidolia, Splash, Showman, Oops! All 6s

**Copy (2):** Blueprint, Brainstorm

**Game parameter (6):** Juggler, Drunkard, Merry Andy, Troubadour, Burglar, Stuntman

**Consumable creation (10):** 8 Ball, Space Joker, Superposition, Vagabond, Hallucination, Sixth Sense, Seance, Cartomancer, Perkeo, Certificate

**Joker creation (1):** Riff-Raff

**Survival/Boss (3):** Mr. Bones, Chicot, Luchador

**Shop modification (4):** Chaos the Clown, Astronomer, Diet Cola, Credit Card

**NOTE:** Many jokers appear in multiple categories (e.g., Seltzer = retrigger + expiring, Stuntman = static chips + game parameter, Scholar = static chips + static mult). This is expected and maps to the multi-flag boolean schema.

---

## Complete Joker List (150)

### Common (60)

1. Joker — +4 Mult
2. Greedy Joker — Played Diamonds give +3 Mult when scored
3. Lusty Joker — Played Hearts give +3 Mult when scored
4. Wrathful Joker — Played Spades give +3 Mult when scored
5. Gluttonous Joker — Played Clubs give +3 Mult when scored
6. Jolly Joker — +8 Mult if hand contains a Pair
7. Zany Joker — +12 Mult if hand contains Three of a Kind
8. Mad Joker — +10 Mult if hand contains Two Pair
9. Crazy Joker — +12 Mult if hand contains a Straight
10. Droll Joker — +10 Mult if hand contains a Flush
11. Sly Joker — +50 Chips if hand contains a Pair
12. Wily Joker — +100 Chips if hand contains Three of a Kind
13. Clever Joker — +80 Chips if hand contains Two Pair
14. Devious Joker — +100 Chips if hand contains a Straight
15. Crafty Joker — +80 Chips if hand contains a Flush
16. Half Joker — +20 Mult if played hand contains 3 or fewer cards
17. Credit Card — Go up to -$20 in debt
18. Banner — +30 Chips for each remaining discard
19. Mystic Summit — +15 Mult when 0 discards remaining
20. 8 Ball — 1 in 4 chance for each played 8 to create a Tarot card
21. Misprint — +0 to +23 Mult (random)
22. Raised Fist — Adds double the rank of lowest held card to Mult
23. Chaos the Clown — 1 free Reroll per shop
24. Scary Face — Played face cards give +30 Chips when scored
25. Abstract Joker — +3 Mult for each Joker owned
26. Delayed Gratification — Earn $2 per discard if no discards used by end of round
27. Hack — Retrigger each played 2, 3, 4, or 5
28. Pareidolia — All cards are considered face cards
29. Gros Michel — +15 Mult. 1 in 6 chance to be destroyed at end of round
30. Even Steven — Played even rank cards give +4 Mult when scored (10, 8, 6, 4, 2)
31. Odd Todd — Played odd rank cards give +31 Chips when scored (A, 9, 7, 5, 3)
32. Scholar — Played Aces give +20 Chips and +4 Mult when scored
33. Business Card — Played face cards have 1 in 2 chance to give $2 when scored
34. Supernova — Adds number of times poker hand has been played this run to Mult
35. Ride the Bus — +1 Mult per consecutive hand played without scoring a face card
36. Space Joker — 1 in 4 chance to upgrade level of played poker hand
37. Egg — +$3 sell value at end of round
38. Burglar — When Blind is selected, gain +3 Hands and lose all discards
39. Runner — +15 Chips if played hand contains a Straight (permanent)
40. Ice Cream — +100 Chips. -5 Chips per hand played
41. Splash — Every played card counts in scoring
42. Blue Joker — +2 Chips per remaining card in deck
43. Fibonacci — Each played Ace, 2, 3, 5, or 8 gives +8 Mult when scored
44. Hiker — Every played card permanently gains +5 Chips when scored
45. Faceless Joker — Earn $5 if 3+ face cards discarded at once
46. Green Joker — +1 Mult per hand played. -1 Mult per discard
47. Superposition — Create a Tarot card if hand contains an Ace and a Straight
48. To Do List — Earn $4 if poker hand matches random target. Target changes each round
49. Cavendish — X3 Mult. 1 in 1000 chance destroyed at end of round
50. Red Card — +3 Mult when any Booster Pack is skipped
51. Square Joker — +4 Chips if played hand has exactly 4 cards (permanent)
52. Stone Joker — +25 Chips per Stone Card in full deck
53. Golden Joker — Earn $4 at end of round
54. Bull — +2 Chips per dollar you have
55. Flash Card — +2 Mult per reroll in shop (permanent)
56. Popcorn — +20 Mult. -4 Mult per round played
57. Walkie Talkie — Each played 10 or 4 gives +10 Chips and +4 Mult when scored
58. Smiley Face — Played face cards give +5 Mult when scored
59. Golden Ticket — Played Gold cards earn $4 when scored
60. Swashbuckler — Adds sell value of all other owned Jokers to Mult

### Uncommon (66)

61. Joker Stencil — X1 Mult per empty Joker slot (self included)
62. Four Fingers — Flushes and Straights can be made with 4 cards
63. Mime — Retrigger all held-in-hand abilities
64. Ceremonial Dagger — On Blind selected, destroy right Joker, permanently add 2x its sell value to Mult
65. Marble Joker — Add one Stone card to deck when Blind selected
66. Loyalty Card — X4 Mult every 6 hands played
67. Dusk — Retrigger all played cards on final hand of round
68. Steel Joker — X0.2 Mult per Steel Card in full deck (added to X1 base)
69. Blackboard — X3 Mult if all held cards are Spades or Clubs
70. DNA — If first hand is single card, add permanent copy to deck and draw it
71. Sixth Sense — If first hand is a single 6, destroy it and create a Spectral card
72. Constellation — +X0.1 Mult per Planet card used (permanent)
73. Card Sharp — X3 Mult if poker hand already played this round
74. Madness — On Blind selected, +X0.5 Mult and destroy a random Joker (not self)
75. Vampire — +X0.1 Mult per scoring Enhanced card, removes the enhancement
76. Shortcut — Straights with gaps of 1 rank allowed
77. Hologram — +X0.25 Mult per playing card added to deck (permanent)
78. Vagabond — Create a Tarot card when hand played with $4 or less
79. Baron — Each King held in hand gives X1.5 Mult
80. Cloud 9 — Earn $1 per 9 in full deck at end of round
81. Rocket — Earn $1 at end of round. Payout +$2 when Boss Blind defeated
82. Obelisk — +X0.2 Mult per consecutive hand without playing most-played type (resets)
83. Midas Mask — All played face cards become Gold cards when scored
84. Luchador — Sell to disable current Boss Blind
85. Photograph — First played face card gives X2 Mult when scored
86. Gift Card — +$1 sell value to every Joker and Consumable at end of round
87. Turtle Bean — +5 hand size. -1 each round
88. Erosion — +4 Mult per card below starting deck size in full deck
89. Reserved Parking — Each face card held in hand has 1 in 2 chance to give $1
90. Mail-In Rebate — Earn $5 per discarded card of target rank. Rank changes each round
91. To the Moon — +$1 interest per $5 at end of round
92. Hallucination — 1 in 2 chance to create Tarot when Booster Pack opened
93. Fortune Teller — +1 Mult per Tarot card used this run
94. Juggler — +1 hand size
95. Drunkard — +1 discard each round
96. Lucky Cat — +X0.25 Mult per Lucky card successful trigger (permanent)
97. Baseball Card — Uncommon Jokers each give X1.5 Mult
98. Diet Cola — Sell to create a free Double Tag
99. Trading Card — If first discard is single card, destroy it and earn $3
100. Spare Trousers — +2 Mult if played hand contains Two Pair (permanent)
101. Ancient Joker — Each played card of target suit gives X1.5 Mult. Suit rotates each round
102. Ramen — X2 Mult. -X0.01 per card discarded
103. Seltzer — Retrigger all played cards for next 10 hands, then destroyed
104. Castle — +3 Chips per discarded card of target suit. Suit rotates each round (permanent)
105. Campfire — +X0.25 Mult per card sold. Resets on Boss Blind defeated
106. Acrobat — X3 Mult on final hand of round
107. Sock and Buskin — Retrigger all played face cards
108. Troubadour — +2 hand size. -1 hand per round
109. Certificate — Round start: add random card with random seal to hand
110. Smeared Joker — Hearts=Diamonds, Spades=Clubs for suit purposes
111. Throwback — X0.25 Mult per Blind skipped this run
112. Hanging Chad — Retrigger first scoring card 2 additional times
113. Rough Gem — Played Diamonds earn $1 when scored
114. Bloodstone — 1 in 2 chance played Hearts give X1.5 Mult when scored
115. Arrowhead — Played Spades give +50 Chips when scored
116. Onyx Agate — Played Clubs give +7 Mult when scored
117. Glass Joker — +X0.75 Mult per Glass Card destroyed (permanent)
118. Showman — Joker, Tarot, Planet, Spectral cards may appear multiple times in shop
119. Flower Pot — X3 Mult if hand contains Diamond, Club, Heart, and Spade
120. Wee Joker — +8 Chips per played 2 scored (permanent)
121. Merry Andy — +3 discards each round. -1 hand size
122. Seeing Double — X2 Mult if hand has scoring Club + scoring non-Club card
123. The Idol — Each played card of target rank+suit gives X2 Mult. Target rotates each round
124. Matador — Earn $8 if played hand triggers Boss Blind ability
125. Hit the Road — +X0.5 Mult per Jack discarded this round (resets each round)
126. Stuntman — +250 Chips. -2 hand size

### Rare (19)

127. The Duo — X2 Mult if hand contains a Pair
128. The Trio — X3 Mult if hand contains Three of a Kind
129. The Family — X4 Mult if hand contains Four of a Kind
130. The Order — X3 Mult if hand contains a Straight
131. The Tribe — X2 Mult if hand contains a Flush
132. Blueprint — Copies ability of Joker to the right
133. Brainstorm — Copies ability of leftmost Joker
134. Invisible Joker — After 2 rounds, sell to duplicate a random Joker
135. Satellite — Earn $1 at end of round per unique Planet card used this run
136. Shoot the Moon — Each Queen held in hand gives +13 Mult
137. Driver's License — X3 Mult if 16+ Enhanced cards in full deck
138. Cartomancer — Create a Tarot card when Blind is selected
139. Astronomer — All Planet cards and Celestial Packs in shop are free
140. Burnt Joker — Upgrade level of first discarded poker hand each round
141. Bootstraps — +2 Mult per $5 you have
142. Mr. Bones — Prevents death if chips >= 25% of required. Self-destructs after saving
143. Oops! All 6s — Doubles all listed probabilities
144. Riff-Raff — When Blind selected, create 2 Common Jokers (if room)
145. Seance — If hand is Straight Flush, create random Spectral card

### Legendary (5)

146. Canio — +X1 Mult when face card destroyed. Starts at X1
147. Triboulet — Played Kings and Queens each give X2 Mult when scored
148. Yorick — +X1 Mult every 23 cards discarded. Starts at X1
149. Chicot — Disables effect of every Boss Blind
150. Perkeo — Creates Negative copy of 1 random consumable at end of shop

---

## Schema Design Decisions Log

### Scaling Block Design Process

The scaling block went through 7 dimensions of design:

1. **What scales:** chips, mult, xmult, economy, sell_value, hand_size
2. **How it scales:** flat_addition, counter_threshold (periodic consolidated into counter_threshold; random_range, proportional, reset_to_base removed as unnecessary)
3. **What drives it:** Always a trigger event from vocabulary. Increment is fixed (static) or derived from game state (Ceremonial Dagger). Fields: scaling_driver, scaling_increment, scaling_increment_source, scaling_increment_detail, scaling_increment_multiplier
4. **Reset conditions:** Some jokers reset to base value on specific triggers (Campfire, Ride the Bus, Obelisk, Hit the Road). Fields: scaling_resets, scaling_reset_trigger, scaling_reset_value
5. **Decay:** Separate from growth. Can coexist with growth (Green Joker). Fields: scaling_decay, scaling_decay_driver, scaling_decay_amount, scaling_decay_floor, scaling_decay_per_item, scaling_destroys_at_floor
6. **Expiry:** Hard cutoff — deterministic (activations/rounds) or probabilistic. Fields: expiry, expiry_type, expiry_threshold, expiry_probability, expiry_check_timing, expiry_outcome
7. **Current value tracking:** Runtime fields for game_state.py (current_value, base_value, scaling_event_counter, expiry_counter). Schema defines scaling_start_value.

### Trigger System Corrections

- `trigger_logic` and `trigger_condition` were redundant — replaced with `trigger_combination` (multi-trigger) and `trigger_detail_logic` (multi-item lists)
- `ante_up_trigger` and `joker_destroyed_trigger` booleans removed — redundant with trigger vocabulary
- Added `trigger_scope` for universal quantifier (Blackboard: "all_cards" must match)
- Added `trigger_hand_type` for poker hand type specification
- Added `event_count_comparison` for exact/minimum/maximum
- Added `trigger_on_probability_success` for Lucky Cat pattern

### Jokers Removed from Scaling Block

- **Hiker** — modifies deck cards, not itself. Handled by `permanent_deck_scaling`
- **Burnt Joker** — modifies poker hand levels. Handled by `hand_upgrade_effect`
- **Marble Joker, Midas Mask, DNA** — deck/card modifications. Handled by `card_creation_effect`/`card_modification_effect`
- **Loyalty Card** — static X4 with periodic trigger, not actually scaling

### Trigger Vocabulary Additions

- `on_booster_pack_skipped` (Red Card)
- `on_card_sold` (Campfire)
- `on_round_start` (Hit the Road reset, Certificate)
- `per_planet_used` (renamed from per_celestial_used — Constellation)
- `per_spectral_used` (future-proofing)

---

## Field Count Summary

- Top-level: 3
- Effect type flags: 9
- Effect values: 4
- Trigger system: 17
- Effect system: 33
- Scaling block: 27
- **Total: 93 fields per joker**

Most are null/false for any given joker. A simple static joker touches ~8-10 fields. A complex scaler touches ~20.

---

## Next Session Instructions

### For Claude Code

Read this document fully before starting. Do not summarize it back to Jonny. Do not ask what he wants to work on. Jump into the next task.

### Next Task

Write `data/jokers.py`:
1. Define the Python structure (recommended: hybrid approach — TypedDict for schema definition, plain dicts for joker data, validation function)
2. Build the empty template
3. Jonny hand-codes all 150 jokers against the schema
4. Validation pass to catch typos/missing fields

### Who Is Jonny

- Warehouse manager, strong Python developer
- Autistic, direct, analytical — no filler
- Understands ML concepts: PPO, shared trunk + 3 heads, state vectors, reward shaping, batch size, advantage estimation, transfer learning, action validity masking
- He is the Balatro domain expert. Claude is the ML/architecture expert.
- Never over-explain. Never redirect to external sources.
