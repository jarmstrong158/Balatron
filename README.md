# Balatron

A reinforcement learning agent that plays [Balatro](https://www.playbalatro.com/) autonomously, combining deep domain knowledge with PPO (Proximal Policy Optimization) to master the game's complex scoring mechanics, economy management, and joker synergy systems.

**Goal:** Achieve consistent Ante 8 clears (Phase 1), then push toward **naneinf** — the score ceiling where 64-bit floats overflow and the game displays "naneinf" (~1.80e308).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
  - [Hybrid Decision System](#hybrid-decision-system)
  - [Neural Network](#neural-network)
  - [State Vector](#state-vector)
  - [Action Space](#action-space)
  - [Reward Shaping](#reward-shaping)
- [Scoring Engine](#scoring-engine)
- [Joker Schema Database](#joker-schema-database)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Resuming from Checkpoint](#resuming-from-checkpoint)
- [Training Phases](#training-phases)
- [Key Design Decisions](#key-design-decisions)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## How It Works

Balatron connects to a live instance of Balatro through the [BalatroBot](https://github.com/coder/balatrobot) mod, which exposes the full game state and accepts action commands via a JSON-RPC 2.0 HTTP API on `127.0.0.1:12346`. No screen capture or computer vision is needed — the agent reads structured game data directly.

The agent observes the game state (hand cards, jokers, economy, blind targets, deck composition, shop contents), encodes it into an 814-dimensional vector, and uses a PPO neural network to select actions. A sophisticated heuristic layer validates and enhances the network's decisions with Balatro-specific domain knowledge — optimal hand selection, joker ordering, consumable usage, pack evaluation, and economy management.

```
Balatro Game
    |
    | JSON-RPC 2.0 (BalatroBot mod)
    v
Game State Encoder (814-dim vector)
    |
    v
PPO Neural Network (shared trunk + 3 policy heads)
    |
    v
Action Mask (domain knowledge biasing)
    |
    v
Heuristic Validator (hand eval, scoring math, economy guards)
    |
    v
API Action Command --> Balatro Game
```

---

## Architecture

### Hybrid Decision System

Balatron uses a **hybrid architecture** that combines neural network learning with hard-coded Balatro expertise:

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **Neural Network** | Strategic decisions under uncertainty | When to reroll, which jokers to prioritize, when to leave the shop, blind skip timing |
| **Action Mask** | Logit biasing to guide exploration | Boost scoring jokers in shop, suppress bad jokers, penalize unaffordable items |
| **Heuristic Guards** | Validate and override bad decisions | Block selling your only joker, prevent buying when it tanks interest, force-buy Blueprint/Brainstorm |
| **Scoring Engine** | Full Balatro math for hand evaluation | Compute exact scores with joker effects, retriggers, editions, enhancements, boss debuffs |
| **Strategic Advisor** | Optimal hand/discard selection | `plan_optimal_action()` — evaluates all possible plays against blind target with draw probability |

### Neural Network

```
Input (814) --> Shared Trunk (814 -> 768 -> 768 -> 512)
                    |
                    |--> Play Head   (512 -> 256 -> 45)  -- SELECTING_HAND state
                    |--> Shop Head   (512 -> 256 -> 45)  -- SHOP state
                    |--> Blind Head  (512 -> 128 -> 45)  -- BLIND_SELECT state
                    |--> Value Head  (512 -> 256 -> 1)   -- all states
```

- **3 policy heads** specialized for different game phases — the shop head never sees hand-play decisions and vice versa
- **Shared trunk** learns representations useful across all phases (joker synergies, economy state, deck composition)
- **Layer normalization** + **SiLU activation** for training stability
- **Action-conditioned target sampling** — after selecting an action type, target logits are masked so only valid targets can be chosen (e.g., `buy_pack` can only target pack slots, not joker slots)

### State Vector

The game state is encoded into an **814-dimensional float vector** with careful normalization:

| Section | Dimensions | Contents |
|---------|-----------|----------|
| Game Meta | 42 | Ante, round, money (log-scaled), hands/discards left, reroll cost, chips/target (log), boss blind info, blind statuses, joker slots |
| Hand Levels | 79 | 13 poker hand types x (level, chips, mult) + play counts, all normalized |
| Deck Composition | 61 | 52 rank x suit counts + 9 enhancement/seal counters |
| Vouchers | 32 | Binary flags for each voucher owned |
| Joker Slots | 270 | 5 slots x 54-field fingerprint: tier weight, effect flags, values (log-scaled), triggers, edition, modifiers, runtime scaling value |
| Hand Cards | 96 | 12 slots x 8 fields: rank, suit, enhancement, seal, edition, debuffed, base chips, is_face |
| Consumables | 12 | 2 slots x 6 fields: type, key hash, cost, value estimate, needs targeting, is negative |
| Shop Contents | 162 | 3 joker fingerprints + 2 vouchers + 2 packs, all with cost/affordability |
| Pack Cards | 10 | 5 slots x 2 fields for opened booster pack contents |
| Boss Blind | 10 | One-hot category + suit debuff encoding |
| Scoring Context | 40 | Projected scores, risk assessment, hand type features |

**Design principle:** Jokers are encoded as **property fingerprints** (what the joker *does*) rather than joker IDs. This means the network generalizes across jokers with similar effects — it doesn't need to memorize 150 individual joker behaviors, it learns that "x2 mult on face cards" is valuable regardless of which joker provides it.

### Action Space

Actions are represented as a **14-element tensor**: `[action_type(1), card_selection(12), target(1)]`

**14 action types:**

| Index | Action | Used In |
|-------|--------|---------|
| 0 | Play hand | SELECTING_HAND |
| 1 | Discard | SELECTING_HAND |
| 2 | Buy joker | SHOP |
| 3 | Buy voucher | SHOP |
| 4 | Buy pack | SHOP |
| 5 | Sell joker | SHOP |
| 6 | Sell consumable | SHOP |
| 7 | Reroll shop | SHOP |
| 8 | Use consumable | SELECTING_HAND, SHOP |
| 9 | Select blind | BLIND_SELECT |
| 10 | Skip blind | BLIND_SELECT |
| 11 | Select pack card | PACK_OPENED |
| 12 | Skip pack | PACK_OPENED |
| 13 | End shop | SHOP |

**19 target slots** map to different game objects depending on the action type:

```
[0-2]   Shop joker slots     (for buy_joker)
[3-4]   Shop voucher slots   (for buy_voucher)
[5-6]   Shop pack slots      (for buy_pack)
[7-11]  Owned joker slots    (for sell_joker)
[12-13] Consumable slots     (for sell/use_consumable)
[14-18] Pack card slots      (for select_pack_card)
```

**Action-conditioned targeting:** After the network selects an action type, the target logits are masked to only allow valid targets for that action. This eliminates the "mismatched target" problem where independent sampling could pair `buy_pack` with a joker sell target.

### Reward Shaping

Rewards are carefully structured to align with Balatro's exponential scoring:

| Reward | Value | Description |
|--------|-------|-------------|
| Game Win (Ante 8 clear) | +10.0 | Phase 1 primary goal |
| Game Loss | -5.0 + 0.3/ante | Harsh base penalty, softened by progress |
| Naneinf Achievement | +50.0 | Phase 2 ultimate goal |
| Ante Cleared | +3.0 + 0.5/ante | Scales with difficulty |
| Blind Cleared | +1.0 (+1.5 boss) | Per-blind progress signal |
| Score vs Target | +0.5 * log10(ratio) | Log-scaled overkill bonus |
| Money Gain | +0.02/dollar | Economy awareness |
| Money Spent | -0.01/dollar | Light spending penalty (spending is necessary) |
| Scaling Growth | +0.05/log-unit | Encourages scaling joker investment |

All score-based rewards use **log10 scaling** because Balatro scores grow exponentially — the agent learns to push the *exponent* higher.

---

## Scoring Engine

The scoring engine (`hand_eval.py`) implements the full Balatro scoring formula:

```
Score = (hand_chips + card_chips + joker_chips) x (hand_mult + joker_mult) x joker_xmult
```

**Features implemented:**

- Complete hand classification (High Card through Flush Five)
- All 150 joker effects — chips, mult, xmult, scaling, conditional triggers
- **Retrigger system** — Hanging Chad (first card +2), Sock and Buskin (face cards +1), Hack (2/3/4/5 +1), Seltzer (all +1), Dusk (last hand), Red Seal (+1)
- Card enhancements: Bonus (+30 chips), Mult (+4 mult), Wild (any suit), Glass (x1.5, may shatter), Steel (x1.5 held), Stone (50 chips, no rank/suit), Gold (+3 money), Lucky (chance mult/money)
- Card editions: Foil (+50 chips), Holographic (+10 mult), Polychrome (x1.5)
- Boss blind debuffs: suit debuffs (Club, Goad, Window, Head), face debuff (Plant), rank debuffs
- Blueprint/Brainstorm copy chain resolution
- Joker ordering optimization (chips -> mult -> xmult left-to-right)
- Draw probability calculation for discard decisions
- Deck composition tracking for suit synergy awareness
- Death tarot and Hanged Man suit-aware targeting

**Strategic advisor** (`plan_optimal_action`) evaluates every possible play and discard against:
- Current blind target score
- Remaining hands/discards
- Draw probabilities for completing better hands
- Joker synergies with specific suits/ranks
- Boss blind debuffs on specific suits

---

## Joker Schema Database

All 150 base-game jokers are encoded in `data/jokers.py` as structured schemas:

```python
"Photograph": make_joker(
    name="Photograph",
    xmult=True,
    xmult_value=2.0,
    triggers=["face_card"],
    score_effect=["xmult"],
    scoring_timing="during_card",
)
```

Each schema captures:
- **Effect type**: chip, mult, xmult, economy, scaling, retrigger, copy, game_param
- **Trigger conditions**: any_hand, specific_hand_type, face_card, specific_suit, specific_rank, scoring_card, periodic, per_dollar_held, per_joker_owned, etc.
- **Scoring timing**: `during_card` (fires per scored card) vs `after_cards` (fires once per hand)
- **Scaling behavior**: flat addition, multiplication, start value, increment, decay
- **Per-card instance**: whether the effect fires once or per qualifying card
- **Effect probability**: for chance-based jokers (Bloodstone 1/3, Lucky Card 1/5, etc.)
- **Tier weights**: competitive value rating for shop decisions

---

## Project Structure

```
balatron/
|-- agent/
|   |-- network.py          # Neural network (shared trunk + 3 heads + value)
|   |-- ppo.py              # PPO trainer, rollout buffer, GAE
|
|-- environment/
|   |-- game_state.py       # 814-dim state vector encoder
|   |-- action_space.py     # Action masks, target mapping, joker evaluation
|   |-- hand_eval.py        # Full scoring engine, hand classifier, strategic advisor
|   |-- reward.py           # Reward shaping (log-scaled, multi-tier)
|
|-- data/
|   |-- jokers.py           # 150 joker schemas + tier weights
|
|-- training/
|   |-- train.py            # Main training loop, episode management, auto-play heuristics
|
|-- scripts/
|   |-- sim_bloodstone.py   # Simulation utilities
|
|-- tests/
|   |-- test_scoring.py     # Scoring engine validation
|
|-- NOTES.md                # Architecture decisions, state vector layout, design rationale
|-- LICENSE                  # MIT License
|-- README.md                # This file
```

---

## Prerequisites

- **Python 3.12+**
- **PyTorch** with CUDA support (GPU strongly recommended for training)
- **Balatro** (Steam version)
- **[Steamodded](https://github.com/Steamopollys/Steamodded)** (>= 0.9.8) — Balatro mod loader
- **[BalatroBot](https://github.com/coder/balatrobot)** mod (v1.4.1+) — JSON-RPC API for game control

### Hardware Used

- AMD Ryzen 9 9800X3D
- NVIDIA RTX 5070 Ti
- Training runs at ~1500 steps/hour with continuous Balatro gameplay

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jarmstrong158/Balatron.git
cd balatron
```

### 2. Install Python dependencies

```bash
pip install torch numpy
```

### 3. Install BalatroBot mod

Follow the [BalatroBot installation guide](https://github.com/coder/balatrobot#installation). The mod requires [Steamodded](https://github.com/Steamopollys/Steamodded) as a dependency.

The mod should be installed to:
```
%APPDATA%\Balatro\Mods\balatrobot\
```

### 4. Install the BalatroBot CLI

```bash
pip install balatrobot
```

---

## Usage

### Training

Training requires two terminals running simultaneously:

**Terminal 1 — Start BalatroBot server + Balatro game:**
```powershell
uvx balatrobot serve --fast
```

This launches the Balatro game with the BalatroBot mod injected and starts the JSON-RPC API server.

**Terminal 2 — Start training:**
```powershell
cd balatron
python -u -m training.train --total-timesteps 1500000 --device cuda
```

Training progress is printed to the console:
```
Runs: 50 (lifetime: 500) | Wins: 3 (lifetime: 12)
  Avg reward: 8.42 | Best: 22.1 | Avg ante: 5.3
  Win rate: 6.0% (lifetime: 2.4%)
  Steps: 125000 / 1500000 (8.3%)
```

### Resuming from Checkpoint

```powershell
python -u -m training.train --total-timesteps 1500000 --device cuda --checkpoint checkpoints/balatron_phase1_final.pt
```

Checkpoints are saved automatically during training.

---

## Training Phases

### Phase 1: General Competence

**Goal:** Reliably clear Ante 8 on White Stake (base difficulty).

The agent learns:
- Basic hand selection (play the highest-scoring hand)
- Joker evaluation (which jokers improve scoring power)
- Economy management (interest tiers, spending discipline)
- Shop decisions (when to buy, reroll, or leave)
- Consumable usage (planet cards to upgrade hand levels, tarots for deck manipulation)
- Blind selection (when to skip for tags vs. accept for money)

### Phase 2: Naneinf Hunting (Planned)

**Goal:** Push scores to naneinf (~1.80e308) using transfer learning on Phase 1 weights.

The agent will learn:
- Infinite scaling combos (scaling jokers that compound over time)
- Deck thinning strategies (reduce deck to increase consistency)
- Suit-focused builds (concentrate on one suit for flush synergies)
- Boss blind manipulation (skip or prepare for specific boss effects)
- Long-game economy (accumulate wealth for later antes)

---

## Key Design Decisions

### Why PPO over DQN?

- **Variable action space** — Balatro's valid actions change dramatically between game phases. PPO handles continuous/discrete mixed spaces better than DQN.
- **Long horizon** — A full Balatro run is 100+ decisions across 8+ antes. PPO's advantage estimation handles long-horizon credit assignment.
- **Training stability** — PPO's clipped objective prevents catastrophic policy updates, critical when the reward signal is sparse (most feedback comes at game end).

### Why Hybrid (NN + Heuristics)?

Pure RL would require millions of games to learn basic Balatro math from scratch. The hybrid approach:
- **Heuristics** handle what's computable — exact scoring, hand classification, joker ordering
- **NN** handles what requires judgment — shop strategy, risk assessment, build direction
- **Action masks** bridge the gap — soft biases that guide exploration without hard-blocking learning

### Why Property Fingerprints over Joker IDs?

Encoding jokers by their properties (chips, mult, xmult, triggers, timing) rather than as one-hot IDs means:
- The network generalizes across similar jokers automatically
- New jokers (mods) work without retraining if their properties are encoded
- The state space stays manageable (54 fields per joker vs. 150-dim one-hot)

---

## Acknowledgments

### BalatroBot API

This project is built on top of **[BalatroBot](https://github.com/coder/balatrobot)**, the JSON-RPC 2.0 API mod that makes programmatic Balatro interaction possible. Without this foundational work, none of this project would exist.

**BalatroBot Authors:**
- **[S1M0N38](https://github.com/S1M0N38)** (primary author)
- **[stirby](https://github.com/stirby)**
- **[phughesion](https://github.com/phughesion)**
- **[besteon](https://github.com/besteon)**
- **[giewev](https://github.com/giewev)**

BalatroBot is licensed under the [MIT License](https://github.com/coder/balatrobot/blob/main/LICENSE).

### Balatro

[Balatro](https://www.playbalatro.com/) is created by **LocalThunk**. This project is a fan-made AI research project and is not affiliated with or endorsed by LocalThunk or Playstack.

### Tools

- **[PyTorch](https://pytorch.org/)** — Neural network framework
- **[Steamodded](https://github.com/Steamopollys/Steamodded)** — Balatro mod loader
- **[Claude Code](https://claude.ai/claude-code)** — AI-assisted development

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The BalatroBot mod (used as a dependency, not included in this repo) is separately licensed under [MIT](https://github.com/coder/balatrobot/blob/main/LICENSE) by its respective authors.
