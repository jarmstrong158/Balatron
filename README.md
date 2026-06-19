[![Tests](https://github.com/jarmstrong158/Balatron/actions/workflows/tests.yml/badge.svg)](https://github.com/jarmstrong158/Balatron/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
  - [Behavior-Cloning Kickstart](#behavior-cloning-kickstart)
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
- [Decisions & Gotchas Log](DECISIONS.md)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## How It Works

Balatron connects to a live instance of Balatro through the [BalatroBot](https://github.com/coder/balatrobot) mod, which exposes the full game state and accepts action commands via a JSON-RPC 2.0 HTTP API on `127.0.0.1:12346`. No screen capture or computer vision is needed — the agent reads structured game data directly.

The agent observes the game state (hand cards, jokers, economy, blind targets, deck composition, shop contents), encodes it into an 838-dimensional vector, and uses a PPO neural network to select actions. A sophisticated heuristic layer validates and enhances the network's decisions with Balatro-specific domain knowledge — optimal hand selection, joker ordering, consumable usage, pack evaluation, and economy management.

```
Balatro Game
    |
    | JSON-RPC 2.0 (BalatroBot mod)
    v
Game State Encoder (838-dim vector)
    |
    v
PPO Neural Network (shared trunk + 3 policy heads)
    |
    v
Action Mask (hard legality only)
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

Balatron uses a **hybrid architecture** — but the split is **judgment vs. computation**, and the policy owns the judgment. As of the 06-13 audit (`policy_authority`), the network's chosen action actually **executes**: it decides play-vs-discard, which joker to buy, when to reroll/skip/sell/leave. The heuristic layer was demoted from *making* those calls (it used to override them, which made the net decorative and unable to learn) to **tactical computation + hard legality only**. That's what makes this a genuine RL agent — the policy's decisions have causal stake in outcomes.

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **Neural Network (policy)** | **All judgment calls** — and they execute | Play vs. discard, which joker to buy, when to reroll/skip/sell/leave shop |
| **Heuristic — tactical math** | The *computation* behind the policy's decision (rules, not strategy) | Which exact cards score best for the chosen play, optimal joker order, draw probabilities |
| **Heuristic — hard legality** | Block only structurally invalid actions | Affordability, slot limits, never sell your only joker, force-buy Blueprint/Brainstorm |
| **Scoring Engine** | Full Balatro math for hand evaluation | Exact scores with joker effects, retriggers, editions, enhancements, boss debuffs |
| **Observation features** | Surface the computed math *to* the policy so its judgment is informed | Hand-eval block (gap-to-target, draw odds), shop context (per-joker marginal value, build coverage) |

### Neural Network

```
Input (838) --> Shared Trunk (838 -> 768 -> 768 -> 512)
                    |
                    |--> Play Head   (512 -> 256 -> 45)  -- SELECTING_HAND state
                    |--> Shop Head   (512 -> 256 -> 45)  -- SHOP state
                    |--> Blind Head  (512 -> 128 -> 45)  -- BLIND_SELECT state
                    |--> Value Head  (512 -> 256 -> 1)   -- all states
```

- **3 policy heads** specialized for different game phases — the shop head never sees hand-play decisions and vice versa
- **Shared trunk** learns representations useful across all phases (joker synergies, economy state, deck composition)
- **Layer normalization** + **ReLU activation** for training stability
- **Action-conditioned target sampling** — after selecting an action type, target logits are masked so only valid targets can be chosen (e.g., `buy_pack` can only target pack slots, not joker slots)

### State Vector

The game state is encoded into an **838-dimensional float vector** with careful normalization:

| Section | Dimensions | Contents |
|---------|-----------|----------|
| Game Meta | 45 | Ante, round, money (log-scaled), hands/discards left, reroll cost, chips/target (log), boss blind info, blind statuses, joker slots |
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
| Shop Context | 16 | Per-shop-joker marginal value, build coverage, economy/slot pressure (SHOP only) |
| Joker Velocity | 5 | Per-owned-joker Δ scaled value over the last 8 hands, signed-log (compounding vs. decaying engine) |

**Design principle:** Jokers are encoded as **property fingerprints** (what the joker *does*) rather than joker IDs. This means the network generalizes across jokers with similar effects — it doesn't need to memorize 150 individual joker behaviors, it learns that "x2 mult on face cards" is valuable regardless of which joker provides it.

### Action Space

Actions are represented as a **14-element tensor**: `[action_type(1), card_selection(12), target(1)]`

**14 action types:**

| Index | Action | Used In |
|-------|--------|---------|
| 0 | Play hand | SELECTING_HAND |
| 1 | Discard | SELECTING_HAND |
| 2 | Buy shop card (joker/planet/tarot) | SHOP |
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
| Game Win (Ante 8 clear) | +15.0 | Phase 1 primary goal — largest single signal |
| Game Loss | -5.0 + 0.3/ante | Harsh base penalty, softened by progress |
| Naneinf Achievement | +50.0 | Phase 2 ultimate goal |
| Ante Cleared | +3.0 + 1.0/ante | Scales steeply with depth |
| Blind Cleared | +1.0 (+1.5 boss) | Per-blind progress signal |
| Score vs Target | +0.5 * log10(ratio) | Log-scaled overkill bonus |
| Score Progress | +0.02 * log10(chips) | Small per-hand nudge (cut from 0.1 — was the dense reward dominating the signal) |
| Money Gain / Spent | +0.01 / -0.01 per $ | Economy is a means, not the goal |
| Scaling Growth | +0.05/log-unit | Encourages scaling joker investment |

> These weights were **rebalanced toward outcomes** (06-13): the dense per-step
> shaping was shrunk and depth/win rewards steepened, after a plateau audit
> showed the policy was optimizing comfortable mid-game scoring instead of
> winning. The single-hand high-water bonus is **Phase-2 only** (it farms chips
> against depth in Phase 1). See [DECISIONS.md](DECISIONS.md) / Context Keeper.

All score-based rewards use **log10 scaling** because Balatro scores grow exponentially — the agent learns to push the *exponent* higher.

Two structural rules keep the signal honest:

- **State bonuses are potential deltas, never per-step accruals.** Joker-diversity and interest-tier bonuses pay only on *change* (acquire a category: +once; lose it: −once; hold: zero). Paid per-step, they accrued +20–40 over a run vs +10 for winning — the agent was being paid more for existing than for winning.
- **Each reward is credited to the action that caused it.** A state delta only becomes observable on the *next* fetch, so it's amended onto the previous transition rather than stored with the current action — otherwise every step reward lands one decision late, often on a different policy head.

### Behavior-Cloning Kickstart

The hybrid design has a structural catch: when heuristics override most consequential decisions (the planner picks play vs discard, shop buys get redirected), the policy's choices barely influence outcomes — so PPO's gradient is honestly near zero and the network coasts. Two mechanisms fix this:

1. **Executed-action storage.** When a heuristic overrides the sampled action, the rollout buffer stores what *actually ran* (re-encoded to the action tensor, with its log-prob under the current policy) — so PPO credits outcomes to the real cause instead of training the policy heads on noise.

2. **Annealed imitation loss.** Overridden steps are flagged as *teacher corrections*, and an auxiliary behavior-cloning term — `bc_coef · (−log π(executed action))`, only on flagged steps — distills the heuristic layer into the policy. `bc_coef` anneals linearly from 0.5 to 0 over 200 updates (anchored to first engagement, persisted in checkpoints across restarts). Early on the policy imitates the teacher; as the coefficient decays, PPO's reward signal — which *can* disagree with the teacher — takes over.

This is the AlphaGo-style recipe (supervised warm-start, then RL surpasses the teacher). **The hard overrides have now been lifted** (Path A / `policy_authority`, 06-13): the policy's judgment calls execute directly, and the override fraction dropped from ~8% to 0%. `entropy_coef` was escalated (0.01 → 0.025 → 0.04) to push exploration under Path A.

3. **The bias masks have come off too (06-14).** The action mask used to inject `log(action_mask)` — `exp(HAND_BIAS_STRENGTH)` heuristic bias — directly into the policy logits. That bias was ~57× the network's own signal, so it *structurally floored entropy at ~0.24* and the policy never actually learned the masked decisions. The mask is now **hard legality only** (legal → raw logit, illegal → −1e9), and the heuristic guidance is re-homed as a **separate annealing prior-KL teacher**: `KL(heuristic_prior ‖ policy)`, weighted by `prior_coef` (0.5, annealed to 0 over 400 updates). The policy keeps the crutch early and owns the decision once it anneals out. Legality masks stay forever; the bias was the last trainer-wheel, and it's off.

---

## Scoring Engine

The scoring engine (`hand_eval.py`) implements the full Balatro scoring formula, in the game's actual order — card/held x-mults (Glass, Polychrome, Steel) fire during card scoring, *before* jokers add flat mult:

```
Score = (hand_chips + card_chips + joker_chips)
        x ((hand_mult + card_mult) x card_xmult + joker_mult)
        x joker_xmult
```

**Features implemented:**

- Complete hand classification (High Card through Flush Five)
- All 150 joker effects — chips, mult, xmult, scaling, conditional triggers
- **Retrigger system** — Hanging Chad (first card +2), Sock and Buskin (face cards +1), Hack (2/3/4/5 +1), Seltzer (all +1), Dusk (last hand), Red Seal (+1)
- Card enhancements: Bonus (+30 chips), Mult (+4 mult), Wild (any suit), Glass (x2, may shatter), Steel (x1.5 held), Stone (50 chips, no rank/suit), Gold (+3 money), Lucky (chance mult/money)
- Card editions: Foil (+50 chips), Holographic (+10 mult), Polychrome (x1.5)
- Boss blind handling: suit debuffs (Club, Goad, Window, Head), face debuff (Plant), scoring debuffs (Flint, Arm), economy bosses (Ox, Tooth), and enforced hand restrictions — The Psychic (plays padded to exactly 5 cards), The Eye (no repeat hand types), The Mouth (locked to the round's first hand type), The Needle (never discard)
- Spectral pack evaluation — ranks all spectral cards, takes Hex/Ankh only as a free single-joker upgrade, targets seals/editions at the best hand card, skips when nothing is safe
- Discard ordering protects held value: Blue Seal, Steel and Gold cards discard last
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
- Chase target viability — only discards when the chase target can realistically beat the blind
- Baseline scoring — uses actual expected hand value (planet levels + joker effects) for multi-hand projections

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
|   |-- game_state.py       # 838-dim state vector encoder
|   |-- action_space.py     # Action masks, target mapping, joker evaluation
|   |-- hand_eval.py        # Full scoring engine, hand classifier, strategic advisor
|   |-- reward.py           # Reward shaping (log-scaled, multi-tier)
|
|-- data/
|   |-- jokers.py           # 150 joker schemas + tier weights
|
|-- training/               # decoupled 06-14 (train.py was a 3.5k-line monolith)
|   |-- train.py            # Orchestrator: rollout loop, PPO cadence, checkpoints, metrics
|   |-- config.py           # TrainConfig (hyperparameters -> PPOConfig)
|   |-- action_executor.py  # ActionExecutor: action->API translation + shop/pack auto-actions
|   |-- env_session.py      # EnvSession: all per-instance state (one per parallel game)
|   |-- episode_tracker.py  # EpisodeTracker: per-episode/lifetime stats, win log
|   |-- joker_order_logger.py # JokerOrderLogger: per-round joker-ordering trace
|
|-- recorder.py             # Automated win recording via ffmpeg gdigrab (+ NullRecorder)
|-- supervise.py            # Memory-guardian supervisor: crawl detection, recovery, Steam reclaim
|-- ensure_supervisor.py    # Watchdog-for-the-watchdog (scheduled task resurrects the supervisor)
|-- dashboard.py            # Live training dashboard (stdlib HTTP, port 8777)
|
|-- scripts/
|   |-- sim_bloodstone.py   # Simulation utilities
|
|-- tests/
|   |-- test_scoring.py     # Scoring engine validation
|
|-- DECISIONS.md            # Running log of design decisions + hard-won gotchas
|-- NOTES.md                # Architecture decisions, state vector layout, design rationale
|-- LICENSE                  # MIT License
|-- README.md                # This file
```

---

## Prerequisites

- **Python 3.12+**
- **PyTorch** (CPU is fine — the bottleneck is live-game rollout collection, not the network)
- **Balatro** (Steam version)
- **[Steamodded](https://github.com/Steamopollys/Steamodded)** (>= 0.9.8) — Balatro mod loader
- **[BalatroBot](https://github.com/coder/balatrobot)** mod (v1.4.1+) — JSON-RPC API for game control

### Hardware Used

- AMD Ryzen 9 9800X3D
- NVIDIA RTX 5070 Ti (unused for training — see above)
- Training runs at ~5,000 steps/hour (8× game speed, continuous Balatro gameplay)

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

> ⚠️ **After installing or updating the mod**, three local crash patches must be
> re-applied — they live in the mod directory, *not* in this repo, and a mod
> update silently wipes them. At high game speed, deferred animation events can
> fire after a fast programmatic state transition destroyed the UI object they
> reference, crashing the game (`round_eval`/`shop` "(a nil value)") or hanging
> `cash_out`. See [DECISIONS.md](DECISIONS.md) gotcha #5 for the three files
> (`round_eval_fix.toml`, `shop_nil_fix.toml`, `cash_out.lua`) and the pattern
> for patching new crashes of this class.

### 4. Install the BalatroBot CLI

```bash
pip install balatrobot
```

---

## Usage

### Supervised launch (recommended)

One detached process keeps the entire stack alive — it starts the server + game if port 12346 isn't listening, and (re)starts the trainer from the newest checkpoint whenever it dies:

```powershell
Start-Process -WindowStyle Hidden python -ArgumentList '-u','supervise.py' -WorkingDirectory 'C:\Users\jarms\repos\balatron'
```

The supervisor runs **multi-instance training**: `NUM_ENVS` parallel Balatro games (ports 12346–12348, currently **N=3**) feed one network through per-env rollout buffers — one brain, many bodies. Measured scaling: 196 steps/min (N=1) → 309 (N=2) → 433 (N=3). There is no "merging" at save time: every update consumes all envs' transitions in one gradient step, and checkpoints are the single network's weights exactly as in single-instance mode. Env 0 records win videos; the others play uncaptured.

Actions are logged to `logs/supervisor.log` (plus a one-line `logs/supervisor_status.txt`), trainer output to `logs/trainer_<timestamp>.log`. Stop it by creating a `SUPERVISOR_STOP` file. For overnight runs, also disable standby (`powercfg /change standby-timeout-ac 0`).

**The real cause of "slow after I'm away for hours" (06-14): external RAM starvation, not the trainer.** For weeks the trainer would crawl after long unattended runs — misdiagnosed three times as an internal "FPS decays ~1/n over the process lifetime." Direct measurement found the truth: Steam's `steamwebhelper.exe` leaks to **13–14 GB** over hours, the machine hits ~94% RAM, Windows pages Balatron (only ~4 GB) out, and the trainer crawls to ~12 steps/min. Reclaiming that one leaked process dropped system RAM **95% → 39%** and throughput recovered instantly. The supervisor now runs a **memory guardian**: when system RAM is critical it restarts the bloated external hog (`steamwebhelper.exe` > 4 GB — Steam respawns a fresh lightweight helper, no game interruption) and logs the diagnosis.

The rebuilt supervisor (06-14) is engineered to stay fast over a 7–8 h unattended run:
- **Fast, reliable crawl detection** from heartbeat steps/min over a 12-min window (floor 80) — *not* the log's cumulative FPS (a misleading lifetime average that hid the crawl). Freeze if no step in 4 min; churn if no checkpoint in 40 min; a **proactive 90-min age recycle** so the trainer never gets old enough to bloat.
- **Cascading kills** — every recycle kills ALL trainers + ALL games + ALL orphan launchers (the old single-PID kill let duplicates and orphans accumulate until RAM was exhausted).
- **Singleton** — kills any rival `supervise.py` on startup (two supervisors each spawned a trainer, doubling load).
- **Orphan reaper** each cycle (the `uvx → serve → serve → Balatro.exe` tree leaks ~1 launcher per restart), **log pruning** that first preserves PPO update lines into `logs/metrics_history.log` so the dashboard keeps its history, and a **watchdog scheduled task** (`ensure_supervisor.py`, every 10 min) that resurrects the supervisor itself if anything ever kills it.

The full recovery hierarchy:

| Failure | Detected by | Healed in |
|---|---|---|
| Game hangs/crashes | Trainer's internal watchdog | ~1 min |
| Trainer process dies | Supervisor existence check | ~30 s |
| Trainer freezes (alive, no progress) | Heartbeat staleness (4 min) | ~4 min |
| Trainer crawls (<80 steps/min) | Heartbeat rate, 12-min window | ~12 min |
| Trainer churns (no checkpoints) | Checkpoint cadence (40 min) | ~40 min |
| Trainer bloats over its lifetime | Proactive age recycle | ~90 min |
| External RAM leak (Steam) starves it | Memory guardian | ~30 s/cycle |
| Orphan launchers / duplicates pile up | Cascading reaper | ~30 s/cycle |
| Supervisor itself dies | `ensure_supervisor.py` watchdog task | ~10 min |
| Machine sleeps (kills everything) | Prevent it | `powercfg /change standby-timeout-ac 0` |

### Monitoring (live dashboard)

A single stdlib-only file serves a live, auto-refreshing dashboard — one place to answer *how many wins, is he improving, is training healthy* without grepping logs:

```powershell
python dashboard.py        # http://localhost:8777, refreshes every 30s
```

It reads `logs/` read-only (safe to run alongside training) and shows:

- **Outcomes** — lifetime wins/win-rate/best-ante, plus mean-ante and deep-run (ante ≥6/≥8) trend per 200-game bucket. Wins are too rare to trend; the depth curves are the real "is he improving" signal.
- **Learning health** (per PPO update) — reward, entropy (a cliff = exploration collapse), `KL` against the `target_kl` line, and `BC` loss against its flat-line watch threshold (the pre-committed "flat after 50 updates → raise `bc_coef`" trigger).
- **Diagnosis** — a final-ante histogram of the last 1,000 games (where the wall is) and a joker win-lift table (which jokers show up in winning runs more than chance).
- **Ops** — live steps/min, 24-hour restart count, and heartbeat age (turns red if stale >5 min).

### Manual training (two terminals)

Training requires two terminals running simultaneously:

**Terminal 1 — Start BalatroBot server + Balatro game:**
```powershell
uvx balatrobot serve --fast
```

This launches the Balatro game with the BalatroBot mod injected and starts the JSON-RPC API server.

Game speed is set via the `BALATROBOT_GAMESPEED` environment variable (e.g. `set BALATROBOT_GAMESPEED=8` before launching). **8× is the recommended setting** — speeds above that (100×/16×) caused stalls and desyncs. A brief drop to 4× during a crash investigation proved the deferred-event nil-crashes were *not* speed-bound (same crash cadence at half speed); the real cause was double-fired transitions, fixed by endpoint lock guards and the trainer's transition debounce. The trainer auto-restarts Balatro if the game hangs or crashes regardless.

**Terminal 2 — Start training:**
```powershell
cd balatron
$env:PYTHONUTF8 = "1"
python -u -m training.train --total-timesteps 1500000 --device cpu --checkpoint-interval 2 --num-envs 2
```

`--num-envs N` plays N parallel game instances (ports 12346..12346+N-1, one server each); default 1.

`PYTHONUTF8=1` is required when output is redirected or piped — the trainer prints emoji that crash on Windows cp1252. `--device cpu` is the practical choice: the wall-clock bottleneck is live-game rollout collection, not the (small) network, so a GPU buys almost nothing here. `--checkpoint-interval 2` saves every 2 PPO updates so a crash loses minutes, not hours.

Training progress is printed per PPO update:
```
Update   395 | Step  814,922 | FPS 975 | Ep 85 | R 2.71 | Ante 3.1 | WR 0.0% | PL -0.0047 | VL 26.92 | Ent 0.468 | KL 0.0022 | CF 0.019 | BC 0.547@0.03(2%) | Pr 0.127@0.49 | LR 1.43e-04
```
`PL`/`VL`/`Ent` are the PPO policy loss, value loss, and entropy. `KL` is the true approximate KL divergence (healthy drift is ~0.01–0.05; the `target_kl=0.03` early-stop trims epochs when it overshoots). `CF` is the clip fraction (healthy ~0.1–0.2). `R` is the genuine per-episode mean reward. `BC loss@coef(frac)` is the behavior-cloning kickstart — the imitation loss, its current (annealing) coefficient, and the fraction of the rollout that was heuristic-overridden. `Pr loss@coef` is the heuristic **prior-KL** teacher — the KL pull toward the heuristic and its annealing coefficient (06-14). (Entropy now reads ~0.4–0.5 since the bias mask was lifted; before 06-14 it was pinned ~0.24, masked by an `≈ln(19)` artifact that read as ~2.6.)

> A note on trusting these numbers: three display metrics were found lying in a single day — `KL` was never accumulated into the printout (read 0.0000 forever), `CF` was computed but never printed, and `R` was a cumulative session sum masquerading as a per-episode mean. All fixed; the training signal itself (PPO buffer rewards) was never affected. Lesson recorded in [DECISIONS.md](DECISIONS.md): verify a metric's computation path before reasoning from it.

### Resuming from Checkpoint

```powershell
python -u -m training.train --total-timesteps 1500000 --device cpu --checkpoint-interval 2 --checkpoint checkpoints/balatron_phase1_update000200.pt
```

Checkpoints are saved automatically during training, every N PPO updates (`--checkpoint-interval`, default 10 — use 2) as `checkpoints/balatron_phase1_updateNNNNNN.pt`, plus a `..._final.pt` on exit. Resume from the latest `updateNNNNNN.pt` (not `_final.pt`, which may be a crash auto-save) to continue where training left off.

### Recording

Only **winning** runs are recorded and kept (via ffmpeg screen capture); every losing run is discarded. Use `--no-record` to disable recording entirely (reduces CPU/disk overhead):

```powershell
python -u -m training.train --total-timesteps 1500000 --device cuda --checkpoint checkpoints/balatron_phase1_final.pt --no-record
```

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

> 📋 **For the full, running log of design decisions *and* hard-won gotchas
> (the `won`-flag trap, pack-pick handling, out-of-repo mod patches, etc.), see
> [DECISIONS.md](DECISIONS.md).** The highlights are below.

### Why PPO over DQN?

- **Variable action space** — Balatro's valid actions change dramatically between game phases. PPO handles continuous/discrete mixed spaces better than DQN.
- **Long horizon** — A full Balatro run is 100+ decisions across 8+ antes. PPO's advantage estimation handles long-horizon credit assignment.
- **Training stability** — PPO's clipped objective prevents catastrophic policy updates, critical when the reward signal is sparse (most feedback comes at game end).

### Why Hybrid (NN + Heuristics)?

Pure RL would require millions of games to learn basic Balatro math from scratch. The hybrid approach:
- **Heuristics** handle what's computable — exact scoring, hand classification, joker ordering
- **NN** handles what requires judgment — shop strategy, risk assessment, build direction
- **Action masks** enforce hard legality only; heuristic guidance enters as a *separate annealing prior-KL teacher* (not baked into the policy softmax) so it can fade and let the policy surpass it

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
