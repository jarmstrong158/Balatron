# Balatron Revamp — Reactor → Planner

**Goal:** consistently beat White Stake (clear ante 8), the way a skilled human does ~85%.
The agent plateaus at ~ante 3.8 fresh because it is a **greedy local optimizer with
incomplete knowledge and no memory** — and greedy-in-isolation *is* a ~ante-4 strategy.

Three deep audits (2026-06-24) converged on three deficits. The revamp closes them in
dependency order: **you can't search on wrong numbers, and you can't steer a search
without a goal.**

The hard prerequisite already exists: a fast, accurate, side-effect-free scoring
simulator (`hand_eval.estimate_score`, `compute_joker_scoring`, `find_best_hands`) and
an RL value head that estimates long-horizon win value (usable as a search leaf).

Audit agent IDs (resume via SendMessage for detail):
- Planning/lookahead: `ab60ad2beadd85b09`
- Joker knowledge/interactions: `ae337cfd1616bd4f3`
- Strategy/commitment: `a617304eba0cded2c`

---

## PILLAR 1 — KNOWLEDGE  (make him *see* correctly)  ← FOUNDATION, START HERE
Build valuation is wrong, so even greedy choices are wrong. ~55-60/150 jokers scored
accurately today.

- [ ] **1a. Scaling jokers in the shop** — `_scaled_value` is owned-only, so shop scaling
      jokers (Hologram/Vampire/Constellation/Green Joker/…) are valued at start (~×1.0 =
      worthless) at the moment of purchase. Project a fair forward value for shop
      candidates. *(highest single impact)*
      Files: `environment/action_space.py:171` `_estimate_joker_value`,
      `environment/hand_eval.py:3290` `estimate_score_for_hand_type`, scaling injection
      in `environment/game_state.py`.
- [ ] **1b. `magnitude_source` jokers** — ~10 jokers carry magnitude metadata never read,
      so they score their flat base or ×1.0. Steel Joker & Joker Stencil score ×1.0 (no
      effect) everywhere; Stone, Erosion, Swashbuckler, Banner, Driver's License,
      Supernova, Throwback, Mystic Summit magnitude. Implement the magnitude table in
      BOTH scoring paths (`compute_joker_scoring` AND `estimate_score_for_hand_type`).
- [ ] **1c. Economy engines** — ~62 economy/utility jokers score 0. Give money-generators
      (Rocket, Bull, To the Moon, Golden, interest jokers) a build value (money → future
      power) so tempo/economy investment is visible.
- [ ] **1d. Unify / cross-check the two scoring paths** so fixes can't silently drift.

**Checkpoint:** agent buys scaling/Steel/economy jokers at sane rates; expect a small
fresh-ante bump even before planning (greedy-but-accurate).

## PILLAR 2 — PLANNING  (give him the computer's superpower)
Zero lookahead today; greedy single-step buys. Add forward search at build decisions.

- [ ] **2a. Transition model** — `simulate_shop_action(state, action) -> state'` (apply
      joker to roster, advance money/interest) + the known Balatro **future-ante
      blind-target curve** (not modeled today).
- [ ] **2b. Search** — shallow expectimax/beam over shop actions (branching ≤ ~8):
      scoring sim as the model, RL **value head** as the leaf evaluator (AlphaZero-style
      search + learned value). Even depth 1–2 lets him value build-*potential*.
- [ ] **2c. Wire into the shop seam** `training/action_executor.py:288-503`, replacing the
      greedy `_estimate_joker_value` core.

**Checkpoint:** agent makes build-around buys (weak-now/strong-later) a greedy agent never would.

## PILLAR 3 — COMMITMENT  (give him a multi-ante through-line)
Feedforward + memoryless: no place for a plan to live. Today he levels random hand
types and his reward fights his own economy.

- [ ] **3a. Build-target representation** — a persistent run-level archetype/win-condition
      he commits to; encode in state; planner + heuristics condition on it.
- [ ] **3b. Deliberate hand-leveling** — acquire/use planets toward the committed hand
      type (fix "use planet on sight / pick index 0").
- [ ] **3c. Economy save-then-spike** — replace the anti-hoard penalty with bankroll-aware
      spike timing.
- [ ] **3d. Boss preparation** — shop conditions on the upcoming boss (perception already
      exists; preparation doesn't).

**Checkpoint (the goal):** fresh-run win rate climbs off ~1% toward a reliable ante-8 clear.

---

## Principles
- Each item: implement → unit test → deploy → measure on the **fresh-only** dashboard
  panel (loaded runs inflate the average). Keep the working agent runnable throughout.
- Keep the near-optimal tactical heuristics (card selection, joker ordering) — the gap is
  strategic, not tactical.
- Context-keeper: dec-034 tracks this revamp; record each pillar's outcome.
