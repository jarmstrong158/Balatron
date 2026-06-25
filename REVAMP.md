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

- [x] **1a. Scaling jokers in the shop** — DONE (commit 66c1ffb). `_project_shop_scaling_value`
      in `hand_eval.py` projects start + inc*(antes_left*2), capped (xmult≤6, flat≤50),
      applied in the valuation scorer's shop fallback. Shop Hologram/Vampire/Green Joker
      now valued by a fair mid-run value instead of ~×1.0. 5 tests.
- [x] **1b. `magnitude_source` jokers** — DONE (commit 3f25153). `_resolve_magnitude_contribution`
      + `_magnitude_count` in `hand_eval.py`; Steel/Stencil xmult, Stone/Bull/Banner chips,
      Erosion/Swashbuckler/Supernova mult, gate jokers (Mystic Summit/Driver's License/
      Bootstraps). Wired into the valuation scorer. 8 tests.
- [x] **1c. Economy engines** — DONE (commit 3f25153). `_estimate_joker_value` gives
      `economy=True` jokers a build value `money_per_round * 30` so all ~62 econ jokers
      are no longer invisible to the buy decision. 1 test.
- [ ] **1d. Live scoring path (`compute_joker_scoring`) magnitude** — the VALUATION path
      (buy decision) is fixed; the live tactical scorer still mis-scores some magnitude
      jokers (affects which hand to play when holding them). Secondary; revisit if it bites.

**Checkpoint:** PILLAR 1 DEPLOYED. agent should buy scaling/Steel/economy at sane rates;
watch fresh-only panel for a small bump even before planning (greedy-but-accurate).

## PILLAR 2 — PLANNING  (give him the computer's superpower)
Zero lookahead today; greedy single-step buys. Add forward search at build decisions.

- [x] **2a. Future-ante blind-target curve** — DONE (commit e148c77). `planner.py`
      `ANTE_BASE_TARGET` + `ante_target(ante, blind)` (boss=2x), extrapolated past 8.
- [x] **2c. Wire into the shop seam (first slice)** — DONE (e148c77). `build_survivability`
      = fractional deepest ante a build clears; `build_value(joker)` = survivability gain;
      `action_executor._planner_pick_joker` makes the PLANNER choose which joker to buy
      under policy_authority (policy still owns buy-vs-reroll-vs-skip tempo). 5 tests.
- [ ] **2b. Full search** — current slice is depth-1 + multi-ante horizon eval. Upgrade to
      shallow expectimax/beam over shop actions (reroll/draw as chance nodes) with the RL
      **value head** as the leaf evaluator. Also: extend `_planner_pick` to the full-slot
      (sell-weakest-then-buy) path and to reroll EV.

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
