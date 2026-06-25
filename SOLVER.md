# Balatron Solver — from approximate planner to win-probability search

**Goal:** consistently beat White Stake (~85%). The dec-035 ceiling audit proved
the dec-034 hybrid caps ~0.5–2% because the planner optimizes a **depth-1 static
survivability proxy**, while winning requires builds whose engines **scale fast
enough to keep pace with exponential blind growth** (each deep ante is a ~35%
gate; 85% needs ~96% → builds must be ~10–100× stronger by ante 5–6).

Fix: turn the approximate planner into a real **search-based solver** — an
accurate win-probability evaluator + lookahead, optimizing the whole run toward
the late-game targets. Built in phases; measured by the **deep-ante conditional
advance rate** (dec-035), not just win rate.

Foundation that already exists: the side-effect-free scoring sim
(`hand_eval.estimate_score_for_hand_type`, handles multiplicative xmult stacking),
the `ANTE_BASE_TARGET` curve, and the RL value head (candidate learned leaf).

---

## PHASE 1 — WIN-PROBABILITY EVALUATOR (trajectory-aware)  ← START HERE
Replace `build_survivability` (static power vs curve) with `win_probability`:
project each engine's value FORWARD to each future ante (scaling growth), score
the build's power at that ante (multiplicative stacking preserved), and compute
P(clear ante a) vs the boss target; win prob = product over remaining antes.
This makes the planner value builds by whether they OUT-SCALE the exponential
curve — directly attacking the ceiling.

- [ ] `_project_jokers(jokers, gamestate, antes_ahead)` — copy jokers, set each
      scaling joker's `_scaled_value` to its projection N antes forward.
- [ ] `projected_power(jokers, gamestate, antes_ahead)` — score the projected build.
- [ ] `win_probability(jokers, gamestate)` — walk current..ante8, multiply per-ante
      clear probs (logistic in power/target).
- [ ] Planner ranks buys/swaps/reroll by **Δwin_probability** instead of Δsurvivability.

**Checkpoint:** deep-ante conditional advance (ante5→6, 6→7, 7→8) rises off ~35%.

## PHASE 2 — LOOKAHEAD / SEARCH
Depth-1 → multi-step. Expectimax/beam over the stochastic shop+reroll+draw tree
using the Phase-1 evaluator at the leaves; account for future shop randomness so
the solver values build PATHS, not single buys.

- [ ] Shop transition model (apply buy/sell/reroll → state').
- [ ] Sampling/expectimax over the next 1–2 shops; reroll as a chance node.
- [ ] (Optional) RL value head as a learned leaf evaluator (AlphaZero-style),
      trained on outcomes, replacing/augmenting the analytic win-prob.

**Checkpoint:** solver makes multi-step build commitments a depth-1 greedy can't.

## PHASE 3 — FULL STRATEGIC SCOPE
The solver owns the whole strategic chain jointly toward win prob: economy
(save→spike), deliberate hand-leveling (planets toward the committed hand,
modeled in the trajectory), reroll budget, boss prep. Keep the near-optimal
tactical heuristics (card selection, joker ordering).

**Checkpoint (the goal):** win rate climbs toward a reliable ante-8 clear.

---

## Principles
- Each phase: implement → unit test → deploy → measure the **conditional advance
  curve** + win rate on clean fresh self-play. Keep the agent runnable throughout.
- The evaluator is the heart — accuracy there caps everything. Validate it against
  real outcomes (does higher win_probability correlate with actually winning?).
- dec-036 tracks this; record each phase's outcome.
