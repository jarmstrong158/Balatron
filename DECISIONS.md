# Decisions & Gotchas

A running log of the **why** behind Balatron's design, and the hard-won lessons
behind its fixes. New sessions (human or AI) should read this before changing
core logic. The machine-queryable mirror lives in `.context/` (Context Keeper);
**the two are kept in sync** — every recorded `dec-NNN` is mirrored here in the
same commit (decisions ≤ dec-034 were also manually maintained; dec-035→047 were
back-filled from Context Keeper on 06-29).

---

## Architecture & Design Decisions

### Hybrid: PPO policy on top of a heuristic layer
The agent is **not** pure RL. A PPO actor-critic (838-dim state → shared trunk
→ 3 state-specific policy heads + value head, ReLU) makes the *judgment* calls
(shop strategy, when to leave, risk, build direction), while a heavy heuristic
layer (`hand_eval.py`, `action_space.py`, shop logic in `train.py`) computes the
*mechanical* parts (optimal hand/discard, joker ordering, scoring, must-buy/sell
guards).

- **Why:** pure RL would need millions of games to learn basic Balatro math
  against a ~0.5% win signal. The hybrid is win-capable fast.
- **Tradeoff:** the network can't learn the tactics the heuristics already
  decide. "Deeper strategy / combo discovery" would require moving decisions
  from heuristics → policy, a relational (attention) encoder, and far more
  training — at the cost of early competence.

### Jokers encoded as property fingerprints, not IDs
Each joker is a 54-field fingerprint (effect flags, values, triggers, edition,
runtime scaling) rather than a one-hot ID. The net generalizes across jokers
with similar effects and tolerates new/modded jokers, and the state stays small.

### Why PPO over DQN
Variable action space across game phases, long-horizon credit assignment
(100+ decisions/run), and clipped-objective stability under a sparse reward.

### Win is gated on `ante > 8`, not the `won` flag
See gotcha #1 below. The win reward (`reward.py`) and win recording (`train.py`)
key off **getting past the ante-8 boss** (ante advances to 9 in endless, or a
post-boss SHOP/ROUND_EVAL state is reached) — never the raw `won` flag.

### BC kickstart: distill heuristics into the policy, then lift the overrides
Weight-delta analysis (update 198→202) proved the policy heads were barely
learning (KL pinned at 0.0000; the blind head literally frozen — one legal
action means zero policy gradient) because heuristics override most
consequential decisions. The path to real learning: (1) store **executed**
actions, not sampled ones (commit `31c527a`); (2) a behavior-cloning
auxiliary loss imitates the heuristic teacher **only on overridden steps**
(`bc_flag`), with `bc_coef` annealed 0.5 → 0 over 200 updates, anchored to
first engagement and persisted in checkpoints (commit `52836ec`); (3) only
*then* lift authority gradually — play/discard tempo first, then shop
overrides one at a time. Legality masks stay forever; bias masks are the
trainer wheels that come off. BC can never exceed the teacher — the anneal
to zero is what lets PPO surpass it. **(06-14: the bias masks have now come
off — the in-softmax bias was removed and re-homed as an annealing prior-KL
teacher; see "Binary mask + prior-KL" below.)**

### Path A: policy authority (the override lift, finally done) — 06-13
A 4-agent deep audit found the lift in the BC-kickstart plan above was never
actually executed: `_action_to_api_call` still let the heuristic re-decide
play-vs-discard, which exact cards, and **scan the whole shop to buy "the best"
joker over the net's pick**. PPO trained on the heuristic's action, so the
policy had zero causal stake — it imitated the teacher to the teacher's ceiling
(~mean ante 4) and then *regressed* (4.0 → 3.4); entropy sat flat ~2.6.
**Fix (`self.policy_authority`, default True):** the policy now executes its own
judgment calls — play-vs-discard and which joker to buy. The heuristic is
demoted to **tactical computation** (the best *cards* for the policy's chosen
action) + hard-legality guards (affordability, BAD/must-buy jokers). This is the
correct NN-for-judgment / heuristic-for-computation split. **Expect performance
to drop first**, then climb past the old ceiling — the policy is taking over
decisions the heuristic used to make perfectly. Flip the flag to revert.
Also fixed in the same batch: a non-learnable `ln(19)` target-entropy constant
that *pinned* entropy (gate target log-prob/entropy on "type has a target", like
the card bits); `REWARD_HAND_HIGH_WATER` farms chips in Phase 1 so it's Phase-2
only; the blind-clear score-ratio read post-SHOP chips (always 0); and a
cross-env `_verdant_leaf_sold` state leak. See Context Keeper `dec-010`.

### Binary mask + prior-KL: the bias-mask wheels finally come off — 06-13/14
Path A lifted the heuristic *override* but left the heuristic *prior* baked into
the softmax. `build_action_mask` returns `exp(HAND_BIAS_STRENGTH·k)` (H=5.0)
**bias multipliers**, and `get_action_and_value` added `log(mask)` straight into
the policy logits — a ±4–5 nat prior, ~57× the head's own signal. Three
independent audits (two subagents + a trajectory audit) converged: this
structurally **floored entropy at ~0.24** and meant the policy never had to learn
the masked decisions. Proof: with a binary mask the head sits at ~ln 2 (uniform)
on play-vs-discard — after 380 updates it had *no independent opinion* on the
game's most common decision. (The earlier "entropy 2.6 → 0.24 collapse" at update
333 was a red herring — it was the `ln(19)` gating fix changing the *measurement*,
not a policy event; real entropy was always ~0.24. See `dec-014`.) Policy-head
softening ×0.4 did nothing because scaling a 0.07 signal inside a sum dominated by
±4 is invisible (`dec-013`).
**Fix (`dec-015`, `con-011`):** `get_action_and_value` now uses the mask as a
**hard legality gate only** (legal → raw head logit, illegal → −1e9). The
heuristic guidance is re-homed as a **separate annealing prior-KL** term:
`KL(heuristic_prior ‖ policy_type_dist)` weighted by `prior_coef` (0.5 → 0 over
400 updates, anchored at first engage, persisted in checkpoints) — the policy
keeps the crutch early and owns the decision once it anneals out. No new params
(prior is computed from the stored mask), so old checkpoints load unchanged.
Smoke-tested: type entropy 0.007 → 0.59 (285× headroom restored). **Expect ante
to regress hard** while the policy relearns play/discard/buy from near-scratch —
the prior-KL cushions but won't erase it. Secondary (deferred stage 2): value &
policy share the trunk and VL=15–30's gradient swamps the policy gradient ~3000×.

### Per-joker growth velocity in the observation — 06-18 (`dec-020`)
The joker fingerprint already encodes scaling *flags*, *increment size*,
*start value*, and *current scaled value* — so "is this a scaler / how big /
how high now" is covered. The genuine gap: a feedforward policy sees ONE state
snapshot and can't infer recent growth *rate*, so it can't tell a compounding
engine firing **now** from a high-increment joker that isn't actually firing
(dead weight). Added a **JOKER_VELOCITY** block (5 dims, one per owned slot;
**STATE_VECTOR_SIZE 833 → 838**). `ScalingTracker` snapshots each scaled value
once **per hand played** into a rolling window (`VELOCITY_WINDOW=8`);
`get_velocity` = current − oldest-in-window, signed-log-normalized (+ compounding,
− decaying e.g. Ice Cream/Popcorn, 0 new/flat). Appended **last**, re-using the
`dec-011` zero-pad migration pattern (verified end-to-end against the live U612
checkpoint: 833 cols byte-preserved, 5 new cols exactly zero). Observation-only,
no reward shaping. **Committed (`eccb883`) but NOT deployed** — deliberately
sequenced behind the exploration lever (`dec-019`): the diagnosed plateau is
exploration/convergence, not observation poverty, so deploying now would muddy
attribution. The next trainer restart will pick it up and cleanly migrate.

### Multi-instance training: one brain, many bodies
N parallel Balatro games (ports 12346+) feed ONE network. Per-env
`RolloutBuffer`s keep amend-last credits and GAE temporal adjacency correct;
`update()` computes GAE per env then concatenates for minibatching — the
"convergence" happens every update, never at save time (checkpoints are the
single network's weights, unchanged). All per-run state lives in `EnvSession`
(`training/env_session.py`; ~24 attributes + game client/reward calc/recorder);
anything left as a Trainer singleton would bleed across games (a stale win flag
from env 0 marking env 1's loss as a win). Env 0 owns the win recorder; others
get a `NullRecorder` (now in `recorder.py`). **Currently N=3** (ports
12346-12348); the supervisor launches with `--num-envs 3`. Per-game kills are
PID-scoped, but the rebuilt supervisor (gotcha #6) deliberately **cascades** on
recycle — killing ALL trainers + ALL games + ALL orphan launchers — so nothing
accumulates. First deployed N=2 on 2026-06-11 (combined ~309 steps/min vs ~196
single, +58%); since raised to N=3.

### The ante ~3.7 plateau and the relational encoder — 06-22/23 (`dec-029/030/031`)
A long attractor at ante ~3.7 survived every *training-signal* lever
(exploration, value_coef, shop-authority, SIL, perception, incentive,
gate-lift). The arc that finally localized it:
- **Un-freeze** (`dec-029`): decoupled the value trunk from the policy trunk
  (`network.py` `value_trunk`) + cut `entropy_coef` 0.10→0.03. NULL — KL settled
  back to ~0.0043, proving the policy was *converged for the reward*, not frozen
  by a mechanism.
- **Curriculum** (`dec-030`): harvest ante-4/5 partial-build states via the
  BalatroBot `save` endpoint and `load` an annealed fraction of rollouts from
  them, so deep-build experience becomes dense. NULL — loads fired flawlessly
  but the fresh-run leading indicator (≥2 xmult by ante 3) stayed flat ~3.4% and
  win density didn't accelerate. Handed deep builds, the agent still couldn't
  convert them.
- **Relational encoder** (`dec-031`): the curriculum null localized the ceiling
  to the build-decision *representation*. Audits confirmed the policy already
  owns the joker-buy decision (`policy_authority`) and the remaining heuristics
  (`find_best_hands`, `compute_optimal_joker_order`) are near-optimal — so the
  problem is that the policy makes that buy from a *flat* 842-vector and can't
  relate joker-to-joker. `agent/set_encoder.py` runs self-attention over the
  joint joker set (5 owned + 3 shop, learned CLS summary) so it can reason about
  pairwise synergy / xmult stacking. Wired **additively** through zero-init
  projections onto both trunks → no-op at load (regression-free), learns from
  update 1. **Pre-committed:** if this too leaves the leading indicator flat over
  hundreds of updates, accept ante-4 competence as the deliverable.

### Reward differentiation, and why it nulled — 06-23/24 (`dec-032/033`)
The encoder didn't move it either, and a deep audit said why: the reward credits
*survival depth*, which additive builds earn as well as xmult ones at the antes
the agent reaches, while the xmult-differentiating reward (the deep/win payoff) is
~0.5%-rare. So `dec-032` made the dense **xmult-engine growth** signal pay 3× the
additive rate (differentiation, not a global knob) and un-suppressed the early
acquisition bonus; `dec-033` added a one-shot **first-engine** bonus to fix
`dec-032`'s chicken-and-egg (the growth premium only fires once you *own* an
xmult). Both **null**: a direct +1.5 reward for the first xmult buy did **not**
move the buy rate at all — proving the ~31% rate is *opportunity/economy-bound*,
not reward-bound. Reward shaping can't manufacture shop RNG/affordability. That
closed the reward/credit lever and the whole reactive-policy category.

### The revamp: reactor → planner — 06-24 (`dec-034`, `REVAMP.md`)
Five reactive levers nulled; the user reset the goal to *consistently beat White
Stake*. Three parallel audits converged: the agent is **a greedy local optimizer
with incomplete knowledge and no memory**, and greedy-in-isolation *is* a ~ante-4
strategy. In the most computer-favorable game possible (fully observable,
deterministic scoring) it uses **none** of the computer's search advantage. The
revamp, in forced dependency order (roadmap + checkboxes in `REVAMP.md`):
- **Pillar 1 KNOWLEDGE** — `_project_shop_scaling_value` (scaling jokers were
  valued ~×1.0 in the shop), `_resolve_magnitude_contribution` (Steel/Stencil
  scored ×1.0, Stone/Bull/Banner/etc. flat-base), economy-joker valuation.
- **Pillar 2 PLANNING** — `environment/planner.py`: a future-ante blind-target
  curve + `build_survivability` + `build_value`. `action_executor._planner_pick_joker`
  makes the **planner** choose which joker to buy (deepest build, multi-ante),
  overriding the policy's slot; PPO records the executed buy so the policy distills
  toward the planner. Verified live: ~23% of buys overridden, KL rose to 0.0168.
- **Pillar 3 COMMITMENT** — `planner.target_hand_type` (build commits to one
  archetype = strongest × achievability); `plan_consumable_use` levels the
  committed hand and **holds** off-build planets (was "use on sight");
  `pick_best_planet` biases the committed hand. Remaining: economy save-then-spike,
  boss prep, sticky archetype memory, full lookahead search (value-head leaf).
Keep the near-optimal tactical heuristics (card selection, joker ordering) — the
gap is *strategic*, not tactical.

### The ceiling audit — 06-25 (`dec-035`, `SOLVER.md`)
The dec-034 hybrid caps ~0.5–2%: the planner optimizes a **depth-1 static
survivability proxy**, but winning needs builds whose engines **scale fast enough
to keep pace with the exponential blind curve**. Each deep ante (5+) is a ~35%
gate; 85% wins needs ~96% per gate → builds must be ~10–100× stronger by ante 5–6.
Reframed the goal from "win occasionally" to "out-scale the curve" → a real
search-based **solver** (`SOLVER.md`).

### Solver phase 1: trajectory-aware evaluator — 06-25 (`dec-036`)
`build_survivability` made trajectory-aware: walk current→ante 8, project each
scaling engine forward (`_project_jokers`) and score the matured build vs the boss
target — so a build is valued by whether it **out-scales** the curve, not by
static current power.

### Deep-research redirect: leveling + economy, not engine count — 06-26 (`dec-037`)
Three audits converged: the binding constraint at depth is the **complete
multiplicative product (hand-level × flat-mult × xmult), NOT xmult count** (advance
rate is flat across 0–4 engines at the 5→6 wall). The evaluator was **blind to the
two biggest levers**: hand-**leveling** (`build_survivability` froze planet level)
and **economy** (no save→spike). Order set: instrument depth deaths → complete +
**validate** the evaluator → then Phase-2 search. Shipped Phase-0 instrumentation
(per-ante money/level/power/margin in `build_progression.jsonl`).

### Evaluator calibration: realization factor — 06-26 (`dec-038`)
5,018 instrumented games showed `build_survivability` is **~2.3× optimistic** (real
boss-blind advance crosses 50% at predicted margin 2.3×, not 1.0×). Added
`REALIZATION_FACTOR = 0.43` so margin ≥ 1 ≈ a real ~50/50 clear. (Later found
stale after dec-040/042 shifted the estimator — flagged for a data-driven re-fit.)

### Training budget 5M → 50M — 06-28 (`dec-039`)
The trainer had silently hit its hardcoded 5M-step budget and was **idling** — each
supervisor restart reloaded the done checkpoint, printed "TRAINING COMPLETE", and
exited (~5-min loop, frozen checkpoint, fake high FPS). Raised `--total-timesteps`
to 50M; the trainer resumes the existing model.

### Deep-audit batch — 06-28 (`dec-040`)
A 5-agent audit found the build under-makes multiplicative xmult (87–109% of the
lethal power gap at depth) plus two compounding RL failures. Shipped: ante-scaled
xmult projection cap (was a flat 6.0, below median realized xmult); per-hand-type
scoring chips (killed the flat +40 that inflated Pair); **wins-only SIL** capture
(the demo buffer was 96–99% *losing* runs → SIL was imitating losses);
`REWARD_GAME_WIN` 15 → 150; a 500-ep WR metric (the 20-ep WR is noise at 0.5%).

### Discard honors the committed hand — 06-28 (`dec-041`)
`find_best_discard` greedily dug for whatever was closest (usually Pair), so the
agent **leveled Flush but played Pair** — the largest cause of the 2.3× realization
gap. Added a 1.4× bias toward the strategy advancing `target_hand_type`.

### Second deep-audit batch — 06-28 (`dec-042`)
Found a self-inflicted regression (win=150 spiked value loss 28→171, EV→0.11 on
wins) and a new binding constraint (the agent is **too broke** to reroll for
xmult). Shipped: **Huber value loss** (tames the win shock; normal value learning
unchanged); `build_survivability` now **projects committed-hand leveling forward**
+ **commit hysteresis** (stops the Pair flip-flop so planets concentrate); reroll
floor relaxed to $5 in antes ≤5; planet vouchers un-blacklisted; **skip-to-harvest**
supply tags enabled; two action bugs fixed (consumable garbage-index, standard-pack
spin); committed-hand + score/target appended to the observation (842→850,
checkpoint-migration-safe).

### Disk exhaustion + first measurement finding — 06-28 (`dec-043`)
C: hit **0 bytes** (silently breaks checkpoint saves). Causes: 43.7 GB of
never-pruned checkpoints + a 1.6 GB unbounded debug log. Fixes: auto-prune
checkpoints to newest 15; disable the joker-order log; (dec-044) a supervisor disk
guard. **Measurement finding** (ante-controlled): realized xmult **VALUE** predicts
deep-ante advance; engine **COUNT** does not — the lever is engine *maturity*.

### Live scoring +40 fix + ops — 06-28 (`dec-044`)
dec-040 fixed the flat +40 only in the planner; the **live** paths
(`estimate_score_for_hand_type`, `pick_best_planet`) still used it, distorting
every shop/planet choice. Made `SCORING_CARD_CHIPS` (in `hand_eval.py`) the single
source of truth. Also fixed a latent ConfigurableReward crash + added the
supervisor disk guard.

### Foundations-first pivot + eval harness — 06-28/29 (`dec-045`/`dec-046`)
After 42 decisions with a flat win rate, the comprehensive audit's verdict: the
project couldn't **measure** improvement (no held-out eval; win rate invisible at
0.5%) and the evaluator was never validated. Decision (with the user): build the
measurement loop first; keep Balatron a **learning** AI via an eventual
AlphaZero-style *learned* evaluator (not a hand-coded solver). Built `eval_report.py`
(advance curve + Wilson CIs + paired A/B), a 300-seed fixed bank, and a held-out
**eval run-loop** (`evaluate.py` / `Trainer.run_eval`, gated behind `eval_mode` so
training is untouched, reusing the real play path). Validated live on 3 seeds.

### Strategy bets: xmult value, depth gradient — 06-29 (`dec-047`)
First properly-measurable changes: reward now targets xmult **value** (dropped the
count-based stack premium per dec-043), not count; **depth-graded loss** so a
shallow death is much worse than a near-win (breaking the "safe ante-5 farm" local
optimum). Training-time changes — validated by train-then-eval vs a baseline on the
seed bank, not an instant A/B.

### Deep audit: three failure layers, Tier-0 fixes — 06-29 (`dec-048`)
A 4-agent data-grounded audit (27,164 reconstructed runs + RL-health + planner
calibration + scoring/regression) explained why dec-040→047 left the curve flat.
**Three converging causes:** (L1) the **planner over-rates builds at depth** — it
projected xmult **uncapped** (`_project_jokers` bypassed the dec-040 cap; Canio→19×),
assumed leveling ~2× too fast, ignored boss effects, and `REALIZATION_FACTOR=0.43`
is stale (fits ante 4; the gating antes imply it's ~6× too high and should be
ante-scaled). (L2) the **build makes too little xmult, too late** (median 1.6
entering ante 5 — xmult magnitude is *the* binding variable; leveling is **not** the
gap) and **dies rich** (78% of ante-8 deaths hold ≥20 gold → needs spend-down, not
the dec-042 economy relax). (L3) the **RL can't learn from the +150 win** (value head
can't represent it → EV craters on win rollouts), and dec-047's depth-loss made dying
**net-positive** from ante 5 (a "safe deep death" basin). Confirmed good: the policy
is no longer decorative (KL healthy), SIL works (14 real wins), the eval harness is
inert for training. **Tier 0 shipped:** revert the depth-loss (terminal loss now ≤0
at every ante), cap the xmult projection like the shop estimator, `LEVELS_PER_ANTE`
0.8→0.45. **Forward plan:** Tier 1 — re-fit RF (ante-scaled, on fresh data) + log
realized end-of-blind score/hands-used (≈40% of deep deaths are adequate-build and
currently undiagnosable); Tier 2 — return/advantage normalization so wins are
learnable; Tier 3 (A/B via eval harness) — magnitude-weighted xmult earlier,
pre-boss spend-down, boss effects in the planner/value path.

### Tier 1 measurement: realized per-blind logging — 06-29 (`dec-049`)
Added `logs/blind_results.jsonl` — one record per blind resolution (beaten→SHOP /
failed→GAME_OVER) with realized score, target, `hands_left`, and the planner's
projected power (`realized_vs_proj`). Closes the audit's #1 blind spot (the per-ante
logs only had the *projection*, so "adequate build, dies anyway" was undiagnosable).
**First data was striking:** the agent realizes only **~7–10% of projected power**
at shallow antes — independently confirming the realization factor should be ~0.075
(not 0.43), and exposing a large **execution gap** (the build projects strong but the
agent under-realizes it: hand-selection/variance plus the 3.0-best-hands projection
assumption). Implication: a calibration scalar alone won't fix it — the *play-side*
realization is implicated. (Beaten-blind realized is floored at target — the tracker
misses the final winning hand; failed-blind realized is accurate via the GAME_OVER
fallback.)

### The realization gap is the policy under-digging — 06-29 (`dec-050`)
Investigated dec-049's gap immediately (no waiting). From 1,349 logged blinds:
dying runs **exhaust all their hands** and die at **~73% of target** — on builds the
planner judged *adequate* (proj ≥ target in 77–100% of deaths) — while winners
one/two-shot. So the build's power is concentrated in the committed hand and
failures never assemble it. Reading the play call site **corrected the hypothesis**:
`plan_optimal_action`'s "hopeless/unviable chase" give-up branches are the *legacy*
path, **bypassed under `policy_authority=True`** — the network owns play-vs-discard
and the heuristic only picks the best available cards. So the gap is the **policy
playing weak hands instead of discarding to dig** toward its committed hand. Added
`discards_left` to `blind_results.jsonl` as the decisive test: a run that uses all
hands but leaves discards **unused** under-dug. **Verdict (clean data, 51 failed
blinds): NOT under-digging** — 84% used *all* discards *and* all hands and still hit
only ~0.71 of target. So the gap is **genuine build under-power masked by an
over-optimistic projection** (greenlights builds at ~3–4× target that realize ~0.7×),
**not** a play-side problem. This rules out a discard fix and points back at
projection honesty + build power (Tier 0's xmult cap + leveling, shipped; plus the
realization-factor re-fit). (Residual: proves discards are *used*, not *optimally*.)

### RF re-fit confirms 0.43; the real wall is the boss plateau — 06-29 (`dec-051`)
Re-fit `REALIZATION_FACTOR` on clean per-blind clear data. On the **deep gating
antes (5–7) boss blinds**, clear-rate is monotonic in projected margin (1×→38%,
2×→47%, 4×→62%), and 50% clear sits at proj/target **~2.3×** — *exactly* what
RF=0.43 already encodes. So **RF is correct; no change.** The earlier "~0.075" /
"~0.71 realized" signals were **confounded** (shallow antes where `proj_power`
underestimates; and the *failure-conditional* tail read as typical). The real
finding: **deep-boss clear plateaus at ~62% even at 4–8× margin** — build power
can't buy past it. That ~38% residual is **boss-debuff + draw variance, which the
planner/scorer don't model**. The next lever is **boss-robustness**, not RF or raw
power. (Discipline win: the re-fit's honest answer was "the constant's already
right," and the measurement redirected the work.)

### Boss-robustness Layer 1: The Mouth setup-override — 06-30 (`dec-052`, `BOSS_ROBUSTNESS.md`)
The boss-death breakdown showed deep deaths are dominated by bosses that punish the
agent's **single-committed-hand** build — led by **The Mouth at 74%**. The Mouth
locks the round to the first hand TYPE *played* (discarding doesn't lock), and the
agent plays its best *current* hand before its strong committed hand is assembled,
locking into a weak type. `mouth_should_dig` (hand_eval.py) + an override in
`action_executor`'s PLAY branch now **dig (discard) to set up the committed hand
before the first play locks the type** — but only pre-lock, with discards left, when
the current best is strictly weaker than the committed target. It's a tactical guard
overriding the policy's PLAY (the executed action is what PPO records). First change
with a **boss-specific A/B signal** (verify via The Mouth's kill rate in
`blind_results`). Remaining bosses + a boss-aware planner (Layer 2) tracked in
`BOSS_ROBUSTNESS.md`.

### Boss-robustness Layer 1: The Needle setup-override — 06-30 (`dec-053`)
Same pattern as The Mouth, applied to **The Needle (63%)** — only *one* hand for the
whole blind. `needle_should_dig` digs with discards to maximize that single hand
(while the best current hand can't clear the target) instead of playing a weak one
immediately, via the same `action_executor` PLAY override. Verify via The Needle's
kill rate in `blind_results`.

### Value-target normalization (PopArt-lite) — 06-30 (`dec-054`, default OFF)
The audit's CRITICAL RL blocker: the value head can't represent the +150 win, so EV
craters to ~0.1 on win rollouts and PPO learns nothing from exactly the trajectories
worth learning. Key realization: the dec-042 Huber tamed the loss *magnitude* but
**caps the value gradient**, so the head never *learns* the win value — and lowering
the win reward is a band-aid (Huber caps the gradient regardless of win size). The
real fix: the value head learns in a **normalized return space** (the win becomes a
representable few-σ target), denormalized for GAE. A running return mean/std (EMA);
`store_transition` denormalizes the head output; the value loss + clipping compare
in normalized space. `config.value_norm` gates **only** the stats update, so OFF
keeps stats at (0,1) → every (de)norm is an identity → byte-identical to before
(109 tests unchanged + 2 new). Enabling has a value-head re-scaling transient (no
PopArt output-layer rescale yet), so **enable + A/B via the eval harness, not blind
on the live trainer.** Highest-leverage RL change; payoff unproven until A/B'd.

### Resumable eval — 06-30 (`dec-055`)
The first baseline eval died with the session (9/300, unusable) — a multi-hour run
tied to the session gets killed on teardown (along with the supervisor-owned game
servers). Fix: `evaluate.py` writes each finished run to a **dedicated** results
file (`logs/eval_<checkpoint>.jsonl`) and **skips seeds already present** on
startup, so a crash costs only a restart — re-run the same command and it continues.
The dedicated file is isolated from training's `game_history` (no `--seeds` filter
needed to analyze). Still needs game servers up + training paused; it does not yet
launch its own Balatro instance (the truly self-contained fix, deferred on RAM).

### Value-head A/B: value_norm ON (live test) — 06-30 (`dec-056`)
After ~18 changes with a flat 0.54% win rate, pulled the one real win-rate lever:
enabled `value_norm` (dec-054) so the policy can finally learn from the +150 win.
Plumbed `TrainConfig.value_norm → PPOConfig`, a `--value-norm` flag, and the
supervisor launch. Chose the **fast live test** (watch EV-on-win-rollouts + WR500)
over the slow formal eval. Rollback point: `checkpoints/rollback_pre_valuenorm.pt`.
Expect a ~10–30-update EV **dip** as the value head re-scales (no PopArt output
rescale yet), then EV-on-wins should recover healthy instead of cratering to ~0.1,
and WR500 should start rising. Revert = drop `--value-norm` + resume from the
rollback checkpoint.

### The plateau audit: architecture ceiling + optimizer damage — 07-02 (`dec-057`)
A 4-agent audit answered "why isn't he learning." **Verdict: both.** (1) **Architecture
ceiling:** a multiplicative per-blind model (51,834 blinds; reproduces the observed
0.7% win) shows a *perfect* play/discard policy caps at ~2.6% — the network's real
levers (play/discard timing, shop tempo) are outcome-inert (tempo r²<0.008). The
agent has already extracted ~half its policy ceiling. **The win rate lives in the
planner:** boss-aware `build_survivability` alone models to ~3.6% (~5×), and with
tactics/power fixes **~11–12% is reachable inside the current hybrid** (boss blinds
sit 15–35 pts below non-boss siblings at every deep ante; `planner.py` hardcodes
`boss: 2.0` and never reads the boss identity). (2) **Optimizer damage:** the policy
*did* learn once (u385→1500, ante 3.46→4.41, LR ≤1e-4, median KL 0.005) until
dec-034 crashed it; then dec-039's 50M budget change **silently raised LR to
2.7e-4** (first-ever KL>1 hit 66 updates later; 40% of updates now blow target KL),
dec-054's ret stats are **not persisted** (value scale resets every 90-min recycle —
the identical VL≈14/EV≈0.3 transient at every session start; EV still craters on win
rollouts), override transitions stored at log-prob −30 create **e²⁵ ratio bombs**,
and the KL early-stop fires only post-epoch (destructive steps land on exactly the
win rollouts). Experience is thin: 41.5% single-choice transitions (blind head 100%
dead), mask prior pre-decides 57% of steps, return ~92% shaping. Throughput was
**exonerated** (395 steps/min healthy; crawls = external contention killing games;
FPS field is garbage after resume). Measured fix feedback: the **Needle override
worked** (realized/proj 0.42→0.58); Mouth's didn't visibly. Plan: Tier-0 optimizer
hygiene (persist ret stats, LR→1e-4, ratio-bomb guard, per-minibatch KL stop,
0·inf guard, torch threads), then **redirect to the planner** (boss-aware, deck
thinning, save→spike economy) — stop tuning PPO strategy.

### Tier-0 optimizer hygiene: stop the active damage — 07-02 (`dec-058`)
Shipped the audit's six optimizer fixes: **persist `ret_mean`/`ret_std` in
checkpoints** (the dec-054 value-norm fix was never operative — the scale reset
every 90-min recycle); **LR locked at 1e-4** (dec-039 had silently raised it to
2.7e-4; the only durable-improvement era ran ≤1e-4), applied at run-start too (the
loaded optimizer carried the old LR); **ratio-bomb guard** — steps with
`|log_ratio|>5` (storage artifacts from override actions stored at log-prob −30)
contribute zero policy gradient and are excluded from `approx_kl`, making it a true
drift measure again; **per-minibatch KL stop** (a batch >1.5× target applies *no*
gradient and halts the update — the old post-epoch check let 4 destructive steps
land on exactly the win rollouts); **0·inf NaN guard** on the prior term;
**`torch.set_num_threads(2)`** (unbounded threading was the ~5-core burn that
starved game instances under external CPU pressure). Bonus: SIL demo capture
truncates at win-detection (demos no longer imitate the post-win endless death
tail). Key subtlety: the ratio-guard must come *before* a per-batch KL stop is
even possible — bomb steps inflated batch KL by ~142 nats each, so a naive SB3
stop would have halted training permanently. Watch after deploy: median KL back
to ~0.005–0.03, the session-start VL/EV transient gone, entropy stabilizing.

### Boss-aware planner v1 — 07-02 (`dec-059`, the audit's ~5× lever)
The planner treated every boss as a generic 2× target; it now gates the
**immediate** ante on the **known** boss's real difficulty. `upcoming_boss()`
reads the boss name from state (UPCOMING/CURRENT/SELECT; `''` once DEFEATED —
next boss unknown); `BOSS_DIFFICULTY` holds multipliers **relative to a typical
boss** — chip facts (The Wall 2.0: it's literally 4× base; Violet Vessel 3.0)
and mechanic haircuts (Needle 3.0 — one hand vs the 3 the power model assumes;
Flint 1.8, Eye 1.5, Water 1.4, Crimson Heart 1.4, Arm 1.3, Manacle/Amber 1.2;
Mouth deliberately 1.0 — dec-052 covers it). **Future/unknown antes stay 1.0:**
`REALIZATION_FACTOR` was fit against average-boss outcomes (dec-051), so an
expected-boss multiplier there would double-count the calibration. Effect:
`build_survivability` (and therefore `build_value` and shop buys) demands
genuinely sufficient builds exactly when a hard boss looms. Verify via per-boss
kill rates in `blind_results` (baselines: Wall 67%, Needle 63%, Eye 62%, Water
61%). Remaining levers (Eye multi-hand builds, suit-debuff pivots, deck
thinning, save→spike) tracked in `BOSS_ROBUSTNESS.md`.

### Save→spike economy v1 — 07-03 (`dec-060`, the power-side companion to dec-059)
dec-059 raises the *bar* before a hard boss, but the ceiling audit found >75% of
deep arrivals lack the *power* to clear a 2× boss — raising the bar does nothing
if the agent can't build past it. This adds the power side, via the existing
*gated reroll* path (no risky new leave-shop logic): (1) `PLANNER_REROLL_THRESHOLD`
0.12→0.25 — hunt for a real engine when the shop is merely *mediocre*, not just
barren, so surplus above the interest reserve buys power-finding instead of junk
jokers (the agent chronically buys weak jokers and never builds power); (2) **the
spike** — `_planner_reroll_ok` relaxes the interest floor to $10 before a hard
(dec-059 difficulty ≥1.5) boss, so the war chest is spent finding/buying power at
the gate it saved for (money is worthless if the run dies there). Reward side
needed no change — `_check_gold_hoarding` already only penalizes above the $25
interest cap. Buys are affordability-gated only (not floor-gated), so the lever
acts through reroll behavior. Verify via money@ante-N + proj-margin in
`build_progression` and per-boss kill rates.

### LR lock at checkpoint-load time — 07-09 (dec-058 follow-up)
<!-- Numbering note: this fix belongs to dec-058 and has NO separate dec-NNN. It
briefly carried a `dec-061` tag that collided with the 07-07 confidence gate, which
owns dec-061 across all code (agent/confidence_gate.py, network.py, config.py,
train.py, action_executor.py). context-keeper filed THIS LR-lock change under the id
dec-061 too (a machine-store artifact, id immutable) — but the canonical dec-061 is
the confidence gate; this LR-lock is a dec-058 follow-up. -->
dec-058 was supposed to lock LR at 1e-4 permanently, but an audit found it still
resetting to the dec-039 damaged **2.7e-4** on trainer recycle (updates 625 and
2465; ~1/3 of training ran damaged, KL to ~7e7, EV cratered). **Root cause:**
`PPOTrainer.load_checkpoint` restores the optimizer `state_dict`, which carries
the LR that was live at save time — repro'd directly (save at 2.7e-4 → load →
2.7e-4 back). dec-058 only counteracted this with a *separate*,
`anneal_lr`-gated `set_learning_rate` inside `run()`: a band-aid far from where
the stale LR re-enters, dependent on timing and on the flag staying True. **Fix
(minimal, co-located):** `load_checkpoint` gains an `lr_override` param and
re-asserts the LR right after `optimizer.load_state_dict` (covers both the
normal and shape-migration load paths); `train.py` passes `1e-4` at the load
site. The lock is now applied *atomically with the load*, so no recycle path or
future caller (eval, resume) can carry a stale LR forward. The schedule itself
and dec-059/060 are untouched; the `run()` locks remain as redundant
belt-and-suspenders. Regression pinned in `tests/test_lr_recycle_lock.py` (one
test documents the old resurrection, two prove the override holds through single
and double recycles). **Lesson:** a locked hyperparameter must be re-asserted
*after* `optimizer.load_state_dict` — the optimizer state carries it verbatim;
never trust a later, separately-gated setter to undo it.

---

### Confidence-gated planner deferral — 07-07 (`dec-061`, inference/eval-only routing)
The policy, planner, and heuristics interact in a *fixed* hierarchy: the policy
owns the action TYPE, the dec-034 build planner owns which-joker (in the shop),
heuristics own tactical card math. This makes that hierarchy **dynamic at decision
time, on the inference/eval path only**: at each decision we read the policy's
confidence off the action-TYPE distribution it *already* computes (no extra
forward pass) and, when it is **uncertain**, route that single decision to the
existing planner instead of the fast policy sample; when it is **confident**, the
policy sample stands (today's behavior). This **routes existing planner compute by
confidence — it adds no new planner and does not change training.**

- **Signal** (`gate_signal`): `top1` = the top-1 action-type probability, or
  `entropy` = normalized certainty `1 − H/log(n_legal)` of the masked type dist.
  Both in [0,1], high = certain. A forced (single-legal) decision is 1.0.
- **Threshold** (`gate_threshold`): defer when `confidence < threshold`. The
  extremes bound today's behavior — `0.0` gates *nothing* (the default), `1.0`
  gates every real (multi-legal) choice — so the feature is a provable **superset**
  of current behavior.
- **Opt-in** (`gate_enabled`, default **OFF**): off ⇒ the play path is byte-for-byte
  unchanged. Deferral reuses the planner via a `buy_joker` action (the planner then
  picks/swaps/rerolls, dec-034); it only has an opinion in the shop, so off-shop
  decisions abstain and keep the policy sample.
- **TRAINING IS UNTOUCHED (deliberate):** the gate is hard-gated behind
  `eval_mode` (`gate_is_active`), which training rollout collection sets False.
  Overriding actions during collection would reintroduce off-policy contamination
  into the on-policy distribution PPO learns from — so the gate never fires there.
  The gate config is *not* forwarded to `PPOConfig`.
- **Measurement:** `run_eval` writes `<out>.gate.json` (deferral rate = planner-call
  count, confidence distribution) and prints a `[GATE]` summary. Compare ON vs OFF
  at a threshold by running `evaluate.py` twice (with/without `--gate`) over the
  same seed bank and diffing advance rate via `eval_report.py` (§ README).
- Files: `agent/confidence_gate.py` (gate + telemetry), `agent/network.py`
  (`return_confidence` flag), `training/action_executor.py`
  (`planner_recommended_action`), the `_collect_rollout` seam, and `evaluate.py`
  flags. Tests: `tests/test_confidence_gate.py`.

### Checkpoint-crawl livelock breaker — 07-10 (`dec-063`)
With `--checkpoint-interval 2`, a slow run (INVALID_STATE desync + deep boss
fights, ~36 steps/min under the 80/min rate floor) is recycled *before* it
completes 2 updates, so it never writes a new `update*.pt` and every relaunch
reloads the SAME checkpoint (stuck at `update003748`) — a livelock. The
supervisor recycles via `kill_pids -> psutil p.kill()` = Windows
`TerminateProcess`, which is **uncatchable**, so a signal-handler teardown save
can never fire on a recycle. Fix (train.py only): a **wall-clock SAFETY
checkpoint** in the update loop saves an untagged `update*.pt` once ≥1 update has
accrued and ≥`SAFETY_CHECKPOINT_S` (480s) since the last save — needs no signal,
survives the hard kill, and is gated on `num_updates` progress so a genuinely
wedged trainer still saves nothing (freeze/churn detectors stay authoritative).
It does NOT change `--checkpoint-interval` (milestone cadence untouched — an
orthogonal time trigger). Also: `newest_checkpoint()` only globs
`balatron_phase1_update*.pt`, so the old `finally` save with `tag="final"` was
silently unresumable — the teardown save is now untagged. Added graceful
SIGINT/SIGTERM/SIGBREAK handlers for the manual-stop / future-graceful path.
Tests: `tests/test_checkpoint_teardown.py`.

### INVALID_STATE desync — abort futile retries — 07-10 (`dec-064`)
~24 INVALID_STATE rejections/session (e.g. `play` fired while in SHOP, `select`
while in SELECTING_HAND) waste RPCs + backoff. Root cause is **structural**: the
action is decoded from the state snapshot taken at the TOP of the
`_collect_rollout` iteration, but the game can leave that state before the send
lands (an animation / blind / run transition completing, or an auto-action
inside `_get_actionable_state`'s poll loop). The retry loop had treated
INVALID_STATE like a transient "buttons not ready" blip and retried the same
stale action 3× into a state the game had already left. Fix (contained to the
retry loop): on INVALID_STATE, parse the accepted states out of the error
message (`_parse_required_states`) and re-read the LIVE state once — if the game
is no longer in a state the method accepts, **abort immediately** (1 send + 1
read) and let the next iteration re-derive from fresh state; only keep retrying
when the live state DOES accept the method (a true timing blip). Failure-path
only, so no cost on the common success path; no buffer/settle change
(`action_succeeded=False` is the pre-existing terminal path). This CUTS the
per-desync cost but does not eliminate the desyncs — the full structural fix
(decode the action from the fetch that immediately precedes the send, which
touches the con-007 settle/store chain) is a recommended follow-up, left out of
this live-run change. Tests: `tests/test_state_guard.py`.

---

### Non-scoring joker valuation + seal acquisition — 07-12 (`dec-065`)
**The blind spot.** A raw-log audit of shop behavior found **71% of planner
rerolls (223/314) fired on a hard `d-surv=0.00`**. Root cause: the survivability
estimator (`_estimate_joker_scoring_for_type`) returns `(0,0,×1)` for any joker
without a `score_effect` field, so **62/150 jokers (41%)** — all 21 economy, 11
card-creation, utility, even the 2 hand-upgrade jokers — moved `build_survivability`
by *exactly* 0.00. The planner rerolled past 41% of the pool and acquired
economy/utility only by accident. (Compounding cause: `build_survivability` is a
fractional-ante on a log10 scale vs exponential targets, so even some
`score_effect` jokers — Bull, Walkie Talkie — round to 0.00.)

**Frame (dec-038).** Money is NOT the binding constraint (agent dies with
$13–49; the multiplicative product is). So the fix is **rankability + tempo, not
hoarding** — every economy/prior term is capped BELOW a real engine.

Four bounded levers in `planner.py`, all feeding the existing survivability curve
so they inherit its bounds:
- **#3 economy (A-model)** — `_economic_survivability_bonus`: expected $/round
  over the spend-horizon → future joker buys → generic-engine survivability,
  **discounted by P(survive to spend) ≈ (base_surv − cur)**. A dying build gets
  ~0 economy credit (can't be lured off buying power); a healthy build's *first*
  strong economy joker clears the reroll bar. `ECON_SURV_CAP=0.26` is coupled
  just above the 0.25 `PLANNER_REROLL_THRESHOLD` on purpose. A 2nd economy joker
  reads ~0 (engine-hunt resumes). Knobs: `ECON_YIELD`, `SURV_PER_ENGINE=0.20`,
  `TYPICAL_JOKER_COST=5`, `ECON_SPEND_HORIZON=3`. **Scaling economy** (Rocket:
  $1/round +$2 per boss defeated) projects the ramp — `current (base + inc·bosses
  beaten, or live _scaled_value) + inc·horizon/2` — instead of the flat
  `money_per_round`, which alone left Rocket *below* the reroll bar; combined with
  the reach discount this values Rocket exactly when you'd buy it (early, while
  ahead of the curve), not late when marginal.
- **#4 tier prior (C)** — `_prior_survivability_bonus`: tiny tier-weight nudge
  (cap 0.08) so scoreless utility (8 Ball, etc.) is *rankable* but never stops an
  engine hunt.
- **#1 hand-upgrade** — Space/Burnt add committed-hand levels/ante through the
  existing `_level_committed_hand` projection (exact).
- **#2 boss-nullifier** — Chicot collapses the dec-059 boss multiplier to 1.0
  (exact).

**Seals** (user: blue/purple seals tailor the deck and are underused — they were
the shop's main seal source, hard-blocked):
- Raised blue (Trance 2.5→4.5, = free planet/round = leveling, dec-037 lever) and
  purple (Medium 2.0→3.5) spectral seal values in `evaluate_pack_spectral`.
- Added `evaluate_pack_standard` (`hand_eval.py`) — pick the best sealed/enhanced
  card, **skip pure dilution**; wired into `train.py`'s ENHANCED branch
  (was "pick index 0"). con-005/con-010-compliant skip within the bounded pack loop.
- Guard-unblocked standard-pack *buying* in `action_executor.py`: allowed when
  FREE or from clear surplus (`money − cost ≥ $20`), so it can't drain interest;
  a scoring joker still wins via the existing REDIRECT.

Verified: strong econ 0.26, weak econ <0.25, dying-build econ ~0, 2nd econ ~0,
real engine > econ on dying builds; standard pack picks blue-seal over Glass and
skips vanilla. Tests: `tests/test_economy_valuation.py` (13 new); 159 pass.
**Follow-up:** A/B that the war chest grows AND deep rerolls land engines (not
burn cash); validate standard-pack buying at scale.

---

### Winning-trend miner + margin potential — 07-13 (`dec-066`)
**The idea (user):** log every win, mine decision trends across runs, reward the
common trends — grouped by joker type (economy / scaling / mult / retrigger).

**The trap:** "most common in winners" ≠ "causes winning" — that's survivorship
bias, and dec-038 already hit it (economy *correlated* with depth but wasn't
causal). Fix: **contrastive, conditioned on reaching each ante** — compare runs
that all reached ante N and ask which reached ante 8. Same depth on both sides
controls for luck.

**Built `tools/analyze_winning_trends.py`** (reconstructs runs from
`build_progression.jsonl`, splits on ante drop; 89,002 runs). Result on
reach-8 rate, effect-size spread at ante 5/6:

| feature | spread | verdict |
|---|---|---|
| **margin** (power/target) | **14.6 pts** | dominant causal spine (ante 6: 4.1→18.7% across buckets) |
| n_xmult | 6.4 pts | real but weaker |
| n_scaling | 0.9 pts | **noise** — the "count scaling jokers" instinct fails |

Plus: **36.5%** of runs that died at ante 4–6 *never acquired an xmult engine*, vs
**4.8%** of deep runs. Emits `logs/trend_calibration.json` (empirical
margin→reach-8 curve) — turns the one-off dec-038 audit into a continuous validator.

**Two supporting changes:**
- **Enriched `build_progression` logging** with `n_economy / n_mult / n_retrigger`
  (train.py `_joker_category_counts`) so the miner can test *every* grouping the
  idea proposed — currently only `n_xmult/n_scaling` existed. Logging only.
- **Product-margin potential** in `reward.py` (`REWARD_MARGIN_POTENTIAL_COEF`,
  **OFF by default**): the reward shapes the xmult *proxy* (dec-032/043) but never
  margin, the actual causal signal. Potential-based (Φ=coef·min(margin,cap),
  paid on delta, con-008) so it telescopes to a bounded boundary term and can't
  recreate the dec-057 "value head calibrated to shaping" failure. Ships
  byte-neutral (scorer only runs when coef>0); flip on as its **own** A/B *after*
  dec-065 can be read cleanly — enabling it now would confound that experiment.

Tests: `tests/test_margin_potential.py` (4); 164 pass. Deploys on the next
supervisor recycle (measurement + off-by-default reward → no forced interruption
of the dec-065 run).

**Follow-up (`5d6767d`): miner defaults to CONTINUOUS depth, not win-rate.**
Wins are ~10–15/day — far too rare to stratify (only 35 reach-8 / 13 real wins in
the first 1.8k-run categorical slice). The tool's primary outcome is now **mean
max-ante reached** (every run informs it; readable today, tightens hourly), with a
configurable binary reach-N as a secondary column. Categorical features are
field-gated (pre-dec-066 records can't count as `0`); the effect-size ranking
ignores buckets < 30 so a lone lucky run can't distort it. On the current-policy
slice this **cross-validates margin** (still monotonic on a different policy's
data) and shows **`n_economy=1` is the sweet spot** — an independent confirmation
of dec-065's first-econ-then-taper.

---

### Margin potential A/B turned ON — 07-14 (`dec-067`)
Flipped `REWARD_MARGIN_POTENTIAL_COEF` **0.0 → 0.1** (live). The dec-066 miner's
one durable finding is that **margin is the causal spine** (0.44-ante mean-depth
spread, monotonic, holds across policies and as the slice tripled to 3.5k runs);
every joker-category grouping washed out — the `n_economy=1` "sweet spot"
regressed to ~flat with more data. The reward shapes the xmult *proxy* but never
margin, so this fills the real gap. Potential-based (Φ=coef·min(margin,4), paid on
delta) → telescopes to a bounded term, can't recreate the dec-057 blowup;
per-step deltas stay in-band with the existing shaping (SCALING_GROWTH 0.05,
DIVERSITY 0.02). **Flipped despite dec-065 lacking a clean win-read** — at ~10–15
wins/day that read is a week+ away, and the miner attributes via the *depth*
distribution (dense, every run), not wins. **Revert = coef→0.0** if KL/EV or
mean max-ante degrade. Watch: does the margin distribution shift up over the next
day, and do KL (≤~0.05) / EV (~0.7) hold. Tests: `tests/test_margin_potential.py`
(4); 164 pass.

---

### Don't overbuild — save when already clearing — 07-14 (`dec-068`)
User watched the agent buy scoring/extra jokers *while already hitting the score
to clear the next ante* — spending money that should compound as interest (and
feed the dec-060 spike). The buy path had a hole: when the planner's best pick had
`d-surv < 0.25`, it rerolled *if* reroll was allowed, but otherwise **fell through
and bought the marginal joker anyway** — no "already clearing → hold" path.

Fix (`action_executor.py`): new `_already_clearing()` — true when
`_score_survivability − ante ≥ AHEAD_BUFFER(=1.0)` (build clears the immediate
ante with a full ante of headroom; uses **score-only** survivability so the
dec-065 economy/prior bonuses can't inflate the check). In the open-slot buy
block, when the pick is marginal (`d-surv < 0.25`) **and** we're already clearing
**and** it isn't a MUST_BUY engine → `return "gamestate"` (hold, bank interest)
instead of buying. Real engines (`d-surv ≥ 0.25`) and Blueprint/Brainstorm still
buy; when *not* ahead, the dec-060 reroll-to-hunt is unchanged. d-surv already
separates redundant near-term power (low when ahead) from deep engines (high even
when ahead), so this suppresses exactly the wasteful buys. Consistent with the
architecture (reroll already overrides the NN's buy; PPO records the executed
skip, so the policy distills toward saving). Buffer/threshold are untuned first
guesses — watch mean max-ante (shouldn't drop from under-buying) and end-of-ante
money (should rise). Tests: `tests/test_save_when_ahead.py` (3); 167 pass.

---

### Log play↔build alignment + synergy for the miner — 07-15 (`dec-069`)
User asked whether the "win-log" analysis captures **WHEN** decisions happen (not
just what), hand levels, hand-play frequency, and play↔joker synergy. Audit's
answer: the **policy already sees all of it** — the state encodes per-hand levels +
**13 play-frequency slots** + a per-joker synergy value + `most_played_hand_type`,
and **SIL replays full winning trajectories**, so timing/sequencing *is* learned.
But the offline **miner** (`build_progression`) was a coarse per-ante snapshot that
couldn't test those hypotheses. Added (train.py `_committed_hand_signals`, logging
only): `ht_level`, `play_share` (committed hand's share of plays), `most_played`,
`committed_is_played` (1 if the committed hand *is* the most-played — the
sharpest "playing what you built for" signal), `n_synergy` (jokers whose trigger
rewards the committed hand). Now the depth-conditioned miner can check whether
play-consistency / synergy predict depth the way margin does. Tests: 167 pass.

**On making the miner "active":** yes — the right form is a **self-tuning
calibration loop**, not a live reward model. The miner periodically emits
`trend_calibration.json` (empirical margin→depth curve); the planner reads it to
replace dec-038's fixed `REALIZATION_FACTOR` scalar with the real outcome curve.
A live *learned win-predictor* would mostly duplicate the PPO critic (skip it).
Key guardrail: only make **validated causal** features (margin) active — auto-
targeting "whatever current winners do" creates a self-reinforcing loop that
amplifies the policy's present biases. Deferred so it doesn't confound the live
dec-065/067/068 A/Bs.

---

### Play-frequency-weighted score projection — the ante wall fix — 07-16 (`dec-070`)
`estimate_score_for_hand_type` (hand_eval.py) took the **best** score among hand
types played at least once. One lucky Straight Flush at ante 2 then pinned the
projection forever to a hand the bot never repeated — which is why
`realized_vs_proj` sat at **~0.30 at every ante**: the estimate described a
ceiling the bot couldn't reach. Now every hand type is scored and **averaged
weighted by its share of actual plays** (`0.05 + 0.95 * play_share`), so the
projection tracks what the bot *typically* does.

**The floor is split across the types (`0.05/12` each), not `0.05` per type** —
this entry originally specified reusing `pick_best_planet`'s per-candidate form
verbatim as "not a new parameter." Measuring it rejected that. `pick_best_planet`
floors a **gain** consumed by an `argmax`, where the floor can't move the winner
much; this function returns an **absolute magnitude** compared against blind
requirements, so floor mass leaks straight into the number. At `0.05` each, the
12 unplayed types take **39% of the weight** — and they're the 1136–3360 point
hands (Straight Flush..Flush Five) the bot never scores, vs a Pair's 56. That
floor becomes the *new* dominant error: modelled realized/proj moves only
**0.18 → 0.28** (target 1.0), and the clean pure-Pair case *regresses* from a
correct 56 to 391 (**1.0 → 0.14**). Split, the floor costs ~5% total: realized/
proj **→ 0.75**, pure-Pair holds at 99 vs a true 56, and every type keeps nonzero
weight so a hand the bot is about to learn still moves the estimate. **Lesson: a
constant tuned for a ranking doesn't transfer to a magnitude without
re-measuring.**

Tradeoffs: a weighted average is strictly lower than a max, so **all**
projections drop — dashboard.py gets a regime boundary at step **4369** so
`realized_vs_proj` isn't read across the discontinuity. The split floor is still
~1.8x over a true pure-Pair score (99 vs 56); zero floor would be exact but
would make a type contribute nothing until first played, which is the
reachability this deliberately keeps. Tests: **172 pass** (5 new in
`tests/test_score_projection.py` lock in the one-lucky-Straight-Flush case).

---

### Consumable-slot clog — never fire a targeted consumable bare — 07-17 (`dec-072`)
Chasing why leveling is slow (dec-069 found **committed-hand level ≥4 by ante 4 →
33% reach-6 vs ~22%**, yet only **9.3%** of runs get there). Planet *supply* was
fine (315 planet buys + 863 celestial opens per log), so acquisition wasn't the
bottleneck — but **144 of ~215 consumable-use attempts were FAILING**, all
targeted tarots/spectrals fired with no `cards` list. The *same* cards failed over
and over (**Trance 30×, Death 22×, Strength 20×**) — proof they were **stuck in a
slot**, not transient. With only 2 consumable slots, unusable tarots clog them so
planets can't be held → **leveling stalls**. This is the `[WARN] INVALID_STATE`
line that had been written off as benign timing noise all session.

Root cause: `action_executor.py`'s `action_type==8` branch fell through to a bare
`use{consumable}` for *any* consumable. `plan_consumable_use` (hand_eval) already
computes each card's correct targets and is called from two *other* paths — the
policy path just never consulted it. Fix: `CONSUMABLE_NEEDS_TARGET` in hand_eval
(single source of truth, contents match the observed failures exactly); if the
picked consumable needs targets and none were supplied, ask the planner, else
**no-op** rather than fire a guaranteed reject — con-005's lesson applied to the
consumable path. Untargeted consumables (planets) unchanged.

Partly an **own-goal from dec-065**, which raised Trance's (blue seal) pick value:
the agent grabbed it more and could never use it, actively defeating dec-065's
leveling intent. Watch: consumable-use failures should collapse from 144, and the
level-4-by-ante-4 rate should rise from 9.3%.
Tests: `tests/test_consumable_targets.py` (4); 176 pass.

---

### Revert the margin reward — null by construction — 07-17 (`dec-073`)
A 6-agent deep audit killed dec-067. **`REWARD_MARGIN_POTENTIAL_COEF` 0.1 → 0.0.**
Two independent fatal reasons: (1) **PBRS is policy-invariant** (Ng et al.) — a
potential term telescopes to a bounded boundary value and **cannot change the
optimal policy at ANY coefficient**; its only benefit is faster value learning, and
EV is already 0.70–0.83, so there was nothing to accelerate. **The A/B was null
before it started.** (2) The lever is small anyway: stratifying 69,894 runs
(survivorship-controlled) shows a *perfect* margin-maximizing policy reaches only
**1.3% win from ante 4 / 2.8% from ante 5** — ~2×, against a needed ~10–20×.
dec-066/067 elevated margin to "the causal spine" **without ever asking "if maxed,
what's the win rate?"** — the question this project has never asked before pulling
a lever. Also: dec-032/033 had *already closed* the reward category ("reward
shaping can't manufacture shop RNG/affordability") and dec-057 said stop tuning
PPO strategy; dec-067 reopened both. The reward is **exonerated**: it's already a
faithful depth surrogate (`R = 11.15·ante − 14.62`, **corr 0.977**; two terms =
97% of mass), correctly ranked. Machinery kept (free at 0); if margin is ever to
matter it must be a **policy-visible observation** (currently computable in
**0.000%** of states) or a planner input — never shaping.

### Unblock ALL measurement — `eval_session.py` — 07-17 (`dec-074`)
**Zero held-out evals have ever completed.** No `logs/eval_*.jsonl` exists;
`eval_baseline.out` died mid-run on 06-30. So **every A/B since dec-045 has been an
eyeball on a confounded live trainer** with 3–8 concurrent uncontrolled changes —
which is how dec-059 (dec-057's predicted ~5× top lever) could be **nulled by its
own named metric** (The Wall: 66% vs a 67% baseline) without anyone noticing, and
how ~40 decisions produced **+0.27 mean ante**.

The cause was **operational, not a code bug**: `evaluate.py` needs the game servers
to itself ("pause training first"), but the supervisor's existence layer (con-010)
**relaunches the trainer within ~30s and steals the ports back** — exactly how the
06-30 attempt died (INVALID_STATE on next_round/play). `supervise.py:642` already
had the right primitive: `SUPERVISOR_STOP` exits cleanly and **leaves the games up**.
`eval_session.py` sequences it: touch stop-file → kill trainer → run the resumable
eval → **`finally:`** remove stop-file + relaunch supervisor (training returns on
Ctrl-C/crash/failure). The harness was always well-designed — `eval_report.py`
already does Wilson-CI conditional-advance curves and **paired seed-matched A/B**,
and already knew win-rate is unmeasurable at 0.5% ("a 500-game sample expects ~2.5
wins"). It only ever needed to be *runnable*. **Cost: an eval pauses training for
hours — accepted; an unmeasured trainer only manufactures unvalidated changes.**

**THE ACTUAL ROOT CAUSE (`88d60b4`) — it was a missing env var, not the game.**
Running `eval_session.py` reproduced the real failure in ~40 seconds:
```
print(f"[SHOP] REDIRECT pack buy → joker buy: ...")     # U+2192
UnicodeEncodeError: 'charmap' codec can't encode character '→'
```
With stdout redirected to a file — which **every** eval is, being a multi-hour
background job — Windows Python encodes **cp1252** and the first non-ASCII log
line **kills the process**. Every eval died on its first shop redirect. Not
desyncs (the 06-30 `INVALID_STATE` tail was incidental), not difficulty, not the
runtime. **The project had already written the rule down — gotcha 6: "always
launch with `PYTHONUTF8=1`" — and `supervise.py:567` already applies it to the
TRAINER, which is why the trainer survives the identical print.** `evaluate.py`
never got it across dec-045/046/055. Fix: `PYTHONUTF8=1` +
`PYTHONIOENCODING=utf-8` on the eval subprocess. **Verified:** the eval now runs
past the crash and is writing `logs/eval_balatron_phase1_update004434.jsonl` —
the first eval artifact in the project's history. *Three decisions built a harness
that a one-line env var kept from ever running once.*

---

### The build escape hatch — 07-17 (`dec-075`) — *first change shipped with a baseline; MEASURED NULL → REVERTED*
> **RESULT (paired A/B, checkpoint 004434, same 300 seeds):** null-to-slightly-negative.
> reach≥4 66.3%→64.3%, reach≥5 45.5%→40.9% (CIs overlap); paired McNemar @ ante-6
> gate **33 better vs 38 worse, P=46% → inconclusive**. WIN 6→1 (Poisson noise).
> **REVERTED** (`c152cd7`) — it didn't help a frozen policy, and "it'll help once
> trained" is the unfalsifiable reasoning the audit indicted. The executor half
> fired **0 times** (untested); the mask half is real con-011 correctness but needs
> the prior-KL guidance channel restored (annealed to 0) to avoid random rerolls —
> a future *measured* experiment, not a bare veto removal. **The value here was the
> process: for the first time a change was measured and shown not to work before it
> could accrete as another unvalidated resident.** Original writeup below.

**The baseline** (`eval_balatron_phase1_update004434.jsonl`, 300 held-out seeds —
the project's first completed eval):
```
reach>=2 85.1% | >=3 86.3% | >=4 66.3% [59.6,72.5] | >=5 45.5% [37.3,54.0]
WIN 2.00% [0.92,4.29]
```
Advance is 85–86% through ante 3, then falls off at **ante 4 (66.3%) and 5
(45.5%)** — exactly where power stops out-scaling the target (2.15× vs 2.50×, then
1.66× vs 2.20×).

**The hole.** 48.7% of ante-4 builds own **zero xmult**; 36.6% of ante-4–6 deaths
never acquired one — not because xmult is unavailable, but because **both**
acquisition routes were closed:
1. **`action_executor`** — the reroll-to-hunt-an-engine block sits inside the
   `if joker_count < joker_limit` branch, so at **full slots** a non-improving swap
   **silently no-op'd**. Slots run **4.74–4.94/5 full by ante 4–5** — the modal
   death antes. The build froze at 5 flat jokers with no way out (**4,960
   zero-effect shop steps + 605 forced random pack-buys** per log).
2. **`action_space`** — the mask **hard-zeroed a legal `ACTION_REROLL`** on
   heuristic grounds ("don't reroll past a buyable joker"), making it **illegal in
   96.3% of shops**. `legal = mask > 0` cannot distinguish a heuristic veto from
   real illegality — a **con-011 violation**. Now legality-only; opinions ride the
   bias value. Measured A/B: every veto **0.0 → 0.3**, while genuine
   unaffordability still returns **0.0**.

**Bonus find:** `any_buyable_joker` is **overloaded** — dec-016 sets it for
*planets* (to keep BUY reachable), which silently vetoed reroll too. **An
affordable planet blocked engine-hunting.** Bias values are inert (prior annealed
to 0), so 0.0→0.3 captures the whole functional effect.

Guards unchanged: `_planner_reroll_ok` (interest floor / per-shop cap /
keep-buy-money) and dec-068's save-gate still apply, so this cannot drain the
economy. Tests: `tests/test_escape_hatch.py` (7); 183 pass.
**Caveat:** the paired eval mostly measures the *executor* half — the policy has
never explored reroll, so a frozen policy won't choose it; the mask half needs
training. Trust ante 4 (n=202) and ante 5 (n=134); ignore the deep rows (n=11–22).

---

### The leaf doesn't discriminate — measured at last — 07-18 (`dec-076`)
Prompted by the observation that **blind targets are deterministic** — each ante
logs exactly **1 distinct target** (`ANTE_BASE_TARGET × BLIND_MULT`), identical
every run, with **The Wall (4× base) and Violet Vessel (6× base) the only
target-inflating bosses**. Verified true. Which means: since the target side is
*exact*, **100% of the margin error lives in the power estimate**.

So we finally measured that estimate. **It barely discriminates:**
```
AUC for predicting whether a blind is beaten (n=51,083):
  ALL 0.527 | ante3 0.635 | ante4 0.655 | ante5 0.647 | ante6 0.650 | ante8 0.661
```
0.5 is a coin flip. **Every build decision — `d-surv`, the reroll threshold,
dec-068's save-gate, the reverted dec-067 margin reward — is ranked by this.**
That is why 12 planner-valuation tweaks moved nothing: *you cannot fix a search by
retuning the weights of a leaf that doesn't discriminate.* dec-035 was right to
gate SOLVER Phase-2 search behind leaf validation — that gate stayed shut 22 days
because the validation was never run. (The pooled 0.527 vs per-ante ~0.65 is
Simpson's paradox — pooling 98%- and 74%-clear antes hides it.)

**Second finding — a metric that lied.** `realized` is floored to `target` on
**100.0% of beaten blinds (46,731/46,732)**: `if beaten: realized = max(realized,
tgt)`, which binds ~always because the per-step tracker never sees the final hand.
So `realized_vs_proj` degenerates to **1/raw_margin — a tautology**. **dec-070 was
justified by exactly that statistic** ("realized/proj ~0.30 at every ante"), i.e.
it changed the estimator on an artifact — and then `REALIZATION_FACTOR` was left
stale on top (the live double-discount). Now flagged `realized_censored` so
nothing builds on it again. The binary `beaten` label is clean.

**Shipped: instrumentation only.** A blind-START replay snapshot (deck rank/suit/
enhancement/seal counts, jokers, hand levels, hand size) so a **distributional
P(clear) estimator** — Monte-Carlo over the real deck vs the *exact* target,
reusing `find_best_hands` — can be built **and validated offline** against
`beaten`. It couldn't be before: no deck state was ever logged. Target to beat:
**AUC 0.65 at antes 4–6.** If a distributional leaf can't beat that, the whole
survivability-proxy approach is dead — and we learn it cheaply, before building
search on it.

---

### Refit the realization factor — end the double-discount — 07-23 (`dec-078`)
**`REALIZATION_FACTOR` 0.43 → 1.0.** dec-038 fit 0.43 (=1/2.30) because the *old*
power estimator crossed 50%-boss-clear at raw margin 2.30×. **dec-070 replaced
that estimator** with a play-frequency-weighted average (~2× lower estimates) and
**never refit RF** — so since 07-16 the planner has been **double-discounting**,
reading every build ~2× too weak. Refit against the *current* estimator:
boss-blind clear stratified by ante (n=16,995 post-dec-070 blinds, con-014
respected) crosses 50% at raw margin **~0.7–1.0** at the deep antes (ante5 `[0,1)`
= 50%; ante6 crossover ~1.0), so RF should be **~1.0**. *Watch the Simpson trap:*
pooled, the curve looks inverted (91% clear at margin <0.5) because it mixes easy
ante-2 bosses (92%) with hard ante-6 (54%); stratified, it's genuinely monotonic.
Single decision-path line (`planner.py`); build_progression stays RAW.

Honest scope: the leaf is weak (**AUC ~0.6**, dec-076) even correctly scaled, so
this **re-centers** the planner (survivability ~doubles → dec-068 save-gate fires
far more, d-surv comparisons shift) but can't make it *discriminate* — expect a
modest effect at most. Sign is uncertain (could bank interest too early), so it's
a **measured A/B**: ckpt 004434, same 300 seeds, RF 0.43 (existing baseline) vs
1.0 — cleanly measurable on a frozen checkpoint because RF drives the *planner*
heuristic, which runs live at eval. **Revert to 0.43 if advance at ante 4/5
doesn't hold.** Tests: `tests/test_planner.py` (23); 176 pass.

> **RESULT (paired A/B, ckpt 005596, same 300 seeds) → KEPT.** Directionally
> positive, not significant: reach≥4 65.0→**68.5%**, reach≥5 40.3→**45.9%**
> (+5.6pp), reach≥8 5→11 runs, WIN 1→4; paired McNemar @ante-6 **35 vs 27,
> P=56% [44,68]** (inconclusive). Only reach≥6 dipped (small n, noise). **Kept**
> because (1) it's the measurably-*correct* calibration (0.43 is a confirmed
> double-discount), and (2) the pre-registered rule was "revert if advance at
> ante 4/5 doesn't hold" — it held/improved. Not a plateau breaker (weak leaf,
> AUC ~0.6). RF is now `BALATRON_RF`-overridable (both arms ran on the current
> checkpoint since 004434 was pruned).

---

### The ante-4/5 plateau is SAVE-gate myopia, not build evaluation — 07-25 (`dec-079`)
**The plateau mechanism, finally located and quantified.** Deep audit of shop
decisions across 50,667 real blind snapshots + live shop telemetry:

*What is NOT broken* (each disproved by measurement, several were my own
hypotheses): the **reward** (97% ante/blind clears, corr 0.977); the **power
model** (`chips × mult × Π xmult` — xmult multiplies correctly); the **shop
ranker** — probed with real `j_` keys, `build_value` tops xmult at *every* stage
(Cavendish x3 ranked #1 even at ante 1), so the "flat-bias" and "first-xmult
marginal-value trap" hypotheses are both **wrong**; and a **learned leaf** —
a net on real build features scores AUC 0.640/0.583/0.573/0.563 at antes 3–6 vs
the analytical leaf's 0.590/0.583/0.560/0.635, i.e. **no better** (an initial
0.89 was a **`deck_n` data leak**; see gotchas). Evaluation is at its
ceiling — the wall is **construction**.

*What IS broken.* `[SHOP] SAVE: already clearing + marginal joker` is the **single
most common shop outcome** (268 events — more than all buys combined). Replaying
dec-068's `_already_clearing` over the real snapshots, the gate fires in **65% of
ante-2** and **54% of ante-3** shops — exactly the cheap engine-building window —
then only ~36% at antes 4–5, when it's too late. The trap is arithmetic: blind
targets **~double per ante**, so `AHEAD_BUFFER=1.0` ("one ante of headroom") is a
vanishing margin. Median headroom is **+1.54 at ante 2**, collapsing to
**+0.70 → +0.30 → +0.00** by antes 4–6. Consequence chain: slots fill at **ante 4
with median 0 xmult**, and **78% of runs never improve xmult composition again**
(slot-lock — once full, a swap needs weakest-slot-only + `+10%` immediate score +
affordability on a chronic $4–7). Net: **xmult acquisition is statistically
indistinguishable from random** (pool 23% xmult → E[xmult] in 5 random jokers
**1.17**; observed median **1**; 84% of ante-4 builds have ≤1). And survivors vs
dyers build *identically* (xmult 0.70 vs 0.73) — the whole population sits at one
weak ceiling where the boss is a coin flip.

**Fix:** demand headroom *proportional to the curve*, and **more of it early**
(a slot spent on an engine compounds for the rest of the run). `AHEAD_BUFFER` is
now ante-scaled via `AHEAD_BUFFER_EARLY_BONUS` × antes-below-4, **env-overridable**
(`BALATRON_AHEAD_EARLY_BONUS`, the dec-078 `BALATRON_RF` pattern) so both arms run
off **one binary** and the control (`bonus=0`) is *provably* byte-identical dec-068.
Simulated on real data, `bonus=1.0` drops SAVE firing to **0.9%/10.6%/25.9%** at
antes 1/2/3 while leaving **antes 4+ unchanged** (dec-060 boss-spike banking
preserved). Note dec-078 predicted this interaction: RF 0.43→1.0 doubled
survivability, so *"the dec-068 save-gate fires far more"* — dec-079 is the other
half of that fix. Tests: `tests/test_save_when_ahead.py` (+3); **179 pass**.

- **Pre-registered rule:** keep only if reach≥5 / reach≥6 improves on a paired
  A/B; revert the flag otherwise.
- Deferred second arm: **floor the prior-KL** (`prior_anneal_updates=250` zeroed
  the planner's guidance ~5,300 updates ago, so buy-*timing* has been unguided) —
  deferred because it only pays off *after* retraining, the unfalsifiable
  reasoning that made dec-075 null.

**MEASURED — NULL. Not shipped** (07-25, ckpt 006268, 300 paired seeds, `dec-080`).
Default stays `BALATRON_AHEAD_EARLY_BONUS=0.0` per the pre-registered rule.

| gate | B better | A better | P(B) |
|---|---|---|---|
| ≥4 | 41 | 33 | 55% [44,66] ns |
| ≥5 | 50 | 46 | 52% [42,62] ns |
| ≥6 | 39 | 28 | 58% [46,69] ns |

Mean ante 4.080 → 4.177, paired bootstrap 95% CI **[−0.100, +0.290]** (includes 0).
Wins 3/300 vs 2/300. Every gate ns; direction consistently positive but never
significant.

**The mechanism DID fire** — this is a real null, not a null implementation.
Replaying the gate over real logged builds: SAVE fires **66.5% → 8.3%** at ante 2
and **57.2% → 24.8%** at ante 3, i.e. the treatment banked in 8% of ante-2 shops
instead of 67% and bought far more aggressively, exactly as designed.

**What it rules out:** buy *timing/quantity* is **not** the binding constraint.
A behaviour change that large producing zero depth gain says acquisition
**quality** is the ceiling — buying more from a pool whose expected xmult content
is fixed (23% of pool → E[xmult]=1.17 in 5 slots, observed median 1) just fills
the same 5 slots sooner with the same composition. Consistent with the learned-
evaluator result (no model beat the analytical leaf once the `deck_n` leak was
removed): boss-clearing is variance-dominated, not under-evaluated.

---

## Gotchas & Hard-Won Lessons

### 1. The `won` flag means "reached the ante-8 boss," NOT "beat it"  *(critical)*
The base game (`functions/state_events.lua` `end_round()`) sets `G.GAME.won = true`
the moment you reach the ante-8 boss — **win or lose** — before the
target-met check. BalatroBot's API `won` field is just `G.GAME.won`. So a boss
**loss** (e.g. round score 90,592 / target 100,000, 0 hands left) reports
`won = true`.

- **Impact:** the win reward paid +10 for *losing* the ante-8 boss (corrupting
  training), and losses were saved as "win" clips (inflating the win count).
- **Rule:** detect a real win only via `ante > 8`, or a post-boss state
  (`SHOP`/`BLIND_SELECT`/`ROUND_EVAL`) seen with `won = true`. A loss goes
  straight to `GAME_OVER`, never to a post-boss shop.
- Fixed in `reward.py` (`_check_terminal` gates on `ante > 8`) and `train.py`
  (`safe_won_states = {SHOP, BLIND_SELECT, ROUND_EVAL}`; `GAME_OVER` won =
  `ante > 8 or already_recorded`). Commit `d387da3`.

### 2. Never sell copy/retrigger jokers — and resolve their copies correctly
All joker-selling paths must route through `_find_weakest_sellable_joker`, which
excludes eternal, negative, MUST_BUY (Blueprint/Brainstorm), retrigger, and copy
jokers. Ad-hoc "weakest joker" loops that only skip eternal jokers once sold a
Brainstorm to make pack room, collapsing the build.

Copy semantics are asymmetric and easy to get wrong: **Blueprint copies the
joker to its RIGHT; Brainstorm copies the LEFTMOST joker** — and a Brainstorm
that *is* leftmost copies itself, i.e. does **nothing** (a copy chain that
resolves to a leftmost Brainstorm is dead). The scoring resolver once resolved
a leftmost Brainstorm to the *next* joker instead — estimates inflated, and
shop/swap logic happily parked Brainstorm in the dead slot ("uses Blueprint
right, but not Brainstorm"). Fixed in `801f538`; the order optimizer was
already correct, so the bug lived purely in the estimate/decision path.

### 3. Card keys have no `the_` prefix
Match cards by their base-game center key, which is **not** the display name:
the Soul is `c_soul` (not `c_the_soul`), Hermit `c_hermit`, Fool `c_fool`,
High Priestess `c_high_priestess`. Verify against the game dump (`game.lua`
`P_CENTERS`). A `c_the_soul` typo silently never matched, so the agent passed
on free Legendary jokers. Key mismatches **fail silently** — only caught by
watching gameplay.

### 4. Booster-pack pick robustness (two halves)
The BalatroBot pack endpoint has a 5s `select_card` timeout waiting for
`G.GAME.pack_choices` to change; a lagging use-animation (or `STATE_COMPLETE`
not yet true) makes a pick return an error.
- **Retry transient failures** — don't skip the whole pack after one failed
  pick (that loses Mega/Black-Hole celestial packs). Commit `874583b`.
- **But skip cleanly when a pick is genuinely impossible** — a joker pack with
  full slots and no worthwhile swap: set `pick_idx = -1` and skip immediately
  instead of falling through and retrying a guaranteed "joker slots full"
  rejection ~12× before bailing. Commit `b5afc8e`.
- Note: the gamestate does **not** expose `pack_choices` (remaining picks) —
  only `count/limit/highlighted_limit`, and `highlighted_limit` is the
  simultaneous-highlight cap (always 1), not the pick count.

### 5. Base-game crash fixes live outside this repo
Seven fixes patch Balatro/BalatroBot itself and are **not** version-controlled
here — they must be re-applied if the mod is reinstalled/updated.

**ROOT CAUSE of the 06-10/11 crash wave (found last, explains everything —
and it was NOT game speed; 4× crashed at the same cadence as 8×):** the game
protects its UI flows with controller locks (`G.CONTROLLER.locks.toggle_shop`
etc.), but the mod's endpoints call `G.FUNCS.*` directly, **bypassing them**.
When a lagging transition times out client-side, the trainer re-issues the
action, and the second invocation's deferred events race the first one's
teardown — every nil-crash site in the wave (`shop`, `screenwipe`,
`blind_select`, `area`) was a double-fire. Two-layer fix:
`next_round.lua` now rejects calls while the toggle_shop lock is held
(original in `next_round.lua.bak`), and the trainer debounces transition
actions (`next_round`/`select`/`skip`/`cash_out` never re-issued within 8s
while the state name is unchanged — commit `fbfefc6`). The nil-guard TOMLs
below remain as defense in depth:
- `%APPDATA%/Balatro/Mods/balatrobot/lovely/blind_select_nil_fix.toml` —
  nil-guard in `button_callbacks.lua` `select_blind` (~2557): its
  0.2s-delayed event indexes `G.blind_select.alignment` while a later event
  in the same flow nils it — fast programmatic blind selection loses the
  race (15 crashes/2.5h, unmasked by the screenwipe fix). Both this patch and
  the screenwipe one were later EXTENDED to also guard their cleanup/remove
  events (crash sites 2577/3231 — double invocations racing teardown).
  **These nil-races surface one at a time as each dominant crash is patched**
  (shop → screenwipe → blind_select → cleanup events); when the 7th site
  appeared at unchanged cadence, the systemic lever fired: game speed 8 → 4
  (see gotcha 7).
- `%APPDATA%/Balatro/Mods/balatrobot/lovely/screenwipe_nil_fix.toml` —
  nil-guards on `G.screenwipe` in `button_callbacks.lua` `wipe_off`
  (~lines 3177/3213): the screen-wipe transition schedules deferred events
  that index `G.screenwipe` after fast programmatic actions tore it down —
  was crashing the game ~every 12 minutes (11 crashes/2.5h, 2026-06-11).
  **Diagnostic tip:** crash tracebacks ARE captured in
  `Mods/lovely/log/*.log` — grep for `attempt to`.
- `%APPDATA%/Balatro/Mods/balatrobot/src/lua/endpoints/cash_out.lua` — a
  ~300-poll timeout fallback so `cash_out` can't hang forever (original in
  `cash_out.lua.bak`).
- `%APPDATA%/Balatro/Mods/balatrobot/src/lua/endpoints/start.lua` — a
  600-poll timeout fallback (original in `start.lua.bak`). The endpoint waits
  for BLIND_SELECT with a `no_delete` condition event and **no timeout**;
  when `start_run` silently no-ops (menu race at speed 8), the connection
  hung ~30s per attempt, zombie events accumulated, and retries never
  succeeded without a game restart — 17 wedges in one night cost ~4×
  throughput. Pairs with the trainer-side 2s menu-settle delay (`d803ecb`).
- `%APPDATA%/Balatro/Mods/balatrobot/lovely/round_eval_fix.toml` — nil-guards
  on `G.round_eval` in `common_events.lua` (lines 1072 & 1195) to stop the
  endless-mode "attempt to index field 'round_eval' (a nil value)" crash.
- `%APPDATA%/Balatro/Mods/balatrobot/lovely/shop_nil_fix.toml` — nil-guard on
  `G.shop` in `game.lua` (~line 3243): `update_shop` schedules a NON-blockable
  0.2s-delayed event reading `G.shop.T.y`; a fast programmatic `next_round`
  exits the shop inside that window and the pending event crashes the game
  ("attempt to index field 'shop' (a nil value)").

Two recurring race classes, both triggered by fast programmatic transitions:
- **Crashes** — deferred animation events firing after the UI object they
  reference was torn down → one-line nil-guard via a lovely TOML patch.
- **Hangs** — endpoint condition-events waiting for a state with no timeout
  (`cash_out`, `start`) → poll-limit fallback that responds with a clean
  error instead of holding the connection forever.

### 6. Print UTF-8 safely / recover from process death
- The trainer prints emoji that crash on Windows `cp1252` when stdout is
  redirected/piped — always launch with `PYTHONUTF8=1`.
- The watchdog restarts **Balatro** on a hung/crashed game, but nothing
  restarted the **trainer process** — and twice the whole stack (server +
  trainer + monitoring shells) died *simultaneously* with no crash trace
  (external kill, likely Windows sleep), sitting idle until noticed.
- **Fix: `supervise.py`** — a detached process that owns the stack: every
  30s it ensures the game ports are listening and exactly one trainer is
  running, relaunching from the newest checkpoint with `PYTHONUTF8=1`.
- **THE REAL "always slow after 7-8h" CAUSE (06-14, dec-016) — it was never
  internal FPS decay.** Three fixes chased a phantom "trainer FPS decays ~1/n
  over its lifetime." The actual cause is **external RAM starvation**:
  `steamwebhelper.exe` leaks to 13–14 GB over hours → system RAM hits ~94% →
  Windows pages Balatron out → the trainer crawls to ~12 steps/min. Balatron's
  own footprint is only ~4 GB; it's the victim. **Killing the one leaked
  steamwebhelper dropped system RAM 95.4% → 38.6%.** The "1/n decay" was just
  progressive paging as RAM filled.
- **06-14 rebuild — bulletproof for long unattended runs.** Detection now uses
  the RELIABLE signal (heartbeat steps/min over a 12-min window, floor 80 — NOT
  the log's cumulative FPS, a misleading average) and acts in MINUTES, not
  hours: FROZEN heartbeat >4 min; CRAWL <80 steps/min over 12 min; CHURN ckpt
  >40 min stale; proactive **90-min** age recycle so the trainer never bloats.
  Kills **cascade** — every recycle kills ALL trainers + ALL games + ALL orphan
  launchers (the old single-PID `Select -First 1` kill let duplicates and
  orphans pile up until RAM was exhausted). The supervisor is a **singleton**
  (kills rival `supervise.py` on startup — two supervisors each spawned a
  trainer). A **memory guardian** restarts the external hog when system RAM is
  critical and the hog is clearly leaked (`steamwebhelper.exe` > 4 GB; normal
  < 1 GB), with a burst guard that backs off and just logs the diagnosis if
  recycling can't help. Logs are pruned (keep 6 newest / 24 h). All process
  management is psutil-based (no per-cycle PowerShell spawns). Health is mirrored
  to `logs/supervisor_status.txt`. Stop via a `SUPERVISOR_STOP` file. For
  overnight runs also disable standby: `powercfg /change standby-timeout-ac 0`.
- **A SECOND crawl source — the win-replay recorder (06-20, `dec-021`).** Not
  every crawl is the steamwebhelper RAM leak. The env-0 `RunRecorder`
  (`recorder.py`, `ffmpeg gdigrab` @30fps libx264) screen-captures env-0's game
  continuously and discards all footage unless the run wins — so at ~0 wins it's
  pure CPU waste, and gdigrab contends with the games' own rendering. On a
  CPU-saturated machine (RAM fine) this helped starve the games into a
  34 steps/min crawl + recycle loop. Fix: supervisor launches the trainer with
  `--no-record` while in the flat-ante/~0-win regime; re-enable once winning.
  When diagnosing a crawl with healthy RAM, check `ffmpeg.exe` / overall CPU,
  not just the leak.

### 7. Don't raise game speed to train faster — it destabilizes the game
Rollout collection (the live game) is the real wall-clock bottleneck, not the
net — so cranking `BALATROBOT_GAMESPEED` *looks* like the obvious speedup. It
isn't: very high speeds (100×/16×) caused stalls and desyncs. Speed history:
`100 → 16 → 8 → 4 → 8`. The 8→4 drop during the 06-11 crash wave was a wrong
theory — **4× crashed at the same cadence as 8×**, which falsified
"speed-bound races" and pointed to the real cause: double-fired transitions
(gotcha 5). Once the lock guard + transition debounce were deployed, 8× was
restored — speed was the wrong lever for that crash class. **Keep it at 8;
never raise above 8.** Speed lives in TWO synced places: `supervise.py` and
`start_balatro.bat` (the trainer's crash-recovery launch path). (The GPU doesn't
help here either — the net is tiny; the minutes go to the game playing, not the
PPO update.)

### 8. The rollout buffer uses the Gym done convention — GAE must too  *(critical)*
`dones[t] == 1` means **action t ended the episode** (that's what `amend_last`
sets). `compute_gae` originally read `dones[t+1]` (the CleanRL convention,
where done marks a *reset state*) — off by one index, every episode: the
terminal action bootstrapped V(next episode's start), diluting the win/loss
reward ~3x, and the second-to-last action was treated as terminal. Any new
GAE/return code must mask with `dones[t]`. Fixed in commit `2f32988`.

### 9. Step rewards settle one fetch later — amend, don't store
A reward computed from the delta `prev_raw → raw_state` describes the
**previous** action (both snapshots predate the current one). It must be
credited via `amend_last_transition`, never stored with the current action —
storing it created a one-step lag that put the blind-clear bonus on the first
SHOP action instead of the winning play. New transitions store reward 0; the
rollout-boundary block does a final settle so the last decision still gets
its reward. Any new code path that stores transitions or restarts the game
must keep this invariant (restart paths close the episode with
`amend_last_transition(done=True)` and call `_reset_run_state()`). Commits
`8f814b3`, `fe0a6dc`.

### 10. Shaping bonuses must be potential DELTAS, never per-step accruals
The joker-diversity and interest bonuses were re-paid every decision and
accrued +20–40 per run vs +10 for winning — the agent was paid more for
existing-while-diverse than for winning. Pay `Φ(s′) − Φ(s)` (acquire: +once,
lose: −once, hold: 0). Apply the same rule to any future shaping term.
Commit `c202a72`.

### 11. Spectral packs need their own evaluator
Routing spectral picks through `pick_best_planet` silently returned index 0
(no spectral key matches a planet), blindly taking Hex/Ankh — which destroy
every joker but one. `evaluate_pack_spectral` ranks all spectral cards,
allows Hex/Ankh only with exactly one joker, and returns None to skip.
Planet picks are now weighted by play frequency (0.05 floor) — absolute gain
always favored high-tier hands the bot never plays. Commit `d68204f`.

### 12. Real Balatro scoring order: card x-mults BEFORE joker flat mult
Glass (×2), Polychrome (×1.5) and held Steel (×1.5) fire during card/held
scoring; jokers trigger last. The model applied `enhance_xmult` as a final
global multiplier — over joker flat mult too — overestimating those hands by
up to ~50% (Glass pair + Joker: model 336, real 224) and triggering false
"wins the round" plays. Fold card x-mults into mult *before* adding joker
mult. Boss filters must compare `hand_type`, never the chase `detail` string
("Flush:Hearts" ≠ "Flush" — The Eye/Mouth filters were silent no-ops); The
Mouth's lock derives from `round_played > 0`, not the chase-commitment field
(reset on every play). The Psychic pads plays to exactly 5 (kickers are
free). Commit `52b5bc7`.

### 13. The 12 card-selection action bits only matter for action type 8
Play/discard cards come from the planner; only "use consumable with hand
targets" reads the bits. Their log-probs/entropy must be gated on
`type_action == 8` in the network — folded in unconditionally they churn the
PPO ratio on causally-dead dimensions, trip `target_kl` on irrelevant drift,
and point the entropy bonus at no-op bits. Target entropy must use the
*conditioned* target distribution (the one actually sampled). Commit
`c352b54`.

### 14. The dec-076 blind snapshot is END-of-blind — it LEAKS the outcome
`env.cur_blind_state` is rewritten at **every** `SELECTING_HAND` step, so a
"blind-start" snapshot actually holds the **last hand before the blind
resolved**. Any feature derived from it that changes *during* the blind encodes
the result: `deck_n` (cards remaining) is high when a blind is beaten on hand 1
and low when the agent dug through 4 hands and failed.

This produced a **fake breakthrough**: a model predicting `beaten` scored
**AUC 0.893** pooled / 0.885 at ante 3 — seemingly proving a better leaf was
possible. Permutation importance exposed it: `deck_n` **+0.282** vs `ante`
+0.049 and *every* real build feature ≤0.011. Dropping the contaminated
features collapsed it to **0.640/0.583/0.573/0.563** at antes 3–6 — no better
than the analytical leaf. The real conclusion is the opposite of the first
result: boss-clearing is **variance-dominated**, and no leaf beats ~0.6.

- **Rule:** never report a learned-model AUC on this data without a leakage
  check. Run permutation importance; if one feature dwarfs the rest, suspect it
  before believing the score. Split **temporally** (train older, test newer).
- Genuinely blind-invariant features only: jokers, hand levels, hand_size,
  ante, boss. Suspect anything mid-blind mutable (`deck_n`, hands/discards
  left, money-at-capture).
- The snapshot itself should be captured **once** at blind start — until then,
  treat every `start` field as end-of-blind.


### Stale-decision aborts don't teach the policy — 07-22 (`dec-077`)
A stuck/stale audit of a live run found ~1 mechanical-failure event/game (~68
`[STATE-GUARD]` aborts + ~74 `[STUCK]` force-plays per run-log) while play is
still centered on the ante-4/5 plateau (`dec-057`), 1% win rate.

`[STATE-GUARD]` fires when the policy's action was valid for the state it was
decided on, but the game raced ahead before the send landed (dominant:
`HAND_PLAYED`/`ROUND_EVAL`/`DRAW_TO_HAND` needing `SELECTING_HAND`). The action
**never executed** — yet a transition was stored with the intended action **and**
the next-iteration settle applied `REWARD_INVALID_ACTION = -0.1` (`reward.py:279`)
because `action_succeeded=False`. So the policy was penalized for **API latency**
and trained on a decision that never happened — mechanical noise fighting the
strategy signal (`dec-076`).

- **Fix (option A):** on a stale abort, skip the whole iteration — store no
  transition and don't advance the reward chain (`prev_raw`/`last_action` stay
  put). The real previous action still settles on the next successful iteration
  (or the rollout boundary). **No `reward.py` change needed:** the stale action
  never becomes `last_action_succeeded=False`, so the −0.1 is never applied.
- **Scope:** only the `STATE-GUARD` live-not-in-required race; generic execute
  failures still store + penalize.
- **con-010:** a skip is a retry, so it's bounded — `env.stale_abort_streak`
  (per-env, `con-013`) escalates to `_restart_balatro` at 8 consecutive aborts so
  a genuine desync can't spin invisibly with no heartbeat.
- **Corrected diagnosis:** the first hypothesis (a missing settle-gate on the
  decision path) was wrong — `_get_actionable_state` already waits out transient
  states; the lever was in the reward accounting, not the game loop.
- **Not yet measured** — recommend an eval A/B once it's picked up on the next
  trainer restart. `train.py`, `env_session.py`.

---

### The build FREEZES after ante 3 — the swap path was unreachable — 07-26 (`dec-081`)

Traced the ante-4/5 plateau to a code defect, not judgement. `action_space.py`'s
full-slot branch vetoed a shop-joker buy unless swapping out the *heuristically
weakest* joker raised **immediate single-hand score** by ≥1.1×. Failing that it
fell through to `continue`, leaving the mask at its `np.zeros` default → buy
**illegal** → `any_buyable_joker` False → `ACTION_BUY_JOKER` hard-blocked → the
agent leaves the shop. Slots fill by ante ~3, so this froze the build for the rest
of every run.

| measured over 6 trainer logs | |
|---|---|
| full-slot shops | 31,185 |
| swaps actually executed | **289 (0.93%)** |
| shops where the PLANNER wanted a swap | **~55%** |

`_planner_pick_swap` — which evaluates all 5 sell candidates by multi-ante
survivability — was therefore **unreachable**. Also a **con-011 violation**: the
mask was making a heuristic value judgement instead of testing legality.

**Fix (env-gated `BALATRON_SWAP_LEGALITY`, default 0):** legality = "some sellable
joker frees the slot and funds it"; neutral weight 1.0, no heuristic thumb; the
worth-it call goes to the planner. Also fixed the affordability precheck, which
counted only the *weakest* joker's sell price. Verified on 120 real builds:
full-slot buys reachable **63% → 98%**.

**MEASURED — WORSE. Not shipped** (ckpt 006344, 300 paired seeds, `dec-081`).

| gate | B | A | P(B better) |
|---|---|---|---|
| ≥4 | 28 | 38 | 42% [31,54] |
| ≥5 | 34 | 41 | 45% [35,57] |
| ≥6 | 25 | 40 | 38% [28,51] |

Mean ante **4.240 → 4.093**; bootstrap CI [−0.330,+0.033]. No single gate is
significant, but **all six favour control — sign test p=0.031**, so this reads as
a small real harm, not noise.

**Why it hurt (hypothesis, unverified):** a swap is inherently money-losing (sell
at ~half, buy at full). That cost only pays if the chooser is a good judge — and
the leaf is AUC ~0.6. Unblocking the path meant acting *more often* on a
near-coin-flip evaluator at a guaranteed cost. The ugly 1.1× veto was accidentally
protecting the agent from that. Testing it properly needs per-shop swap/money
logging. *Kept dormant (default 0) rather than deleted — it encodes the finding.*

---

### Snapshot capture-once — the `deck_n` leak that faked AUC 0.89 — 07-26 (`dec-082`)

The dec-076 replay snapshot sits in the per-**step** `SELECTING_HAND` handler, so
despite its "once per blind" comment it re-ran every hand and the surviving row
described the blind's **END** state. That made `deck_n` an outcome proxy — beaten
on hand 1 leaves a full deck, a 4-hand grind leaves a stub. A model trained on the
logged features read **AUC 0.89** almost entirely off that one column; with
`deck_n` removed it fell to **0.58–0.64**, i.e. no better than the analytical leaf
it was meant to beat.

**Fix:** guard the capture on an `(ante, blind_name)` key and tag rows
`at_blind_start: True` so validation can filter pre-fix rows explicitly. Verified
live after a trainer restart — clean rows show median `deck_n` **44 vs 33**.

⚠️ **Regime boundary (con-014):** `start` semantics changed from end-of-blind to
start-of-blind. Never trend `deck_n` across this fix. All 51k prior rows remain
contaminated and cannot validate any deck-reading estimator.

---

### Monte-Carlo rollout evaluator — built, GATED, unvalidated — 07-26 (`dec-083`)

Three A/Bs (dec-075, dec-079, dec-081) came back null-or-worse *despite their
mechanisms provably firing*. The common factor: they all improved **access** to
shop decisions while the thing **ranking** them predicts `beaten` at AUC
0.56–0.63. The leaf's structural flaw is that it computes one deterministic point
estimate (`committed hand score × hands × RF`) assuming the build always draws its
hand type — so it is blind to **consistency**, which is what decides a boss.

`environment/rollout.py` returns `P(clear target)` by *simulating* the blind:
reconstruct a deck, deal, pick the best play with the real scoring engine, play
out the hands, repeat N times. 9ms/40 samples, deterministic per seed.

**NOT in the decision path.** `BALATRON_ROLLOUT` defaults to 0; the planner
integration covers the current ante only (its deck is known; the rollout costs
~100× the leaf) and is fail-safe (`None` → keep the analytical estimate).

**Pre-registered gate — offline first:** `tools/validate_rollout.py` must show
**≥ +0.05 AUC** over the leaf at antes 4–6 on post-dec-082 clean rows *before* any
A/B is run. This deliberately converts a 2.5h A/B into a cheap offline check.

**Early evidence is against it:**

| | AUC |
|---|---|
| rollout, deck as-logged (leaked) | 0.774 |
| **rollout, deck size normalised** | **0.669** |
| analytical leaf | 0.638 |

The leak was worth **+0.105** of the apparent gain; the honest edge is **+0.031**,
*below* the gate. It may well fail — which is itself informative: it would mean
boss-clearing is not predictable from build state and this architecture is at its
ceiling regardless of evaluator.

---

### SIL was cloning lucky wins, not learning from them — 07-26 (`dec-084`)

Audit question: *does balatron take the right signal from wins and learn from it?*
**Signal: yes. Learning: crude.**

**What was already right** (verified, not assumed):
- **Win detection is sound on both paths.** `GAME_OVER` requires `ante > 8`; the
  post-boss path trusts the API `won` flag **only** in `SHOP`/`BLIND_SELECT`/
  `ROUND_EVAL`, because the base game sets `won=true` on merely *reaching* the
  ante-8 boss. Confirmed in logs: `ante=8 … api_won=True` → **`won=False`**.
  con-001's trap is closed. Recorded wins land at ante {9:35, 10:24, 11:4, 12:1}.
- Wins-only capture (dec-040), FIFO buffer holding the most recent ~165 wins,
  SIL live and learning (loss **0.123 → 0.115** over 106 updates).

**The defect:** `_sil_loss` was `-mean(log π(a|s))` — uniform **behaviour
cloning** wearing SIL's name, no advantage weighting. A winning run is ~180 steps
at a ~1.3% win rate in a game we proved is **variance-dominated** (dec-083: no
learned evaluator beat AUC ~0.6), so many banked wins are **lucky**. Uniform
cloning teaches *"the average behaviour of runs that happened to win"* rather than
the decisions that won them. Amplified because `BC 0.000@0.00` and `Pr …@0.00` —
both annealed out, leaving **SIL as the only live guidance channel**.

**Fix:** real SIL (Oh et al.) — weight by `(R − V(s))⁺`. Returns were previously
unrecoverable because rewards settle one step late (`amend_last_transition`), so
a parallel `env.episode_rewards` track mirrors that same lag, the win bonus lands
on the last captured step (capture already stops at win detection, dec-058), and
discounted return-to-go is computed at flush with `config.gamma`.

- Pre-fix transitions load with **NaN** returns and keep the old uniform weight —
  the corpus is weeks of 1.3%-rate wins and is irreplaceable.
- Normalised by **total weight, not count**, so `sil_coef` keeps its meaning.
- `sil_advantage_filter` (default **True**); `False` restores the byte-identical
  old loss, pinned by a test. Tests: `test_sil_advantage.py` (+6); **201 pass**.

⚠️ **UNMEASURED** — deploys on the next trainer restart, no A/B run. Efficacy
depends on critic calibration: wins are rare, so the critic under-predicts them
and `(R−V)⁺` stays positive for most steps of a winning run — this **re-weights**
more than it sharply filters. If win rate or mean ante degrades, set
`sil_advantage_filter=False` before investigating anything else.

---

### Evaluation is a dead end — four measured levers — 07-26 (`dec-085`)

`dec-083`'s Monte-Carlo rollout leaf **FAILED** its pooled A/B. Not shipped
(`BALATRON_ROLLOUT` stayed 0 throughout, so the revert was a no-op).

| | mean ante | 95% CI |
|---|---|---|
| batch 1 (300 seeds) | +0.107 | [−0.070, +0.283] |
| **batch 2 (300 FRESH seeds)** | **+0.010** | [−0.180, +0.197] |
| **pooled (600)** | **+0.058** | **[−0.072, +0.187]** |

Batch 1 looked directionally positive (4 gates favouring treatment, 0 against).
**It did not replicate.** In batch 2 the DEEP gates reverse — ante≥7 12/21 to
control (36%), ante≥8 8/13 (38%) — so pooled, shallow gates lean slightly
positive and deep gates lean negative: the signature of no effect plus noise. A
real evaluator gain should persist or grow with depth. Wins 6 vs 8 of 600.

**This is the important result.** The rollout was *measurably better at its job* —
**+0.064 AUC** on clean leak-controlled data (a gate set before it was built) and
it changed **32% of shop picks** — and run depth did not move. Four independent
levers now, each with a mechanism **verified to fire before the outcome was read**:

| lever | mechanism proven | outcome |
|---|---|---|
| dec-075 escape hatch | — | null |
| dec-079 buy timing | save gate 66.5%→8.3% | null |
| dec-081 buy legality | reachability 63%→98% | **worse** (p=0.031) |
| dec-083 rollout leaf | +0.064 AUC, 32% picks changed | null |

⇒ **Better build evaluation does not produce deeper runs.** Boss-clearing is
variance-dominated. **Stop tuning the planner's evaluation** (constraint recorded).

---

### The gamma fix was silently undone by lambda — 07-26 (`dec-086`)

`gae_lambda` **0.95 → 0.99**. GAE discounts by γ·**λ**, not γ — so the earlier
`gamma` 0.99→0.995 raise, made *specifically* so early-ante decisions would feel
the win, was neutralised:

| step-40 shop decision, win at step 179 (median run) | credit |
|---|---|
| γ alone (`0.995^139`) | **0.50** ← what the gamma fix intended |
| γ·λ (`0.9452^139`) | **0.0004** ← what actually reached it |
| γ·0.99 (`0.9850^139`) | **0.12** ← with this change |

Credit decayed to 10% after **41 steps** against runs of median **179**. The build
decisions that determine a run were learning from ~nothing. `0.97` was rejected —
0.7% at the same distance doesn't solve it. This is the last untested mechanism
attacking **construction** rather than evaluation, which dec-085 just closed off.

⚠️ **This CANNOT be A/B'd.** `gae_lambda` is used only in `compute_gae` during
TRAINING, so it has no effect on a fixed checkpoint. It is evaluated by training
with it and watching the trend — a different, weaker design than every planner
lever, and stated as such rather than dressed up as an A/B.

**Regime boundary (con-014) — do not trend across it.** Pre-registered *before*
deploying:

- **Baseline** (last ~150 updates, ~6399–6491): mean ante **4.30** (sd 0.43),
  WR500 **1.37%**, value loss **0.178**, EV **0.620**.
- **Window:** ≥500 updates (~1–2 days). The horizon change only pays off once the
  policy has *learned* differently; earlier readings are noise.
- **Success:** mean ante or WR500 clearly above the baseline noise band.
- **REVERT trigger:** WR500 sustained below ~0.7%, **or** a variance blow-up —
  value loss ≫0.178 or EV ≪0.62 — which is the known cost of raising lambda.
- **Revert:** `gae_lambda` back to 0.95.

**Co-deployed with `dec-084`** (advantage-filtered SIL) on the same restart —
recorded here so neither is misattributed later. Accepted because dec-084's
expected effect is small (it re-weights more than it filters) while this changes
the credit horizon by ~4x; if the regime degrades, revert both and bisect.

---

### Codebase audit — dead code out, silent failures surfaced — 07-26 (`dec-087`)

Sweep over 23.5k lines. **Behaviour is unchanged** — this removes traps for
future work, it does not move the plateau.

**The real defect:** `derisk_saveload.py` sat importable at the repo root calling
`asyncio.run(main())` at **module level**. Merely importing every module to check
for import errors **connected to the live game on port 12346 and round-tripped a
`.jkr` save/load against the trainer's in-progress run at ante 3.** Now
`__main__`-guarded.

**Eight boss effects** (Psychic, Water, Tooth, Hook, Amber Acorn, Cerulean Bell,
Serpent, Manacle) were detected into locals nothing read — *looked* like coverage,
was dead weight. Verified each is really handled elsewhere before removing
(Psychic has its own "must play 5" wrapper; Water/Manacle are in
`BOSS_DIFFICULTY`; all 8 are in `BOSS_BLIND_INFO`, so the policy sees them).

**Exception handlers: classified, not swept.** 104 broad handlers, 55 silent, 37
unexplained. Sweeping would have been wrong — most exist so instrumentation can
never take the trainer down. The dangerous subset is the four on the **decision
path**, where swallowing changes what the agent *does*:

| site | silent consequence |
|---|---|
| `find_best_hands` | `current_score`=0.0 → mask treats the hand as worthless |
| `find_best_discard` | `discard_ev`=0.0 → agent stops digging |
| `_estimate_joker_value` | shop card reads worthless **in the state vector** |
| joker flags | that joker drops out of the state vector entirely |

These now call `diagnostics.warn_once` — loud on first failure, then backing off
(1st, 2nd, 5th, 10th, 100th…), with exact counts via `swallowed_counts()`. Bare
`print` was rejected: per-step hot paths would emit thousands of lines a minute,
get muted, and the signal would be lost. The helper never raises. The two
handlers that are *correctly* silent are now documented as such.

**Found and deliberately NOT fixed:** the dagger-sacrifice heuristic has a
horizon term that was started and never finished (`remaining_antes` computed and
discarded), so that ratio has **always** been horizon-free. Completing it would
change which jokers get sacrificed — a behaviour change belonging in its own
measured commit, not a cleanup. Recorded in the code so it isn't re-found as a
mystery.

Also: duplicate `TIER_WEIGHTS` key (same value, harmless), 31 unused imports, 47
dead locals, 11 orphaned expressions. `ruff.toml` added — **F + E9 only, no style
rules**, because this file's dense decision-history comments would be buried by a
reformat.

⚠️ **Two tooling traps, now guarded (`con-019`):** `ruff --fix` rewrote **12 files
in the cambium worktree nested inside `.git`** (restored; now excluded).

**Verification — and a methodology correction.** The first fingerprint sampled
*randomly* from `logs/blind_results.jsonl`, a file the live trainer **appends
to**, so its input changed between runs and the hash wasn't reproducible; it
briefly looked like the refactor had broken something. Re-done with the first 250
rows pinned in file order:

| | fingerprint |
|---|---|
| pre-refactor `061c9f2` | `6eb4c7fe2aab400d8d0c0f90` |
| post-refactor `47d35ca` | `6eb4c7fe2aab400d8d0c0f90` |
| final | `6eb4c7fe2aab400d8d0c0f90` |

Plus action mask identical on 60 real shop states. **207 tests pass, ruff clean.**

---

### `gae_lambda` reverted on its own variance trigger — 07-27/28 (`dec-088`)

`dec-086` (λ 0.95→0.99) **REVERTED** at ~185 of its pre-registered 500 updates.

| | baseline | post-dec-086 (105 sampled) |
|---|---|---|
| **EV** | **0.620** | **0.387** (43% of updates <0.40) |
| VL median | 0.178 | 0.159 *(fine)* |
| VL mean | — | **0.327** (18% of updates spiked >0.5, one 1.14) |
| mean ante | 4.30 | 4.19 *(inside the ±0.43 noise band)* |

The variance trigger — written **before** the data was seen, precisely so it
couldn't be rationalised past — fired. Fat-tailed value loss plus a 38% relative
EV drop, with **zero hint of upside**.

**The counter-argument, stated rather than buried:** some EV loss is
*mechanically expected* as λ→1, because GAE then leans on realised returns rather
than the critic, so a worse-fitting critic matters less for the advantage
estimate. EV alone is not proof of harm. What decided it was the combination — a
real stability cost, no upside, and ~2 more days needed to confirm what the
variance signal already showed.

⚠️ **This reverts the CURE, not the diagnosis.** The credit-assignment problem is
untouched and still real: at λ=0.95 a step-40 shop decision receives **~0.04%** of
the win signal for a win at step 179, so build decisions still learn from almost
nothing. **0.97 was considered and rejected** — it reaches only 0.7% at that
distance, so it pays the variance without solving the problem. Any next attempt
should **shorten the episode or reshape the reward** so a shorter horizon
suffices, rather than stretching the horizon to cover 179 steps.

`dec-084` (advantage-filtered SIL) **stays on** — barely confounded anyway, since
the trainer reports only 496/30000 demo transitions carry a return, so the filter
is ~1.7% active and phases in as new wins bank.

---

### The plateau is a two-ante STEP-DOWN, not a decay or a cliff — 07-28 (`dec-089`)

**Analysis only** — no code, config or experiment. Conditional survival decomposed
from existing logs (5,000 runs / 50,003 blinds, all post-U4417; dec-086 window
split as a robustness check: boss clear 83.7% vs 84.1% post-revert, so pooling is
safe).

| ante | 1 | 2 | 3 | **4** | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| conditional clear | 93.8% | 85.2% | 83.5% | **67.3%** | **47.0%** | 39.8% | 39.0% | 34.8% |
| Δ | | −8.7 | −1.7 | **−16.2** | **−20.3** | −7.2 | −0.7 | −4.3 |

**Not smooth decay** (that needs a *constant* rate; spread is 2.70× max/min).
**Not a single cliff** (two adjacent drops, and the pre-drop region isn't flat).
**45.6% of all runs die at antes 4–5.**

**The boss is the killer** — gap vs small blind widens 4.6 → 9.8 → 20.1 → **31.1
pp** by ante 5. But **no single boss is the wall**: ante-stratified (essential —
pooled, The Psychic reads 74.3%, yet at ante 6 it is the *worst* at 26.9%), the
ante-6 spread is 26.9%–83.8% and the hard ones cluster by **mechanism**
(score-suppressors: Psychic, Mouth, Wall, Eye, Water).

**The decisive number:** the boss target curve *decelerates* (2.67× → 2.50× →
2.20× → 1.82× → 1.43×) exactly where survival collapses (87.8% → 75.7% → 60.3%).
The wall gets *relatively easier* to keep pace with and the agent falls behind
anyway ⇒ **build power stops growing around ante 4.**

Chip deficit at loss (uncensored: 0/4518 losses censored): median **0.721**,
broad — 28.7% near misses (≥0.85) **and** 24.5% blowouts (<0.50). Both problems
coexist.

⚠️ **Two requested failure classes are NOT derivable — reported, not estimated
around.** "Failed to meet chips" and "ran out of hands" are *the same event*
(`hands_left` reads 1 on 4233/4273 losses — it is recorded *before* the fatal
hand; a first classifier returned "100% died with hands left" and was discarded
as an artifact). **"Economy death" needs a shop-level log that does not exist.**
Smallest fix: one row per shop visit — `{ante, money, offers:[{key,cost}],
n_affordable, bought_key|null, rerolls_used}`.

⚠️ **Unresolved:** whether the post-ante-5 plateau (~35–40%) is real stabilisation
or survivorship — antes 6/7/8 have overlapping CIs on n=1055/420/165.

---

### Antes 4–5 are VARIANCE-dominated, not strength-gated — 07-28 (`dec-090`)

**Analysis only.** Resolves the question `dec-089` left open — and corrects its
claim that the logs couldn't answer it. `blind_results.start` carries full build
state at **every** blind including deep antes, so strength-stratified clear rates
were computable all along. That was a failure to look, not a real gap.

**The test:** survivorship and real-difficulty make opposite predictions once
strength is held fixed. Survivorship ⇒ matched-strength rates are FLAT across
antes; real difficulty ⇒ they still FALL. Measured with **two independent
proxies** — planner margin, and a structural one (xmult count × committed hand
level) chosen so the answer can't inherit dec-076's weak-leaf problem. They agree.

**Answer: BOTH, and they partially cancel** — which is exactly why the raw curve
looked flat.

| | ante 3 | ante 5 | ante 8 |
|---|---|---|---|
| mean xmult | 0.67 | 1.03 | 1.84 |
| % strong (2x+) | 12.8% | 26.9% | **61.0%** |

Survivorship is large. But at *matched* strength, rates still fall: mid(0x,lvl≥3)
78%→28%, good(1x,lvl≥3) 78%→46%, **strong(2x+) 76%→61%**. The drop *shrinks* as
strength rises — stronger builds are more ante-robust.

**THE UNEXPECTED FINDING — build strength barely matters where runs actually die:**

| ante | weak (0x) | good (1x) | strong (2x+) | spread |
|---|---|---|---|---|
| **4** | 76% | 78% | **76%** | **2 pp** |
| 5 | 54% | 59% | 64% | 10 pp |
| 7 | — | 47% | 60% | **32 pp** |

At ante 4 a build with **2+ xmult engines clears the boss at the same rate as one
with none** (n=490 vs n=254). Strength-dependence only emerges from ante 6.

⇒ The antes where **45.6% of runs die are variance-dominated**; the deep antes
that only **3.3%** of runs reach are strength-gated. This is outcome-side
confirmation of what the AUC work found from the predictor side, and it explains
why four build-QUALITY levers (dec-075/079/081/083) came back null or worse — at
the antes that kill runs, build quality is not the binding variable.

⚠️ It does **not** identify what the binding variable *is* at antes 4–5, only that
xmult count and hand level don't capture it. Untested candidates: draw variance,
boss-ability interaction, in-blind play quality. Deep-ante tiers rest on n=29–129.

---

### In-blind play is NOT the lever either — 07-28 (`dec-091`)

The last unexamined decision surface. Every build-side one was already closed
(dec-073 reward, dec-085 evaluation, dec-079/081 access, dec-088 credit horizon,
dec-090 no build feature predicts clearing), and discard usage measured sensible.
Chips only come from played hands — nobody had checked whether the agent played
the **best** one.

`play_quality.py` logs, per played hand, the agent's cards and score against the
**best available** hand — computed under the **same boss debuffs** (an undebuffed
optimum would manufacture a gap that isn't the agent's fault). Logging-only, one
guarded call site placed *after* the rearrange/fallback mutations so it records
the cards actually sent.

**Pre-registered before any data:** <0.90 = a real lever; ≥0.95 = play is fine
and antes 4–5 are variance-dominated.

**Result (329 clean plays): mean capture 0.9829, median 1.0000, exact-best 97.0%.**

| ante | 1 | 2 | 3 | **4** | **5** |
|---|---|---|---|---|---|
| capture | 1.000 | 0.999 | 0.999 | **0.957** | **0.935** |
| exact-best | 100% | 98.3% | 98.5% | 94.3% | **90.0%** |

Play *does* degrade exactly where runs die — a real finding. **But the arithmetic
rules it out:** at ante 5 the median loss sits at **0.70 of target** (needs +43%),
while perfect play adds **~7%**. An order of magnitude too small.

⚠️ **Two bugs in my own instrumentation, both caught by invariants, not
inspection:**
1. Discarded `classify_hand`'s `scoring_indices` and passed `range(len(cards))`,
   crediting every played card. Live data read **capture 1.061** — impossible
   against a true maximum, which is how it surfaced. Fixed + `capture ≤ 1.0` test
   and a stricter "best hand ⇒ capture exactly 1.0" test.
2. **Scoring-path disagreement — now DIAGNOSED and FIXED.** 171/16,545 (1.03%)
   of plays used the *same cards* as the baseline yet the two paths returned
   different scores. **All 171 were boss blinds and all 171 had a debuff active**
   (151 suit, 20 face), the direct path reading higher in 163. Cause:
   `classify_hand` has **no debuff awareness**, so it returns scoring indices
   still including debuffed cards that `find_best_hands` correctly excludes. Fix:
   score BOTH sides by looking the agent's play up inside `find_best_hands`' own
   enumeration — removing the discrepancy class by construction rather than
   clamping over it. Verified across every debuffed-boss case: 0 disagreements,
   0 captures >1.0, 0 enumeration misses. Pre-fix rows keep their flag.

**RE-READ ON 16,545 PLAYS (50× the first sample) — finding STRENGTHENED:**

| ante | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| capture | 0.997 | 0.993 | 0.989 | 0.983 | 0.965 | 0.961 | 0.940 | 0.936 |
| exact-best | 98.4% | 98.6% | 97.8% | 96.4% | 94.3% | 93.8% | 92.0% | 91.8% |

Ante 5 is **0.9648**, not the 0.935 the thin sample suggested — so perfect play
adds only **~3.6%** against the **+43%** a median loss needs. The conclusion holds
with a wider margin than first reported.

---

### Every build-side null was measured inside a band the agent never leaves — 07-29 (`dec-093`)

**The economy hypothesis died first.** dec-090 swept seven build features and
never tested *money* — which mattered, because dec-031 concluded the agent is
*rationally* not stacking xmult (it never pays by ante 4) and dec-080 showed
acquisition is random-like, which is what you get when you cannot afford to
reroll for better. Blind-level AUC for money predicting a boss clear: **0.528 /
0.551 / 0.541 / 0.524** at antes 3–6. But that is the *wrong test* — cash scores
no chips, so it cannot predict clearing the blind you are standing in. Money buys
*future* power. The right frame is run-level, reconstructing runs (5,732, split
per env on ante resets) and asking whether state at ante A predicts **reaching
ante A+2**:

| from ante | n | money | xmult | hand lvl |
|---|---|---|---|---|
| 3 | 3943 | 0.540 | 0.476 | 0.486 |
| 4 | 3418 | 0.556 | 0.491 | 0.523 |
| 5 | 2299 | 0.562 | 0.558 | 0.555 |

Nothing predicts depth either. Economy closes exactly like every other build-side
hypothesis.

**But the obvious reading of all of this is wrong.** "Builds don't matter" is NOT
supported, because of **range restriction**: at ante 4 the agent's entire build
space is 0–2 xmult engines — **3 engines = 2.9% of the field, 4 = 0.1%**. A real
Balatro engine is a stacked multiplicative loop worth 5–10×. Every measurement in
dec-085/090 and above compared *mediocre builds against other mediocre builds*
and correctly found no difference. **Mediocre-vs-engine has never been measured,
because the agent never builds an engine.**

So the premise underneath the entire investigation — *better builds go deeper* —
is untested. Four levers optimized toward it without anyone verifying it.

**The probe.** `engine_forcing.py` (env-gated `BALATRON_FORCE_ENGINE=1`, default
OFF so control is byte-identical): buy by flat engine priority (xmult > copier /
retrigger > additive scaling > economy), **bypass the dec-068
bank-when-already-clearing hold**, and reroll to hunt when no engine piece is
affordable. That hold is the key bypass — it is locally correct (why buy power
you don't need for *this* ante?) and is exactly what leaves the agent arriving at
ante 6 with the build that only ever cleared ante 3.

This is an **instrument, not a proposed policy**. It is deliberately crude and is
certainly a worse policy in the short run, so a raw win-rate drop alone would not
refute the premise.

**Pre-registered before any data:**
- forced engines reach **deeper** → engines *are* the lever; the problem is
  acquisition/affordability, not evaluation. Reopens everything.
- forced engines **don't** → builds don't determine depth even at the top of the
  reachable range, and the ceiling is somewhere nobody has looked.

**Mandatory manipulation check, also pre-registered:** if the forced arm does not
actually shift the xmult distribution upward vs control, the instrument did not
fire and the run is **INVALID, not null**.

⚠️ That check exists because the first draft of `_tier` keyed off
`scaling_type`/`scaling_increment` — **fields that do not exist anywhere in
`data/jokers.py`**. Every joker would have scored 0, the forced arm would have
been byte-identical to control, and the experiment would have reported a
confident null while measuring nothing. Caught by dumping the real schema before
wiring. `test_tiers_are_not_all_zero` and `test_every_field_tier_reads_actually_exists`
now pin it. *A vacuous instrument that reports "no effect" is worse than no
instrument.*

**RESULT — three arms, same 60 seeds, same checkpoint (`update007070`):**

| arm | med $ | buys/run | t5/run | t5 share | afford/run | capture | **mean ante** | ≥5 |
|---|---|---|---|---|---|---|---|---|
| control | 9 | 5.67 | 0.67 | 12% | 3.33 | 20% | **4.267** | 46.7% |
| mode 1 (spend) | 6 | 4.58 | 0.63 | 14% | 0.82 | 78% | **3.283** | 23.3% |
| mode 2 (bank) | 8 | 4.77 | **0.90** | 19% | 2.78 | 32% | **3.817** | 35.0% |

Control reproduces the 600-seed baseline (4.267 vs 4.277), so the 60-seed subset
is representative.

**Mode 2 shifted engines/run 0.67 → 0.90 (+34%) and outcomes got WORSE.** By the
pre-registered rule the manipulation fired, so this is a real result — but it is
a result about *slightly more engines*, not about engines. 0.90/run is still deep
inside the 0–2 band. **The hypothesis that a real 3–5 engine build wins remains
untested, because no shop policy can reach that range.**

⚠️ **Two of my own numbers in this session were wrong, both flattering the story
I was already telling — recorded because the errors are the useful part:**

1. **The "affordability cliff" was an artifact.** I reported median bankroll $6
   against $7 engines, a $4-wide cliff, "the agent lives under the wall". That
   was measured on **mode 1's log — the arm that had spent its own bankroll
   down**. Control's real median is **$9**, it can afford **3.33** engines/run and
   declines **80%** of them. Engines were never priced out. I measured the wall
   inside the one arm guaranteed to manufacture it.
2. **"Mode 2 suppressed buying (0.92/run vs control's 5.67)"** compared mode 2's
   FORCE-ENGINE *subset* against control's *total*. Consistent accounting across
   all buy paths gives 4.77 vs 5.67 — banking barely changed buy rate.

**A third limitation, unresolved:** `_tier` scores any `xmult=True` joker as
tier 5, but that set includes heavily conditional cards (Blackboard needs all-black
hands, Loyalty Card fires every 6th, Cavendish is a 1-in-1000 lottery). "Tier 5"
is therefore a proxy for *nominal* xmult, not for a working engine. Forcing
nominal-xmult share from 12%→19% buying more conditional jokers that rarely fire
is a plausible mechanism for mode 2 being worse, and it means **the arms tested
"buy more cards labelled xmult", not "build an engine".** Any future attempt must
score jokers by *realised* contribution, not by schema flag.

**Where this leaves the plateau:** engine acquisition is not affordability-limited
(control declines 80% of what it can pay for) and not obviously policy-limited in
a way that helps (forcing it up hurt twice). The 0–2 band is not a budget
constraint — it is what the shop supplies (3.42 tier-5 offered/run against 5
slots and 5.67 buys/run) crossed with the agent preferring non-engine cards.
Whether a genuine engine wins is still unmeasured.

**Invalid first pilot, retained deliberately.** Before the fix, the forced branch
returned a no-op whenever it had no engine to buy AND no affordable reroll — the
normal state at antes 1–2 ($4–5 money, $7 engines, hunting needs $8). The arm did
nothing on every early shop step and spun until runs died: **mean ante 2.06, 21
of 64 runs dead at ante 1**, a blind no build strategy can fail. Read naively
that is a crushing refutation of the engine hypothesis. It was the instrument
destroying itself. An override may only ever ADD a decision it can act on; with
nothing to do it must defer. `test_early_ante_money_leaves_the_forcing_with_nothing_to_do`
pins how *common* that state is so it can never be treated as a rare edge case.

---

### 226 updates of training bought nothing — the plateau is intact — 07-30 (`dec-094`)

Training-log win counts looked far above the historical ~1% and raised the
question of whether the plateau had finally broken. It had not.

**Training log first** (5,000 non-curriculum runs, 07-29/07-30, chronological
chunks of 1000): 1.00 / 0.80 / 1.30 / 1.10 / **1.70** percent, overall 1.18%
[0.92%, 1.52%], mean ante 4.158. The trailing 1.70% is the source of the "more
wins" impression, but its CI [1.06%, 2.71%] overlaps every earlier chunk and the
sequence is non-monotone — noise around ~1.2%.

**Held-out eval, the EXACT 600 seeds the baseline used** (`eval_seeds.txt` +
`eval_seeds_batch2.txt`), so the only difference is 226 updates of training — no
code change is active, since the dec-093 forcing is env-gated off and the rest of
that commit was additive:

| metric | baseline `006964` | current `007190` |
|---|---|---|
| mean ante | 4.277 | **4.242** [4.102, 4.381] |
| win % (ante>8) | 1.00% | **0.83%** [0.36%, 1.94%] |
| reach ≥5 | 47.3% | 47.7% [43.7, 51.7] |
| reach ≥6 | 20.3% | 22.7% [19.5, 26.2] |
| reach ≥7 | — | 7.5% |
| reach ≥8 | — | 2.8% |

Wins counted per con-001 (ante advanced PAST 8); the `won` flag agreed exactly
(5 and 5). Nothing separates from baseline; win% is nominally *down*, and the
one positive (reach≥6, +2.4pp) sits inside its own CI.

**Endpoints were chosen BEFORE the run, from a power calculation.** Distinguishing
1.0% from 1.7% needs **~4,261 seeds per arm**; at n=600 the win% CI half-width is
±0.87% on a ~1.2% rate. So win% cannot be settled at this sample size by
construction, and mean ante / reach depth were designated primary. Reporting
win% as the headline would have been a claim the design could not support.

⚠️ **The comparison is UNPAIRED, and that is a self-inflicted limitation.** No
per-seed results file survived from the baseline run — only the summary
aggregates and the checkpoint. Pairing by seed would tighten the interval
materially. Judged not worth another hour of stopped training here, because a
−0.035 delta with a ±0.139 half-width cannot become a gain under pairing, only a
sharper null — but a marginal or positive future result must re-run the preserved
baseline checkpoint rather than trust remembered numbers. Going forward the
per-seed jsonl is preserved (`baselines/eval_600seeds_update007190_PERSEED.jsonl`).

**Two rules out of this:** never read a win-rate improvement off training-log
counts (volume moves counts without moving rate), and always keep the per-seed
output of anything that may later serve as a baseline.

---

### The agent's evaluator cannot SEE the scaling-engine archetype — 07-30 (`dec-095`)

dec-093 couldn't test the engine hypothesis because `_tier` scored jokers by the
`xmult=True` **schema flag**, identical for a real engine and for Blackboard
(X3 only if every card *held in hand* is black), Loyalty Card (every 6th) or
Cavendish (1-in-1000). So: measure what each joker **actually contributes**.

`compute_joker_scoring` gained an optional `breakdown` list; `play_quality` logs
it per played hand. Deltas are captured at the **iteration boundary** of the
joker loop — snapshot on entry, attribute to the previous joker, flush the last
after the loop — rather than at the 11 accumulation sites inside it, because the
body has many `continue` paths and a per-site hook silently misses whichever
branch a joker took. Default `None` ⇒ the default path is untouched.

**Validated against theory:** Acrobat (`final_hand_of_round`, implemented) fires
**11%** of hands with mean **1.226×**, against a predicted 1 + 0.11×2 = **1.22**
for an X3 firing 11% of the time. Three decimals on 62 samples — the delta
arithmetic is right.

**Then it found something much bigger.** Ranking nominal tier-5 engines by how
often they actually fire:

| joker | held | fire% | mean × |
|---|---|---|---|
| The Idol | 27 | **0%** | 1.00 |
| Obelisk | 57 | **0%** | 1.00 |
| Acrobat | 62 | 11% | 1.23 |
| Campfire | 53 | 28% | 1.07 |
| Flower Pot | 25 | 48% | 1.96 |
| Photograph | 44 | 61% | 2.30 |

The obvious read — *four of six engines are dead* — is **wrong**, and checking
saved it. `hand_eval` contains **zero** occurrences of `xmult_scaling` and
**zero** of `rotating_condition`. Those jokers don't read 0% because they're
dead; they read 0% because **the estimator does not model them**.

**17 of 150 jokers are affected, 14 of them nominal tier-5 engines** — the entire
scaling archetype: Glass Joker, Hologram, Lucky Cat, Vampire, Constellation,
Madness, Yorick, Canio, Ramen, Obelisk, The Idol, Ancient Joker, Campfire, Hit
the Road.

### ⛔ RETRACTED, 07-31 — the instrument was blind, not the evaluator

**The conclusion above is wrong.** `hand_eval` implements scaling correctly. The
retraction is kept in place of a silent edit because the error chain is the
lesson.

What actually happened, in order:

1. `play_quality` passes `raw_state` jokers **without** the `_scaled_value`
   injection that `GameState.inject_scaling_values()` performs from its
   `ScalingTracker`. So every scaling joker reads ×1.0 in the instrument no matter
   how large it has grown. Verified directly: Vampire **without** `_scaled_value`
   → ×1.0, **with** `_scaled_value=2.5` → ×2.5.
2. I grepped `hand_eval` for the literal strings `xmult_scaling` and
   `rotating_condition`, got zero hits, and concluded the effects were
   unimplemented. But the implementation keys off **`scaling_type`** (which does
   exist — `'xmult'` for all 12) and `_scaled_value`. `hand_eval.py:~1008` already
   applies a tracked `_scaled_value` unconditionally, carrying a comment stating
   the exact reasoning I later "discovered" independently: *these jokers' trigger
   field describes what makes them grow, not when they score.*
3. I asserted `scaling_type` was absent from the schema on the basis of a
   truncated `head -40` key dump. A test written to pin that absence **failed**,
   which is what exposed the whole chain.
4. Acting on the wrong conclusion I wrote a parallel scaling implementation into
   `hand_eval`. It was **reverted before commit** — on the agent's live path
   `_scaled_value` *is* injected, so the new branch would have **double-applied
   every scaling multiplier** on real scoring.

**What survives:** the breakdown instrument itself (validated to three decimals
against Acrobat), and the fact that its scaling-joker numbers are a **floor, not a
measurement**. Fix is to thread the GameState (or the already-injected joker list)
through to `log_play` from its `train.py` call site.

**What does not survive:** "the agent can't see the scaling archetype", and with
it the mechanism it appeared to give for the plateau. No such mechanism has been
demonstrated.

Twice now a confident plateau explanation has come from an instrument measuring
its own defect — dec-093's no-op stall, and this. Both were caught by a check
written to confirm the story rather than to tell it.

---

### Recording HUMAN play, to reach a build region the agent cannot — 07-31 (`dec-096`)

Every build-side result is uninterpretable for one reason: **range restriction**.
The agent holds ~0.67 xmult engines per run and never leaves the 0–2 band (3
engines = 2.9% of the field, 4 = 0.1%), so dec-085/090/093 all compared mediocre
builds against other mediocre builds and correctly found nothing. dec-093 then
showed **no shop policy reaches that range** — forcing it made outcomes worse in
both directions tried.

Human wins supply the missing arm **observationally**. Secondarily, `demo_buffer`
currently holds only the agent's own ~1% wins, so SIL imitates mediocre play.

`human_record.py` polls the gamestate the mod already exposes and **never sends an
action**, so it cannot interfere with play or corrupt a run.

**The hard part is that a human does not announce actions.** The trainer knows
what it did because it chose it; here the action must be reconstructed from
consecutive states. The two halves are biased in *opposite* directions on purpose:

- `infer_action` returns `None` on anything ambiguous, so the transition is
  **dropped rather than guessed** — a mislabelled action teaches the agent the
  wrong lesson from a good run, which is worse than missing data.
- `is_decision` calls a transition a decision **when unsure**, so genuinely missed
  actions stay visible in coverage instead of being excused as engine noise.

Together those mean coverage cannot be flattered from either side.

**The filter was necessary, not cosmetic.** The first measurement reported 47%,
computed as labelled ÷ *all* state changes. A live diagnostic showed the misses
were dominated by `state`-only and `chips`+`state` changes — scoring animations,
round evaluation, draws and round-boundary counter refills. None are decisions;
putting the game's own state machine in the denominator measured the wrong thing.
With the filter: **60%** overall, 75% in the steady-state window.

⚠️ **Measured against the AGENT at `GAMESPEED=8`, not a human.** A human plays far
slower than the 0.35s poll, so real coverage should be higher — but that is an
assumption, not a measurement, and must be re-checked on a real session.

**Deliberately not built:** the `demo_buffer` path (state-vector encoding +
`head_indices`). That is the expensive half and is pointless until input quality
is known; a pipeline fed mislabelled actions trains on a fiction. Unlabelled
decisions are printed with their changed fields so a low number can be diagnosed
rather than merely reported — and if they stay high, it means the *agent's* action
space cannot represent moves a human treats as routine, which is worth knowing on
its own.

---

### Hand levels vs the hand actually played — and why reward can't fix it — 08-01 (`dec-097`)

`pick_best_planet` levels the build's **committed archetype**, not the hand being
played. That is deliberate: its docstring says Jupiter beats Mercury *"even if
you've played more Pairs"*, and the Pillar 3b comment dismisses play frequency as
*"lagging frequency"*. On clean rows the committed hand is the most-played hand
only **50.4%** of the time — half the celestial investment goes into a hand the
agent isn't primarily playing.

⚠️ **The existing alignment signal was fabricating data.** `_committed_hand_signals`
opened its max-search at `most_n = -1`, so before any hand was played `0 > -1`
made the **first hand in dict order** the reported `most_played`, and strict `>`
stopped any other zero-count hand displacing it. The game lists **Flush House**
first.

That produced 47,474 rows reading *"commits Flush but plays Flush House"* — 36% of
all apparent misalignment — while `play_quality` ground truth shows Flush House
played **zero times in 31,041 hands**. 47,472 of them were rows where nothing had
been played yet. It polluted **31.8%** of `build_progression` and inflated
misalignment from its true **49.6%** to 65.6%. Fixed; `most_played` is now `""`
when nothing has been played.

**The cost is still unmeasured, because the obvious instrument can't see it.**
`build_progression.margin` is `power/target` with
`power = estimate_score_for_hand_type(COMMITTED)` — it measures how strong the
*committed* hand looks and structurally cannot see the loss from playing a
different one. It gave a meaningless 1.00–1.06 ratio. `blind_results` now carries
`committed_ht` / `played_ht` / their levels / `ht_aligned` beside the binary
`beaten` flag (con-001). Reading pre-registered in
`audit_ht_alignment_cost.py` before any data.

**A reward signal was requested; it cannot work, for a structural reason.**
`pick_best_planet` is called by the trainer at `train.py:2063`, so planet
selection is **heuristic-owned** under dec-002's hybrid split. The policy never
emits that action, so PPO has nothing to reinforce and no gradient path to the
behaviour — a reward would pay the agent for an outcome it does not control.
Separately, con-008 requires potential-based shaping and dec-073 established PBRS
is **policy-invariant by construction**, which is exactly why margin shaping was
null; an alignment potential would be null by the same theorem.

The lever is the **picker** — a heuristic we edit directly, no training, effect
immediate. Deliberately not changed yet: the cost is unmeasured, and *fix-then-
check* is the pattern behind four nulls (dec-075/079/081/083) and two
instrument-artifact retractions.

**General rule out of this:** don't add a reward term for an outcome the policy
doesn't control. Heuristic-owned decisions — planet picks, hand selection, joker
order — have no action for PPO to reinforce. Fix the heuristic.

**RESULT — 35,945 tagged blinds. Alignment does NOT help; the surface closes.**

| ante | aligned | misaligned | diff |
|---|---|---|---|
| 3 | 93.6% | 94.5% | −1.0% |
| **4** | **88.1%** | **88.6%** | **−0.5%** |
| **5** | **80.1%** | **80.7%** | **−0.7%** |
| 6 | 76.1% | 78.9% | −2.8% |
| 7 | 72.5% | 78.4% | −5.9% |

Aligned clears the same or *slightly worse* at every ante but 2 and 8. At the
plateau antes the gap is −0.5% / −0.7%, inside the CIs. Per the pre-registered
reading, levelling the committed hand is fine and the 50% misalignment is
harmless. **Not changing `pick_best_planet` before measuring saved a fifth null
lever.**

**A second null fell out:** hand LEVEL doesn't predict clearing either. Within
antes 4–5, levels 1–4 clear at **86.0 / 84.6 / 84.8 / 85.7%** — flat. Consistent
with dec-090, where hand level scored AUC 0.544 at ante 4.

⚠️ **Confound caught in my own output.** The first per-level table showed clear
rate falling 92.5% → 81.8% from level 1 to 5, which reads as *"levelling hurts"*.
It was pure ante confound — mean ante rises monotonically with level (2.87 →
6.34), so the table was measuring ante difficulty. Stratifying within antes 4–5
erases it. **Any per-level or per-build statistic must be stratified by ante
before it is interpreted** — the same mistake dec-090 was built to avoid.

---

### The plateau is exponential decay, and every A/B was underpowered — 08-02 (`dec-098`)

Ten decision surfaces investigated, ten reported nulls, plateau unmoved. The
framing was that one component must be broken. **The framing was wrong, and the
evidence used to eliminate each candidate could not have detected a real
improvement anyway.**

A run clears ~24 blinds, so **win rate = per-blind clear rate ^24**:

| ante | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| clear | .978 | .953 | .942 | **.886** | **.804** | .772 | .749 | .777 |

Product over 3 blinds/ante = **2.22%**, against 0.83–1.18% measured — right
order, gap from skips and non-independence.

**Consequence 1 — no single lever can fix this.** A 10% win rate needs **0.909
per blind: +5pp at every ante simultaneously**. No shop policy or evaluator
delivers that. The hunt for the broken component was chasing something that
doesn't exist.

**Consequence 2 — the old nulls are non-results.** A good lever is worth ~+1pp
per blind: a **1.3× win rate**, genuinely worth having. Detecting that as win
rate needs **18,627 runs/arm**. dec-079/081/083/093 ran **60–600**.

**Per-blind clear is ~24× more sample-efficient** — and the number isn't
arbitrary: a run yields 24 blinds of evidence but only **one bit** of win/loss,
so scoring per blind recovers what the binary throws away.

| effect | blinds/arm | ≈ |
|---|---|---|
| +3pp | 1,934 | hours |
| +2pp | 4,498 | ~a day |
| +1pp | 18,537 | ~a week |
| +0.5pp | 75,416 | out of reach |

So 2–3pp levers are now testable **from training logs alone** — no stopped
training, no eval sessions. `audit_blind_clear.py` does it, with `--split-step`
for A/B at a deploy boundary and `--min-step` for con-014 regimes.

⚠️ **Two arithmetic errors of mine, both caught by tests written to pin the
claims rather than by re-reading them:**
1. I reported the efficiency gain as **~7×**. It is **24.1×**. The 7× compared an
   ante-4 base (0.886) at Δ0.005 against a 0.858 base at Δ0.01 — varying *both*
   the base rate and the effect size across the two sides, which makes the ratio
   meaningless.
2. I conflated the uniform `0.858^24 = 2.53%` with the per-ante product `2.22%`.
   Different quantities; the per-ante product is harsher because the late antes
   sit far below the mean and the exponent punishes them.

**Caveat:** per-blind clear conditions on *reaching* that ante, so it is a
conditional rate — a lever that changes reach-depth alters the population being
compared. Check arm sizes alongside any difference.

---

### ENGINES WIN — the hypothesis dec-093 couldn't test — 08-02 (`dec-099`)

Triggered by a real win the user spotted: seed `ZHAYBFDH`, **ante 10**, verified
per con-001 as advancing *past* 8 rather than trusting the `won` flag. It carried
Photograph + Madness (tier-5) plus Hanging Chad + Brainstorm.

**Stratified by REACHED ante** — every run in a row had ~the same number of shops:

| engines | reached 6+ | reached 7+ | reached 8+ |
|---|---|---|---|
| 0 | 0.8% | 2.7% | 8.3% |
| 1 | 4.1% | 9.9% | 27.7% |
| 2 | 6.2% | 14.3% | 36.0% |
| **3+** | **18.0%** | **32.7%** | **62.1%** |

At ante 6+, 3+ engines wins **22× more often** than 0 (18.0% [11.7–26.7] vs 0.8%
[0.2–2.9]) — **non-overlapping CIs at every depth**.

The stratification is not optional. Raw win-rate-by-engine-count is severely
survivorship-biased: mean final ante rises monotonically with engine count
(3.63 → 6.56), so engines and survival feed each other circularly.

**This reconciles the two results that appeared to contradict it:**
- **dec-090** found no build feature predicts *per-blind* clearing at antes 4–6 —
  correct, because engines barely rescue the blind in front of you. They
  **compound** across all remaining blinds, exactly the dec-098 frame where a
  +2pp per-blind edge is invisible at one blind and decisive over twenty.
- **dec-093** found forcing engines made things *worse* — also consistent, because
  forced acquisition of nominal tier-5 cards (including dead conditionals) at the
  cost of tempo is a different population from natural assembly.

**The bottleneck is FREQUENCY, not value:** the agent assembles 3+ engines in only
**4.5%** of runs (219/4,906).

**And the constraint is SLOT ALLOCATION, not acquisition.** Comparing the
*non-engine* composition at ante 6+ (comparing engine counts would be circular):

| tier | 3+ engines | 0 engines |
|---|---|---|
| copier/retrigger | 0.41 | 0.55 |
| additive scaling | 0.17 | 0.43 |
| economy | 0.05 | 0.08 |
| **tier-0 (no value)** | **1.45** | **3.67** |
| total held | 5.21 | 4.73 |

Engine builds don't hold *more* of anything — they hold **less junk**. Engine-free
runs carry **3.67 worthless jokers out of 4.73**, 78% of a near-full board, and
never sell them. No individual non-engine joker stands out (every lift ≈1%), so
it is the aggregate junk load, not a missing enabler.

⚠️ **Not fully causal.** Conditioning on reached-ante removes the dominant
confound but not all of it. `_tier` is also the nominal schema flag (the dec-095
limitation), counting Blackboard and Cavendish as engines — that adds noise and
biases *toward* the null, so the true effect is plausibly larger.

**This reopens `dec-081` (swap-legality)**, which exists precisely to stop the
mask vetoing full-slot buys, and was closed on a 300-seed win-rate measurement —
a non-result under dec-098's power arithmetic.

---

## Operations

See the [Usage](README.md#usage) section for launch commands. Key points:

- **Two processes:** start the server+game (`uvx balatrobot serve --fast`,
  `BALATROBOT_GAMESPEED` for speed) and wait for `127.0.0.1:12346`, then start
  the trainer.
- **Always** `PYTHONUTF8=1` (see gotcha #6).
- **Resume from the newest** `checkpoints/balatron_phase1_updateNNNNNN.pt`, not
  `_final.pt` (a stop/crash auto-save).
- `--checkpoint-interval 2` saves every 2 PPO updates (~10–30 min) so a crash
  loses minimal training; the default 10 risks losing hours.
- Recording is **win-only** — winning runs are kept in `recordings/wins/`,
  everything else is discarded.
