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
