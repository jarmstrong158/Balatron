# Decisions & Gotchas

A running log of the **why** behind Balatron's design, and the hard-won lessons
behind its fixes. New sessions (human or AI) should read this before changing
core logic. The machine-queryable mirror lives in `.context/` (Context Keeper).

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
