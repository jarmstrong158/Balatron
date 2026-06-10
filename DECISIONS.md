# Decisions & Gotchas

A running log of the **why** behind Balatron's design, and the hard-won lessons
behind its fixes. New sessions (human or AI) should read this before changing
core logic. The machine-queryable mirror lives in `.context/` (Context Keeper).

---

## Architecture & Design Decisions

### Hybrid: PPO policy on top of a heuristic layer
The agent is **not** pure RL. A PPO actor-critic (817-dim state → shared trunk
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
to zero is what lets PPO surpass it.

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

### 2. Never sell copy/retrigger jokers
All joker-selling paths must route through `_find_weakest_sellable_joker`, which
excludes eternal, negative, MUST_BUY (Blueprint/Brainstorm), retrigger, and copy
jokers. Ad-hoc "weakest joker" loops that only skip eternal jokers once sold a
Brainstorm to make pack room, collapsing the build.

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
Two fixes patch Balatro/BalatroBot itself and are **not** version-controlled
here — they must be re-applied if the mod is reinstalled/updated:
- `%APPDATA%/Balatro/Mods/balatrobot/src/lua/endpoints/cash_out.lua` — a
  ~300-poll timeout fallback so `cash_out` can't hang forever (original in
  `cash_out.lua.bak`).
- `%APPDATA%/Balatro/Mods/balatrobot/lovely/round_eval_fix.toml` — nil-guards
  on `G.round_eval` in `common_events.lua` (lines 1072 & 1195) to stop the
  endless-mode "attempt to index field 'round_eval' (a nil value)" crash.

### 6. Print UTF-8 safely / recover from process death
- The trainer prints emoji that crash on Windows `cp1252` when stdout is
  redirected/piped — always launch with `PYTHONUTF8=1`.
- The watchdog restarts **Balatro** on a hung/crashed game (~48s), but nothing
  restarts the **trainer process** if it dies — it can sit idle. Relaunch from
  the newest checkpoint. (A supervisor loop is the proper fix.)

### 7. Don't raise game speed to train faster — it destabilizes the game
Rollout collection (the live game) is the real wall-clock bottleneck, not the
net — so cranking `BALATROBOT_GAMESPEED` *looks* like the obvious speedup. It
isn't: high speeds repeatedly cause UNKNOWN-state stalls, desyncs, `round_eval`
nil-crashes, and hung packs. Speed went `100 → 16 → 8` for exactly this reason.
**Keep it at 8.** Stability at 8 beats churn at higher speeds. (The GPU doesn't
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
