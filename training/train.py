"""
Balatron — Training Orchestrator

Ties together all components:
- GameStateManager (API + state vector + scaling tracker)
- ActionDecoder + action masks (action space)
- RewardCalculator (reward shaping)
- BalatronNetwork (actor-critic)
- PPOTrainer (rollout buffer + PPO updates)

Main loop:
1. Connect to BalatroBot
2. Collect transitions by playing the game
3. When buffer full -> PPO update
4. Log metrics, save checkpoints
5. Repeat

Module layout (06-14 monolith decoupling — see Context Keeper dec-017):
- training/config.py            TrainConfig
- training/episode_tracker.py   EpisodeTracker
- training/joker_order_logger.py JokerOrderLogger
- training/env_session.py       EnvSession (one per parallel game)
- training/action_executor.py   ActionExecutor — action->API translation +
                                shop/pack auto-actions (the heuristic planner seam)
This file keeps the orchestrator itself: the rollout loop (_collect_rollout /
_get_actionable_state), PPO update cadence, checkpointing, and metrics.

Usage:
    python -m training.train
    python -m training.train --checkpoint path/to/checkpoint.pt
    python -m training.train --phase 2 --checkpoint phase1_best.pt
"""

import argparse
import asyncio
import json
import glob
import os
import random
import string
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from agent.network import BalatronNetwork, create_network, get_head_index
from agent.ppo import PPOConfig, PPOTrainer
from environment.game_state import GameStateManager, STATE_VECTOR_SIZE
from environment.action_space import (
    build_action_mask, ActionDecoder, ACTION_HEAD_SIZE,
    get_action_type_name,
)
from environment.hand_eval import (
    find_best_discard, find_best_hands, estimate_score_for_hand_type,
    plan_optimal_action, compute_optimal_joker_order,
    plan_consumable_use, optimize_play_order,
    evaluate_pack_tarot, evaluate_pack_spectral, pick_best_planet,
)
from recorder import RunRecorder
from demo_buffer import DemoBuffer
from data.jokers import JOKERS
from training.config import TrainConfig
from training.episode_tracker import EpisodeTracker
from training.env_session import EnvSession
from training.action_executor import (
    ActionExecutor, _find_weakest_sellable_joker,
)


def _get_blind_target_from_state(raw_state: dict) -> float:
    """Extract the current blind's target score."""
    blinds = raw_state.get("blinds", {})
    if isinstance(blinds, dict):
        for b in blinds.values():
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                return b.get("score", 300)
    return 300.0
from environment.reward import RewardCalculator


def _build_composition(joker_cards: list) -> tuple:
    """Count (xmult, scaling, total) jokers in a build. xmult = the
    multiplicative engine that gates depth (data: ante-7+ runs avg 1.36 xmult
    vs 0.71 at ante 3-4). Classified via the JOKERS schema, matching the
    death-build analysis. Used by the build-progression leading indicator."""
    nx = ns = 0
    for c in joker_cards:
        name = c.get("label") or ""
        s = JOKERS.get(name)
        if not s:
            continue
        if s.get("xmult") or s.get("xmult_scaling") or s.get("scaling_type") == "xmult":
            nx += 1
        if (s.get("scaling_type") or s.get("mult_scaling")
                or s.get("xmult_scaling") or s.get("chip_scaling")):
            ns += 1
    return nx, ns, len(joker_cards)


# ============================================================
# Training Orchestrator
# ============================================================

class Trainer:
    """Main training loop.

    Connects to N BalatroBot instances (ports 12346..12346+N-1), plays
    them in parallel via asyncio, collects transitions into per-env
    buffers, and runs PPO updates over the combined data.
    """

    def __init__(self, config: TrainConfig, checkpoint_path: Optional[str] = None):
        self.config = config

        # Create components
        self.network = BalatronNetwork()
        self.ppo = PPOTrainer(self.network, config.to_ppo_config())
        self.action_decoder = ActionDecoder()
        self.episode_tracker = EpisodeTracker()
        # env 0 owns the real recorder; rec_port=12346 lets it target env 0's
        # specific game window (not an arbitrary same-titled parallel game).
        self.recorder = RunRecorder(enabled=config.record_wins, rec_port=12346)

        # Self-imitation demo buffer (Phase 1: capture-only). Persisted across
        # restarts so wins aren't lost on the frequent supervisor recycles.
        # ALL envs feed it (not just env 0 — unlike the video recorder, demo
        # capture is just data, so every parallel game's wins are worth saving).
        self.demo_buffer = DemoBuffer(
            capacity=config.demo_capacity,
            state_dim=STATE_VECTOR_SIZE,
            action_dim=14,
            mask_dim=ACTION_HEAD_SIZE,
            path=config.demo_path,
        ) if config.collect_demos else None
        self._demo_captures = 0  # count of trajectories captured this session
        # SIL Phase 2: let the PPO update sample this buffer for self-imitation
        # (active only when config.sil_coef > 0; capture alone is Phase 1).
        self.ppo.demo_buffer = self.demo_buffer

        # POLICY AUTHORITY (real-RL mode). When True, the network's chosen
        # action actually executes: it owns play-vs-discard and which joker to
        # buy. The heuristic layer is demoted to TACTICAL COMPUTATION only
        # (which exact cards score best for the chosen action, hard-legality
        # guards) — it no longer makes the judgment calls. This is the fix for
        # the "decorative network" finding (audit 06-13): previously the
        # heuristic re-decided everything and PPO trained on its choices, so
        # the policy had no causal stake and couldn't learn. Set False to
        # revert to the legacy heuristic-drives-everything behavior.
        self.policy_authority = True
        self.action_executor = ActionExecutor(
            policy_authority=self.policy_authority)

        # One session per parallel game instance; env 0 owns the real
        # win recorder, the rest get no-ops.
        num_envs = max(1, getattr(config, "num_envs", 1))
        self.sessions = [
            EnvSession(env_id=i, port=12346 + i, phase=config.phase,
                       recorder=self.recorder if i == 0 else None)
            for i in range(num_envs)
        ]
        # Single-env compatibility aliases (summary/util paths)
        self.game = self.sessions[0].game

        # Training state
        self.global_step = 0
        self.num_updates = 0
        self.start_time = 0.0

        # dec-045: EVAL mode (held-out fixed-seed evaluation). Off for training;
        # set True only by evaluate.py. Everything eval-specific is gated behind
        # this flag so the training path is byte-identical when it is False.
        self.eval_mode = False

        # Load checkpoint if provided
        if checkpoint_path:
            self.ppo.load_checkpoint(checkpoint_path)
            self.global_step = self.ppo.total_steps
            self.num_updates = self.ppo.total_updates
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"  Resuming from step {self.global_step}, update {self.num_updates}")

        # Ensure checkpoint directory exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _touch_heartbeat(self):
        """Update logs/heartbeat so the supervisor can tell a frozen or
        DEGRADED trainer from a working one. Called on every stored
        transition. Format: "<unix_time> <global_step>" — the timestamp
        catches freezes (stale mtime), the step counter lets the
        supervisor compute steps/min and catch slow-churn stalls (chronic
        wedge cycles kept the heartbeat fresh while throughput fell 4x)."""
        try:
            hb = os.path.join("logs", "heartbeat")
            with open(hb, "w") as f:
                f.write(f"{time.time()} {self.global_step}")
        except OSError:
            pass  # never let liveness reporting break training

    def _maybe_capture_demo(self, env):
        """Flush this episode's trajectory to the demo buffer iff it WON or
        reached a high ante; otherwise drop it. Called on every run-finished
        path via _reset_run_state, BEFORE the win/ante flags are cleared.

        Phase-1 self-imitation capture: pure logging, never reads back into
        the policy. Every env feeds it (demo capture is just data, so unlike
        the single-window video recorder, all parallel games' wins count)."""
        if self.demo_buffer is None:
            return
        if self.eval_mode:                      # dec-045: don't pollute SIL with eval runs
            env.episode_traj = []
            env.max_ante_seen = 1
            return
        traj = env.episode_traj
        # dec-040: WINS-ONLY capture. The old condition also banked any run that
        # merely REACHED demo_min_ante (6) even if it LOST. With ~0 wins the buffer
        # filled ~96-99% with losing ante-6/7/8 trajectories, and _sil_loss has no
        # positive-advantage filter — so SIL was imitating LOSSES (a degenerate
        # attractor pinning the plateau; RL audit's #1 finding). Imitate only true
        # successes; once real wins appear, SIL becomes a correct flywheel.
        keep = bool(traj) and env.win_recorded
        if keep:
            n = self.demo_buffer.add_trajectory(
                [t[0] for t in traj], [t[1] for t in traj],
                [t[2] for t in traj], [t[3] for t in traj],
            )
            self._demo_captures += 1
            print(f"[DEMO] captured: env{env.env_id} ante={env.max_ante_seen} "
                  f"won={env.win_recorded} steps={n} | "
                  f"buffer={len(self.demo_buffer)} "
                  f"(lifetime {self.demo_buffer.trajectories_added} traj)",
                  flush=True)
            if self._demo_captures % self.config.demo_save_every == 0:
                self.demo_buffer.save()
        # Always clear the accumulator for the next run.
        env.episode_traj = []
        env.max_ante_seen = 1

    def _curriculum_prob(self) -> float:
        """Annealed probability of starting a run from a banked seed. Anchored
        at the FIRST call (curriculum start), not absolute num_updates — the
        trainer resumes from a checkpoint already past the anneal window, so an
        absolute anneal would be 0 from the start (the bug that made loads never
        fire)."""
        cfg = self.config
        if not cfg.curriculum_enabled:
            return 0.0
        if getattr(self, "_curriculum_start_update", None) is None:
            self._curriculum_start_update = self.num_updates
        elapsed = self.num_updates - self._curriculum_start_update
        frac = max(0.0, 1.0 - elapsed / max(cfg.curriculum_anneal_updates, 1))
        return cfg.curriculum_prob * frac

    def _pick_curriculum_seed(self):
        """Return an absolute path to a random banked seed to load, or None
        (fresh start) — with annealed probability. None if disabled / no seeds."""
        p = self._curriculum_prob()
        if p <= 0.0 or random.random() > p:
            return None
        seeds = glob.glob(os.path.join(self.config.seed_dir, "*.jkr"))
        return os.path.abspath(random.choice(seeds)) if seeds else None

    async def _harvest_seed(self, env, ante: int, nx: int):
        """Bank the current run as a curriculum seed (capped, FIFO). Called when
        a FRESH run reaches ante 4/5 with an xmult engine started. The save is
        a non-disruptive snapshot."""
        try:
            sd = self.config.seed_dir
            os.makedirs(sd, exist_ok=True)
            seeds = sorted(glob.glob(os.path.join(sd, "*.jkr")),
                           key=os.path.getmtime)
            while len(seeds) >= self.config.seed_capacity:
                try:
                    os.remove(seeds.pop(0))
                except OSError:
                    break
            path = os.path.abspath(os.path.join(
                sd, f"ante{ante}_x{nx}_e{env.env_id}_{self.global_step}.jkr"))
            await env.game.execute_action("save", {"path": path})
            print(f"[CURRICULUM] harvest OK ante{ante} x{nx} (env {env.env_id})",
                  flush=True)
        except Exception as e:
            print(f"[CURRICULUM] harvest failed: {e}", flush=True)

    def _reset_run_state(self, env):
        """Clear all per-run flags and pending state.

        Must run on every path that abandons or finishes a run (GAME_OVER
        and every crash-recovery restart). A stale _win_recorded surviving
        a post-win wedge restart made the NEXT run's loss count as a win
        at GAME_OVER (won = ante > 8 or already_recorded) — logged, stat-
        counted, and video-recorded as a win. Stale _pending_* could fire
        actions aimed at the previous run's shop.
        """
        # Capture the just-finished run's trajectory BEFORE clearing the win/
        # ante flags this decision depends on.
        self._maybe_capture_demo(env)

        env.win_recorded = False
        env.win_reward_stored = False
        env.last_logged_ante = 0   # build-progression: re-log antes next run
        env.from_curriculum = False  # curriculum: re-decided at next run-start
        env.pending_upgrade_buy = None
        env.pending_rearrange = None
        env.pending_hand_rearrange = None
        env.pending_hand_rearrange_fallback = None
        env.last_api_method = None
        env.last_action_succeeded = True
        env.prev_actionable_state = None
        env.shop_rerolls = 0
        env.shop_noop_count = 0
        env.prev_shop_fingerprint = None
        env.state_stuck_count = 0
        env.prev_state_fingerprint = None
        env.menu_loop_count = 0
        env.current_ante = 1
        env.current_score = 0

    async def _restart_balatro(self, env):
        """Kill Balatro and relaunch it after a crash.

        Kills any running Balatro.exe process, waits briefly, then
        relaunches via start_balatro.bat. Reconnects the API session.
        """
        print("[CRASH-RECOVERY] Balatro appears stuck — restarting...", flush=True)

        # Save recording before killing everything — crash runs at ante 7+
        # are still worth keeping for review
        try:
            ante = env.current_ante
            score = env.current_score
            won = env.win_recorded
            env.recorder.end_run(
                won=won,
                ante_reached=ante,
                final_score=score,
            )
        except Exception:
            pass  # never block restart for recording

        # Close the crashed episode in the tracker (unless the win-fallback
        # already ended it) so its reward/length/ante don't bleed into the
        # next run's stats. Guard on length so a menu-time restart doesn't
        # count a phantom zero-length episode.
        try:
            if (not env.win_recorded
                    and self.episode_tracker.episode_length(env.env_id) > 0):
                self.episode_tracker.end_episode(False, env_id=env.env_id)
        except Exception:
            pass

        # Recording saved and episode closed — now clear ALL per-run state
        # so nothing leaks into the next run.
        self._reset_run_state(env)

        # Kill THIS env's game only — by the PID that owns its port, plus
        # this env's tracked wrapper. A blanket taskkill /IM would murder
        # every parallel instance's game at once.
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-NetTCPConnection -LocalPort {env.port} "
                 f"-State Listen -ErrorAction SilentlyContinue)"
                 f".OwningProcess | Select-Object -First 1"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            if out:
                subprocess.run(["taskkill", "/F", "/PID", out, "/T"],
                               capture_output=True, timeout=10)
                print(f"[CRASH-RECOVERY] killed port {env.port} owner "
                      f"PID {out}", flush=True)
        except Exception as e:
            print(f"[CRASH-RECOVERY] port-kill failed: {e}", flush=True)
        if env.balatro_process is not None:
            try:
                env.balatro_process.kill()
            except Exception:
                pass

        await asyncio.sleep(3.0)  # Wait for process to fully die

        # Relaunch directly (NOT via start_balatro.bat — the bat has no
        # port awareness) with this env's port.
        try:
            relaunch_env = dict(
                os.environ,
                BALATROBOT_PORT=str(env.port),
                BALATROBOT_GAMESPEED="8",
                BALATROBOT_ANIMATION_FPS="120",
            )
            env.balatro_process = subprocess.Popen(
                [r"C:\Users\jarms\.local\bin\uvx.exe",
                 "balatrobot", "serve", "--fast"],
                env=relaunch_env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[CRASH-RECOVERY] Relaunched Balatro (PID {env.balatro_process.pid})", flush=True)
        except Exception as e:
            print(f"[CRASH-RECOVERY] Failed to relaunch: {e}", flush=True)
            print("[CRASH-RECOVERY] Please restart Balatro manually", flush=True)
            return

        # Wait for the API to come back up
        print("[CRASH-RECOVERY] Waiting for API to come back...", flush=True)
        for attempt in range(60):  # Up to 60 seconds
            await asyncio.sleep(1.0)
            try:
                # Reconnect the HTTP session
                await env.game.disconnect()
                await env.game.connect()
                raw = await env.game.fetch_gamestate()
                state = raw.get("state", "")
                print(f"[CRASH-RECOVERY] API responded — state={state}", flush=True)

                # If we're at menu, start a new run (and record it — this
                # path bypasses the MENU branch in _get_actionable_state,
                # so without start_run() a win in this run had no video)
                if state == "MENU":
                    env.recorder.start_run()
                    seed = ''.join(random.choices(string.ascii_uppercase, k=8))
                    await env.game.execute_action(
                        "start", {"deck": "RED", "stake": "WHITE", "seed": seed}
                    )
                    await asyncio.sleep(0.5)

                env.consecutive_api_failures = 0
                print("[CRASH-RECOVERY] Recovery complete!", flush=True)
                return
            except Exception:
                if attempt % 10 == 9:
                    print(f"[CRASH-RECOVERY] Still waiting... ({attempt+1}s)", flush=True)

        print("[CRASH-RECOVERY] Timed out waiting for API. Please restart manually.", flush=True)

    async def run(self):
        """Main training loop."""
        cfg = self.config
        self.start_time = time.time()

        print("=" * 60)
        print("BALATRON TRAINING")
        print(f"  Phase: {cfg.phase}")
        print(f"  Total timesteps: {cfg.total_timesteps:,}")
        print(f"  Rollout steps: {cfg.rollout_steps}")
        print(f"  Device: {cfg.device}")
        print(f"  Network params: {sum(p.numel() for p in self.network.parameters()):,}")
        print("=" * 60)

        # Connect to all BalatroBot instances
        for env in self.sessions:
            await env.game.connect()
        print(f"Connected to {len(self.sessions)} BalatroBot instance(s) "
              f"(ports {', '.join(str(e.port) for e in self.sessions)})")

        try:
            while self.global_step < cfg.total_timesteps:
                # Collect one rollout across all envs in parallel. Each
                # env task plays until the COMBINED buffer total reaches
                # rollout_steps (natural load balancing — faster envs
                # contribute more).
                results = await asyncio.gather(
                    *[self._collect_rollout(env) for env in self.sessions]
                )
                last_values = [r[0] for r in results]
                last_dones = [r[1] for r in results]

                # PPO update over the combined per-env buffers
                metrics = self.ppo.update(last_values, last_dones)
                self.num_updates += 1

                # LR annealing — FLOORED. At ~75% of total_timesteps the
                # un-floored LR had decayed to 7.5e-5 and the policy stopped
                # moving (KL ~0.002, plateaued). Keep a usable minimum so
                # learning never dies just because of the schedule.
                if cfg.anneal_lr:
                    frac = 1.0 - self.global_step / cfg.total_timesteps
                    new_lr = max(cfg.learning_rate * frac, 1.0e-4)
                    self.ppo.set_learning_rate(new_lr)

                # Logging
                if self.num_updates % cfg.log_interval == 0:
                    self._log_update(metrics)

                # Checkpoint
                if self.num_updates % cfg.checkpoint_interval == 0:
                    self._save_checkpoint()

                # Check recording file size every 5 updates (~10k steps)
                if self.num_updates % 5 == 0:
                    self.recorder.check_file_size()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise
        finally:
            # Final checkpoint
            self._save_checkpoint(tag="final")
            # Persist the demo buffer so captures below the save-every
            # threshold aren't lost on this (frequent) recycle. Atomic save.
            if self.demo_buffer is not None:
                self.demo_buffer.save()
            self.recorder.cleanup()
            for env in self.sessions:
                await env.game.disconnect()
            print("Disconnected from BalatroBot")

        self._print_summary()

    async def run_eval(self):
        """Held-out evaluation (dec-045). Play each env's fixed forced_seeds bank
        with the loaded (frozen) checkpoint and NO learning, writing per-run
        outcomes to game_history.jsonl tagged by seed (analyze with eval_report.py).

        Reuses the EXACT training play path (_collect_rollout) so eval behavior
        matches real play; only the PPO update is skipped and the buffer discarded.
        The game server(s) on the session ports must be up and NOT in use by a
        training trainer (pause training first)."""
        self.eval_mode = True
        total = sum(len(getattr(e, "forced_seeds", []) or []) for e in self.sessions)
        print(f"[EVAL] {total} seeds across {len(self.sessions)} env(s) "
              f"on ports {[e.port for e in self.sessions]}", flush=True)
        for env in self.sessions:
            await env.game.connect()
        try:
            while not all(getattr(e, "eval_finished", False) for e in self.sessions):
                active = [e for e in self.sessions
                          if not getattr(e, "eval_finished", False)]
                await asyncio.gather(*[self._collect_rollout(e) for e in active])
                for buf in self.ppo.buffers:      # discard — eval never learns
                    buf.reset()
        finally:
            for env in self.sessions:
                await env.game.disconnect()
        print(f"[EVAL] done — {total} seeds played; outcomes in "
              f"logs/game_history.jsonl (seed-tagged). Run: "
              f"python eval_report.py logs/game_history.jsonl", flush=True)

    async def _collect_rollout(self, env) -> tuple[float, bool]:
        """Collect transitions for ONE env until the COMBINED total
        across all envs reaches rollout_steps.

        Returns:
            (last_value, last_done) for this env's GAE bootstrap
        """
        cfg = self.config
        prev_raw = None
        last_done = False

        while self.ppo.total_collected() < cfg.rollout_steps:

            # dec-045: in eval, stop this env once its fixed-seed bank is played
            # out (set in the run-start seam when forced_seeds empties).
            if self.eval_mode and getattr(env, "eval_finished", False):
                break

            # Get current game state
            raw_state = await self._get_actionable_state(env)

            if raw_state is None:
                # Couldn't get an actionable state — treat as terminal
                self.ppo.store_transition(
                    np.zeros(STATE_VECTOR_SIZE, dtype=np.float32),
                    np.zeros(14, dtype=np.float32),
                    0.0, 0.0, 0.0, True,
                    np.zeros(ACTION_HEAD_SIZE, dtype=np.float32),
                    "GAME_OVER", env_id=env.env_id,
                )
                self.global_step += 1
                last_done = True
                continue

            game_state_name = raw_state.get("state", "")

            # Detect round completion: SELECTING_HAND -> SHOP means blind was beaten
            prev_state_name = prev_raw.get("state", "") if prev_raw else ""
            if prev_state_name == "SELECTING_HAND" and game_state_name == "SHOP":
                env.joker_logger.round_end()
                self._log_blind_result(env, beaten=True)  # dec-049

            # Detect win via API 'won' flag (endless mode auto-continues)
            # The Lua mod auto-dismisses the win screen, so GAME_OVER may
            # never fire. Record the win when we first see won=True.
            #
            # IMPORTANT: Only trust 'won' in POST-BOSS states (SHOP,
            # BLIND_SELECT, ROUND_EVAL). The base game sets G.GAME.won=true the
            # moment you reach the ante-8 boss — win OR lose (state_events.lua
            # end_round) — so it is true during SELECTING_HAND while playing the
            # boss and at GAME_OVER after dying on it. A LOSS goes straight to
            # GAME_OVER, never to a post-boss SHOP/ROUND_EVAL, so seeing won in
            # those states is the only reliable proof the boss was beaten.
            api_won = raw_state.get("won", False)
            safe_won_states = {"SHOP", "BLIND_SELECT", "ROUND_EVAL"}
            if (api_won and not env.win_recorded
                    and game_state_name in safe_won_states):
                env.win_recorded = True
                ante = raw_state.get("ante_num", 1)
                print(f"[WIN] Won flag detected in state={game_state_name} "
                      f"ante={ante}", flush=True)
                self.episode_tracker.end_episode(True, raw_state, env_id=env.env_id)

                # Fallback: the Lua mod auto-dismisses the win screen, so
                # GAME_OVER may never fire and the win reward would never reach
                # the PPO buffer. Push a terminal transition carrying the win
                # reward here, attached to the last real state/action. Guarded
                # by _win_reward_stored so GAME_OVER won't double-count it.
                if not env.win_reward_stored:
                    win_reward = env.reward_calc.terminal_win_reward(raw_state)
                    # Credit the win to the real decision that won the run by
                    # amending the last stored transition in place (set
                    # done=True, add the reward). Appending a fresh done=True
                    # transition would make GAE train only the value head on a
                    # phantom state, never the policy on the winning action.
                    amended = self.ppo.amend_last_transition(
                        reward_delta=win_reward, done=True,
                        env_id=env.env_id,
                    )
                    if not amended:
                        # No real transition yet this rollout — fall back to a
                        # placeholder so the win reward isn't lost.
                        self.ppo.store_transition(
                            np.zeros(STATE_VECTOR_SIZE, dtype=np.float32),
                            np.zeros(14, dtype=np.float32),
                            0.0, win_reward, 0.0, True,
                            np.zeros(ACTION_HEAD_SIZE, dtype=np.float32),
                            game_state_name, env_id=env.env_id,
                        )
                        self.global_step += 1
                    env.win_reward_stored = True
                    # Consume the last real transition so the GAME_OVER block
                    # below can't credit a second terminal from the same one.
                    env.last_transition = None
                    print(f"[WIN] Credited terminal win reward "
                          f"{win_reward:+.2f} to last transition", flush=True)
                # DON'T end recording here — let GAME_OVER handle it so the
                # FULL run is captured: the winning hand, the endless-mode
                # continuation past ante 8, and the eventual end. (Window
                # targeting — filming env 0's own window — is fixed separately
                # in recorder.py; that was the real "wrong footage" bug.)

            # Handle GAME_OVER — end episode, start new one
            if game_state_name == "GAME_OVER":
                env.joker_logger.round_end()  # flush any pending round data
                self._log_blind_result(env, beaten=False, raw=raw_state)  # dec-049
                ante = raw_state.get("ante_num", 1)
                api_won_flag = raw_state.get("won", False)
                already_recorded = env.win_recorded
                # Win only if we got PAST the ante-8 boss: ante > 8 (advanced in
                # endless) OR already_recorded by the post-boss win-fallback.
                # NOT "ante >= 8 and api_won": the base game sets won=true on
                # reaching the ante-8 boss, so that clause counted boss LOSSES
                # (e.g. 90,592/100,000 at ante 8) as wins.
                won = ante > 8 or already_recorded
                print(f"[GAME_OVER] ante={ante} won={won} "
                      f"already_recorded={already_recorded} "
                      f"api_won={api_won_flag}", flush=True)

                # Append to game history log (rolling buffer)
                try:
                    import json as _json
                    from datetime import datetime as _dt
                    _hist_path = os.path.join("logs", "game_history.jsonl")
                    _entry = _json.dumps({
                        "ante": ante, "won": won,
                        "ts": _dt.now().isoformat(timespec="seconds"),
                        "episode": getattr(self, 'episode_count', 0),
                        "seed": getattr(env, "current_seed", None),  # dec-045: eval attribution
                        # Tag curriculum-loaded runs (started from a banked
                        # ante-4/5 seed) so the dashboard can show FRESH-only
                        # outcomes — loaded runs start deep and inflate the
                        # ante/reach-rate averages (dec-030/031 head-start).
                        "from_curriculum": bool(getattr(env, "from_curriculum", False)),
                        "jokers": [j.get("label", "?") for j in
                                   raw_state.get("jokers", {}).get("cards", [])],
                    })
                    with open(_hist_path, "a") as _hf:
                        _hf.write(_entry + "\n")
                    # Rotate: keep last 5000 entries
                    _MAX_HIST = 5000
                    with open(_hist_path, "r") as _hf:
                        _lines = _hf.readlines()
                    if len(_lines) > _MAX_HIST:
                        with open(_hist_path, "w") as _hf:
                            _hf.writelines(_lines[-_MAX_HIST:])
                except Exception:
                    pass  # never crash training for logging

                # Compute terminal reward (kwargs describe the action that
                # led into GAME_OVER — the one this reward will amend)
                reward = env.reward_calc.step(
                    prev_raw, raw_state,
                    action=env.last_api_method,
                    action_succeeded=env.last_action_succeeded,
                )
                # If the win reward was already pushed to the buffer via the
                # won-flag fallback above, don't count it again here. Capture
                # the flag before reset() clears it below so we can also skip
                # the duplicate terminal store.
                win_already_stored = env.win_reward_stored
                if win_already_stored:
                    reward = 0.0
                self.episode_tracker.step(reward, ante, raw_state, env_id=env.env_id)
                # Only call end_episode if win wasn't already recorded
                if not env.win_recorded:
                    self.episode_tracker.end_episode(won, raw_state, env_id=env.env_id)
                # End recording — save if won, discard if lost
                # Wait a moment on wins so the win screen / final scoring
                # animation gets captured before ffmpeg stops
                if won:
                    await asyncio.sleep(3.0)
                env.recorder.end_run(
                    won=won,
                    ante_reached=ante,
                    final_score=int(raw_state.get("round", {}).get("chips", 0)),
                    checkpoint_path=os.path.join(
                        self.config.checkpoint_dir,
                        f"balatron_phase{self.config.phase}_step{self.global_step}.pt",
                    ),
                    total_steps=self.global_step,
                )

                env.reward_calc.reset()
                env.game.reset()
                self._reset_run_state(env)  # win flags, pending actions, etc.
                prev_raw = None

                # Credit the terminal reward to the real decision that ended
                # the run by amending the last stored transition in place (set
                # done=True, add the reward) rather than appending a phantom
                # done=True step, which GAE would only use to train the value
                # head. Skip entirely if the won-flag fallback already amended
                # the last transition — otherwise we'd double-count the end.
                if not win_already_stored:
                    amended = self.ppo.amend_last_transition(
                        reward_delta=reward, done=True,
                        env_id=env.env_id,
                    )
                    if not amended:
                        # Buffer empty (terminal on the first step of a fresh
                        # rollout) — no real decision to credit, so fall back
                        # to a placeholder terminal transition.
                        self.ppo.store_transition(
                            np.zeros(STATE_VECTOR_SIZE, dtype=np.float32),
                            np.zeros(14, dtype=np.float32),
                            0.0, reward, 0.0, True,
                            np.zeros(ACTION_HEAD_SIZE, dtype=np.float32),
                            "GAME_OVER", env_id=env.env_id,
                        )
                        self.global_step += 1
                env.last_transition = None
                last_done = True

                # Navigate back to menu so next poll doesn't see GAME_OVER again
                try:
                    await env.game.execute_action("menu")
                except Exception:
                    pass

                continue

            # Get state vector from game manager
            try:
                state_vec = await env.game.step()
            except Exception:
                # API dropped (e.g. menu escape killed server) — wait and retry.
                # Count consecutive silent skips so a persistent failure trips
                # recovery instead of looping forever before the stuck-detector.
                env.silent_skip_count = env.silent_skip_count + 1
                if env.silent_skip_count >= 40:
                    print(f"[STUCK] {env.silent_skip_count} consecutive "
                          f"state-encode failures — restarting Balatro", flush=True)
                    env.silent_skip_count = 0
                    # Close the crashed episode in the buffer — otherwise
                    # GAE bootstraps the new run's values into the dead run.
                    self.ppo.amend_last_transition(done=True, env_id=env.env_id)
                    last_done = True
                    await self._restart_balatro(env)
                    env.reward_calc.reset()
                    env.game.reset()
                    prev_raw = None
                    continue
                await asyncio.sleep(0.3)
                continue

            # Build action mask
            action_mask = build_action_mask(raw_state)

            # Expose reroll cap to mask — prevent NN from selecting reroll
            # when the hard guard will block it anyway
            if game_state_name == "SHOP":
                from environment.action_space import ACTION_REROLL
                rerolls = env.shop_rerolls
                money_now = raw_state.get("money", 0)
                reroll_cost = raw_state.get("round", {}).get("reroll_cost", 5)
                min_joker_cost = 4
                money_after = money_now - reroll_cost

                # Hard block: can't afford reroll or can't buy anything after
                if money_now < reroll_cost or money_after < min_joker_cost:
                    action_mask[ACTION_REROLL] = 0.0
                else:
                    # Cap based on surplus above interest floor
                    interest_floor = 25
                    surplus = max(money_now - interest_floor, 0)
                    affordable_rerolls = surplus // max(reroll_cost, 1)
                    reroll_cap = max(1, min(affordable_rerolls, 8))
                    if rerolls >= reroll_cap:
                        action_mask[ACTION_REROLL] = 0.0

            # Check if any actions are valid
            if action_mask[:14].sum() == 0:
                # No valid actions — skip this step. Count consecutive skips so
                # a wedged state the agent can't act in (e.g. a shop with an
                # all-zero mask) trips recovery instead of looping forever past
                # the spin/stuck detectors below.
                env.silent_skip_count = env.silent_skip_count + 1
                if env.silent_skip_count >= 40:
                    print(f"[STUCK] {env.silent_skip_count} steps with no valid "
                          f"action in {game_state_name} — restarting Balatro",
                          flush=True)
                    env.silent_skip_count = 0
                    # Close the crashed episode in the buffer (see above)
                    self.ppo.amend_last_transition(done=True, env_id=env.env_id)
                    last_done = True
                    await self._restart_balatro(env)
                    env.reward_calc.reset()
                    env.game.reset()
                    prev_raw = None
                    continue
                prev_raw = raw_state
                continue

            # Valid action available — clear the silent-skip watchdog.
            env.silent_skip_count = 0

            # Get head index
            head_idx = get_head_index(game_state_name)

            # Network forward pass
            with torch.no_grad():
                state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

                if cfg.device == "cuda" and torch.cuda.is_available():
                    state_t = state_t.cuda()
                    mask_t = mask_t.cuda()

                action_t, log_prob_t, _, value_t, _ = self.network.get_action_and_value(
                    state_t, head_idx, mask_t
                )

            action_np = action_t[0].cpu().numpy()
            log_prob = log_prob_t[0].cpu().item()
            value = value_t[0].cpu().item()

            # Decode sampled action tensor to API call
            api_method, api_params = self.action_executor._action_to_api_call(
                env, action_np, raw_state
            )

            # Shop spin detection: track ANY repeated shop state, not just
            # gamestate calls — failed buy attempts also cause loops
            if game_state_name == "SHOP":
                _shop_fp = (
                    raw_state.get("money", -1),
                    str(raw_state.get("shop", {}).get("cards", [])),
                    str(raw_state.get("packs", {}).get("cards", [])),
                    len(raw_state.get("jokers", {}).get("cards", [])),
                )
                _prev_shop_fp = env.prev_shop_fingerprint
                if _shop_fp == _prev_shop_fp:
                    env.shop_noop_count = env.shop_noop_count + 1
                else:
                    env.shop_noop_count = 0
                env.prev_shop_fingerprint = _shop_fp

                noop_count = env.shop_noop_count
                if noop_count > 0 and noop_count <= 5:
                    noop_action_type = int(action_np[0])
                    noop_target = int(action_np[13])
                    print(f"[SHOP-SPIN] #{noop_count} action={noop_action_type} "
                          f"target={noop_target} api={api_method} "
                          f"params={api_params}", flush=True)

                if noop_count >= 3:
                    from environment.action_space import (
                        _is_joker_card as _noop_ij,
                    )
                    noop_money = raw_state.get("money", 0)
                    noop_jokers = raw_state.get("jokers", {}).get("cards", [])
                    noop_jcount = len(noop_jokers)
                    noop_jlimit = raw_state.get("jokers", {}).get("limit", 5)
                    noop_cons = raw_state.get("consumables", {}).get("cards", [])
                    noop_climit = raw_state.get("consumables", {}).get("limit", 2)

                    # Priority 1: buy affordable packs (spectral > arcana > celestial > other)
                    noop_packs = raw_state.get("packs", {}).get("cards", [])
                    best_pack = -1
                    best_pack_priority = -1
                    for pi, pc in enumerate(noop_packs):
                        pc_cost = pc.get("cost", {}).get("buy", 999)
                        if pc_cost > noop_money:
                            continue
                        pk = pc.get("key", "")
                        if "spectral" in pk:
                            prio = 4
                        elif "arcana" in pk:
                            prio = 3
                        elif "celestial" in pk:
                            prio = 2
                        elif "buffoon" in pk and noop_jcount < noop_jlimit:
                            prio = 1
                        else:
                            prio = 0
                        if prio > best_pack_priority:
                            best_pack = pi
                            best_pack_priority = prio
                    if best_pack >= 0 and best_pack_priority > 0:
                        pk_name = noop_packs[best_pack].get("key", "?")
                        print(f"[SHOP] SPIN recovery: buying pack {pk_name} "
                              f"(slot {best_pack})", flush=True)
                        api_method = "buy"
                        api_params = {"pack": best_pack}
                        env.shop_noop_count = 0
                    else:
                        # Priority 2: buy affordable shop card (joker/consumable)
                        noop_shop = raw_state.get("shop", {}).get("cards", [])
                        best_noop_buy = -1
                        for ni, nc in enumerate(noop_shop):
                            nc_cost = nc.get("cost", {}).get("buy", 999)
                            if nc_cost > noop_money:
                                continue
                            if _noop_ij(nc):
                                if noop_jcount < noop_jlimit and best_noop_buy < 0:
                                    best_noop_buy = ni
                            elif nc.get("set", "").upper() in ("PLANET", "TAROT"):
                                if len(noop_cons) < noop_climit and best_noop_buy < 0:
                                    best_noop_buy = ni
                        if best_noop_buy >= 0:
                            bk = noop_shop[best_noop_buy].get("key", "?")
                            print(f"[SHOP] SPIN recovery: buying card {bk} "
                                  f"(slot {best_noop_buy})", flush=True)
                            api_method = "buy"
                            api_params = {"card": best_noop_buy}
                            env.shop_noop_count = 0
                        elif noop_count >= 8:
                            print(f"[SHOP] SPIN recovery: forcing next_round "
                                  f"after {noop_count} spins", flush=True)
                            api_method = "next_round"
                            api_params = None
                            env.shop_noop_count = 0
            else:
                env.shop_noop_count = 0
                env.prev_shop_fingerprint = None

            # Stuck-state detector: if we keep getting the same state,
            # escalate from force-play → restart
            state_fingerprint = (
                game_state_name,
                str(raw_state.get("hand", {}).get("cards", [])),
                raw_state.get("round", {}).get("hands_left", -1),
                raw_state.get("round", {}).get("discards_left", -1),
                raw_state.get("money", -1),
                str(raw_state.get("shop", {}).get("cards", [])) if game_state_name == "SHOP" else "",
            )
            prev_fingerprint = env.prev_state_fingerprint
            if state_fingerprint == prev_fingerprint:
                env.state_stuck_count = env.state_stuck_count + 1
                stuck = env.state_stuck_count

                if stuck == 3 and game_state_name == "SELECTING_HAND":
                    # First escalation: force-play best hand
                    print(f"[STUCK] State unchanged {stuck}x — "
                          f"force-playing best hand", flush=True)
                    hand_cards = raw_state.get("hand", {}).get("cards", [])
                    jokers_raw = raw_state.get("jokers", {}).get("cards", [])
                    try:
                        best = find_best_hands(hand_cards, jokers_raw, raw_state, top_n=1)
                        forced_cards = list(best[0]["card_indices"])[:5] if best else [0]
                    except Exception:
                        forced_cards = [0] if hand_cards else []
                    api_method = "play"
                    api_params = {"cards": forced_cards}
                    env.pending_hand_rearrange = None
                    env.pending_rearrange = None

                elif stuck >= 5 and game_state_name == "SHOP":
                    # Shop loop — force-leave via next_round BEFORE the generic
                    # stuck>=8 restart below (which would otherwise preempt this
                    # branch and hard-restart instead of just leaving the shop).
                    # The mod's next_round is reliable, so this escapes cheaply.
                    print(f"[STUCK] Shop unchanged {stuck}x — "
                          f"forcing next_round", flush=True)
                    api_method = "next_round"
                    api_params = None
                    env.state_stuck_count = 0

                elif stuck >= 8:
                    # Buttons are permanently broken — restart Balatro
                    print(f"[STUCK] State unchanged {stuck}x — "
                          f"game desynced, restarting Balatro", flush=True)
                    # INSTRUMENTATION: dump the frozen state + the action that
                    # isn't landing, so we can see what desynced (esp. post-win).
                    import json as _json
                    print(f"[STATE-DUMP] desync state={game_state_name} "
                          f"won={raw_state.get('won')} ante={raw_state.get('ante_num')} "
                          f"action={api_method} params={api_params} "
                          f"keys={sorted(raw_state.keys())} "
                          f"raw={_json.dumps(raw_state, default=str)[:1500]}", flush=True)
                    env.state_stuck_count = 0
                    env.prev_state_fingerprint = None
                    # Close the crashed episode in the buffer, and drop the
                    # stale prev_raw — otherwise the next settle computes a
                    # junk delta between the dead run and the fresh one.
                    self.ppo.amend_last_transition(done=True, env_id=env.env_id)
                    last_done = True
                    await self._restart_balatro(env)
                    env.reward_calc.reset()
                    env.game.reset()
                    prev_raw = None
                    continue
            else:
                env.state_stuck_count = 0
            env.prev_state_fingerprint = state_fingerprint

            # Reorder jokers before playing a hand (using actual cards)
            _intended_joker_order = None
            if api_method == "play" and env.pending_rearrange is not None:
                hand_cards, deck_cards = env.pending_rearrange
                env.pending_rearrange = None
                try:
                    _intended_joker_order = await self.action_executor._auto_rearrange_jokers(
                        env, raw_state, hand_cards=hand_cards, deck_cards=deck_cards
                    )
                except Exception as e:
                    err_msg = f"Pre-play: {e}"
                    print(f"[WARN] Joker rearrange failed: {e}")
                    env.joker_logger.log_rearrange_failure("pre-play", str(e))

            # Rearrange hand cards for optimal scoring order (face card first
            # for Photograph, highest chips first for Hanging Chad, etc.)
            if api_method == "play" and env.pending_hand_rearrange is not None:
                new_hand_order = env.pending_hand_rearrange
                fallback_cards = env.pending_hand_rearrange_fallback
                env.pending_hand_rearrange = None
                env.pending_hand_rearrange_fallback = None
                try:
                    await env.game.execute_action("rearrange", {"hand": new_hand_order})
                except Exception as e:
                    print(f"[WARN] Hand rearrange failed: {e}")
                    # Rearrange failed — play cards at original indices instead
                    # of the post-rearrange 0..N-1 indices
                    if fallback_cards and api_params:
                        api_params["cards"] = fallback_cards

            # Log play details for joker order review
            if api_method == "play":
                self.action_executor._log_play_for_joker_order(
                    env, raw_state, _intended_joker_order
                )

            # Debounce state-TRANSITION actions: re-issuing next_round /
            # select / skip while the previous one is still animating
            # double-fires the game's UI flow (the mod bypasses the
            # game's own controller locks) — the second invocation's
            # deferred events race the first's teardown and CRASH the
            # game ('shop'/'blind_select'/'area' nil). If we issued the
            # same transition action within the last 8s and the state
            # name hasn't changed, wait instead of re-firing.
            TRANSITION_ACTIONS = {"next_round", "select", "skip", "cash_out"}
            if api_method in TRANSITION_ACTIONS:
                last_m, last_t, last_s = env.last_transition_fire
                if (api_method == last_m
                        and game_state_name == last_s
                        and time.time() - last_t < 8.0):
                    await asyncio.sleep(1.0)
                    continue
                env.last_transition_fire = (
                    api_method, time.time(), game_state_name)

            # Execute action (with retry on UI timing errors)
            action_succeeded = True
            max_retries = 3
            for _attempt in range(max_retries):
                try:
                    await env.game.execute_action(api_method, api_params)
                    break  # success
                except Exception as exc:
                    exc_str = str(exc)
                    if ("buttons" in exc_str or "INVALID_STATE" in exc_str) and _attempt < max_retries - 1:
                        # UI buttons not ready yet — wait and retry
                        await asyncio.sleep(0.3 * (_attempt + 1))
                        continue
                    action_succeeded = False
                    print(f"[WARN] execute_action({api_method}, {api_params}) "
                          f"FAILED: {exc}", flush=True)
                    break

            # Skip shop rearrange — joker order only matters before plays,
            # and we already rearrange at round start + before each play.
            # This eliminates ~120-perm brute force per buy/sell action.

            # Reconcile sampled vs EXECUTED action. When a heuristic
            # override changed what ran, store the executed action with its
            # log-prob under the current policy so PPO credits the outcome
            # to the real cause (the old behavior trained the shop head on
            # noise: outcomes of redirected buys credited to whatever the
            # net happened to sample).
            was_override = False
            if action_succeeded:
                exec_action = self.action_executor._encode_executed_action(
                    api_method, api_params, action_np)
                if (exec_action is not None
                        and not np.array_equal(exec_action, action_np)):
                    with torch.no_grad():
                        ea_t = torch.tensor(
                            exec_action, dtype=torch.float32).unsqueeze(0)
                        if cfg.device == "cuda" and torch.cuda.is_available():
                            ea_t = ea_t.cuda()
                        _, exec_lp_t, _, _, _ = self.network.get_action_and_value(
                            state_t, head_idx, mask_t, action=ea_t)
                    exec_lp = exec_lp_t[0].cpu().item()
                    # Guard: an executed action that is ~impossible under
                    # the masked policy (e.g. a force-buy of a masked
                    # target) would store log_prob -> -inf and blow up the
                    # PPO ratio. Keep the sampled action in that case.
                    if np.isfinite(exec_lp) and exec_lp > -30.0:
                        action_np = exec_action
                        log_prob = exec_lp
                        # Teacher correction — flag for the BC kickstart
                        # loss (imitates only overridden steps).
                        was_override = True

            # Settle the PREVIOUS action's outcome. The delta prev_raw ->
            # raw_state is what the last stored decision caused — it only
            # became observable on this fetch. It must be credited to that
            # transition (amend in place), not stored with the current
            # action: storing it here put every step reward one decision
            # late, often on a different policy head (e.g. the blind-clear
            # bonus landed on the first SHOP action instead of the winning
            # play). The action kwargs are likewise the PREVIOUS action's —
            # the sell-penalty and invalid-action penalty describe the
            # action that produced this delta.
            _cached_contribs = getattr(env.game, '_joker_eval_cache', {}).get('contributions')
            settled_reward = env.reward_calc.step(
                prev_raw, raw_state,
                action=env.last_api_method,
                action_succeeded=env.last_action_succeeded,
                scaling_values=self._get_scaling_snapshot(env),
                joker_contributions=_cached_contribs,
                skip_economy=env.auto_action_this_step,
            )
            env.auto_action_this_step = False
            env.last_api_method = api_method
            env.last_action_succeeded = action_succeeded

            store_reward = 0.0
            if settled_reward != 0.0:
                if not self.ppo.amend_last_transition(reward_delta=settled_reward, env_id=env.env_id):
                    # Rollout boundary: the causing transition was consumed
                    # with the previous buffer — keep the reward on this
                    # step rather than dropping it.
                    store_reward = settled_reward

            # Track episode
            ante = raw_state.get("ante_num", 1)
            env.current_ante = ante
            env.current_score = int(raw_state.get("round", {}).get("chips", 0))
            self.episode_tracker.step(settled_reward, ante, raw_state, env_id=env.env_id)

            # Build-progression leading indicator: once per ante boundary, log
            # the xmult-engine composition of THIS run. Each record where
            # ante==N is one run reaching ante N, so offline we compute
            # "fraction with >=2 xmult by ante 3" — the upstream metric that
            # moves in a few updates vs ~150 for mean ante. Logging only.
            if ante > env.last_logged_ante:
                env.last_logged_ante = ante
                try:
                    jcards = raw_state.get("jokers", {}).get("cards", [])
                    nx, ns, nj = _build_composition(jcards)
                    record = {
                        "ante": ante, "n_xmult": nx, "n_scaling": ns,
                        "n_jokers": nj, "env": env.env_id,
                        "step": self.global_step,
                    }
                    # dec-037 depth-death instrumentation: log the FULL picture at
                    # each ante boundary so deep deaths (economy vs LEVELING vs
                    # composition) are diagnosable and the evaluator is validatable.
                    # money / committed-hand base chips+mult (encodes planet level)
                    # / board power vs boss target / chips-mult-xmult decomposition.
                    try:
                        from environment.planner import (
                            target_hand_type, ante_target, HANDS_PER_BLIND,
                        )
                        from environment.hand_eval import (
                            _estimate_joker_scoring_for_type, BASE_HAND_SCORES,
                            estimate_score_for_hand_type,
                        )
                        ht = target_hand_type(jcards, raw_state)
                        hinfo = raw_state.get("hands", {}).get(ht, {})
                        bc, bm = BASE_HAND_SCORES.get(ht, (5, 1))
                        jc, jm, jx = _estimate_joker_scoring_for_type(ht, jcards, raw_state)
                        # NOTE: logged power/margin are RAW (no REALIZATION_FACTOR) on
                        # purpose — this is the unbiased yardstick the calibration is
                        # validated against (dec-038). Do NOT apply the discount here.
                        power = estimate_score_for_hand_type(jcards, raw_state) * HANDS_PER_BLIND
                        tgt = ante_target(ante, "boss")
                        record.update({
                            "money": raw_state.get("money", 0),
                            "ht": ht,
                            "base_chips": round(float(hinfo.get("chips", bc)), 1),
                            "base_mult": round(float(hinfo.get("mult", bm)), 1),
                            "j_chips": round(float(jc), 1),
                            "j_mult": round(float(jm), 1),
                            "xmult": round(float(jx), 2),
                            "power": round(float(power), 0),
                            "target": round(float(tgt), 0),
                            "margin": round(float(power) / max(float(tgt), 1.0), 3),
                        })
                        env.last_proj_power = float(power)  # dec-049: for realized-vs-projected
                    except Exception:
                        pass  # diagnostics are best-effort; never block the log
                    with open(os.path.join("logs", "build_progression.jsonl"),
                              "a") as _bf:
                        _bf.write(json.dumps(record) + "\n")
                    # Curriculum harvest: bank ante-4/5 partial-build states
                    # (with an xmult engine started) from FRESH runs as seeds.
                    if (self.config.curriculum_enabled and ante in (4, 5)
                            and nx >= 1 and not env.from_curriculum):
                        await self._harvest_seed(env, ante, nx)
                except Exception:
                    pass  # never let instrumentation break training

            # Store transition with reward 0 — its own outcome is settled
            # (amended in) on the next iteration, or by the boundary settle
            # after the loop, or by the terminal paths.
            self.ppo.store_transition(
                state_vec, action_np, log_prob, store_reward, value,
                False, action_mask, game_state_name,
                bc_flag=was_override, env_id=env.env_id,
            )
            # Self-imitation capture (Phase 1): accumulate this REAL step into
            # the episode trajectory; _reset_run_state flushes it to the demo
            # buffer iff the run wins / reaches a high ante. Pure logging.
            if self.demo_buffer is not None:
                env.episode_traj.append((
                    state_vec.copy(), action_np.copy(), action_mask.copy(),
                    get_head_index(game_state_name),
                ))
                if ante > env.max_ante_seen:
                    env.max_ante_seen = ante
            # Liveness heartbeat: touch on every REAL step so the
            # supervisor can distinguish a frozen trainer from a working
            # one. Wedges keep finding new shapes (post-win start hang,
            # boot-splash freeze with a live socket) — progress is the
            # only signal that covers all of them.
            self._touch_heartbeat()
            # Remember the last real (non-terminal) transition so terminal
            # rewards can be attached to the actual last state/action rather
            # than a zeroed placeholder.
            env.last_transition = (
                state_vec, action_np, log_prob, value,
                action_mask, game_state_name,
            )

            prev_raw = raw_state
            self.global_step += 1
            last_done = False

        # Bootstrap value for GAE
        last_value = 0.0
        if not last_done:
            try:
                raw_state = await env.game.fetch_gamestate()

                # Settle the final transition's outcome before the buffer is
                # consumed — without this, the last decision of every rollout
                # would never receive its step reward (the settle normally
                # happens on the NEXT iteration, which won't come).
                if prev_raw is not None:
                    boundary_terminal = raw_state.get("state", "") == "GAME_OVER"
                    final_reward = env.reward_calc.step(
                        prev_raw, raw_state,
                        action=env.last_api_method,
                        action_succeeded=env.last_action_succeeded,
                        scaling_values=self._get_scaling_snapshot(env),
                    )
                    if final_reward != 0.0 or boundary_terminal:
                        self.ppo.amend_last_transition(
                            reward_delta=final_reward,
                            done=True if boundary_terminal else None,
                            env_id=env.env_id,
                        )
                    if boundary_terminal:
                        # Don't bootstrap V(GAME_OVER) into the dead episode.
                        last_done = True

                if not last_done:
                    state_vec = await env.game.step()
                    head_idx = get_head_index(raw_state.get("state", ""))
                    with torch.no_grad():
                        state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                        if cfg.device == "cuda" and torch.cuda.is_available():
                            state_t = state_t.cuda()
                        _, value_t = self.network.forward(state_t, head_idx)
                        last_value = value_t[0].cpu().item()
            except Exception:
                last_value = 0.0

        return last_value, last_done

    async def _get_actionable_state(self, env) -> Optional[dict]:
        """Poll until we get a game state where the agent can act.

        Handles non-decision transitions automatically:
        - ROUND_EVAL → call cash_out → SHOP
        - MENU → call start → BLIND_SELECT
        - Transient states → poll until resolved
        """
        cfg = self.config
        pack_attempts = 0  # prevent infinite pack retry loops
        unknown_state_count = 0  # detect stuck on crash/error screen

        consecutive_fetch_fails = 0
        for _ in range(cfg.max_poll_attempts):
            try:
                raw = await env.game.fetch_gamestate()
                consecutive_fetch_fails = 0
            except Exception:
                consecutive_fetch_fails += 1
                # With the 8s fetch timeout, 6 consecutive failures ≈ 48s — a
                # hung/crashed game recovers in ~1 min instead of ~10 min.
                if consecutive_fetch_fails >= 6:
                    print(f"[CRASH-RECOVERY] {consecutive_fetch_fails} consecutive fetch failures — triggering restart", flush=True)
                    await self._restart_balatro(env)
                    env.reward_calc.reset()
                    env.game.reset()
                    return None
                await asyncio.sleep(cfg.api_poll_delay)
                continue

            state = raw.get("state", "")

            # Actionable states — the agent makes decisions here
            if state in ("SELECTING_HAND", "SHOP", "GAME_OVER",
                          "BLIND_SELECT"):
                env.consecutive_api_failures = 0  # API is alive
                pack_attempts = 0  # reset pack retry counter
                unknown_state_count = 0  # reset stuck counter
                env.round_eval_count = 0  # reset win-screen detector
                env.menu_loop_count = 0  # a run is live — menu loop over
                if state == "SHOP":
                    # Reset the reroll budget only on ENTRY to a shop —
                    # resetting on every poll made the per-shop reroll cap
                    # dead code (the counter never survived to the guards).
                    if env.prev_actionable_state != "SHOP":
                        env.shop_rerolls = 0
                    # ── RAW STATE DUMP — see exactly what the API returns ──
                    import json as _json
                    _raw_shop = raw.get("shop", {})
                    _raw_packs = raw.get("packs", {})
                    _money = raw.get("money", 0)
                    _jcount = len(raw.get("jokers", {}).get("cards", []))
                    _jlimit = raw.get("jokers", {}).get("limit", 5)
                    print(f"[SHOP-RAW] money=${_money} jokers={_jcount}/{_jlimit}",
                          flush=True)
                    print(f"[SHOP-RAW] shop={_json.dumps(_raw_shop, default=str)[:800]}",
                          flush=True)
                    print(f"[SHOP-RAW] packs={_json.dumps(_raw_packs, default=str)[:400]}",
                          flush=True)
                    # Also log parsed view
                    from environment.action_space import _is_joker_card as _debug_ij
                    shop_cards_debug = _raw_shop.get("cards", []) if isinstance(_raw_shop, dict) else []
                    shop_items = []
                    for sc in shop_cards_debug:
                        sc_set = sc.get("set", "")
                        sc_key = sc.get("key", "")
                        sc_cost = sc.get("cost", {}).get("buy", "?")
                        is_j = _debug_ij(sc)
                        shop_items.append(f"{sc_key}(set={sc_set},joker={is_j})${sc_cost}")
                    print(f"[SHOP-ENTER] shop_cards({len(shop_cards_debug)})=[{', '.join(shop_items)}]",
                          flush=True)
                if state == "BLIND_SELECT":
                    # Auto-skip for Investment Tag (free money)
                    # At BLIND_SELECT, the offered blind has status "SELECT"
                    # and its tag is the skip reward. If we skip, we face the
                    # next UPCOMING blind instead.
                    blinds = raw.get("blinds", {})
                    select_blind = None
                    next_blind_score = 0
                    if isinstance(blinds, dict):
                        for b in blinds.values():
                            if isinstance(b, dict) and b.get("status") == "SELECT":
                                select_blind = b
                        # Find the next blind we'd face after skipping
                        for b in blinds.values():
                            if isinstance(b, dict) and b.get("status") == "UPCOMING":
                                next_blind_score = b.get("score", 0)
                                break  # first upcoming = next in line

                    if select_blind and select_blind.get("tag_name") == "Investment Tag":
                        ante_num = raw.get("ante_num", 1)
                        jokers_raw = raw.get("jokers", {}).get("cards", [])
                        scoring = estimate_score_for_hand_type(jokers_raw, raw) * 4
                        # Investment Tag = $25 when boss blind is beaten.
                        # That's MASSIVE value — almost always worth skipping.
                        # Only don't skip if we can't even beat the next blind.
                        should_skip = False
                        if ante_num <= 4:
                            # Early-mid game: always skip for $25 payout.
                            # Blinds are very beatable through ante 4.
                            should_skip = True
                        elif next_blind_score > 0 and scoring > next_blind_score * 0.8:
                            # Late game: skip even if we're only at 80% of the
                            # next blind's score — the $25 is worth the risk.
                            should_skip = True
                        elif next_blind_score == 0:
                            # Can't determine next blind — skip anyway
                            should_skip = True
                        if should_skip:
                            try:
                                await env.game.execute_action("skip")
                            except Exception:
                                pass
                            await asyncio.sleep(0.2)
                            continue
                if state == "SELECTING_HAND":
                    # Log round start for joker order tracking
                    blinds_data = raw.get("blinds", {})
                    current_blind_name = ""
                    current_blind_score = 0.0
                    env._verdant_leaf_sold = False  # reset per blind (per-env!)
                    if isinstance(blinds_data, dict):
                        for b in blinds_data.values():
                            if isinstance(b, dict) and b.get("status") == "CURRENT":
                                current_blind_name = b.get("name", "")
                                current_blind_score = b.get("score", 0)
                                break
                    # dec-049: track realized per-blind progress (flushed on resolve)
                    _rd = raw.get("round", {})
                    env.cur_blind_name = current_blind_name
                    env.cur_blind_target = float(current_blind_score or 0)
                    env.cur_realized = float(_rd.get("chips", 0) or 0)
                    env.cur_hands_left = int(_rd.get("hands_left", -1) or -1)
                    env.cur_discards_left = int(_rd.get("discards_left", -1) or -1)  # dec-050
                    env.cur_blind_ante = int(raw.get("ante_num", 1) or 1)
                    joker_cards = raw.get("jokers", {}).get("cards", [])
                    joker_keys = [j.get("key", "") for j in joker_cards]
                    env.joker_logger.round_start(
                        ante=raw.get("ante_num", 1),
                        round_num=raw.get("round_num", 1),
                        blind_name=current_blind_name,
                        blind_score=current_blind_score,
                        joker_keys=joker_keys,
                    )

                    # ── BOSS-SPECIFIC AUTO-ACTIONS ──
                    # Verdant Leaf: sell weakest joker immediately to un-debuff cards
                    if current_blind_name == "Verdant Leaf":
                        joker_cards_vl = raw.get("jokers", {}).get("cards", [])
                        # Only sell once per blind — check if we already sold
                        _vl_sold = getattr(env, '_verdant_leaf_sold', False)
                        if len(joker_cards_vl) > 1 and not _vl_sold:
                            env._verdant_leaf_sold = True
                            from environment.action_space import (
                                _api_key_to_name as _vl_name,
                            )
                            # Find the weakest SELLABLE joker — the shared
                            # guard never returns eternal, negative, MUST_BUY
                            # (Blueprint/Brainstorm), retrigger, or copy jokers.
                            weakest_idx, _ = _find_weakest_sellable_joker(
                                joker_cards_vl, raw,
                            )
                            if weakest_idx >= 0:
                                vk = joker_cards_vl[weakest_idx].get("key", "")
                                vn = _vl_name(vk) or vk
                                print(f"[BOSS] Verdant Leaf — selling {vn} "
                                      f"(idx {weakest_idx}) to un-debuff cards",
                                      flush=True)
                                try:
                                    await env.game.execute_action(
                                        "sell", {"joker": weakest_idx})
                                    await asyncio.sleep(0.3)
                                    raw = await env.game.fetch_gamestate()
                                except Exception as e:
                                    print(f"[BOSS] Verdant Leaf sell failed: {e}",
                                          flush=True)

                    # Amber Acorn: jokers are shuffled — re-arrange immediately
                    if current_blind_name == "Amber Acorn":
                        print(f"[BOSS] Amber Acorn — re-arranging shuffled jokers",
                              flush=True)

                    # Auto-rearrange jokers at start of each hand round
                    await self.action_executor._auto_rearrange_jokers(env, raw)
                    # Auto-use consumables (Planet cards, well-timed Tarots, etc.)
                    pre_auto_money = raw.get("money", 0)
                    raw = await self.action_executor._auto_use_consumables(env, raw)
                    post_auto_money = raw.get("money", 0)
                    env.auto_action_this_step = (pre_auto_money != post_auto_money)
                elif state == "SHOP":
                    # Auto-use non-targeting consumables in shop
                    # (Hermit for money doubling, Temperance, Wheel, etc.)
                    pre_auto_money = raw.get("money", 0)
                    raw = await self.action_executor._auto_use_consumables(env, raw)
                    # Auto-buy vouchers the NN doesn't understand
                    raw = await self.action_executor._auto_buy_vouchers(env, raw)
                    post_auto_money = raw.get("money", 0)
                    env.auto_action_this_step = (pre_auto_money != post_auto_money)
                env.prev_actionable_state = state
                return raw

            # SMODS_BOOSTER_OPENED — select best card from pack
            if state == "SMODS_BOOSTER_OPENED":
                pack_attempts += 1

                # Delay to let pack opening animation finish.
                # First attempt needs the longest wait; retries are shorter.
                if pack_attempts == 1:
                    await asyncio.sleep(1.0)
                elif pack_attempts == 2:
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.3)

                # Hard bailout: if stuck too long, force skip
                # (raised from 8 — multi-pick packs can legitimately need many attempts)
                if pack_attempts > 15:
                    print(f"[PACK-DBG] BAILOUT after {pack_attempts} attempts — "
                          f"force skipping (took nothing)", flush=True)
                    try:
                        await env.game.execute_action("pack", {"skip": True})
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                    # If we've bailed out too many times, game is probably broken
                    if pack_attempts > 25:
                        print(f"⚠️  PACK STUCK beyond recovery — triggering restart", flush=True)
                        await self._restart_balatro(env)
                        env.reward_calc.reset()
                        env.game.reset()
                        return None
                    continue

                pack_cards = raw.get("pack", {}).get("cards", [])
                if not pack_cards:
                    # Pack is still opening (card creation animation in progress)
                    continue

                # Determine pack type from card sets (use majority — handles
                # mixed packs like Celestial with Black Hole which is "SPECTRAL")
                set_counts: dict[str, int] = {}
                for pc in pack_cards:
                    s = pc.get("set", "")
                    set_counts[s] = set_counts.get(s, 0) + 1
                card_set = max(set_counts, key=set_counts.get) if set_counts else ""

                # INSTRUMENTATION: log every opened pack so the occasional
                # multi-pick / Black-Hole freeze-then-skip is fully captured.
                _pk_meta = {k: v for k, v in raw.get("pack", {}).items() if k != "cards"}
                print(f"[PACK-DBG] open type={card_set} attempt={pack_attempts} "
                      f"meta={_pk_meta} cards={[c.get('key', '?') for c in pack_cards]}",
                      flush=True)

                # ── Pick best card from pack ──
                pick_idx = 0

                # ── SOUL CARD CHECK (all pack types) ──
                # The Soul creates a Legendary joker — ALWAYS pick it.
                # If joker slots are full, sell weakest to make room.
                soul_idx = -1
                for sc_idx, sc in enumerate(pack_cards):
                    # The Soul's API key is "c_soul" (its display name is
                    # "The Soul"); "c_the_soul" never matches, so the agent
                    # was passing on free Legendary jokers.
                    if sc.get("key", "") == "c_soul":
                        soul_idx = sc_idx
                        break
                if soul_idx >= 0:
                    jokers_info_soul = raw.get("jokers", {})
                    soul_jcount = len(jokers_info_soul.get("cards", []))
                    soul_jlimit = jokers_info_soul.get("limit", 5)
                    if soul_jcount >= soul_jlimit:
                        # Sell weakest joker to make room for Legendary
                        weakest_soul_idx, _ = _find_weakest_sellable_joker(
                            jokers_info_soul.get("cards", []), raw)
                        if weakest_soul_idx >= 0:
                            w_name = jokers_info_soul["cards"][weakest_soul_idx].get("label") or "?"
                            print(f"[PACK] SOUL CARD! Selling {w_name} (idx {weakest_soul_idx}) "
                                  f"to make room for Legendary joker", flush=True)
                            try:
                                await env.game.execute_action("sell", {"joker": weakest_soul_idx})
                                await asyncio.sleep(0.5)
                            except Exception as e:
                                print(f"[PACK] Soul sell failed: {e}", flush=True)
                                # Sell failed — skip picking, can't make room
                                await asyncio.sleep(cfg.api_poll_delay)
                                continue
                        else:
                            print(f"[PACK] SOUL CARD found but can't sell any joker — "
                                  f"picking anyway (may fail)", flush=True)
                    else:
                        print(f"[PACK] SOUL CARD found! Picking Legendary joker", flush=True)
                    pick_idx = soul_idx
                    # Skip all other evaluation — go straight to pick logic
                    try:
                        await env.game.execute_action("pack", {"card": pick_idx})
                    except Exception as e:
                        print(f"[PACK] Soul pick failed: {e}", flush=True)
                    await asyncio.sleep(cfg.api_poll_delay)
                    continue

                preferred_targets: Optional[list] = None
                if card_set == "PLANET":
                    # Joker-aware planet selection: pick the planet that gives
                    # the biggest marginal score increase with current jokers
                    jokers_for_planet = raw.get("jokers", {}).get("cards", [])
                    pick_idx = pick_best_planet(pack_cards, jokers_for_planet, raw)

                elif card_set == "SPECTRAL":
                    # Spectral packs need their own evaluator — routing them
                    # through pick_best_planet (which knows no spectral keys)
                    # silently picked index 0, including build-destroying
                    # cards like Hex/Ankh (destroy every joker but one).
                    hand_for_spec = raw.get("hand", {}).get("cards", [])
                    jokers_for_spec = raw.get("jokers", {}).get("cards", [])
                    spec_result = evaluate_pack_spectral(
                        pack_cards, hand_for_spec, jokers_for_spec, raw)
                    if spec_result is None:
                        spec_keys = [c.get("key", "?") for c in pack_cards]
                        print(f"[PACK] No safe spectral pick in {spec_keys} "
                              f"— skipping pack", flush=True)
                        try:
                            await env.game.execute_action("pack", {"skip": True})
                        except Exception:
                            pass
                        await asyncio.sleep(cfg.api_poll_delay)
                        continue
                    pick_idx, preferred_targets = spec_result
                    print(f"[PACK] spectral pick: "
                          f"{pack_cards[pick_idx].get('key', '?')} "
                          f"targets={preferred_targets}", flush=True)

                elif card_set == "TAROT":
                    # Evaluate tarot cards and pick the best one with targets
                    hand_cards_for_tarot = raw.get("hand", {}).get("cards", [])
                    jokers_for_tarot = raw.get("jokers", {}).get("cards", [])

                    result = evaluate_pack_tarot(
                        pack_cards, hand_cards_for_tarot,
                        jokers_for_tarot, raw
                    )

                    if result is not None:
                        pick_idx, target_indices = result
                        try:
                            pack_params: dict = {"card": pick_idx}
                            if target_indices:
                                pack_params["targets"] = target_indices
                            await env.game.execute_action("pack", pack_params)
                        except Exception as e:
                            pass  # print(f"PACK tarot {pick_idx} failed: {e}")
                            # Re-check state before skipping — the select may have
                            # partially succeeded and the pack is already closing.
                            await asyncio.sleep(0.5)
                            try:
                                recheck = await env.game.fetch_gamestate()
                                if recheck.get("state", "") == "SMODS_BOOSTER_OPENED":
                                    await env.game.execute_action("pack", {"skip": True})
                            except Exception:
                                pass
                        await asyncio.sleep(cfg.api_poll_delay)
                        continue
                    else:
                        # No worthwhile tarot — skip
                        try:
                            await env.game.execute_action("pack", {"skip": True})
                        except Exception:
                            pass
                        await asyncio.sleep(cfg.api_poll_delay)
                        continue

                elif card_set == "JOKER":
                    # Joker packs: evaluate which pack joker adds most scoring power
                    JOKER_BLACKLIST = {
                        "j_four_fingers", # Four Fingers — we don't build 4-card flushes/straights
                        "j_drunkard",    # Drunkard — +1 discard but -1 hand
                    }
                    # Must-pick jokers in packs — always grab these regardless of delta
                    PACK_MUST_PICK = {
                        "j_blueprint", "j_brainstorm",
                    }
                    jokers_info = raw.get("jokers", {})
                    joker_count = len(jokers_info.get("cards", []))
                    joker_limit = jokers_info.get("limit", 5)
                    current_jokers = jokers_info.get("cards", [])

                    if joker_count < joker_limit:
                        # Check for must-pick jokers first
                        must_pick_idx = -1
                        for pc_idx, pc in enumerate(pack_cards):
                            pc_key = pc.get("key", "")
                            if pc_key in PACK_MUST_PICK:
                                must_pick_idx = pc_idx
                                print(f"[PACK] MUST-PICK joker found: {pc_key} "
                                      f"at index {pc_idx}", flush=True)
                                break

                        if must_pick_idx >= 0:
                            pick_idx = must_pick_idx
                        else:
                            # Slots available — pick the best joker from the pack
                            current_score = estimate_score_for_hand_type(current_jokers, raw)
                            best_improvement = 0.0
                            best_pack_idx = 0
                            for pc_idx, pc in enumerate(pack_cards):
                                pc_key = pc.get("key", "")
                                if pc_key in JOKER_BLACKLIST:
                                    continue
                                test_jokers = current_jokers + [pc]
                                score_with = estimate_score_for_hand_type(test_jokers, raw)
                                improvement = score_with - current_score
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_pack_idx = pc_idx
                            pick_idx = best_pack_idx

                    else:
                        # Slots full — evaluate swap with weakest owned joker
                        current_score = estimate_score_for_hand_type(current_jokers, raw)

                        # Find the weakest SELLABLE joker to make room. The
                        # shared guard never returns eternal, negative,
                        # MUST_BUY (Blueprint/Brainstorm), retrigger, or copy
                        # jokers, so none of those can be sold to swap for a
                        # pack card.
                        worst_idx, _ = _find_weakest_sellable_joker(
                            current_jokers, raw,
                        )
                        if worst_idx < 0:
                            # Nothing safe to sell — skip the pack rather than
                            # swap out a protected joker.
                            try:
                                await env.game.execute_action("pack", {"skip": True})
                            except Exception:
                                pass
                            await asyncio.sleep(cfg.api_poll_delay)
                            continue

                        # Check for must-pick jokers first (always swap for these)
                        must_pick_idx = -1
                        for pc_idx, pc in enumerate(pack_cards):
                            pc_key = pc.get("key", "")
                            if pc_key in PACK_MUST_PICK:
                                must_pick_idx = pc_idx
                                print(f"[PACK] MUST-PICK joker {pc_key} found "
                                      f"(slots full, will swap)", flush=True)
                                break

                        # Check each pack joker as a replacement for the weakest
                        best_swap = must_pick_idx if must_pick_idx >= 0 else None
                        best_swap_score = current_score
                        if best_swap is None:
                            for pc_idx, pc in enumerate(pack_cards):
                                pc_key = pc.get("key", "")
                                if pc_key in JOKER_BLACKLIST:
                                    continue
                                jokers_with_swap = [j for i, j in enumerate(current_jokers) if i != worst_idx]
                                jokers_with_swap.append(pc)
                                swap_score = estimate_score_for_hand_type(jokers_with_swap, raw)
                                if swap_score > best_swap_score:
                                    best_swap_score = swap_score
                                    best_swap = pc_idx

                        if best_swap is None:
                            # Slots full and no pack joker beats the current
                            # build — skip cleanly. Without this, pick_idx stays
                            # at its default 0 and the bot falls through to pick
                            # a joker into full slots, which the mod rejects
                            # ("joker slots full") on every retry until bailout.
                            pick_idx = -1

                        if best_swap is not None:
                            # Sell the weakest joker then IMMEDIATELY pick
                            # the replacement in the same iteration.
                            try:
                                worst_name = current_jokers[worst_idx].get("label") or current_jokers[worst_idx].get("key", "?")
                                print(f"[PACK] selling joker {worst_idx} ({worst_name}) "
                                      f"to swap for pack card {best_swap}", flush=True)
                                await env.game.execute_action("sell", {"joker": worst_idx})
                                await asyncio.sleep(0.5)
                                # Verify pack is still open before picking
                                recheck = await env.game.fetch_gamestate()
                                if recheck.get("state", "") == "SMODS_BOOSTER_OPENED":
                                    pick_idx = best_swap
                                    # Fall through to generic pick logic below
                                else:
                                    print(f"[PACK] state changed to "
                                          f"{recheck.get('state', '?')} after sell — "
                                          f"pack closed, can't pick", flush=True)
                                    await asyncio.sleep(cfg.api_poll_delay)
                                    continue
                            except Exception as e:
                                print(f"[PACK] sell failed: {e}", flush=True)
                                pick_idx = -1  # sell failed, skip below

                        if pick_idx < 0:
                            # No improvement or sell failed — skip remaining picks
                            try:
                                await env.game.execute_action("pack", {"skip": True})
                            except Exception as e:
                                pass
                            await asyncio.sleep(cfg.api_poll_delay)
                            continue

                elif card_set == "ENHANCED":
                    # Standard/Buffoon packs with playing cards — just pick first
                    pass  # fall through to generic pick logic below

                # Build ordered list of cards to try. For each card, determine
                # whether it needs targets based on its key. The Lua side uses
                # G.P_CENTERS to validate — we mirror that logic here.
                # Cards that need hand card targets (highlighted cards).
                NEEDS_TARGET_KEYS = {
                    # Tarots that enhance/modify specific hand cards
                    "c_magician", "c_empress", "c_heirophant",
                    "c_lovers", "c_chariot", "c_justice",
                    "c_hanged_man", "c_death",
                    "c_tower", "c_star",
                    "c_moon", "c_sun", "c_world",
                    "c_devil", "c_strength",
                    # Spectral cards that need hand targets
                    "c_aura", "c_sigil", "c_ouija", "c_immolate",
                    # Spectral seal cards (1 selected hand card)
                    "c_medium", "c_talisman", "c_deja_vu", "c_trance",
                    # Spectral cards that destroy/modify hand cards
                    "c_familiar", "c_grim", "c_incantation",
                    # Spectral cards that duplicate hand cards
                    "c_cryptid",
                }
                # Cards that don't need ANY targets (act on jokers, deck, or RNG)
                # c_the_hermit, c_temperance, c_judgement, c_the_fool,
                # c_wheel_of_fortune, c_the_high_priestess, c_the_soul,
                # c_black_hole, c_ankh, c_ectoplasm, c_hex,
                # c_wraith, all planet cards

                # Try cards in priority order: pick_idx first, then others.
                # On first 2 attempts, only try the ideal card (game may still be animating).
                # After that, try all cards to avoid getting stuck.
                if pack_attempts <= 2:
                    card_order = [pick_idx]
                else:
                    card_order = [pick_idx] + [i for i in range(len(pack_cards)) if i != pick_idx]

                selected = False
                for try_idx in card_order:
                    if selected:
                        break
                    card_key = pack_cards[try_idx].get("key", "") if try_idx < len(pack_cards) else ""
                    needs_targets = card_key in NEEDS_TARGET_KEYS

                    if needs_targets:
                        # Death/Strength need exactly 2; most others need 1-3.
                        # Try 2 targets first (covers Death), then 1, then 3.
                        NEEDS_TWO_TARGETS = {"c_death"}
                        if card_key in NEEDS_TWO_TARGETS:
                            target_attempts = [[0, 1], [0, 1, 2]]
                        else:
                            target_attempts = [[0], [0, 1], [0, 1, 2]]
                        # The evaluator's chosen target(s) (e.g. the best
                        # hand card for a seal/edition spectral) go first
                        if preferred_targets and try_idx == pick_idx:
                            target_attempts = [preferred_targets] + target_attempts
                        for targets in target_attempts:
                            try:
                                await env.game.execute_action("pack", {"card": try_idx, "targets": targets})
                                selected = True
                                break
                            except Exception as e:
                                pass  # print(f"PACK card {try_idx} targets={targets} failed: {e}")
                    else:
                        try:
                            await env.game.execute_action("pack", {"card": try_idx})
                            selected = True
                        except Exception as e:
                            pass  # print(f"PACK card {try_idx} failed: {e}")

                # INSTRUMENTATION: did the pick land?
                _tried_key = pack_cards[pick_idx].get("key", "?") if pick_idx < len(pack_cards) else "?"
                print(f"[PACK-DBG] pick result selected={selected} pick_idx={pick_idx} "
                      f"key={_tried_key} attempt={pack_attempts} order={card_order}",
                      flush=True)

                # On a failed pick, DON'T skip the pack — the failure is usually
                # transient (the mod's 5s use_card timeout when a planet/Black-
                # Hole use-animation lags, or STATE_COMPLETE not yet true). Just
                # retry on the next poll; the pack stays open. Only give up and
                # skip near the bailout, so an occasional hiccup no longer
                # discards the whole pack ("freeze then take nothing").
                if not selected and pack_attempts >= 12:
                    try:
                        recheck = await env.game.fetch_gamestate()
                        if recheck.get("state", "") == "SMODS_BOOSTER_OPENED":
                            print(f"[PACK-DBG] giving up after {pack_attempts} "
                                  f"failed picks — skipping pack", flush=True)
                            await env.game.execute_action("pack", {"skip": True})
                    except Exception:
                        pass

                await asyncio.sleep(cfg.api_poll_delay)
                continue

            # ROUND_EVAL — auto cash out
            if state == "ROUND_EVAL":
                round_eval_count = env.round_eval_count + 1
                env.round_eval_count = round_eval_count
                try:
                    await env.game.execute_action("cash_out")
                    env.round_eval_count = 0
                except Exception:
                    pass

                # If stuck at ROUND_EVAL for many polls, likely the win overlay
                # is blocking (game paused after beating ante 8). Go to menu.
                if round_eval_count > 10:
                    ante = raw.get("ante_num", 1)
                    print(f"[WIN-RECOVERY] Stuck at ROUND_EVAL (ante={ante}, "
                          f"polls={round_eval_count}) — going to menu", flush=True)
                    env.round_eval_count = 0
                    try:
                        await env.game.execute_action("menu")
                    except Exception:
                        # Menu may also fail if paused — try restart
                        print("[WIN-RECOVERY] Menu failed — triggering restart", flush=True)
                        await self._restart_balatro(env)
                        env.reward_calc.reset()
                        env.game.reset()
                        return None
                    await asyncio.sleep(1.0)

                await asyncio.sleep(cfg.api_poll_delay)
                continue

            # MENU — start a new run with random seed
            if state == "MENU":
                # Count consecutive MENU visits: after WIN #36 the mod's
                # start endpoint wedged (hung, then dropped the connection
                # with no response) while gamestate still answered — the
                # old silent except:pass retried forever, visibly spamming
                # run-starts for 25+ minutes with a frozen log. Covers both
                # failure shapes: start raising AND start "succeeding" but
                # bouncing straight back to MENU.
                env.menu_loop_count = env.menu_loop_count + 1
                if env.menu_loop_count == 1:
                    # Let the menu settle before the first start attempt.
                    # Firing start mid-transition (GAME_OVER overlay still
                    # closing at speed 8) can make the mod's start_run a
                    # silent no-op that wedges every subsequent start until
                    # a full game restart — 17 of those wedges in one night
                    # cost ~4x throughput.
                    await asyncio.sleep(2.0)
                if env.menu_loop_count >= 8:
                    print(f"[MENU] {env.menu_loop_count} consecutive MENU "
                          f"polls — start endpoint wedged, restarting "
                          f"Balatro", flush=True)
                    env.menu_loop_count = 0
                    await self._restart_balatro(env)
                    env.reward_calc.reset()
                    env.game.reset()
                    return None

                # Start recording the new run
                env.recorder.start_run()
                # CURRICULUM (dec-030): with annealed probability, LOAD a banked
                # ante-4/5 partial-build state instead of a fresh run so the
                # policy gets dense experience where engines matter (wins and
                # advantages no longer 0.5%-rare). Falls back to a fresh start
                # on any failure.
                _seed_path = self._pick_curriculum_seed()
                if _seed_path is not None:
                    try:
                        await env.game.execute_action("load", {"path": _seed_path})
                        env.from_curriculum = True
                        env.menu_loop_count = 0
                        print(f"[CURRICULUM] load OK p={self._curriculum_prob():.3f} "
                              f"{os.path.basename(_seed_path)} (env {env.env_id})",
                              flush=True)
                        await asyncio.sleep(0.5)
                        continue
                    except Exception as e:
                        print(f"[CURRICULUM] load failed ({e}) -> fresh start",
                              flush=True)
                env.from_curriculum = False
                # dec-045: eval seam. If this env has a forced-seed queue (set by
                # the eval harness) use the next fixed seed so runs are reproducible
                # and two checkpoints can be A/B'd on identical seeds. Production
                # leaves forced_seeds unset -> random, unchanged behavior.
                forced = getattr(env, "forced_seeds", None)
                if forced:
                    seed = forced.pop(0)
                elif self.eval_mode:
                    # Eval bank exhausted — stop starting runs for this env.
                    env.eval_finished = True
                    return None
                else:
                    seed = ''.join(random.choices(string.ascii_uppercase, k=8))
                env.current_seed = seed   # tagged into game_history for eval attribution
                try:
                    await env.game.execute_action(
                        "start", {"deck": "RED", "stake": "WHITE", "seed": seed}
                    )
                except Exception as e:
                    print(f"[MENU] start failed "
                          f"(attempt {env.menu_loop_count}): {e}", flush=True)
                await asyncio.sleep(0.5)
                continue

            # Transient game states (card animations, state transitions)
            # These are normal during pack processing and other game events.
            # Don't panic — just wait for them to resolve.
            TRANSIENT_STATES = {
                "PLAY_TAROT", "HAND_PLAYED", "DRAW_TO_HAND", "NEW_ROUND",
                "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK",
                "STANDARD_PACK", "BUFFOON_PACK",
                # Pack opening transitions
                "SMODS_BOOSTER_OPENING", "BOOSTER_PACK",
                # Scoring animations
                "SCORING", "EVAL_HAND",
                # Consumable use animations
                "USE_CONSUMABLE", "PLAY_SPECTRAL",
                # Sell/buy transitions
                "SELLING_CARD",
                # Blind selection transitions
                "SKIP_BLIND",
                # Generic unknown — mod returns this during win screen,
                # run summary, and other unmapped transitions
                "UNKNOWN",
            }
            if state in TRANSIENT_STATES:
                # These resolve on their own — give extra time
                await asyncio.sleep(0.5)
                # Only count toward stuck if we've been in transient for a while
                unknown_state_count += 1
                # INSTRUMENTATION: dump the raw state at a few checkpoints so we
                # can see what the game is actually showing during a stall —
                # especially post-win win-screen / run-summary wedges.
                if unknown_state_count in (5, 30, 60):
                    import json as _json
                    print(f"[STATE-DUMP] transient '{state}' poll#{unknown_state_count} "
                          f"won={raw.get('won')} ante={raw.get('ante_num')} "
                          f"keys={sorted(raw.keys())} "
                          f"raw={_json.dumps(raw, default=str)[:1500]}", flush=True)
                if unknown_state_count > 60:  # 30+ seconds in transient
                    print(f"[CRASH-RECOVERY] Stuck in transient state '{state}' for "
                          f"{unknown_state_count} polls — triggering restart", flush=True)
                    await self._restart_balatro(env)
                    env.reward_calc.reset()
                    env.game.reset()
                    return None
                continue

            # Unknown/unrecognized state — wait briefly
            unknown_state_count += 1
            if unknown_state_count == 1 or unknown_state_count % 10 == 0:
                print(f"[UNKNOWN-STATE] '{state}' (poll #{unknown_state_count})", flush=True)
            if unknown_state_count > 30:
                print(f"[CRASH-RECOVERY] Stuck in unknown state '{state}' for {unknown_state_count} polls — triggering restart", flush=True)
                await self._restart_balatro(env)
                env.reward_calc.reset()
                env.game.reset()
                return None
            await asyncio.sleep(cfg.api_poll_delay)

        # All poll attempts exhausted — Balatro is likely crashed
        env.consecutive_api_failures += 1
        if env.consecutive_api_failures >= 3:
            print(f"[CRASH-RECOVERY] {env.consecutive_api_failures} consecutive poll failures — triggering restart", flush=True)
            await self._restart_balatro(env)
            # Reset episode state after restart
            env.reward_calc.reset()
            env.game.reset()
        return None  # Timed out

    def _get_scaling_snapshot(self, env) -> dict[int, float]:
        """Get current scaling values from game state manager."""
        tracker = env.game._scaling_tracker
        return dict(tracker._joker_values)

    def _log_update(self, metrics: dict):
        """Print training metrics to console."""
        elapsed = time.time() - self.start_time
        fps = self.global_step / max(elapsed, 1)
        ep_stats = self.episode_tracker.get_recent_stats()

        print(
            f"Update {self.num_updates:>5d} | "
            f"Step {self.global_step:>8,d} | "
            f"FPS {fps:>6.0f} | "
            f"Ep {ep_stats['episodes']:>4d} | "
            f"R {ep_stats['mean_reward']:>7.2f} | "
            f"Ante {ep_stats['mean_ante']:>4.1f} | "
            f"WR {ep_stats['win_rate']:>5.1%} | "
            f"WR500 {ep_stats.get('win_rate_long', 0.0):>5.2%}"
            f"(W{ep_stats.get('lifetime_wins', 0)}) | "
            f"PL {metrics['policy_loss']:>7.4f} | "
            f"VL {metrics['value_loss']:>7.4f} | "
            f"Ent {metrics['entropy']:>6.3f} | "
            f"KL {metrics['approx_kl']:>8.6f} | "
            f"CF {metrics.get('clip_fraction', 0.0):>5.3f} | "
            f"BC {metrics.get('bc_loss', 0.0):>6.3f}"
            f"@{metrics.get('bc_coef', 0.0):.2f}"
            f"({metrics.get('bc_fraction', 0.0):.0%}) | "
            f"Pr {metrics.get('prior_kl', 0.0):>6.3f}"
            f"@{metrics.get('prior_coef', 0.0):.2f} | "
            f"SIL {metrics.get('sil_loss', 0.0):>5.2f}"
            f"@{self.config.sil_coef:.2f} | "
            f"LR {self.ppo.get_learning_rate():.2e} | "
            f"EV {metrics.get('explained_variance', 0.0):>6.3f}"
        )

    def _log_blind_result(self, env, beaten: bool, raw: dict = None):
        """dec-049 (Tier 1 measurement): write one realized per-blind outcome to
        logs/blind_results.jsonl when a blind resolves. Separates an UNDER-POWERED
        build (realized << target) from an ADEQUATE build that died to variance /
        boss-debuff (realized ~ target, or realized << projected power). Closes the
        ~40% 'adequate build, dies anyway' blind spot from the deep audit, and gives
        realized data to re-fit the realization factor. Best-effort; never blocks."""
        try:
            tgt = env.cur_blind_target
            if tgt <= 0:
                return  # nothing tracked for this blind
            realized = env.cur_realized
            if raw is not None:
                realized = max(realized, float(raw.get("round", {}).get("chips", 0) or 0))
            # The per-step tracker misses the FINAL hand (state jumps
            # SELECTING_HAND->SHOP), so beaten blinds undercount. A beaten blind
            # scored >= its target by definition -> floor realized at target.
            # (Failed-blind realized is accurate via the GAME_OVER raw fallback.)
            if beaten:
                realized = max(realized, tgt)
            proj = env.last_proj_power
            rec = {
                "ante": env.cur_blind_ante,
                "blind": env.cur_blind_name,
                "beaten": bool(beaten),
                "realized": round(realized, 0),
                "target": round(tgt, 0),
                "realized_margin": round(realized / max(tgt, 1.0), 3),
                "hands_left": env.cur_hands_left,
                "discards_left": env.cur_discards_left,  # dec-050: under-dig signal
                "proj_power": round(proj, 0),
                "realized_vs_proj": round(realized / max(proj, 1.0), 3),
                "env": env.env_id, "step": self.global_step,
            }
            path = os.path.join("logs", "blind_results.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            env.cur_blind_target = 0.0  # consumed; avoid double-logging the same blind
            # light rotation (dec-043 lesson: don't grow unbounded)
            self._blind_log_count = getattr(self, "_blind_log_count", 0) + 1
            if self._blind_log_count % 2000 == 0:
                with open(path) as f:
                    lines = f.readlines()
                if len(lines) > 50000:
                    with open(path, "w") as f:
                        f.writelines(lines[-50000:])
        except Exception:
            pass  # instrumentation must never break training

    def _save_checkpoint(self, tag: Optional[str] = None):
        """Save training checkpoint."""
        if tag:
            filename = f"balatron_phase{self.config.phase}_{tag}.pt"
        else:
            filename = f"balatron_phase{self.config.phase}_update{self.num_updates:06d}.pt"

        path = os.path.join(self.config.checkpoint_dir, filename)
        self.ppo.save_checkpoint(path)

        # Also save training config and episode stats
        meta_path = path.replace(".pt", "_meta.json")
        meta = {
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "config": asdict(self.config),
            "episode_stats": self.episode_tracker.get_recent_stats(),
            "elapsed_time": time.time() - self.start_time,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"  Checkpoint saved: {path}")
        self._prune_checkpoints(keep=15)

    def _prune_checkpoints(self, keep: int = 15):
        """Keep only the newest `keep` numbered checkpoints (+ any final/best),
        deleting older .pt and their _meta.json. dec-043: checkpoints saved every
        2 updates were NEVER pruned and grew to 43.7 GB / 1183 files, filling the
        disk to 0 bytes — which silently breaks future saves. Runs after each save."""
        try:
            import glob
            cps = glob.glob(os.path.join(self.config.checkpoint_dir,
                                         f"balatron_phase{self.config.phase}_update*.pt"))
            cps.sort(key=os.path.getmtime, reverse=True)
            for old in cps[keep:]:
                for p in (old, old.replace(".pt", "_meta.json")):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
        except Exception:
            pass  # never let pruning break training

    def _print_summary(self):
        """Print final training summary."""
        elapsed = time.time() - self.start_time
        stats = self.episode_tracker.get_recent_stats()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"  Total steps: {self.global_step:,}")
        print(f"  Total updates: {self.num_updates}")
        print(f"  Total episodes: {stats['episodes']}")
        print(f"  Total time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
        print(f"  Avg FPS: {self.global_step / max(elapsed, 1):.0f}")
        print(f"  Mean reward (last 20): {stats['mean_reward']:.2f}")
        print(f"  Mean ante (last 20): {stats['mean_ante']:.1f}")
        print(f"  Max ante (last 20): {stats['max_ante']}")
        print(f"  Win rate (last 20): {stats['win_rate']:.1%}")
        print(f"  Total wins: {self.episode_tracker.wins}")
        print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

def parse_args() -> TrainConfig:
    """Parse command-line arguments into TrainConfig."""
    parser = argparse.ArgumentParser(description="Balatron Training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase (1=general, 2=naneinf)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total environment steps")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Log every N updates")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N updates")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable ffmpeg win recording")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Parallel Balatro instances "
                             "(ports 12346..12346+N-1; env 0 records)")

    args = parser.parse_args()

    config = TrainConfig(
        phase=args.phase,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        record_wins=not args.no_record,
        num_envs=args.num_envs,
    )

    return config, args.checkpoint


async def async_main():
    config, checkpoint_path = parse_args()
    trainer = Trainer(config, checkpoint_path)
    await trainer.run()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
