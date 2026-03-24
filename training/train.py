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

Usage:
    python -m training.train
    python -m training.train --checkpoint path/to/checkpoint.pt
    python -m training.train --phase 2 --checkpoint phase1_best.pt
"""

import argparse
import asyncio
import json
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
    evaluate_pack_tarot, pick_best_planet,
)
from recorder import RunRecorder


def _get_blind_target_from_state(raw_state: dict) -> float:
    """Extract the current blind's target score."""
    blinds = raw_state.get("blinds", {})
    if isinstance(blinds, dict):
        for b in blinds.values():
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                return b.get("score", 300)
    return 300.0
from environment.reward import RewardCalculator


# ============================================================
# Training Config
# ============================================================

@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Training
    total_timesteps: int = 1_000_000     # Total environment steps
    rollout_steps: int = 2048            # Steps per rollout
    phase: int = 1                       # 1 = general, 2 = naneinf

    # PPO (forwarded to PPOConfig)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_epochs: int = 4
    num_minibatches: int = 4
    target_kl: float = 0.03

    # Logging
    log_interval: int = 1               # Log every N updates
    checkpoint_interval: int = 10       # Save every N updates
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cpu"

    # LR annealing
    anneal_lr: bool = True

    # Game
    api_poll_delay: float = 0.05        # Seconds to wait when game is resolving
    max_poll_attempts: int = 100        # Max polls before giving up

    # Recording
    record_wins: bool = True            # Record runs and save wins to recordings/wins/

    def to_ppo_config(self) -> PPOConfig:
        return PPOConfig(
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            num_epochs=self.num_epochs,
            num_minibatches=self.num_minibatches,
            rollout_steps=self.rollout_steps,
            target_kl=self.target_kl,
            anneal_lr=self.anneal_lr,
            device=self.device,
        )


# ============================================================
# Episode Tracker
# ============================================================

class EpisodeTracker:
    """Tracks per-episode statistics across training."""

    STATS_FILE = "logs/lifetime_stats.json"
    WIN_LOG_FILE = "logs/win_log.json"

    def __init__(self):
        self.current_reward = 0.0
        self.current_length = 0
        self.current_ante = 1

        self.completed_episodes = 0
        self.total_rewards: list[float] = []
        self.total_lengths: list[int] = []
        self.max_antes: list[int] = []
        self.wins = 0

        # Session-wide records
        self.session_highest_ante = 0
        self.session_highest_score = 0.0
        self.session_highest_score_round = ""
        self._prev_round_chips = 0.0

        # Per-episode highest hand (reset each run)
        self._episode_highest_hand = 0.0
        self._episode_highest_hand_type = ""

        # Load lifetime stats from disk
        self._lifetime_wins = 0
        self._lifetime_episodes = 0
        self._lifetime_highest_ante = 0
        self._load_lifetime_stats()

    def _load_lifetime_stats(self):
        """Load cumulative stats from disk."""
        try:
            with open(self.STATS_FILE, "r") as f:
                data = json.load(f)
            self._lifetime_wins = data.get("wins", 0)
            self._lifetime_episodes = data.get("episodes", 0)
            self._lifetime_highest_ante = data.get("highest_ante", 0)
            print(f"[STATS] Loaded lifetime stats: {self._lifetime_wins} wins / "
                  f"{self._lifetime_episodes} runs "
                  f"(best ante: {self._lifetime_highest_ante})", flush=True)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Fresh start

    def _save_lifetime_stats(self):
        """Persist cumulative stats to disk."""
        os.makedirs(os.path.dirname(self.STATS_FILE) or ".", exist_ok=True)
        data = {
            "wins": self._lifetime_wins,
            "episodes": self._lifetime_episodes,
            "highest_ante": self._lifetime_highest_ante,
        }
        try:
            with open(self.STATS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _append_win_log(self, record: dict):
        """Append a win record to the persistent win log."""
        os.makedirs(os.path.dirname(self.WIN_LOG_FILE) or ".", exist_ok=True)
        try:
            try:
                with open(self.WIN_LOG_FILE, "r") as f:
                    wins_list: list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                wins_list = []
            wins_list.append(record)
            with open(self.WIN_LOG_FILE, "w") as f:
                json.dump(wins_list, f, indent=2)
        except OSError:
            pass

    def step(self, reward: float, ante: int, raw_state: dict = None):
        """Record a single step."""
        self.current_reward += reward
        self.current_length += 1
        self.current_ante = max(self.current_ante, ante)

        # Track highest hand score from chip deltas
        if raw_state:
            round_chips = raw_state.get("round", {}).get("chips", 0)
            if round_chips > self._prev_round_chips and self._prev_round_chips >= 0:
                hand_score = round_chips - self._prev_round_chips
                # Per-episode tracking
                if hand_score > self._episode_highest_hand:
                    self._episode_highest_hand = hand_score
                    # Try to get hand type from the [HAND] log context
                    blind_name = ""
                    blinds_ep = raw_state.get("blinds", {})
                    if isinstance(blinds_ep, dict):
                        for b in blinds_ep.values():
                            if isinstance(b, dict) and b.get("status") == "CURRENT":
                                blind_name = b.get("name", "")
                                break
                    self._episode_highest_hand_type = f"Ante {ante} ({blind_name})" if blind_name else f"Ante {ante}"
                if hand_score > self.session_highest_score:
                    self.session_highest_score = hand_score
                    blind_name = ""
                    blinds = raw_state.get("blinds", {})
                    if isinstance(blinds, dict):
                        for b in blinds.values():
                            if isinstance(b, dict) and b.get("status") == "CURRENT":
                                blind_name = b.get("name", "")
                                break
                    self.session_highest_score_round = f"Ante {ante} ({blind_name})" if blind_name else f"Ante {ante}"
            self._prev_round_chips = round_chips

        # Check for new highest ante
        if ante > self.session_highest_ante:
            self.session_highest_ante = ante
            score_info = f" | Best hand: {self.session_highest_score:,.0f} in {self.session_highest_score_round}" if self.session_highest_score > 0 else ""
            print(f"🏆 NEW HIGHEST ANTE: {ante}{score_info}")

    def end_episode(self, won: bool, raw_state: dict = None):
        """Record episode completion."""
        self.total_rewards.append(self.current_reward)
        self.total_lengths.append(self.current_length)
        self.max_antes.append(self.current_ante)
        self.completed_episodes += 1

        # Update lifetime counters
        self._lifetime_episodes += 1
        self._lifetime_highest_ante = max(self._lifetime_highest_ante, self.current_ante)

        if won:
            self.wins += 1
            self._lifetime_wins += 1
            print(f"\n{'='*50}")
            print(f"WIN #{self._lifetime_wins} (lifetime) | "
                  f"#{self.wins} this session")
            print(f"   Run #{self._lifetime_episodes} lifetime | "
                  f"#{self.completed_episodes} this session | "
                  f"Ante {self.current_ante}")
            print(f"   Lifetime: {self._lifetime_wins}/{self._lifetime_episodes} "
                  f"({self._lifetime_wins/max(self._lifetime_episodes,1)*100:.1f}%)")
            win_record: dict = {
                "win_number": self._lifetime_wins,
                "episode": self._lifetime_episodes,
                "ante": self.current_ante,
                "timestamp": datetime.now().isoformat(),
                "reward": round(self.current_reward, 2),
            }
            if raw_state:
                jokers = raw_state.get("jokers", {}).get("cards", [])
                if jokers:
                    joker_names = [j.get("label") or j.get("key", "?") for j in jokers]
                    print(f"   Jokers: {' | '.join(joker_names)}")
                    win_record["jokers"] = []
                    for j in jokers:
                        jinfo: dict = {
                            "name": j.get("label") or j.get("key", "?"),
                            "edition": j.get("edition", ""),
                        }
                        # Capture scaling values if present
                        for k in ("_scaled_value", "x_mult", "extra"):
                            if k in j:
                                jinfo[k] = j[k]
                        win_record["jokers"].append(jinfo)
                # Hand levels
                poker_hands = raw_state.get("poker_hands", {})
                if poker_hands:
                    levels: dict = {}
                    for hname, hdata in poker_hands.items():
                        lvl = hdata if isinstance(hdata, int) else hdata.get("level", 1)
                        if lvl > 1:
                            levels[hname] = lvl
                    if levels:
                        win_record["hand_levels"] = levels
                win_record["money"] = raw_state.get("money", 0)
                win_record["deck"] = raw_state.get("deck_name", "")
                win_record["stake"] = raw_state.get("stake", "")
                if self._episode_highest_hand > 0:
                    win_record["highest_hand"] = round(self._episode_highest_hand)
                    win_record["highest_hand_context"] = self._episode_highest_hand_type
            print(f"{'='*50}\n")
            self._append_win_log(win_record)
        else:
            # Print a compact loss line every 10 episodes so progress is visible
            if self.completed_episodes % 10 == 0:
                print(f"Runs: {self.completed_episodes} (lifetime: {self._lifetime_episodes}) | "
                      f"Wins: {self.wins} (lifetime: {self._lifetime_wins}) | "
                      f"Best ante: {self._lifetime_highest_ante} | "
                      f"Avg ante (last 20): {sum(self.max_antes[-20:])/min(len(self.max_antes), 20):.1f}")

        # Save after every episode
        self._save_lifetime_stats()

        self.current_reward = 0.0
        self.current_length = 0
        self.current_ante = 1
        self._prev_round_chips = 0.0
        self._episode_highest_hand = 0.0
        self._episode_highest_hand_type = ""

    def get_recent_stats(self, window: int = 20) -> dict:
        """Get statistics over the last N episodes."""
        if not self.total_rewards:
            return {
                "episodes": 0,
                "mean_reward": 0.0,
                "mean_length": 0,
                "mean_ante": 0.0,
                "max_ante": 0,
                "win_rate": 0.0,
            }

        recent_r = self.total_rewards[-window:]
        recent_l = self.total_lengths[-window:]
        recent_a = self.max_antes[-window:]
        recent_wins = sum(1 for a in recent_a if a > 8)

        return {
            "episodes": self.completed_episodes,
            "mean_reward": np.mean(recent_r),
            "mean_length": int(np.mean(recent_l)),
            "mean_ante": np.mean(recent_a),
            "max_ante": max(recent_a),
            "win_rate": recent_wins / len(recent_a),
        }


# ============================================================
# Joker Order Round Logger
# ============================================================

class JokerOrderLogger:
    """Logs per-round joker ordering data to a rolling file.

    Tracks intended vs confirmed joker orders, Brainstorm copy targets,
    and rearrange failures for post-training review.
    """

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "joker_order_log.jsonl")
        self._current_round: dict = {}
        self._plays: list[dict] = []
        self._rearrange_failures: list[str] = []
        self._active = False

    def round_start(self, ante: int, round_num: int, blind_name: str,
                    blind_score: float, joker_keys: list[str]):
        """Called when a new SELECTING_HAND state begins."""
        # Flush previous round if it wasn't closed
        if self._active:
            self._flush()

        from environment.hand_eval import _api_key_to_name
        joker_names = [_api_key_to_name(k) or k for k in joker_keys]

        self._current_round = {
            "ante": ante,
            "round": round_num,
            "blind": blind_name,
            "blind_score": blind_score,
            "jokers_at_start": joker_names,
        }
        self._plays = []
        self._rearrange_failures = []
        self._active = True

    def log_play(self, hand_type: str, played_cards: list[str],
                 intended_order: list[str] | None,
                 confirmed_order: list[str] | None,
                 brainstorm_copies: str | None,
                 order_matched: bool | None):
        """Called before each hand play."""
        if not self._active:
            return
        self._plays.append({
            "hand_type": hand_type,
            "cards": played_cards,
            "intended_order": intended_order,
            "confirmed_order": confirmed_order,
            "brainstorm_copies": brainstorm_copies,
            "order_matched": order_matched,
        })

    def log_rearrange_failure(self, context: str, error: str):
        """Called when a rearrange API call fails."""
        if not self._active:
            return
        self._rearrange_failures.append(f"{context}: {error}")

    def round_end(self):
        """Called when blind is beaten (SELECTING_HAND → SHOP) or game over."""
        if self._active:
            self._flush()

    def _flush(self):
        """Write the current round entry to the log file."""
        if not self._current_round:
            self._active = False
            return

        entry = {
            **self._current_round,
            "plays": self._plays,
            "rearrange_failures": self._rearrange_failures,
            "total_plays": len(self._plays),
        }

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Don't crash training for logging

        self._current_round = {}
        self._plays = []
        self._rearrange_failures = []
        self._active = False


# ============================================================
# Sell-guard helpers
# ============================================================

def _find_weakest_sellable_joker(
    jokers_raw: list[dict],
    raw_state: dict,
    *,
    exclude_indices: set[int] | None = None,
) -> tuple[int, float]:
    """Find the weakest joker that is safe to sell.

    Skips: eternal, negative edition, MUST_BUY (Blueprint/Brainstorm),
    retrigger jokers, and copy jokers.

    Returns (index, value) of weakest sellable joker, or (-1, inf) if none.
    """
    from environment.action_space import (
        _estimate_joker_value, _api_key_to_name, MUST_BUY_JOKERS,
    )
    from data.jokers import JOKERS

    weakest_idx = -1
    weakest_val = float("inf")
    exclude = exclude_indices or set()

    for i, j in enumerate(jokers_raw):
        if i in exclude:
            continue
        mod = j.get("modifier", {})
        if isinstance(mod, dict):
            if mod.get("eternal", False):
                continue
            if mod.get("edition", "") == "NEGATIVE":
                continue

        # Never sell MUST_BUY jokers
        jk = j.get("joker_key", "") or j.get("key", "")
        name = _api_key_to_name(jk)
        if name in MUST_BUY_JOKERS:
            continue

        # Never sell retrigger or copy jokers
        if name and name in JOKERS:
            schema = JOKERS[name]
            if schema.get("retrigger_effect") or schema.get("copy"):
                continue

        val = _estimate_joker_value(j, jokers_raw, raw_state)
        if val < weakest_val:
            weakest_val = val
            weakest_idx = i

    return weakest_idx, weakest_val


# ============================================================
# Training Orchestrator
# ============================================================

class Trainer:
    """Main training loop.

    Connects to BalatroBot, plays the game, collects transitions,
    and runs PPO updates.
    """

    def __init__(self, config: TrainConfig, checkpoint_path: Optional[str] = None):
        self.config = config

        # Create components
        self.network = BalatronNetwork()
        self.ppo = PPOTrainer(self.network, config.to_ppo_config())
        self.game = GameStateManager()
        self.reward_calc = RewardCalculator(phase=config.phase)
        self.action_decoder = ActionDecoder()
        self.episode_tracker = EpisodeTracker()
        self.joker_logger = JokerOrderLogger()
        self.recorder = RunRecorder(enabled=config.record_wins)

        # Training state
        self.global_step = 0
        self.num_updates = 0
        self.start_time = 0.0
        self._consecutive_api_failures = 0
        self._balatro_process: Optional[subprocess.Popen] = None

        # Load checkpoint if provided
        if checkpoint_path:
            self.ppo.load_checkpoint(checkpoint_path)
            self.global_step = self.ppo.total_steps
            self.num_updates = self.ppo.total_updates
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"  Resuming from step {self.global_step}, update {self.num_updates}")

        # Ensure checkpoint directory exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    async def _restart_balatro(self):
        """Kill Balatro and relaunch it after a crash.

        Kills any running Balatro.exe process, waits briefly, then
        relaunches via start_balatro.bat. Reconnects the API session.
        """
        print("[CRASH-RECOVERY] Balatro appears stuck — restarting...", flush=True)

        # Kill existing Balatro process
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "Balatro.exe"],
                capture_output=True, timeout=10,
            )
        except Exception as e:
            print(f"[CRASH-RECOVERY] taskkill failed: {e}", flush=True)

        await asyncio.sleep(3.0)  # Wait for process to fully die

        # Relaunch via start_balatro.bat (runs in background)
        bat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "start_balatro.bat")
        if not os.path.exists(bat_path):
            bat_path = r"C:\Users\jarms\repos\balatron\start_balatro.bat"

        try:
            self._balatro_process = subprocess.Popen(
                ["cmd", "/c", bat_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[CRASH-RECOVERY] Relaunched Balatro (PID {self._balatro_process.pid})", flush=True)
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
                await self.game.disconnect()
                await self.game.connect()
                raw = await self.game.fetch_gamestate()
                state = raw.get("state", "")
                print(f"[CRASH-RECOVERY] API responded — state={state}", flush=True)

                # If we're at menu, start a new run
                if state == "MENU":
                    seed = ''.join(random.choices(string.ascii_uppercase, k=8))
                    await self.game.execute_action(
                        "start", {"deck": "RED", "stake": "WHITE", "seed": seed}
                    )
                    await asyncio.sleep(0.5)

                self._consecutive_api_failures = 0
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

        # Connect to BalatroBot
        await self.game.connect()
        print("Connected to BalatroBot API")

        try:
            while self.global_step < cfg.total_timesteps:
                # Collect rollout
                last_value, last_done = await self._collect_rollout()

                # PPO update
                metrics = self.ppo.update(last_value, last_done)
                self.num_updates += 1

                # LR annealing
                if cfg.anneal_lr:
                    frac = 1.0 - self.global_step / cfg.total_timesteps
                    new_lr = cfg.learning_rate * frac
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
            self.recorder.cleanup()
            await self.game.disconnect()
            print("Disconnected from BalatroBot")

        self._print_summary()

    async def _collect_rollout(self) -> tuple[float, bool]:
        """Collect rollout_steps transitions by playing the game.

        Returns:
            (last_value, last_done) for GAE bootstrapping
        """
        cfg = self.config
        prev_raw = None
        last_done = False

        for step in range(cfg.rollout_steps):

            # Get current game state
            raw_state = await self._get_actionable_state()

            if raw_state is None:
                # Couldn't get an actionable state — treat as terminal
                self.ppo.store_transition(
                    np.zeros(STATE_VECTOR_SIZE, dtype=np.float32),
                    np.zeros(14, dtype=np.float32),
                    0.0, 0.0, 0.0, True,
                    np.zeros(ACTION_HEAD_SIZE, dtype=np.float32),
                    "GAME_OVER",
                )
                self.global_step += 1
                last_done = True
                continue

            game_state_name = raw_state.get("state", "")

            # Detect round completion: SELECTING_HAND -> SHOP means blind was beaten
            prev_state_name = prev_raw.get("state", "") if prev_raw else ""
            if prev_state_name == "SELECTING_HAND" and game_state_name == "SHOP":
                self.joker_logger.round_end()

            # Detect win via API 'won' flag (endless mode auto-continues)
            # The Lua mod auto-dismisses the win screen, so GAME_OVER may
            # never fire. Record the win when we first see won=True.
            #
            # IMPORTANT: Only trust 'won' when we're in a safe state (SHOP,
            # BLIND_SELECT, ROUND_EVAL) — NOT during SELECTING_HAND where
            # the bot could still die on the boss blind.  The API may set
            # won=True after beating ante 8's small/big blind, before the
            # boss is beaten.
            api_won = raw_state.get("won", False)
            safe_won_states = {"SHOP", "BLIND_SELECT", "ROUND_EVAL", "SELECTING_HAND", "GAME_OVER"}
            if (api_won and not getattr(self, '_win_recorded', False)
                    and game_state_name in safe_won_states):
                self._win_recorded = True
                ante = raw_state.get("ante_num", 1)
                print(f"[WIN] Won flag detected in state={game_state_name} "
                      f"ante={ante}", flush=True)
                self.episode_tracker.end_episode(True, raw_state)
                # DON'T end recording here — let GAME_OVER handle it so the
                # winning hand, scoring animation, and win screen get captured

            # Handle GAME_OVER — end episode, start new one
            if game_state_name == "GAME_OVER":
                self.joker_logger.round_end()  # flush any pending round data
                ante = raw_state.get("ante_num", 1)
                api_won_flag = raw_state.get("won", False)
                already_recorded = getattr(self, '_win_recorded', False)
                # Win if: ante >= 8 AND api says won, OR ante > 8 (survived past 8)
                won = (ante >= 8 and api_won_flag) or ante > 8 or already_recorded
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

                # Compute terminal reward
                reward = self.reward_calc.step(prev_raw, raw_state)
                self.episode_tracker.step(reward, ante, raw_state)
                # Only call end_episode if win wasn't already recorded
                if not getattr(self, '_win_recorded', False):
                    self.episode_tracker.end_episode(won, raw_state)
                # End recording — save if won, discard if lost
                # Wait a moment on wins so the win screen / final scoring
                # animation gets captured before ffmpeg stops
                if won:
                    await asyncio.sleep(3.0)
                self.recorder.end_run(
                    won=won,
                    ante_reached=ante,
                    final_score=int(raw_state.get("round", {}).get("chips", 0)),
                    checkpoint_path=os.path.join(
                        self.config.checkpoint_dir,
                        f"balatron_phase{self.config.phase}_step{self.global_step}.pt",
                    ),
                    total_steps=self.global_step,
                )

                self.reward_calc.reset()
                self.game.reset()
                self._win_recorded = False  # reset for next run
                prev_raw = None

                # Store terminal transition
                state_vec = np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)
                self.ppo.store_transition(
                    state_vec, np.zeros(14, dtype=np.float32),
                    0.0, reward, 0.0, True,
                    np.zeros(ACTION_HEAD_SIZE, dtype=np.float32),
                    "GAME_OVER",
                )
                self.global_step += 1
                last_done = True

                # Navigate back to menu so next poll doesn't see GAME_OVER again
                try:
                    await self.game.execute_action("menu")
                except Exception:
                    pass

                continue

            # Get state vector from game manager
            try:
                state_vec = await self.game.step()
            except Exception:
                # API dropped (e.g. menu escape killed server) — wait and retry
                await asyncio.sleep(0.3)
                continue

            # Build action mask
            action_mask = build_action_mask(raw_state)

            # Expose reroll cap to mask — prevent NN from selecting reroll
            # when the hard guard will block it anyway
            if game_state_name == "SHOP":
                from environment.action_space import ACTION_REROLL
                rerolls = getattr(self, '_shop_rerolls', 0)
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
                # No valid actions — skip this step
                prev_raw = raw_state
                continue

            # Get head index
            head_idx = get_head_index(game_state_name)

            # Network forward pass
            with torch.no_grad():
                state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

                if cfg.device == "cuda" and torch.cuda.is_available():
                    state_t = state_t.cuda()
                    mask_t = mask_t.cuda()

                action_t, log_prob_t, _, value_t = self.network.get_action_and_value(
                    state_t, head_idx, mask_t
                )

            action_np = action_t[0].cpu().numpy()
            log_prob = log_prob_t[0].cpu().item()
            value = value_t[0].cpu().item()

            # Decode sampled action tensor to API call
            api_method, api_params = self._action_to_api_call(
                action_np, raw_state
            )

            # Shop spin detection: if the NN keeps choosing no-op actions
            # (gamestate) while in SHOP, try to force a buy before leaving
            if game_state_name == "SHOP":
                if api_method == "gamestate":
                    self._shop_noop_count = getattr(self, '_shop_noop_count', 0) + 1
                    noop_action_type = int(action_np[0])
                    noop_target = int(action_np[13])
                    noop_money = raw_state.get("money", 0)
                    if self._shop_noop_count <= 3:
                        mask_val = action_mask[noop_action_type] if noop_action_type < len(action_mask) else -1
                        print(f"[SHOP-NOOP] action_type={noop_action_type} "
                              f"target={noop_target} money=${noop_money} "
                              f"mask[{noop_action_type}]={mask_val:.4f} "
                              f"(noop #{self._shop_noop_count})", flush=True)
                    if self._shop_noop_count >= 3:
                        # Before forcing leave, try to buy the best affordable joker
                        from environment.action_space import (
                            _is_joker_card as _noop_ij,
                        )
                        # Try to buy ANY affordable card in shop (joker or consumable)
                        noop_shop = raw_state.get("shop", {}).get("cards", [])
                        noop_jokers = raw_state.get("jokers", {}).get("cards", [])
                        noop_jcount = len(noop_jokers)
                        noop_jlimit = raw_state.get("jokers", {}).get("limit", 5)
                        noop_cons = raw_state.get("consumables", {}).get("cards", [])
                        noop_climit = raw_state.get("consumables", {}).get("limit", 2)
                        best_noop_buy = -1
                        for ni, nc in enumerate(noop_shop):
                            nc_key = nc.get("key", "")
                            nc_set = nc.get("set", "").upper()
                            nc_cost = nc.get("cost", {}).get("buy", 999)
                            print(f"[SHOP-NOOP-SCAN] slot {ni}: "
                                  f"key={nc_key} set={nc_set} cost=${nc_cost}",
                                  flush=True)
                            if nc_cost > noop_money:
                                continue
                            if _noop_ij(nc):
                                if noop_jcount < noop_jlimit and best_noop_buy < 0:
                                    best_noop_buy = ni
                            elif nc_set in ("PLANET", "TAROT") and len(noop_cons) < noop_climit:
                                if best_noop_buy < 0:
                                    best_noop_buy = ni
                        if best_noop_buy >= 0:
                            bk = noop_shop[best_noop_buy].get("key", "?")
                            print(f"[SHOP] NOOP recovery: force-buying {bk} "
                                  f"(slot {best_noop_buy})", flush=True)
                            api_method = "buy"
                            api_params = {"card": best_noop_buy}
                            self._shop_noop_count = 0
                        elif self._shop_noop_count > 5:
                            print(f"[SHOP] Forcing next_round after "
                                  f"{self._shop_noop_count} no-ops", flush=True)
                            api_method = "next_round"
                            api_params = None
                            self._shop_noop_count = 0
                else:
                    self._shop_noop_count = 0
            else:
                self._shop_noop_count = 0

            # Stuck-state detector: if we keep getting the same state,
            # escalate from force-play → restart
            state_fingerprint = (
                game_state_name,
                str(raw_state.get("hand", {}).get("cards", [])),
                raw_state.get("round", {}).get("hands_left", -1),
                raw_state.get("round", {}).get("discards_left", -1),
            )
            prev_fingerprint = getattr(self, '_prev_state_fingerprint', None)
            if state_fingerprint == prev_fingerprint:
                self._state_stuck_count = getattr(self, '_state_stuck_count', 0) + 1
                stuck = self._state_stuck_count

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
                    self._pending_hand_rearrange = None
                    self._pending_rearrange = None

                elif stuck >= 4 and game_state_name == "SHOP":
                    # Shop loop — force leave instead of restarting
                    print(f"[STUCK] Shop unchanged {stuck}x — "
                          f"forcing next_round", flush=True)
                    api_method = "next_round"
                    api_params = None
                    self._state_stuck_count = 0

                elif stuck >= 8:
                    # Buttons are permanently broken — restart Balatro
                    print(f"[STUCK] State unchanged {stuck}x — "
                          f"game desynced, restarting Balatro", flush=True)
                    self._state_stuck_count = 0
                    self._prev_state_fingerprint = None
                    await self._restart_balatro()
                    self.reward_calc.reset()
                    self.game.reset()
                    continue
            else:
                self._state_stuck_count = 0
            self._prev_state_fingerprint = state_fingerprint

            # Reorder jokers before playing a hand (using actual cards)
            _intended_joker_order = None
            if api_method == "play" and getattr(self, '_pending_rearrange', None) is not None:
                hand_cards, deck_cards = self._pending_rearrange
                self._pending_rearrange = None
                try:
                    _intended_joker_order = await self._auto_rearrange_jokers(
                        raw_state, hand_cards=hand_cards, deck_cards=deck_cards
                    )
                except Exception as e:
                    err_msg = f"Pre-play: {e}"
                    print(f"[WARN] Joker rearrange failed: {e}")
                    self.joker_logger.log_rearrange_failure("pre-play", str(e))

            # Rearrange hand cards for optimal scoring order (face card first
            # for Photograph, highest chips first for Hanging Chad, etc.)
            if api_method == "play" and getattr(self, '_pending_hand_rearrange', None) is not None:
                new_hand_order = self._pending_hand_rearrange
                fallback_cards = getattr(self, '_pending_hand_rearrange_fallback', None)
                self._pending_hand_rearrange = None
                self._pending_hand_rearrange_fallback = None
                try:
                    await self.game.execute_action("rearrange", {"hand": new_hand_order})
                except Exception as e:
                    print(f"[WARN] Hand rearrange failed: {e}")
                    # Rearrange failed — play cards at original indices instead
                    # of the post-rearrange 0..N-1 indices
                    if fallback_cards and api_params:
                        api_params["cards"] = fallback_cards

            # Log play details for joker order review
            if api_method == "play":
                self._log_play_for_joker_order(
                    raw_state, _intended_joker_order
                )

            # Execute action (with retry on UI timing errors)
            action_succeeded = True
            max_retries = 3
            for _attempt in range(max_retries):
                try:
                    await self.game.execute_action(api_method, api_params)
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

            # Compute reward — pass cached joker contributions to avoid
            # redundant leave-one-out scoring in reward._sync_state
            _cached_contribs = getattr(self.game, '_joker_eval_cache', {}).get('contributions')
            reward = self.reward_calc.step(
                prev_raw, raw_state,
                action=api_method,
                action_succeeded=action_succeeded,
                scaling_values=self._get_scaling_snapshot(),
                joker_contributions=_cached_contribs,
                skip_economy=getattr(self, '_auto_action_this_step', False),
            )
            self._auto_action_this_step = False

            # Track episode
            ante = raw_state.get("ante_num", 1)
            self.episode_tracker.step(reward, ante, raw_state)

            # Store transition
            self.ppo.store_transition(
                state_vec, action_np, log_prob, reward, value,
                False, action_mask, game_state_name,
            )

            prev_raw = raw_state
            self.global_step += 1
            last_done = False

        # Bootstrap value for GAE
        last_value = 0.0
        if not last_done:
            try:
                raw_state = await self.game.fetch_gamestate()
                state_vec = await self.game.step()
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

    async def _get_actionable_state(self) -> Optional[dict]:
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
                raw = await self.game.fetch_gamestate()
                consecutive_fetch_fails = 0
            except Exception:
                consecutive_fetch_fails += 1
                if consecutive_fetch_fails >= 20:
                    print(f"[CRASH-RECOVERY] {consecutive_fetch_fails} consecutive fetch failures — triggering restart", flush=True)
                    await self._restart_balatro()
                    self.reward_calc.reset()
                    self.game.reset()
                    return None
                await asyncio.sleep(cfg.api_poll_delay)
                continue

            state = raw.get("state", "")

            # Actionable states — the agent makes decisions here
            if state in ("SELECTING_HAND", "SHOP", "GAME_OVER",
                          "BLIND_SELECT"):
                self._consecutive_api_failures = 0  # API is alive
                pack_attempts = 0  # reset pack retry counter
                unknown_state_count = 0  # reset stuck counter
                self._round_eval_count = 0  # reset win-screen detector
                if state == "SHOP":
                    self._shop_rerolls = 0
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
                                await self.game.execute_action("skip")
                            except Exception:
                                pass
                            await asyncio.sleep(0.2)
                            continue
                if state == "SELECTING_HAND":
                    # Log round start for joker order tracking
                    blinds_data = raw.get("blinds", {})
                    current_blind_name = ""
                    current_blind_score = 0.0
                    if isinstance(blinds_data, dict):
                        for b in blinds_data.values():
                            if isinstance(b, dict) and b.get("status") == "CURRENT":
                                current_blind_name = b.get("name", "")
                                current_blind_score = b.get("score", 0)
                                break
                    joker_cards = raw.get("jokers", {}).get("cards", [])
                    joker_keys = [j.get("key", "") for j in joker_cards]
                    self.joker_logger.round_start(
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
                        if len(joker_cards_vl) > 1:
                            # Check if cards are still debuffed (first hand only)
                            round_chips = raw.get("round", {}).get("chips", 0)
                            hands_played = raw.get("round", {}).get("hands_left", 0)
                            total_hands = raw.get("round", {}).get("hands", 0)
                            if round_chips == 0 and hands_played == total_hands:
                                # Find weakest joker by delta
                                from environment.action_space import (
                                    _estimate_joker_value as _vl_est,
                                    _joker_is_scoring as _vl_scoring,
                                    _api_key_to_name as _vl_name,
                                )
                                weakest_idx = -1
                                weakest_delta = float('inf')
                                for vi, vj in enumerate(joker_cards_vl):
                                    vmod = vj.get("modifier", {})
                                    if isinstance(vmod, dict) and vmod.get("eternal", False):
                                        continue
                                    ved = vmod.get("edition", "") if isinstance(vmod, dict) else ""
                                    if ved == "NEGATIVE":
                                        continue  # never sell negative
                                    vdelta = _vl_est(vj, joker_cards_vl, raw)
                                    if vdelta < weakest_delta:
                                        weakest_delta = vdelta
                                        weakest_idx = vi
                                if weakest_idx >= 0:
                                    vk = joker_cards_vl[weakest_idx].get("key", "")
                                    vn = _vl_name(vk) or vk
                                    print(f"[BOSS] Verdant Leaf — selling {vn} "
                                          f"(idx {weakest_idx}) to un-debuff cards",
                                          flush=True)
                                    try:
                                        await self.game.execute_action(
                                            "sell", {"joker": weakest_idx})
                                        await asyncio.sleep(0.3)
                                        raw = await self._fetch_state()
                                    except Exception as e:
                                        print(f"[BOSS] Verdant Leaf sell failed: {e}",
                                              flush=True)

                    # Amber Acorn: jokers are shuffled — re-arrange immediately
                    if current_blind_name == "Amber Acorn":
                        print(f"[BOSS] Amber Acorn — re-arranging shuffled jokers",
                              flush=True)

                    # Auto-rearrange jokers at start of each hand round
                    await self._auto_rearrange_jokers(raw)
                    # Auto-use consumables (Planet cards, well-timed Tarots, etc.)
                    pre_auto_money = raw.get("money", 0)
                    raw = await self._auto_use_consumables(raw)
                    post_auto_money = raw.get("money", 0)
                    self._auto_action_this_step = (pre_auto_money != post_auto_money)
                elif state == "SHOP":
                    # Auto-use non-targeting consumables in shop
                    # (Hermit for money doubling, Temperance, Wheel, etc.)
                    pre_auto_money = raw.get("money", 0)
                    raw = await self._auto_use_consumables(raw)
                    # Auto-buy vouchers the NN doesn't understand
                    raw = await self._auto_buy_vouchers(raw)
                    post_auto_money = raw.get("money", 0)
                    self._auto_action_this_step = (pre_auto_money != post_auto_money)
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
                    # print(f"PACK BAILOUT after {pack_attempts} attempts — force skipping", flush=True)
                    try:
                        await self.game.execute_action("pack", {"skip": True})
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                    # If we've bailed out too many times, game is probably broken
                    if pack_attempts > 25:
                        print(f"⚠️  PACK STUCK beyond recovery — triggering restart", flush=True)
                        await self._restart_balatro()
                        self.reward_calc.reset()
                        self.game.reset()
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

                # ── Pick best card from pack ──
                pick_idx = 0

                # ── SOUL CARD CHECK (all pack types) ──
                # The Soul creates a Legendary joker — ALWAYS pick it.
                # If joker slots are full, sell weakest to make room.
                soul_idx = -1
                for sc_idx, sc in enumerate(pack_cards):
                    if sc.get("key", "") == "c_the_soul":
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
                                await self.game.execute_action("sell", {"joker": weakest_soul_idx})
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
                        await self.game.execute_action("pack", {"card": pick_idx})
                    except Exception as e:
                        print(f"[PACK] Soul pick failed: {e}", flush=True)
                    await asyncio.sleep(cfg.api_poll_delay)
                    continue

                if card_set in ("PLANET", "SPECTRAL"):
                    # Joker-aware planet selection: pick the planet that gives
                    # the biggest marginal score increase with current jokers
                    jokers_for_planet = raw.get("jokers", {}).get("cards", [])
                    pick_idx = pick_best_planet(pack_cards, jokers_for_planet, raw)

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
                            await self.game.execute_action("pack", pack_params)
                        except Exception as e:
                            pass  # print(f"PACK tarot {pick_idx} failed: {e}")
                            # Re-check state before skipping — the select may have
                            # partially succeeded and the pack is already closing.
                            await asyncio.sleep(0.5)
                            try:
                                recheck = await self.game.fetch_gamestate()
                                if recheck.get("state", "") == "SMODS_BOOSTER_OPENED":
                                    await self.game.execute_action("pack", {"skip": True})
                            except Exception:
                                pass
                        await asyncio.sleep(cfg.api_poll_delay)
                        continue
                    else:
                        # No worthwhile tarot — skip
                        try:
                            await self.game.execute_action("pack", {"skip": True})
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

                        # Find the weakest current joker (lowest score contribution)
                        worst_idx = 0
                        worst_score_without = 0
                        for j_idx in range(len(current_jokers)):
                            # Skip eternal jokers — can't sell them
                            mod = current_jokers[j_idx].get("modifier", {})
                            if isinstance(mod, dict) and mod.get("eternal", False):
                                continue
                            jokers_without = [j for i, j in enumerate(current_jokers) if i != j_idx]
                            score_without = estimate_score_for_hand_type(jokers_without, raw)
                            if score_without > worst_score_without or j_idx == 0:
                                worst_score_without = score_without
                                worst_idx = j_idx

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

                        if best_swap is not None:
                            # Sell the weakest joker then IMMEDIATELY pick
                            # the replacement in the same iteration.
                            try:
                                worst_name = current_jokers[worst_idx].get("label") or current_jokers[worst_idx].get("key", "?")
                                print(f"[PACK] selling joker {worst_idx} ({worst_name}) "
                                      f"to swap for pack card {best_swap}", flush=True)
                                await self.game.execute_action("sell", {"joker": worst_idx})
                                await asyncio.sleep(0.5)
                                # Verify pack is still open before picking
                                recheck = await self.game.fetch_gamestate()
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
                                await self.game.execute_action("pack", {"skip": True})
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
                        for targets in target_attempts:
                            try:
                                await self.game.execute_action("pack", {"card": try_idx, "targets": targets})
                                selected = True
                                break
                            except Exception as e:
                                pass  # print(f"PACK card {try_idx} targets={targets} failed: {e}")
                    else:
                        try:
                            await self.game.execute_action("pack", {"card": try_idx})
                            selected = True
                        except Exception as e:
                            pass  # print(f"PACK card {try_idx} failed: {e}")

                # Last resort: skip the pack (re-check state first — a card
                # select may have partially succeeded and closed the pack)
                if not selected:
                    try:
                        recheck = await self.game.fetch_gamestate()
                        if recheck.get("state", "") == "SMODS_BOOSTER_OPENED":
                            await self.game.execute_action("pack", {"skip": True})
                    except Exception as e:
                        pass  # print(f"PACK skip failed (attempt {pack_attempts}): {e}")
                        await asyncio.sleep(0.5)

                await asyncio.sleep(cfg.api_poll_delay)
                continue

            # ROUND_EVAL — auto cash out
            if state == "ROUND_EVAL":
                round_eval_count = getattr(self, '_round_eval_count', 0) + 1
                self._round_eval_count = round_eval_count
                try:
                    await self.game.execute_action("cash_out")
                    self._round_eval_count = 0
                except Exception:
                    pass

                # If stuck at ROUND_EVAL for many polls, likely the win overlay
                # is blocking (game paused after beating ante 8). Go to menu.
                if round_eval_count > 10:
                    ante = raw.get("ante_num", 1)
                    print(f"[WIN-RECOVERY] Stuck at ROUND_EVAL (ante={ante}, "
                          f"polls={round_eval_count}) — going to menu", flush=True)
                    self._round_eval_count = 0
                    try:
                        await self.game.execute_action("menu")
                    except Exception:
                        # Menu may also fail if paused — try restart
                        print("[WIN-RECOVERY] Menu failed — triggering restart", flush=True)
                        await self._restart_balatro()
                        self.reward_calc.reset()
                        self.game.reset()
                        return None
                    await asyncio.sleep(1.0)

                await asyncio.sleep(cfg.api_poll_delay)
                continue

            # MENU — start a new run with random seed
            if state == "MENU":
                # Start recording the new run
                self.recorder.start_run()
                seed = ''.join(random.choices(string.ascii_uppercase, k=8))
                try:
                    await self.game.execute_action(
                        "start", {"deck": "RED", "stake": "WHITE", "seed": seed}
                    )
                except Exception:
                    pass
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
            }
            if state in TRANSIENT_STATES:
                # These resolve on their own — give extra time
                await asyncio.sleep(0.5)
                # Only count toward stuck if we've been in transient for a while
                unknown_state_count += 1
                if unknown_state_count > 60:  # 30+ seconds in transient
                    print(f"[CRASH-RECOVERY] Stuck in transient state '{state}' for "
                          f"{unknown_state_count} polls — triggering restart", flush=True)
                    await self._restart_balatro()
                    self.reward_calc.reset()
                    self.game.reset()
                    return None
                continue

            # Unknown/unrecognized state — wait briefly
            unknown_state_count += 1
            if unknown_state_count == 1 or unknown_state_count % 10 == 0:
                print(f"[UNKNOWN-STATE] '{state}' (poll #{unknown_state_count})", flush=True)
            if unknown_state_count > 30:
                print(f"[CRASH-RECOVERY] Stuck in unknown state '{state}' for {unknown_state_count} polls — triggering restart", flush=True)
                await self._restart_balatro()
                self.reward_calc.reset()
                self.game.reset()
                return None
            await asyncio.sleep(cfg.api_poll_delay)

        # All poll attempts exhausted — Balatro is likely crashed
        self._consecutive_api_failures += 1
        if self._consecutive_api_failures >= 3:
            print(f"[CRASH-RECOVERY] {self._consecutive_api_failures} consecutive poll failures — triggering restart", flush=True)
            await self._restart_balatro()
            # Reset episode state after restart
            self.reward_calc.reset()
            self.game.reset()
        return None  # Timed out

    def _action_to_api_call(self, action: np.ndarray,
                            raw_state: dict) -> tuple[str, Optional[dict]]:
        """Convert sampled action tensor to BalatroBot API call.

        Action format: [type(1), cards(12), target(1)]
        All card references are 0-based positional indices.
        """
        action_type = int(action[0])
        card_selections = action[1:13]
        target_idx = int(action[13])

        # Play or Discard — strategic advisor decides based on full math:
        # blind target, joker synergies, deck composition, draw probabilities
        if action_type in (0, 1):
            hand_cards = raw_state.get("hand", {}).get("cards", [])
            deck_cards = raw_state.get("cards", {}).get("cards", [])
            jokers_raw = raw_state.get("jokers", {}).get("cards", [])

            # Inject scaling tracker values so compute_joker_scoring knows
            # the accumulated values for scaling jokers (Square, Ride the Bus, etc.)
            self.game.inject_scaling_values(jokers_raw)

            try:
                plan = plan_optimal_action(hand_cards, deck_cards, jokers_raw, raw_state)
                action = plan["action"]
                cards = plan["cards"]

                if action == "play":
                    cards = cards[:5]
                    if not cards and hand_cards:
                        cards = [0]
                    optimal_order = optimize_play_order(cards, hand_cards, jokers_raw)
                    # Rearrange hand so played cards appear in optimal scoring
                    # order (face card first for Photograph, etc.).
                    # Balatro scores left-to-right by hand position, so we must
                    # physically reorder the hand before playing.
                    if optimal_order != sorted(optimal_order):
                        played_set = set(optimal_order)
                        non_played = [i for i in range(len(hand_cards)) if i not in played_set]
                        # New hand order: optimal play order first, then non-played
                        new_hand_order = list(optimal_order) + non_played
                        # Store rearrange request — executed before play in the main loop
                        self._pending_hand_rearrange = new_hand_order
                        # Store original indices as fallback if rearrange fails
                        self._pending_hand_rearrange_fallback = list(optimal_order)
                        # After rearrange, played cards will be at indices 0..N-1
                        cards = list(range(len(optimal_order)))
                    else:
                        cards = optimal_order
                    # Store hand/deck cards for joker reordering before play
                    self._pending_rearrange = (hand_cards, deck_cards)
                    return "play", {"cards": cards}
                else:
                    cards = list(cards)[:5]
                    if not cards and hand_cards:
                        cards = [0]
                    return "discard", {"cards": cards}

            except Exception as exc:
                import traceback
                print(f"[WARN] plan_optimal_action CRASHED: {exc}", flush=True)
                traceback.print_exc()
                # Fallback: play or discard based on network's original choice
                if action_type == 0:
                    try:
                        top_hands = find_best_hands(hand_cards, jokers_raw, raw_state, top_n=1)
                        selected = list(top_hands[0]["card_indices"])[:5] if top_hands else [0]
                    except Exception:
                        selected = [0] if hand_cards else []
                    selected = optimize_play_order(selected, hand_cards, jokers_raw)
                    return "play", {"cards": selected}
                else:
                    try:
                        advice = find_best_discard(hand_cards, deck_cards, jokers_raw, raw_state)
                        discard_indices = list(advice["discard_indices"])[:5]
                    except Exception:
                        discard_indices = [0] if hand_cards else []
                    return "discard", {"cards": discard_indices}

        # ================================================================
        # SHOP / BLIND-SELECT ACTIONS — heuristic-guarded
        # The RL network picks the action type, but we verify every
        # buy/sell/reroll against the scoring engine before executing.
        # Bad decisions are redirected to safer alternatives.
        # ================================================================

        game_state = raw_state.get("state", "")

        # Actions 2-7, 11-13 are shop-only. Block them outside SHOP.
        # Actions 8 (use consumable), 9 (select blind), 10 (skip blind)
        # can work outside SHOP and must NOT be blocked here.
        SHOP_ONLY_ACTIONS = {2, 3, 4, 5, 6, 7, 11, 12, 13}
        if action_type in SHOP_ONLY_ACTIONS and game_state != "SHOP":
            return "gamestate", None

        jokers_raw = raw_state.get("jokers", {}).get("cards", [])
        joker_count = len(jokers_raw)
        joker_limit = raw_state.get("jokers", {}).get("limit", 5)
        money = raw_state.get("money", 0)
        current_score = estimate_score_for_hand_type(jokers_raw, raw_state)

        # ── Pending upgrade buy: after selling a joker for an upgrade,
        # force-buy the target on the very next shop action ──
        pending_buy = getattr(self, '_pending_upgrade_buy', None)
        if pending_buy is not None and game_state == "SHOP":
            self._pending_upgrade_buy = None
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            p_idx = pending_buy
            if p_idx < len(shop_cards):
                p_cost = shop_cards[p_idx].get("cost", {}).get("buy", 999)
                if p_cost <= money and joker_count < joker_limit:
                    print(f"[SHOP] Completing upgrade: buying shop card {p_idx} "
                          f"(${p_cost})", flush=True)
                    return "buy", {"card": p_idx}
                else:
                    print(f"[SHOP] Pending upgrade buy failed: "
                          f"cost=${p_cost} money=${money} "
                          f"slots={joker_count}/{joker_limit}", flush=True)

        # Buy joker (target 0-2 = shop card index)
        if action_type == 2:
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            if target_idx >= len(shop_cards):
                return "gamestate", None  # invalid index

            card = shop_cards[target_idx]
            from environment.action_space import (
                _estimate_joker_value, _joker_is_scoring, _joker_sell_value,
                MUST_BUY_JOKERS, BAD_JOKERS, _api_key_to_name, _is_non_joker_card,
            )
            cost = card.get("cost", {}).get("buy", 999)
            if cost > money:
                return "gamestate", None  # can't afford

            # Non-joker cards (planets/tarots/Hermit) — buy directly if mask allowed it
            if _is_non_joker_card(card):
                card_key = card.get("key", "")
                card_set = card.get("set", "").upper()
                cons_count = len(raw_state.get("consumables", {}).get("cards", []))
                cons_limit = raw_state.get("consumables", {}).get("limit", 2)
                if cons_count >= cons_limit:
                    return "gamestate", None  # no consumable slots
                card_label = card.get("label") or card_key
                print(f"[SHOP] Buying consumable: {card_label} "
                      f"(${cost}, {card_set})", flush=True)
                return "buy", {"card": target_idx}

            joker_key = card.get("joker_key", "") or card.get("key", "")
            name = _api_key_to_name(joker_key)

            # Always buy must-buy jokers (Blueprint, Brainstorm, etc.)
            if name in MUST_BUY_JOKERS:
                if joker_count >= joker_limit:
                    # Slots full — sell weakest to make room
                    from environment.action_space import _joker_sell_value as _jsv2
                    weakest_i, _ = _find_weakest_sellable_joker(
                        jokers_raw, raw_state)
                    if weakest_i >= 0:
                        sell_price = _jsv2(jokers_raw[weakest_i])
                        if cost <= money + sell_price:
                            weak_name = jokers_raw[weakest_i].get("label", "?")
                            print(f"[SHOP] Selling {weak_name} (idx {weakest_i}) "
                                  f"to buy MUST-BUY {name}", flush=True)
                            self._pending_upgrade_buy = target_idx
                            return "sell", {"joker": weakest_i}
                    return "gamestate", None  # can't make room
                return "buy", {"card": target_idx}

            # Never buy bad jokers
            if name in BAD_JOKERS:
                return "gamestate", None

            # Check edition (negative bypasses slot limit, poly/holo = high value)
            from environment.action_space import HIGH_VALUE_JOKERS
            shop_mod = card.get("modifier", {})
            shop_edition = shop_mod.get("edition", "") if isinstance(shop_mod, dict) else ""
            is_negative = shop_edition == "NEGATIVE"
            is_high_value = (name in HIGH_VALUE_JOKERS or
                             shop_edition in ("POLYCHROME", "HOLO", "NEGATIVE"))

            if joker_count < joker_limit or is_negative:
                # Open slot — scan ALL shop jokers and buy the best one,
                # not just whichever one the NN happened to pick.
                shop_cards_all = raw_state.get("shop", {}).get("cards", [])
                best_idx = -1
                best_delta = -999999
                best_is_high = False
                best_is_scoring = False
                best_name = ""
                for si, sc in enumerate(shop_cards_all):
                    if _is_non_joker_card(sc):
                        continue
                    sc_cost = sc.get("cost", {}).get("buy", 999)
                    if sc_cost > money:
                        continue
                    sc_key = sc.get("joker_key", "") or sc.get("key", "")
                    sc_name = _api_key_to_name(sc_key)
                    if sc_name and sc_name in BAD_JOKERS:
                        continue
                    sc_mod = sc.get("modifier", {})
                    sc_ed = sc_mod.get("edition", "") if isinstance(sc_mod, dict) else ""
                    sc_high = (sc_name in HIGH_VALUE_JOKERS if sc_name else False) or \
                              sc_ed in ("POLYCHROME", "HOLO", "NEGATIVE")
                    sc_scoring = _joker_is_scoring(sc)
                    sc_delta = _estimate_joker_value(sc, jokers_raw, raw_state)
                    # Priority: must-buy > high-value > highest delta > scoring > economy
                    sc_priority = sc_delta
                    if sc_name and sc_name in MUST_BUY_JOKERS:
                        sc_priority = 999999
                    elif sc_high:
                        sc_priority = max(sc_priority, 10000)
                    elif sc_scoring and sc_delta <= 0:
                        sc_priority = max(sc_priority, 1)  # scoring beats economy
                    if sc_priority > best_delta:
                        best_delta = sc_priority
                        best_idx = si
                        best_is_high = sc_high
                        best_is_scoring = sc_scoring
                        best_name = sc_name or sc_key

                if best_idx >= 0 and (best_delta > 0 or best_is_high or best_is_scoring):
                    if best_idx != target_idx:
                        print(f"[SHOP] Redirecting buy → {best_name} "
                              f"(delta={best_delta:.0f}, better than NN pick)",
                              flush=True)
                    return "buy", {"card": best_idx}
                elif best_idx >= 0 and best_delta == 0:
                    # Economy joker, better than empty slot
                    print(f"[SHOP] Buying {best_name} (delta=0, "
                          f"scoring={best_is_scoring}) — filling empty slot",
                          flush=True)
                    return "buy", {"card": best_idx}
                else:
                    # Nothing worth buying
                    print(f"[SHOP] BLOCKED buy {name or joker_key} "
                          f"(no viable joker in shop)", flush=True)
                    return "gamestate", None
            else:
                # Slots full — only buy if it's a meaningful upgrade over weakest
                weakest_idx, _ = _find_weakest_sellable_joker(
                    jokers_raw, raw_state)

                if weakest_idx < 0:
                    return "gamestate", None  # all protected

                # Score with swap
                swapped = [j for i, j in enumerate(jokers_raw) if i != weakest_idx]
                swapped.append(card)
                swap_score = estimate_score_for_hand_type(swapped, raw_state)

                sell_price = _joker_sell_value(jokers_raw[weakest_idx])
                if cost > money + sell_price:
                    return "gamestate", None  # can't afford even after sell

                # High-value jokers use a lower swap threshold (5% vs 10%)
                swap_threshold = 1.05 if is_high_value else 1.1
                if swap_score > current_score * swap_threshold:
                    # Sell weakest joker first to make room, then buy on next step
                    weak_name = jokers_raw[weakest_idx].get("label", "?")
                    shop_name = name or joker_key
                    print(f"[SHOP] Selling {weak_name} (idx {weakest_idx}) "
                          f"to upgrade to {shop_name} "
                          f"(score {current_score:.0f} → {swap_score:.0f}, "
                          f"+{(swap_score/current_score - 1)*100:.0f}%)",
                          flush=True)
                    # Queue the buy for the very next step
                    self._pending_upgrade_buy = target_idx
                    return "sell", {"joker": weakest_idx}
                else:
                    return "gamestate", None

        # Buy voucher (target 3-4 -> voucher index 0-1)
        if action_type == 3:
            v_idx = target_idx - 3 if target_idx >= 3 else target_idx
            shop_vouchers = raw_state.get("vouchers", {}).get("cards", [])
            if v_idx < 0 or v_idx >= len(shop_vouchers):
                return "gamestate", None
            vcost = shop_vouchers[v_idx].get("cost", {}).get("buy", 999)
            if vcost > money:
                return "gamestate", None

            # Voucher tiers — must match auto_buy_vouchers logic
            CRITICAL_VOUCHERS = {
                "v_grabber", "v_nacho_tong",       # +1 hand
                "v_wasteful", "v_recyclomancy",     # +1 discard
                "v_paint_brush", "v_palette",       # +1 hand size
                "v_seed_money", "v_money_tree",     # interest cap
                "v_antimatter",                     # +1 joker slot
            }
            GOOD_VOUCHERS = {
                "v_hieroglyph", "v_petroglyph",     # -1 ante (nice but not critical)
                "v_overstock", "v_overstock_plus",  # +1 shop slot
                "v_directors_cut",                  # reroll boss blind
            }
            vkey = shop_vouchers[v_idx].get("key", "")

            if vkey not in CRITICAL_VOUCHERS and vkey not in GOOD_VOUCHERS:
                print(f"[SHOP] BLOCKED voucher buy ({vkey}, ${vcost}) — "
                      f"not valuable", flush=True)
                return "gamestate", None

            # Economy check: don't buy if it drops an interest tier
            # (critical vouchers bypass this)
            remaining_after = money - vcost
            current_tiers = min(money // 5, 5)
            after_tiers = min(max(remaining_after, 0) // 5, 5)
            if after_tiers < current_tiers and vkey not in CRITICAL_VOUCHERS:
                print(f"[SHOP] BLOCKED voucher buy ({vkey}, ${vcost}) — "
                      f"would lose interest tier ({current_tiers} → {after_tiers})",
                      flush=True)
                return "gamestate", None

            # GOOD (non-critical) vouchers need $10 cushion
            if vkey in GOOD_VOUCHERS and money < vcost + 10:
                print(f"[SHOP] BLOCKED voucher buy ({vkey}, ${vcost}) — "
                      f"not enough cushion (${money})", flush=True)
                return "gamestate", None

            return "buy", {"voucher": v_idx}

        # Buy pack (target 5-6 -> pack index 0-1)
        if action_type == 4:
            p_idx = target_idx - 5 if target_idx >= 5 else target_idx
            shop_packs = raw_state.get("packs", {}).get("cards", [])
            if p_idx < 0 or p_idx >= len(shop_packs):
                return "gamestate", None
            cost = shop_packs[p_idx].get("cost", {}).get("buy", 999)
            if cost > money:
                return "gamestate", None

            # Guard: if there's a buyable joker in shop, buy that instead
            pack_key = shop_packs[p_idx].get("key", "")
            is_free = cost <= 0
            if not is_free and joker_count < joker_limit:
                from environment.action_space import (
                    _joker_is_scoring as _pack_jis,
                    _api_key_to_name as _pack_aktn,
                    _is_non_joker_card as _pack_nonj,
                )
                shop_joker_cards = raw_state.get("shop", {}).get("cards", [])
                best_joker_idx = -1
                for sji, sjc in enumerate(shop_joker_cards):
                    if _pack_nonj(sjc):
                        continue
                    sj_cost = sjc.get("cost", {}).get("buy", 999)
                    if sj_cost > money:
                        continue
                    if _pack_jis(sjc):
                        best_joker_idx = sji
                        break
                if best_joker_idx >= 0:
                    sj_key = shop_joker_cards[best_joker_idx].get("key", "")
                    sj_name = _pack_aktn(sj_key) or sj_key
                    print(f"[SHOP] REDIRECT pack buy → joker buy: {sj_name}",
                          flush=True)
                    return "buy", {"card": best_joker_idx}

            # Block standard packs
            if "standard" in pack_key:
                return "gamestate", None

            return "buy", {"pack": p_idx}

        # Sell joker (target 7-11 -> joker index 0-4)
        if action_type == 5:
            j_idx = target_idx - 7 if target_idx >= 7 else target_idx
            if j_idx >= joker_count:
                return "gamestate", None  # invalid index

            # Never sell your only joker
            if joker_count <= 1:
                return "gamestate", None

            from environment.action_space import (
                _estimate_joker_value, _api_key_to_name,
                _joker_sell_value, _safe_modifier,
                MUST_BUY_JOKERS,
            )

            joker = jokers_raw[j_idx]
            mod = _safe_modifier(joker)

            # Hard blocks: eternal, negative, MUST_BUY jokers
            if mod.get("eternal", False):
                return "gamestate", None
            ed = mod.get("edition", "") if isinstance(mod, dict) else ""
            if ed == "NEGATIVE":
                return "gamestate", None
            joker_key = joker.get("joker_key", "") or joker.get("key", "")
            name = _api_key_to_name(joker_key) or joker_key
            if name in MUST_BUY_JOKERS:
                return "gamestate", None

            # Check if there's a better joker available in shop to replace
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            has_upgrade = False
            sell_price = _joker_sell_value(joker)
            for sc in shop_cards:
                from environment.action_space import _is_non_joker_card
                if _is_non_joker_card(sc):
                    continue
                sc_key = sc.get("joker_key", "") or sc.get("key", "")
                sc_name = _api_key_to_name(sc_key)
                if not sc_name:
                    continue
                sc_cost = sc.get("cost", {}).get("buy", 999)
                if sc_cost <= money + sell_price:
                    swapped = [j for i, j in enumerate(jokers_raw) if i != j_idx]
                    swapped.append(sc)
                    swap_score = estimate_score_for_hand_type(swapped, raw_state)
                    if swap_score > current_score * 1.05:
                        has_upgrade = True
                        print(f"[SHOP] Selling {name} (idx {j_idx}) — "
                              f"shop has {sc_name} as upgrade "
                              f"(score {current_score:.0f} -> {swap_score:.0f}, "
                              f"+{(swap_score/max(current_score,1)-1)*100:.0f}%)",
                              flush=True)
                        break

            if not has_upgrade:
                return "gamestate", None

            return "sell", {"joker": j_idx}

        # Sell consumable (target 12-13 -> consumable index 0-1)
        if action_type == 6:
            c_idx = target_idx - 12 if target_idx >= 12 else target_idx
            consumables = raw_state.get("consumables", {}).get("cards", [])
            if c_idx >= len(consumables):
                return "gamestate", None
            # Block selling planets (always useful) and hermit (money doubler)
            c_set = consumables[c_idx].get("set", "")
            c_key = consumables[c_idx].get("key", "")
            if c_set == "PLANET" or c_key in ("c_hermit", "c_temperance"):
                return "gamestate", None
            return "sell", {"consumable": c_idx}

        # Reroll — cap at 3 per shop normally, but uncap when desperate
        if action_type == 7:
            reroll_cost = raw_state.get("round", {}).get("reroll_cost", 5)
            if money < reroll_cost:
                return "gamestate", None

            # HARD GUARD: When there's a good buyable joker in shop, BUY IT
            # instead of rerolling. Don't just block — force the buy action.
            from environment.action_space import (
                _estimate_joker_value as _ejv, _joker_is_scoring as _jis,
                HIGH_VALUE_JOKERS, _api_key_to_name as _aktn,
                _is_non_joker_card as _reroll_nonj,
            )
            shop_cards = raw_state.get("shop", {}).get("cards", [])
            best_buy_idx = -1
            best_buy_delta = -999
            best_buy_name = ""
            for sci, sc in enumerate(shop_cards):
                if _reroll_nonj(sc):
                    continue
                sc_cost = sc.get("cost", {}).get("buy", 999)
                if sc_cost > money:
                    continue
                sc_key = sc.get("joker_key", "") or sc.get("key", "")
                sc_name = _aktn(sc_key)
                sc_mod = sc.get("modifier", {})
                sc_edition = sc_mod.get("edition", "") if isinstance(sc_mod, dict) else ""
                is_high = (sc_name in HIGH_VALUE_JOKERS or
                           sc_edition in ("POLYCHROME", "HOLO", "NEGATIVE"))
                is_scoring = _jis(sc)
                if is_high or is_scoring:
                    delta = _ejv(sc, jokers_raw, raw_state)
                    # Buy if positive delta, or high-value, or scoring with open slot
                    should_buy = (delta > 0 or is_high or
                                  (is_scoring and joker_count < joker_limit))
                    if should_buy and delta > best_buy_delta:
                        best_buy_delta = delta
                        best_buy_idx = sci
                        best_buy_name = sc_name or sc_key

            if best_buy_idx >= 0 and joker_count < joker_limit:
                # FORCE BUY instead of rerolling — don't waste the joker
                print(f"[SHOP] OVERRIDE reroll → buying {best_buy_name} "
                      f"(delta={best_buy_delta:.0f})", flush=True)
                return "buy", {"card": best_buy_idx}

            # SIMPLE RULE: after rerolling, can you still buy a joker?
            # Cheapest jokers cost ~$2. If money after reroll < $4, don't bother.
            min_joker_cost = 4
            money_after = money - reroll_cost
            if money_after < min_joker_cost:
                return "gamestate", None

            # Track rerolls per shop
            self._shop_rerolls = getattr(self, '_shop_rerolls', 0) + 1

            # Reroll cap: only spend surplus above interest floor.
            # Interest floor = $25 base, $50 with Seed Money, $125 with Money Tree.
            vouchers = raw_state.get("vouchers", {}).get("owned", [])
            v_set = set(vouchers) if isinstance(vouchers, list) else set()
            if "v_money_tree" in v_set:
                interest_floor = 125
            elif "v_seed_money" in v_set:
                interest_floor = 50
            else:
                interest_floor = 25
            surplus = max(money - interest_floor, 0)
            affordable = surplus // max(reroll_cost, 1)
            # Allow 1 peek reroll even with 0 surplus, but cap at 8
            max_rerolls = max(1, min(affordable, 8))
            if self._shop_rerolls > max_rerolls:
                return "gamestate", None

            return "reroll", None

        # Use consumable (target 12-13 -> consumable index 0-1)
        if action_type == 8:
            c_idx = target_idx - 12 if target_idx >= 12 else target_idx
            game_state = raw_state.get("state", "")
            if game_state == "SELECTING_HAND":
                hand_cards = raw_state.get("hand", {}).get("cards", [])
                target_cards = [
                    i for i, sel in enumerate(card_selections)
                    if sel > 0.5 and i < len(hand_cards)
                ]
                if target_cards:
                    return "use", {"consumable": c_idx, "cards": target_cards}
            return "use", {"consumable": c_idx}

        # Select blind
        if action_type == 9:
            return "select", None

        # Skip blind — ONLY for Investment Tag + strong scoring
        if action_type == 10:
            blinds = raw_state.get("blinds", {})
            for b in (blinds.values() if isinstance(blinds, dict) else []):
                if isinstance(b, dict) and b.get("status") in ("SELECT", "CURRENT"):
                    tag_name = b.get("tag_name", "")
                    if tag_name == "Investment Tag":
                        scoring_power = estimate_score_for_hand_type(
                            jokers_raw, raw_state) * 4
                        blind_target = b.get("score", 0)
                        if blind_target > 0 and scoring_power > blind_target * 2.0:
                            return "skip", None
                    break
            # Conditions not met — select instead
            return "select", None

        # Select pack card (target 14-18 -> pack card index 0-4)
        if action_type == 11:
            pk_idx = target_idx - 14 if target_idx >= 14 else target_idx
            # Tarot/Spectral pack cards may need hand card targets
            pack_cards = raw_state.get("pack", {}).get("cards", [])
            card_set = pack_cards[0].get("set", "") if pack_cards else ""
            if card_set in ("TAROT", "SPECTRAL"):
                return "pack", {"card": pk_idx, "targets": [0]}
            return "pack", {"card": pk_idx}

        # Skip pack
        if action_type == 12:
            return "pack", {"skip": True}

        # End shop -> next_round
        if action_type == 13:
            # Guard: don't leave shop with empty joker slots if there's a
            # buyable scoring joker. Only check actual jokers (not tarots/planets).
            # Use a counter to prevent infinite loops.
            # Re-read joker count fresh to avoid stale data after buys
            _fresh_jokers = raw_state.get("jokers", {}).get("cards", [])
            _fresh_jcount = len(_fresh_jokers)
            _fresh_jlimit = raw_state.get("jokers", {}).get("limit", 5)
            shop_block_count = getattr(self, '_shop_block_count', 0)
            if _fresh_jcount < _fresh_jlimit and money >= 2 and shop_block_count < 3:
                shop_cards = raw_state.get("shop", {}).get("cards", [])
                from environment.action_space import (
                    _estimate_joker_value, _joker_is_scoring, _api_key_to_name,
                    MUST_BUY_JOKERS, BAD_JOKERS, HIGH_VALUE_JOKERS,
                    _is_non_joker_card as _end_nonj,
                )
                # Also check for must-buy consumables (Hermit, etc.)
                from environment.action_space import MUST_BUY_CONSUMABLES
                _end_cons = raw_state.get("consumables", {}).get("cards", [])
                _end_climit = raw_state.get("consumables", {}).get("limit", 2)
                for i, sc in enumerate(shop_cards):
                    sc_cost = sc.get("cost", {}).get("buy", 999)
                    if sc_cost > money:
                        continue
                    sc_key = sc.get("joker_key", "") or sc.get("key", "")
                    # Check consumables first (Hermit etc.)
                    if _end_nonj(sc):
                        card_key = sc.get("key", "")
                        if card_key in MUST_BUY_CONSUMABLES and len(_end_cons) < _end_climit:
                            self._shop_block_count = shop_block_count + 1
                            label = sc.get("label") or card_key
                            print(f"[SHOP] BLOCKED leaving → buying consumable {label} "
                                  f"(cost=${sc_cost})", flush=True)
                            return "buy", {"card": i}
                        continue
                    sc_name = _api_key_to_name(sc_key)
                    if sc_name and sc_name in BAD_JOKERS:
                        continue
                    sc_mod = sc.get("modifier", {})
                    sc_ed = sc_mod.get("edition", "") if isinstance(sc_mod, dict) else ""
                    is_high = (sc_name in HIGH_VALUE_JOKERS if sc_name else False) or \
                              sc_ed in ("POLYCHROME", "HOLO", "NEGATIVE")
                    is_scoring = _joker_is_scoring(sc)
                    delta = _estimate_joker_value(sc, _fresh_jokers, raw_state)
                    # Force-buy if: positive delta, must-buy, high-value,
                    # or ANY scoring joker when we have open slots
                    if delta > 0 or (sc_name and sc_name in MUST_BUY_JOKERS) or \
                       is_high or is_scoring:
                        self._shop_block_count = shop_block_count + 1
                        label = sc_name or sc_key or f"slot{i}"
                        print(f"[SHOP] BLOCKED leaving → buying {label} "
                              f"(delta={delta:.0f}, cost=${sc_cost}, "
                              f"scoring={is_scoring}, high={is_high})", flush=True)
                        return "buy", {"card": i}
            self._shop_block_count = 0

            # ── Riff-Raff optimization ──
            # If we have Riff-Raff, it creates 2 Common Jokers when the next
            # blind is selected — but only if there are empty joker slots.
            # Sell any genuinely weak jokers before leaving the shop to let
            # Riff-Raff fill those slots with new randoms.
            # SAFETY: only sell if we can still beat the upcoming blind without
            # the joker we're selling.
            has_riff_raff = False
            riff_raff_idx = -1
            for ji, j in enumerate(jokers_raw):
                jk = (j.get("joker_key", "") or j.get("key", "")).lower()
                if "riff" in jk and "raff" in jk:
                    has_riff_raff = True
                    riff_raff_idx = ji
                    break

            if has_riff_raff and joker_count >= joker_limit and joker_count > 1:
                # Joker slots are full — Riff-Raff can't fire.
                # Find the weakest non-essential joker and sell it to make room.

                # Figure out the upcoming blind target so we don't sell
                # ourselves into a loss.
                blinds_data = raw_state.get("blinds", {})
                next_blind_target = 0
                if isinstance(blinds_data, dict):
                    for bkey in ("small", "big", "boss"):
                        b = blinds_data.get(bkey, {})
                        if isinstance(b, dict) and b.get("status") == "UPCOMING":
                            next_blind_target = b.get("score", 0)
                            break

                # Current scoring power (per hand × default 4 hands)
                hands_available = raw_state.get("round", {}).get("hands_left", 4)
                if hands_available <= 0:
                    hands_available = 4  # default for next round
                current_total_score = estimate_score_for_hand_type(
                    jokers_raw, raw_state) * hands_available

                worst_idx, worst_val = _find_weakest_sellable_joker(
                    jokers_raw, raw_state,
                    exclude_indices={riff_raff_idx})

                # Only sell if the joker is genuinely weak (below threshold).
                # A random Common joker averages ~4.0 value — sell if worse.
                if worst_idx >= 0 and worst_val < 5.0:
                    # Check scoring power WITHOUT this joker — can we still win?
                    jokers_without = [j for i, j in enumerate(jokers_raw)
                                      if i != worst_idx]
                    score_without = estimate_score_for_hand_type(
                        jokers_without, raw_state) * hands_available

                    if next_blind_target > 0 and score_without < next_blind_target:
                        pass  # Can't sell — would lose next blind
                    else:
                        return "sell", {"joker": worst_idx}

            # Reset reroll counter for next shop visit
            self._shop_rerolls = 0
            return "next_round", None

        return "gamestate", None

    def _log_play_for_joker_order(self, raw_state: dict,
                                   intended_order: list[str] | None):
        """Log joker order details for the current play action."""
        from environment.hand_eval import (
            _api_key_to_name, classify_hand, _resolve_copy_source, JOKERS
        )

        # Get current joker order from API state
        joker_cards = raw_state.get("jokers", {}).get("cards", [])
        confirmed_names = []
        for j in joker_cards:
            jk = j.get("key", "")
            jn = _api_key_to_name(jk) or jk
            confirmed_names.append(jn)

        # Determine order match
        order_matched = None
        if intended_order is not None:
            order_matched = intended_order == confirmed_names

        # Figure out what hand is being played
        hand_cards = raw_state.get("hand", {}).get("cards", [])
        played_card_strs = []
        hand_type = "Unknown"
        if hand_cards:
            # Best guess: use the first 5 cards (the play hasn't happened yet,
            # but we stored the planned cards in _pending data)
            card_strs = [
                f"{c.get('value', '?')}{c.get('suit', '?')[0]}"
                for c in hand_cards[:5]
            ]
            played_card_strs = card_strs

        # Resolve Brainstorm copy target
        brainstorm_copies = None
        for idx, j in enumerate(joker_cards):
            jk = j.get("key", "")
            jn = _api_key_to_name(jk)
            if jn == "Brainstorm":
                schema = JOKERS.get(jn, {})
                target_dir = schema.get("copy_target", "left")
                copy_src = _resolve_copy_source(joker_cards, idx, target_dir)
                if copy_src is not None:
                    ck = copy_src.get("key", "")
                    brainstorm_copies = _api_key_to_name(ck) or ck
                else:
                    brainstorm_copies = "NONE (no valid target)"
                break

        self.joker_logger.log_play(
            hand_type=hand_type,
            played_cards=played_card_strs,
            intended_order=intended_order,
            confirmed_order=confirmed_names,
            brainstorm_copies=brainstorm_copies,
            order_matched=order_matched,
        )

    async def _auto_rearrange_jokers(self, raw_state: dict,
                                      hand_cards: list[dict] | None = None,
                                      deck_cards: list[dict] | None = None
                                      ) -> list[str] | None:
        """Automatically reorder jokers for optimal scoring.

        Called after any action that changes the joker lineup (buy, sell, pack swap)
        or before playing a hand. When hand/deck cards are available, uses them
        for accurate scoring. Otherwise builds representative hands from deck.

        Returns list of joker names in the new order, or None if no change.
        """
        from environment.hand_eval import _api_key_to_name

        jokers = raw_state.get("jokers", {}).get("cards", [])
        if len(jokers) <= 1:
            return None

        try:
            new_order = compute_optimal_joker_order(
                jokers, gamestate=raw_state,
                hand_cards=hand_cards, deck_cards=deck_cards
            )
            if new_order is not None:
                joker_names = []
                for idx in new_order:
                    if idx < len(jokers):
                        jk = jokers[idx].get("key", "")
                        jn = _api_key_to_name(jk) or jk
                        joker_names.append(jn)
                await self.game.execute_action("rearrange", {"jokers": new_order})
                return joker_names
        except Exception as e:
            print(f"[WARN] Joker rearrange failed: {e}", flush=True)
        return None

    async def _auto_buy_vouchers(self, raw_state: dict) -> dict:
        """Auto-buy vouchers in the shop when affordable.

        Most vouchers are strong upgrades. Skip only the ones that add
        more tarot/celestial cards to packs (low value for the bot).

        Timing awareness: if we're in the first shop of an ante (after
        Small Blind) and the voucher isn't critical, defer the purchase
        to a later shop if buying now would cost us interest income.
        """
        # Vouchers to skip — extra tarot/celestial pack cards aren't useful
        SKIP_VOUCHERS = {
            "v_tarot_merchant",   # Tarot cards appear more in shop
            "v_tarot_tycoon",     # Even more tarots
            "v_planet_merchant",  # Planet cards appear more in shop
            "v_planet_tycoon",    # Even more planets
            "v_crystal_ball",     # +1 consumable slot (marginal)
            "v_omen_globe",       # Spectral cards in Arcana packs
        }

        shop_vouchers = raw_state.get("vouchers", {}).get("cards", [])
        if not shop_vouchers:
            return raw_state

        money = raw_state.get("money", 0)

        # Check interest cap for safe spending
        owned_vouchers = raw_state.get("used_vouchers", [])
        v_set = set(owned_vouchers) if isinstance(owned_vouchers, list) else set()
        if "v_money_tree" in v_set:
            interest_cap = 25   # $25 max interest (from $125)
            interest_cap_money = 125
        elif "v_seed_money" in v_set:
            interest_cap = 10   # $10 max interest (from $50)
            interest_cap_money = 50
        else:
            interest_cap = 5    # $5 max interest (from $25)
            interest_cap_money = 25

        # Determine which shop this is within the ante by checking
        # blind statuses. If Small Blind is defeated but Big Blind
        # is not, we're in the first shop (more shops coming).
        blinds = raw_state.get("blinds", {})
        small_status = ""
        big_status = ""
        boss_status = ""
        if isinstance(blinds, dict):
            small_info = blinds.get("small", {})
            big_info = blinds.get("big", {})
            boss_info = blinds.get("boss", {})
            small_status = small_info.get("status", "") if isinstance(small_info, dict) else ""
            big_status = big_info.get("status", "") if isinstance(big_info, dict) else ""
            boss_status = boss_info.get("status", "") if isinstance(boss_info, dict) else ""

        # shops_remaining: how many more shop visits after this one
        # After Small Blind: Big + Boss shops remain = 2
        # After Big Blind: Boss shop remains = 1
        # After Boss Blind: next ante = 0
        if boss_status == "DEFEATED":
            shops_remaining = 0
        elif big_status == "DEFEATED":
            shops_remaining = 1
        else:
            shops_remaining = 2

        # Only truly game-changing vouchers bypass interest protection
        CRITICAL_VOUCHERS = {
            "v_grabber", "v_nacho_tong",       # +1 hand (huge)
            "v_wasteful", "v_recyclomancy",     # +1 discard (huge)
            "v_paint_brush", "v_palette",       # +1 joker slot (huge)
            "v_seed_money", "v_money_tree",     # interest cap increase (pays for itself)
            "v_antimatter",                     # +1 joker slot (huge)
        }

        for v_idx, voucher in enumerate(shop_vouchers):
            key = voucher.get("key", "")
            cost = voucher.get("cost", {}).get("buy", 999)

            if key in SKIP_VOUCHERS:
                continue

            if cost > money:
                continue

            # Don't buy if it would drop us below an interest tier
            remaining_after = money - cost
            current_tiers = min(money // 5, interest_cap)
            after_tiers = min(max(remaining_after, 0) // 5, interest_cap)
            if after_tiers < current_tiers and key not in CRITICAL_VOUCHERS:
                continue

            # ── Economy guard for non-critical vouchers ──
            # Vouchers are unique per shop visit (don't persist), so we
            # can't defer. But we CAN refuse to buy non-critical vouchers
            # that cost us too much economy.  Only buy if we keep $10+
            # after purchase (preserve interest and shop flexibility).
            if key not in CRITICAL_VOUCHERS:
                if remaining_after < 10:
                    continue

            # Buy it
            try:
                # print(f"[SHOP] AUTO-BUY voucher {v_idx} ({key}) for ${cost}", flush=True)
                await self.game.execute_action("buy", {"voucher": v_idx})
                await asyncio.sleep(0.3)
                # Re-fetch state after purchase
                try:
                    raw_state = await self.game.fetch_gamestate()
                    money = raw_state.get("money", 0)
                except Exception:
                    pass
            except Exception as e:
                print(f"[SHOP] Voucher buy failed: {e}", flush=True)

        return raw_state

    async def _auto_use_consumables(self, raw_state: dict) -> dict:
        """Automatically use consumable cards when heuristics say it's optimal.

        Called at the start of SELECTING_HAND. Uses Planet cards immediately,
        Tarot cards with smart targeting, and economy cards at right timing.
        Returns the (possibly updated) raw_state after any consumable use.
        """
        # Loop: may use multiple consumables in sequence (e.g., Planet then Tarot)
        max_uses = 2  # At most 2 consumable slots
        for _ in range(max_uses):
            consumables = raw_state.get("consumables", {}).get("cards", [])
            if not consumables:
                break

            try:
                action = plan_consumable_use(raw_state)
                if action is None:
                    break

                await self.game.execute_action("use", action)
                # Re-fetch state after use (cards may have changed)
                raw_state = await self.game.fetch_gamestate()
            except Exception:
                break  # consumable use is best-effort

        return raw_state

    def _get_scaling_snapshot(self) -> dict[int, float]:
        """Get current scaling values from game state manager."""
        tracker = self.game._scaling_tracker
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
            f"PL {metrics['policy_loss']:>7.4f} | "
            f"VL {metrics['value_loss']:>7.4f} | "
            f"Ent {metrics['entropy']:>6.3f} | "
            f"KL {metrics['approx_kl']:>6.4f} | "
            f"LR {self.ppo.get_learning_rate():.2e}"
        )

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
