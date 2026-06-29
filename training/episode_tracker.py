"""Per-episode statistics tracking across training.

Extracted verbatim from training/train.py (06-14 monolith decoupling).
Multi-instance aware: each env tracks its own in-flight episode; lifetime and
session records are shared (one process).
"""

import json
import os
from datetime import datetime

import numpy as np


class EpisodeTracker:
    """Tracks per-episode statistics across training."""

    STATS_FILE = "logs/lifetime_stats.json"
    WIN_LOG_FILE = "logs/win_log.json"

    def __init__(self):
        # Per-env per-episode accumulators (multi-instance: each game
        # tracks its own in-flight episode; lifetime/session stats stay
        # shared since this is all one process)
        self._eps: dict = {}

        self.completed_episodes = 0
        self.total_rewards: list[float] = []
        self.total_lengths: list[int] = []
        self.max_antes: list[int] = []
        self.wins = 0

        # Session-wide records
        self.session_highest_ante = 0
        self.session_highest_score = 0.0
        self.session_highest_score_round = ""

        # Load lifetime stats from disk
        self._lifetime_wins = 0
        self._lifetime_episodes = 0
        self._lifetime_highest_ante = 0
        self._load_lifetime_stats()

    def _ep(self, env_id: int):
        """Per-env in-flight episode accumulator."""
        if env_id not in self._eps:
            from types import SimpleNamespace
            self._eps[env_id] = SimpleNamespace(
                reward=0.0, length=0, ante=1, prev_round_chips=0.0,
                highest_hand=0.0, highest_hand_type="")
        return self._eps[env_id]

    def episode_length(self, env_id: int = 0) -> int:
        return self._ep(env_id).length

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
        self._atomic_json(self.STATS_FILE, data)

    @staticmethod
    def _atomic_json(path: str, data):
        """Write JSON atomically (temp + os.replace) so a kill mid-write can't
        corrupt the file — a torn lifetime_stats/win_log resets the displayed
        lifetime totals / truncates win history to a single entry."""
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError:
            pass

    def _append_win_log(self, record: dict):
        """Append a win record to the persistent win log."""
        os.makedirs(os.path.dirname(self.WIN_LOG_FILE) or ".", exist_ok=True)
        try:
            with open(self.WIN_LOG_FILE, "r") as f:
                wins_list: list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            wins_list = []
        wins_list.append(record)
        self._atomic_json(self.WIN_LOG_FILE, wins_list)

    def step(self, reward: float, ante: int, raw_state: dict = None, env_id: int = 0):
        """Record a single step."""
        ep = self._ep(env_id)
        ep.reward += reward
        ep.length += 1
        ep.ante = max(ep.ante, ante)

        # Track highest hand score from chip deltas
        if raw_state:
            round_chips = raw_state.get("round", {}).get("chips", 0)
            if round_chips > ep.prev_round_chips and ep.prev_round_chips >= 0:
                hand_score = round_chips - ep.prev_round_chips
                # Per-episode tracking
                if hand_score > ep.highest_hand:
                    ep.highest_hand = hand_score
                    # Try to get hand type from the [HAND] log context
                    blind_name = ""
                    blinds_ep = raw_state.get("blinds", {})
                    if isinstance(blinds_ep, dict):
                        for b in blinds_ep.values():
                            if isinstance(b, dict) and b.get("status") == "CURRENT":
                                blind_name = b.get("name", "")
                                break
                    ep.highest_hand_type = f"Ante {ante} ({blind_name})" if blind_name else f"Ante {ante}"
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
            ep.prev_round_chips = round_chips

        # Check for new highest ante
        if ante > self.session_highest_ante:
            self.session_highest_ante = ante
            score_info = f" | Best hand: {self.session_highest_score:,.0f} in {self.session_highest_score_round}" if self.session_highest_score > 0 else ""
            print(f"[RECORD] NEW HIGHEST ANTE: {ante}{score_info}")

    def end_episode(self, won: bool, raw_state: dict = None, env_id: int = 0):
        """Record episode completion."""
        ep = self._ep(env_id)
        self.total_rewards.append(ep.reward)
        self.total_lengths.append(ep.length)
        self.max_antes.append(ep.ante)
        self.completed_episodes += 1

        # Update lifetime counters
        self._lifetime_episodes += 1
        self._lifetime_highest_ante = max(self._lifetime_highest_ante, ep.ante)

        if won:
            self.wins += 1
            self._lifetime_wins += 1
            print(f"\n{'='*50}")
            print(f"WIN #{self._lifetime_wins} (lifetime) | "
                  f"#{self.wins} this session")
            print(f"   Run #{self._lifetime_episodes} lifetime | "
                  f"#{self.completed_episodes} this session | "
                  f"Ante {ep.ante}")
            print(f"   Lifetime: {self._lifetime_wins}/{self._lifetime_episodes} "
                  f"({self._lifetime_wins/max(self._lifetime_episodes,1)*100:.1f}%)")
            win_record: dict = {
                "win_number": self._lifetime_wins,
                "episode": self._lifetime_episodes,
                "ante": ep.ante,
                "timestamp": datetime.now().isoformat(),
                "reward": round(ep.reward, 2),
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
                if ep.highest_hand > 0:
                    win_record["highest_hand"] = round(ep.highest_hand)
                    win_record["highest_hand_context"] = ep.highest_hand_type
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

        # Discard ONLY this env's accumulator so the next step() starts it
        # fresh — the printed R is then a true per-episode value, not a
        # running session sum (the old single-stream code reset only
        # highest_hand, never reward/length/ante, so R was a cumulative mean).
        # Do NOT clear the whole dict: with N>1 that wipes the OTHER envs'
        # in-flight episodes (caught by test_episode_per_env_isolation).
        self._eps.pop(env_id, None)

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

        # dec-040: a 20-ep WR window is pure noise at a ~0.5% win rate (expect ~0
        # wins/window, with occasional 5-10% spikes) — it cannot show learning.
        # Add a long (500-ep) window + lifetime cumulative as the real trend.
        long_a = self.max_antes[-500:]
        long_wins = sum(1 for a in long_a if a > 8)
        lifetime_wins = sum(1 for a in self.max_antes if a > 8)

        return {
            "episodes": self.completed_episodes,
            "mean_reward": np.mean(recent_r),
            "mean_length": int(np.mean(recent_l)),
            "mean_ante": np.mean(recent_a),
            "max_ante": max(recent_a),
            "win_rate": recent_wins / len(recent_a),
            "win_rate_long": long_wins / len(long_a),
            "win_rate_long_n": len(long_a),
            "lifetime_wins": lifetime_wins,
        }
