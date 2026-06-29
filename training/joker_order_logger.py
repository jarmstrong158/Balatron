"""Per-round joker-ordering logger.

Extracted verbatim from training/train.py (06-14 monolith decoupling). Tracks
intended vs confirmed joker orders, Brainstorm copy targets, and rearrange
failures to a rolling JSONL file for post-training review.
"""

import json
import os


class JokerOrderLogger:
    """Logs per-round joker ordering data to a rolling file.

    Tracks intended vs confirmed joker orders, Brainstorm copy targets,
    and rearrange failures for post-training review.
    """

    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "joker_order_log.jsonl")
        # dec-043: gate the append. In production this file grew UNBOUNDED to
        # 1.6 GB (write-only, never analyzed) and helped fill the disk to 0 bytes,
        # which silently breaks checkpoint saves. Off in env_session; tests keep it on.
        self._enabled = enabled
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
        if not self._enabled:
            self._current_round = {}
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
