"""Per-environment session for multi-instance training.

Extracted verbatim from training/train.py (06-14 monolith decoupling).
Everything that belongs to ONE Balatro instance lives here so N parallel
games don't bleed state into each other.
"""

import subprocess
from typing import Optional

from environment.game_state import GameStateManager
from environment.reward import RewardCalculator
from recorder import NullRecorder
from training.joker_order_logger import JokerOrderLogger


class EnvSession:
    """Everything that belongs to ONE Balatro instance.

    The trainer's per-run mutable state used to live as singleton
    attributes; with N parallel games each needs its own copy or runs
    bleed into each other (a stale win flag from env 0 would mark env
    1's loss as a win). All defaults here mirror the old getattr
    defaults — the env methods read these directly now.
    """

    def __init__(self, env_id: int, port: int, phase: int,
                 recorder=None):
        self.env_id = env_id
        self.port = port
        self.game = GameStateManager(port=port)
        self.reward_calc = RewardCalculator(phase=phase)
        self.joker_logger = JokerOrderLogger(enabled=False)  # dec-043: off (unbounded disk)
        self.recorder = recorder if recorder is not None else NullRecorder()
        self.balatro_process: Optional[subprocess.Popen] = None

        # Per-run flags / pending actions (mirrors _reset_run_state)
        self.win_recorded = False
        self.win_reward_stored = False
        self._verdant_leaf_sold = False  # per-blind boss auto-sell guard (per-env)
        self.pending_upgrade_buy = None
        self.pending_rearrange = None
        self.pending_hand_rearrange = None
        self.pending_hand_rearrange_fallback = None
        self.last_api_method = None
        self.last_action_succeeded = True
        self.prev_actionable_state = None
        self.shop_rerolls = 0
        self.shop_noop_count = 0
        self.prev_shop_fingerprint = None
        self.state_stuck_count = 0
        self.prev_state_fingerprint = None
        self.menu_loop_count = 0
        self.current_ante = 1
        self.current_score = 0
        self.silent_skip_count = 0
        self.round_eval_count = 0
        self.consecutive_api_failures = 0
        self.auto_action_this_step = False
        self.last_transition_fire = (None, 0.0, None)
        self.last_transition = None

        # Self-imitation capture (Phase 1): accumulate this episode's REAL
        # (state, action, mask, head_idx) steps; flushed to the demo buffer on
        # a win / high-ante finish, discarded otherwise. Behavior-neutral —
        # pure logging, never reads back into the policy in Phase 1.
        self.episode_traj: list = []
        self.max_ante_seen: int = 1

        # Build-progression instrumentation: highest ante already logged for
        # this run, so we record the xmult-engine composition ONCE per ante
        # boundary (the leading indicator: stacked-xmult-by-ante-3). Logging
        # only — does not touch the policy.
        self.last_logged_ante: int = 0

        # Curriculum (dec-030): True when this run was LOADED from a banked
        # ante-4/5 seed (don't re-harvest it, and tag its experience).
        self.from_curriculum: bool = False
