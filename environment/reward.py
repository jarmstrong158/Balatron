"""
Balatron — Reward Shaping

Translates gamestate transitions into scalar rewards for PPO.

Reward tiers (largest → smallest):
1. Terminal    — game win/loss, naneinf achievement
2. Ante        — ante cleared milestone
3. Round       — blind cleared, score vs target
4. Per-action  — economy delta, score progress during round

All score-based rewards use log10 scaling because Balatro scores
grow exponentially. The agent learns to push the exponent higher.

See NOTES.md for reward design rationale.
"""

import math
from typing import Optional


# ============================================================
# Reward Weights — Tunable Hyperparameters
# ============================================================

# Terminal rewards
REWARD_GAME_WIN = 10.0           # Cleared Ante 8 (Phase 1 goal)
REWARD_GAME_LOSS = -5.0          # Game over base penalty (harsh — dying is bad)
REWARD_NANEINF = 50.0            # Achieved naneinf (Phase 2 goal)
REWARD_PER_ANTE_SURVIVED = 0.3   # Bonus per ante reached before dying (less forgiveness)

# Ante/round rewards
REWARD_ANTE_CLEARED = 3.0        # Base reward for clearing an ante
REWARD_ANTE_SCALING = 0.5        # Additional per ante number (ante 5 = 3.0 + 5*0.5 = 5.5)
REWARD_BLIND_CLEARED = 1.0       # Base reward for clearing any blind
REWARD_BOSS_BLIND_CLEARED = 1.5  # Extra reward for boss blind

# Score rewards (log-scaled)
REWARD_SCORE_RATIO = 0.5         # Multiplied by log10(score / target) when clearing
REWARD_SCORE_PROGRESS = 0.1      # Per-action: log10(chips_gained) during round

# Economy rewards
REWARD_MONEY_GAIN = 0.02         # Per dollar gained
REWARD_MONEY_LOSS = -0.01        # Per dollar spent (lighter penalty — spending is necessary)
REWARD_INTEREST_THRESHOLD = 0.01 # Bonus for maintaining $5+ increments (interest)

# Scaling joker rewards
REWARD_SCALING_GROWTH = 0.05     # Per log-unit of scaling value growth

# Joker build diversity
REWARD_DIVERSITY_PER_CATEGORY = 0.02  # Per core category represented (chip/mult/xmult/econ)
REWARD_SCALING_BONUS = 0.01           # Extra for having at least one scaling joker
REWARD_RETRIGGER_BONUS = 0.01         # Extra for having at least one retrigger joker

# Sell penalty — penalize selling jokers that contribute significantly to scoring
REWARD_SELL_PENALTY_MAX = -0.15  # Cap: worst case penalty for selling best joker
REWARD_SELL_PENALTY_SCALE = 0.3  # Multiplier on normalized contribution (0-1 range)

# Penalty for wasted actions
REWARD_INVALID_ACTION = -0.1     # Attempted an action that failed/was invalid

# Phase-aware reward constants
REWARD_XMULT_ACQUIRE_FIXED = 0.3      # Acquiring a fixed xMult joker
REWARD_XMULT_ACQUIRE_SCALING = 0.5    # Acquiring a scaling xMult joker (rarer, more impactful)
REWARD_GOLD_HOARD_PENALTY = -0.02     # Per-dollar penalty above reroll buffer (Scale phase)
REWARD_GOLD_HOARD_BUFFER = 10         # Dollar threshold for hoarding penalty


# ============================================================
# Phase Weight Functions
# ============================================================

def _sigmoid_ramp(ante: float, center: float, width: float = 0.8) -> float:
    """Smooth sigmoid transition centered at `center` ante."""
    x = (ante - center) / width
    x = max(min(x, 10.0), -10.0)  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-x))


def compute_phase_weights(ante: int) -> tuple[float, float, float]:
    """Returns (w_stabilize, w_scale, w_execute) for the given ante.

    Three phases with smooth sigmoid transitions:
    - Stabilize (antes 1-2): Survive, acquire functional joker base
    - Scale (antes 3-5): Spend aggressively on scaling/xMult jokers
    - Execute (antes 6-8): Optimize plays, stop experimenting
    """
    ramp_early = _sigmoid_ramp(ante, 2.5, 0.8)
    ramp_late = _sigmoid_ramp(ante, 5.5, 0.8)
    w_stabilize = 1.0 - ramp_early
    w_scale = ramp_early * (1.0 - ramp_late)
    w_execute = ramp_late
    return (w_stabilize, w_scale, w_execute)


# ============================================================
# Reward Calculator
# ============================================================

class RewardCalculator:
    """Computes shaped rewards from gamestate transitions.

    Call step() after each action with the previous and new raw gamestates.
    Returns a scalar reward.

    Tracks cumulative state across the run for ante/round transition
    detection and running totals.
    """

    def __init__(self, phase: int = 1):
        """
        Args:
            phase: 1 = general competence (target Ante 8 clear),
                   2 = naneinf hunting (reward extreme scores)
        """
        self.phase = phase
        self.reset()

    def reset(self):
        """Reset all state for a new run."""
        self._prev_ante = 1
        self._prev_round = 0
        self._prev_money = 0
        self._prev_chips = 0.0
        self._prev_score_total = 0.0
        self._max_ante_reached = 0
        self._run_reward = 0.0
        self._prev_scaling_values: dict[int, float] = {}  # slot_id → value
        self._prev_joker_ids: set[int] = set()  # track joker IDs to detect sells
        self._prev_joker_contributions: dict[int, float] = {}  # id → normalized contribution

    def step(self, prev_state: Optional[dict], new_state: dict,
             action: Optional[str] = None,
             action_succeeded: bool = True,
             scaling_values: Optional[dict[int, float]] = None,
             joker_contributions: Optional[list[float]] = None,
             skip_economy: bool = False) -> float:
        """Compute reward for a single state transition.

        Args:
            prev_state: previous raw gamestate (None on first step)
            new_state: new raw gamestate after action
            action: the action that was taken (for context)
            action_succeeded: False if the action was invalid/failed
            scaling_values: current scaling tracker values {slot_id: value}
            joker_contributions: pre-computed leave-one-out contributions
                from game_state encoder cache (avoids recomputing)
            skip_economy: True if auto-actions (voucher buy, consumable use)
                fired this step — economy delta shouldn't be attributed to agent

        Returns:
            Scalar reward for this transition.
        """
        if prev_state is None:
            self._sync_state(new_state, scaling_values, joker_contributions)
            return 0.0

        reward = 0.0

        # Invalid action penalty
        if not action_succeeded:
            reward += REWARD_INVALID_ACTION

        # Check for terminal state
        terminal_reward = self._check_terminal(prev_state, new_state)
        if terminal_reward is not None:
            reward += terminal_reward
            self._run_reward += reward
            return reward

        # Ante progression
        reward += self._check_ante_cleared(prev_state, new_state)

        # Round/blind cleared
        reward += self._check_blind_cleared(prev_state, new_state)

        # Score progress during round
        reward += self._check_score_progress(prev_state, new_state)

        # Economy changes (skip if auto-actions changed money this step)
        if not skip_economy:
            reward += self._check_economy(prev_state, new_state)

        # Scaling joker growth
        if scaling_values is not None:
            reward += self._check_scaling_growth(scaling_values)

        # Joker build diversity
        reward += self._check_joker_diversity(new_state)

        # Sell penalty — penalize selling high-contribution jokers
        if action == "sell":
            reward += self._check_sell_penalty(prev_state, new_state)

        # xMult acquisition bonus (phase-scaled)
        reward += self._check_xmult_acquisition(prev_state, new_state)

        # Gold hoarding penalty (Scale phase only)
        if not skip_economy:
            reward += self._check_gold_hoarding(new_state)

        self._sync_state(new_state, scaling_values, joker_contributions)
        self._run_reward += reward
        return reward

    def get_run_reward(self) -> float:
        """Get total accumulated reward for the current run."""
        return self._run_reward

    def get_max_ante(self) -> int:
        """Get the highest ante reached this run."""
        return self._max_ante_reached

    # --------------------------------------------------------
    # Reward components
    # --------------------------------------------------------

    def _check_terminal(self, prev_state: dict, new_state: dict) -> Optional[float]:
        """Check for game over / game won. Returns reward or None."""
        new_game_state = new_state.get("state", "")
        prev_game_state = prev_state.get("state", "")

        if new_game_state != "GAME_OVER" or prev_game_state == "GAME_OVER":
            return None

        reward = 0.0
        ante = new_state.get("ante_num", self._prev_ante)

        # Check if this was a win (cleared Ante 8)
        if ante > 8:
            reward += REWARD_GAME_WIN
            # Phase 2: check for naneinf
            if self.phase == 2:
                reward += self._check_naneinf(new_state)
        else:
            # Loss — base penalty + partial credit for progress
            reward += REWARD_GAME_LOSS
            reward += ante * REWARD_PER_ANTE_SURVIVED

        return reward

    def _check_naneinf(self, state: dict) -> float:
        """Check if the run achieved naneinf scores."""
        # Look at the highest score achieved
        # BalatroBot may report this differently — check round score
        round_data = state.get("round", {})
        score = round_data.get("chips", 0)

        # naneinf threshold: score exceeds float64 max (~1.80e308)
        # In practice, if we see "nan" or "inf" or score > 1e300
        if score > 1e300 or math.isinf(score) or math.isnan(score):
            return REWARD_NANEINF

        # Phase 2 bonus: reward extremely high scores on log scale
        if score > 1e100:
            log_score = math.log10(score)
            return log_score * 0.1  # Gentle scaling reward

        return 0.0

    def _check_ante_cleared(self, prev_state: dict, new_state: dict) -> float:
        """Reward for clearing an ante (boss blind defeated)."""
        prev_ante = prev_state.get("ante_num", self._prev_ante)
        new_ante = new_state.get("ante_num", prev_ante)

        if new_ante > prev_ante:
            self._max_ante_reached = max(self._max_ante_reached, new_ante)
            reward = REWARD_ANTE_CLEARED + prev_ante * REWARD_ANTE_SCALING
            return reward

        return 0.0

    def _check_blind_cleared(self, prev_state: dict, new_state: dict) -> float:
        """Reward for clearing a blind (transitioning to shop or next blind)."""
        prev_game_state = prev_state.get("state", "")
        new_game_state = new_state.get("state", "")

        # Round → Shop transition means blind was cleared
        if prev_game_state == "SELECTING_HAND" and new_game_state == "SHOP":
            reward = REWARD_BLIND_CLEARED

            # Was it a boss blind?
            blinds = prev_state.get("blinds", {})
            if isinstance(blinds, dict):
                for b in blinds.values():
                    if isinstance(b, dict) and b.get("status") == "CURRENT":
                        if b.get("type") == "BOSS":
                            reward += REWARD_BOSS_BLIND_CLEARED
                        break

            # Score ratio bonus — how much did we exceed the target?
            round_data = new_state.get("round", {})
            score = round_data.get("chips", 0)
            target = self._get_blind_target(prev_state)
            if target > 0 and score > 0:
                ratio = score / target
                if ratio > 1.0:
                    reward += REWARD_SCORE_RATIO * math.log10(ratio)

            return reward

        return 0.0

    def _check_score_progress(self, prev_state: dict, new_state: dict) -> float:
        """Small reward for scoring chips during a round."""
        # Only during active play
        if new_state.get("state") != "SELECTING_HAND":
            return 0.0

        prev_chips = prev_state.get("round", {}).get("chips", 0)
        new_chips = new_state.get("round", {}).get("chips", 0)

        if new_chips > prev_chips:
            gained = new_chips - prev_chips
            if gained > 0:
                return REWARD_SCORE_PROGRESS * math.log10(gained + 1)

        return 0.0

    def _check_economy(self, prev_state: dict, new_state: dict) -> float:
        """Reward/penalty for money changes."""
        prev_money = prev_state.get("money", 0)
        new_money = new_state.get("money", 0)
        delta = new_money - prev_money

        reward = 0.0

        if delta > 0:
            reward += delta * REWARD_MONEY_GAIN
        elif delta < 0:
            reward += delta * REWARD_MONEY_LOSS  # Note: delta is negative, MONEY_LOSS is negative

        # Interest bonus: reward maintaining money in $5 increments
        # Balatro gives $1 interest per $5 held, up to $5 max (at $25)
        # Dampened during Scale phase — stop rewarding gold hoarding
        if new_money >= 5:
            interest_tiers = min(new_money // 5, 5)
            ante = new_state.get("ante_num", 1)
            _, w_scale, _ = compute_phase_weights(ante)
            interest_damping = 1.0 - 0.7 * w_scale  # 30% of normal at peak Scale
            reward += interest_tiers * REWARD_INTEREST_THRESHOLD * interest_damping

        return reward

    def _check_joker_diversity(self, state: dict) -> float:
        """Reward having a diverse joker build across core categories.

        Categories (from joker schema flags):
          - chip: primary effect adds chips
          - mult: primary effect adds flat mult
          - xmult: primary effect multiplies mult
          - econ: generates money per round
          - retrigger: has retrigger_effect (bonus category)
          - scaling: has scaling_increment > 0 (bonus category)

        Returns a small additive reward. Max = 4 × 0.02 + 0.01 + 0.01 = 0.10
        """
        joker_cards = state.get("jokers", {}).get("cards", [])
        if not joker_cards:
            return 0.0

        from data.jokers import JOKERS
        from environment.hand_eval import _api_key_to_name

        has_chip = False
        has_mult = False
        has_xmult = False
        has_econ = False
        has_retrigger = False
        has_scaling = False

        for jc in joker_cards:
            key = jc.get("key", "")
            name = _api_key_to_name(key)
            if not name or name not in JOKERS:
                continue
            schema = JOKERS[name]

            if schema.get("chip") or schema.get("chip_scaling"):
                has_chip = True
            if schema.get("mult") or schema.get("mult_scaling"):
                has_mult = True
            if schema.get("xmult") or schema.get("xmult_scaling"):
                has_xmult = True
            if schema.get("economy") or (schema.get("money_per_round") or 0) > 0:
                has_econ = True
            if schema.get("retrigger_effect"):
                has_retrigger = True
            if (schema.get("scaling_increment") or 0) > 0:
                has_scaling = True

        reward = 0.0

        # Core categories: +0.02 each (max +0.08)
        core_count = sum([has_chip, has_mult, has_xmult, has_econ])
        reward += core_count * REWARD_DIVERSITY_PER_CATEGORY

        # Bonus categories: +0.01 each
        if has_retrigger:
            reward += REWARD_RETRIGGER_BONUS
        if has_scaling:
            reward += REWARD_SCALING_BONUS

        return reward

    def _check_sell_penalty(self, prev_state: dict, new_state: dict) -> float:
        """Penalize selling jokers that contributed significantly to scoring.

        Compares joker IDs between states to find which joker was sold,
        then uses its cached contribution score to compute a proportional penalty.
        Capped at REWARD_SELL_PENALTY_MAX to avoid paralyzing the agent.
        """
        prev_jokers = prev_state.get("jokers", {}).get("cards", [])
        new_jokers = new_state.get("jokers", {}).get("cards", [])

        prev_ids = {j.get("id") for j in prev_jokers if j.get("id") is not None}
        new_ids = {j.get("id") for j in new_jokers if j.get("id") is not None}

        sold_ids = prev_ids - new_ids
        if not sold_ids:
            return 0.0

        # Find the contribution of the sold joker(s)
        total_penalty = 0.0
        for sold_id in sold_ids:
            contribution = self._prev_joker_contributions.get(sold_id, 0.0)
            if contribution <= 0.05:
                # Near-zero or negative contribution — minimal/no penalty
                continue
            # Penalty proportional to contribution (0-1 normalized)
            # contribution of 1.0 = strongest joker → full penalty
            penalty = -abs(contribution) * REWARD_SELL_PENALTY_SCALE
            penalty = max(penalty, REWARD_SELL_PENALTY_MAX)  # cap
            total_penalty += penalty

        # Amplify during Execute phase — discourage late-game experimentation
        ante = new_state.get("ante_num", 1)
        _, _, w_execute = compute_phase_weights(ante)
        total_penalty *= (1.0 + w_execute)  # up to 2× penalty in Execute

        return total_penalty

    def _check_xmult_acquisition(self, prev_state: dict, new_state: dict) -> float:
        """Reward acquiring xMult jokers, scaled by phase weight.

        xMult jokers are the strongest category — multiplicative scaling
        compounds with everything else. Identified via data/jokers.py schema:
        schema.get("xmult") or schema.get("xmult_scaling").
        """
        prev_jokers = prev_state.get("jokers", {}).get("cards", [])
        new_jokers = new_state.get("jokers", {}).get("cards", [])

        prev_ids = {j.get("id") for j in prev_jokers if j.get("id") is not None}
        new_ids = {j.get("id") for j in new_jokers if j.get("id") is not None}

        acquired_ids = new_ids - prev_ids
        if not acquired_ids:
            return 0.0

        from data.jokers import JOKERS
        from environment.hand_eval import _api_key_to_name

        reward = 0.0
        for jc in new_jokers:
            if jc.get("id") not in acquired_ids:
                continue
            key = jc.get("key", "")
            name = _api_key_to_name(key)
            if not name or name not in JOKERS:
                continue
            schema = JOKERS[name]

            if schema.get("xmult_scaling"):
                reward += REWARD_XMULT_ACQUIRE_SCALING
            elif schema.get("xmult"):
                reward += REWARD_XMULT_ACQUIRE_FIXED

        if reward == 0.0:
            return 0.0

        # Scale by phase — xmult is most valuable in Scale phase
        # but always worth at least 30% even outside it
        ante = new_state.get("ante_num", 1)
        _, w_scale, _ = compute_phase_weights(ante)
        phase_multiplier = 0.3 + 0.7 * w_scale
        return reward * phase_multiplier

    def _check_gold_hoarding(self, new_state: dict) -> float:
        """Penalize holding excess gold during Scale phase (antes 3-5).

        Buffer = max(reroll_cost * 2, 10). Gold above this threshold
        incurs a small per-dollar penalty scaled by w_scale.
        """
        ante = new_state.get("ante_num", 1)
        _, w_scale, _ = compute_phase_weights(ante)
        if w_scale < 0.1:
            return 0.0  # negligible outside Scale phase

        money = new_state.get("money", 0)
        reroll_cost = new_state.get("round", {}).get("reroll_cost", 5)
        buffer = max(reroll_cost * 2, REWARD_GOLD_HOARD_BUFFER)

        excess = money - buffer
        if excess <= 0:
            return 0.0

        # Gradual penalty: small per-dollar, capped
        penalty = excess * REWARD_GOLD_HOARD_PENALTY * w_scale
        return max(penalty, -0.15)  # hard floor

    def _check_scaling_growth(self, scaling_values: dict[int, float]) -> float:
        """Reward growth in scaling joker values."""
        reward = 0.0

        for slot_id, new_val in scaling_values.items():
            old_val = self._prev_scaling_values.get(slot_id, 0.0)
            if new_val > old_val and new_val > 0:
                # Log-scale growth reward
                growth = math.log10(new_val + 1) - math.log10(old_val + 1)
                if growth > 0:
                    reward += growth * REWARD_SCALING_GROWTH

        return reward

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _get_blind_target(self, state: dict) -> float:
        """Extract the current blind's target score."""
        blinds = state.get("blinds", {})
        if isinstance(blinds, dict):
            for b in blinds.values():
                if isinstance(b, dict) and b.get("status") == "CURRENT":
                    return b.get("score", 0)
        return 0.0

    def _sync_state(self, state: dict, scaling_values: Optional[dict[int, float]],
                    joker_contributions: Optional[list[float]] = None):
        """Update internal tracking state."""
        self._prev_ante = state.get("ante_num", self._prev_ante)
        self._prev_round = state.get("round_num", self._prev_round)
        self._prev_money = state.get("money", self._prev_money)
        self._prev_chips = state.get("round", {}).get("chips", 0)
        self._max_ante_reached = max(self._max_ante_reached, self._prev_ante)

        if scaling_values is not None:
            self._prev_scaling_values = dict(scaling_values)

        # Track joker IDs and their contributions for sell penalty
        joker_cards = state.get("jokers", {}).get("cards", [])
        self._prev_joker_ids = {j.get("id") for j in joker_cards if j.get("id") is not None}

        # Use pre-computed contributions from game_state encoder cache
        # instead of recomputing leave-one-out scores every step
        if joker_cards and joker_contributions is not None:
            max_c = max((abs(c) for c in joker_contributions), default=1.0)
            max_c = max(max_c, 1.0)
            contribs: dict[int, float] = {}
            for i, jc in enumerate(joker_cards):
                jid = jc.get("id")
                if jid is None or i >= len(joker_contributions):
                    continue
                contribs[jid] = max(joker_contributions[i] / max_c, 0.0)
            self._prev_joker_contributions = contribs
        elif not joker_cards:
            self._prev_joker_contributions = {}


# ============================================================
# Reward Config — For Hyperparameter Sweeps
# ============================================================

class RewardConfig:
    """Mutable reward weights for hyperparameter tuning.

    Pass to RewardCalculator to override defaults.
    """

    def __init__(self, **overrides):
        # Copy all defaults
        self.game_win = overrides.get("game_win", REWARD_GAME_WIN)
        self.game_loss = overrides.get("game_loss", REWARD_GAME_LOSS)
        self.naneinf = overrides.get("naneinf", REWARD_NANEINF)
        self.per_ante_survived = overrides.get("per_ante_survived", REWARD_PER_ANTE_SURVIVED)
        self.ante_cleared = overrides.get("ante_cleared", REWARD_ANTE_CLEARED)
        self.ante_scaling = overrides.get("ante_scaling", REWARD_ANTE_SCALING)
        self.blind_cleared = overrides.get("blind_cleared", REWARD_BLIND_CLEARED)
        self.boss_blind_cleared = overrides.get("boss_blind_cleared", REWARD_BOSS_BLIND_CLEARED)
        self.score_ratio = overrides.get("score_ratio", REWARD_SCORE_RATIO)
        self.score_progress = overrides.get("score_progress", REWARD_SCORE_PROGRESS)
        self.money_gain = overrides.get("money_gain", REWARD_MONEY_GAIN)
        self.money_loss = overrides.get("money_loss", REWARD_MONEY_LOSS)
        self.interest_threshold = overrides.get("interest_threshold", REWARD_INTEREST_THRESHOLD)
        self.scaling_growth = overrides.get("scaling_growth", REWARD_SCALING_GROWTH)
        self.invalid_action = overrides.get("invalid_action", REWARD_INVALID_ACTION)


class ConfigurableRewardCalculator(RewardCalculator):
    """RewardCalculator that uses a RewardConfig for weights.

    Drop-in replacement for RewardCalculator with tunable weights.
    """

    def __init__(self, config: RewardConfig, phase: int = 1):
        super().__init__(phase)
        self._config = config

        # Override module-level constants with config values
        # We do this by overriding the methods that use them
        self._weights = config

    def _check_terminal(self, prev_state: dict, new_state: dict) -> Optional[float]:
        new_game_state = new_state.get("state", "")
        prev_game_state = prev_state.get("state", "")

        if new_game_state != "GAME_OVER" or prev_game_state == "GAME_OVER":
            return None

        reward = 0.0
        ante = new_state.get("ante_num", self._prev_ante)
        w = self._weights

        if ante > 8:
            reward += w.game_win
            if self.phase == 2:
                reward += self._check_naneinf(new_state)
        else:
            reward += w.game_loss
            reward += ante * w.per_ante_survived

        return reward

    def _check_ante_cleared(self, prev_state: dict, new_state: dict) -> float:
        prev_ante = prev_state.get("ante_num", self._prev_ante)
        new_ante = new_state.get("ante_num", prev_ante)

        if new_ante > prev_ante:
            self._max_ante_reached = max(self._max_ante_reached, new_ante)
            w = self._weights
            return w.ante_cleared + prev_ante * w.ante_scaling

        return 0.0

    def _check_blind_cleared(self, prev_state: dict, new_state: dict) -> float:
        prev_game_state = prev_state.get("state", "")
        new_game_state = new_state.get("state", "")

        if prev_game_state == "SELECTING_HAND" and new_game_state == "SHOP":
            w = self._weights
            reward = w.blind_cleared

            blinds = prev_state.get("blinds", {})
            if isinstance(blinds, dict):
                for b in blinds.values():
                    if isinstance(b, dict) and b.get("status") == "CURRENT":
                        if b.get("type") == "BOSS":
                            reward += w.boss_blind_cleared
                        break

            round_data = new_state.get("round", {})
            score = round_data.get("chips", 0)
            target = self._get_blind_target(prev_state)
            if target > 0 and score > 0:
                ratio = score / target
                if ratio > 1.0:
                    reward += w.score_ratio * math.log10(ratio)

            return reward

        return 0.0

    def _check_score_progress(self, prev_state: dict, new_state: dict) -> float:
        if new_state.get("state") != "SELECTING_HAND":
            return 0.0

        prev_chips = prev_state.get("round", {}).get("chips", 0)
        new_chips = new_state.get("round", {}).get("chips", 0)

        if new_chips > prev_chips:
            gained = new_chips - prev_chips
            if gained > 0:
                return self._weights.score_progress * math.log10(gained + 1)

        return 0.0

    def _check_economy(self, prev_state: dict, new_state: dict) -> float:
        prev_money = prev_state.get("money", 0)
        new_money = new_state.get("money", 0)
        delta = new_money - prev_money
        w = self._weights

        reward = 0.0
        if delta > 0:
            reward += delta * w.money_gain
        elif delta < 0:
            reward += delta * w.money_loss

        if new_money >= 5:
            interest_tiers = min(new_money // 5, 5)
            reward += interest_tiers * w.interest_threshold

        return reward

    def _check_scaling_growth(self, scaling_values: dict[int, float]) -> float:
        reward = 0.0
        w = self._weights

        for slot_id, new_val in scaling_values.items():
            old_val = self._prev_scaling_values.get(slot_id, 0.0)
            if new_val > old_val and new_val > 0:
                growth = math.log10(new_val + 1) - math.log10(old_val + 1)
                if growth > 0:
                    reward += growth * w.scaling_growth

        return reward
