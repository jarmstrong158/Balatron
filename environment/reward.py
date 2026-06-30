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
REWARD_GAME_WIN = 150.0          # Cleared Ante 8 (Phase 1 goal) — largest single signal.
                                 # dec-040: raised 15->150. The RL audit found the
                                 # old +15 was DWARFED by ~28-60 of accumulated
                                 # shaping per run, so the value head was calibrated
                                 # to shaping, not winning, and PPO had almost no
                                 # gradient pointing at win-vs-deep-loss. 150 makes
                                 # the terminal win dominate the trajectory return.
REWARD_GAME_LOSS = -5.0          # Game over base penalty (harsh — dying is bad)
REWARD_NANEINF = 50.0            # Achieved naneinf (Phase 2 goal)
REWARD_PER_ANTE_SURVIVED = 0.3   # Bonus per ante reached before dying (less forgiveness)

# Ante/round rewards
REWARD_ANTE_CLEARED = 3.0        # Base reward for clearing an ante
REWARD_ANTE_SCALING = 1.0        # Additional per ante number (ante 7 = 3.0 + 7*1.0 = 10.0) — steepen depth gradient
REWARD_BLIND_CLEARED = 1.0       # Base reward for clearing any blind
REWARD_BOSS_BLIND_CLEARED = 1.5  # Extra reward for boss blind

# Score rewards (log-scaled)
REWARD_SCORE_RATIO = 0.5         # Multiplied by log10(score / target) when clearing
REWARD_SCORE_PROGRESS = 0.02     # Per-action: log10(chips_gained) during round.
                                 # Cut 0.1->0.02: was the dominant dense reward,
                                 # paying for comfortable mid-game scoring the
                                 # heuristics already do (the ante-4 plateau). Now
                                 # a nudge so depth/win rewards dominate the signal.
REWARD_HAND_HIGH_WATER = 0.1     # Per log10-DECADE when a new best single-hand
                                 # score is set this run. Potential-delta (only
                                 # the increase is paid, so the run telescopes
                                 # to a bounded coef*log10(best)); log scaling
                                 # makes "bigger = more" pay per order of
                                 # magnitude, never linearly. Set 0 to disable
                                 # (A/B vs REWARD_SCORE_PROGRESS).

# Economy rewards
REWARD_MONEY_GAIN = 0.01         # Per dollar gained (halved — economy is a means to depth, not the goal)
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
# Stacking premium (dec-026): the data shows depth is gated by STACKING xmult
# (ante-7+ runs avg 1.36 xmult vs 0.71 at ante 3-4), but the flat per-acquisition
# reward priced the 2nd/3rd xmult identically to the 1st. Multiply the
# acquisition reward by (1 + this * #xmult_already_held), so 2nd xmult ~2x, 3rd
# ~3x — pulling hardest at the 1->2 transition that distinguishes deep runs.
REWARD_XMULT_STACK_BONUS = 0.0   # dec-047 (#4): DROPPED. The eval-validated
                                 # finding (dec-043) is that realized xmult VALUE
                                 # predicts deep-ante advance while engine COUNT
                                 # does NOT (flat once ante-controlled; 3+ engines
                                 # is diminishing). This premium rewarded the 2nd/
                                 # 3rd engine (COUNT) — the wrong target. Value is
                                 # still rewarded via REWARD_XMULT_GROWTH_PREMIUM;
                                 # the first engine via REWARD_XMULT_FIRST_ENGINE.
# xMult-engine GROWTH premium (dec-032). The audits found the plateau is bound
# by the reward not differentiating xmult from additive builds: _check_scaling_
# growth paid an xmult engine compounding X1.5->X3.0 the SAME per-log-unit as an
# additive joker gaining +mult — yet multiplicative compounding is the mechanism
# that breaks Balatro's exponential ante wall. This pays xmult-scaler growth
# this-many-times the additive rate. It's the DENSE causal signal (every firing
# hand) the value function can credit — unlike the ~0.5%-rare terminal depth
# payoff that was the only other xmult-differentiating reward.
REWARD_XMULT_GROWTH_PREMIUM = 3.0
# First-engine bootstrap bonus (dec-033). dec-032's growth premium is chicken-
# and-egg: it only fires once an xmult engine is OWNED, so it reinforces xmult
# AFTER the fact but can't push the very FIRST buy — and the audit showed the
# initial buy competes against a survival gradient ~100x larger. This is a
# substantial ONE-SHOT bonus paid on the run's 0->1 xmult transition,
# UN-phase-scaled so it lands in the early antes where the foundation engine
# must be bought. One-shot (fires once per run) so it can't accrue; survival
# reward still dwarfs the run total, so a junk-xmult rush that dies still loses.
REWARD_XMULT_FIRST_ENGINE = 1.5
REWARD_GOLD_HOARD_PENALTY = -0.02     # Per-dollar penalty above reroll buffer (Scale phase)
REWARD_GOLD_HOARD_BUFFER = 10         # Dollar threshold for hoarding penalty
# Balatro interest cap (dec-034 Pillar 3c): $1 interest per $5 held, capped at
# $5/round = $25 held. Money UP TO this earns interest and is the optimal
# "war chest" a White-Stake player banks toward a power-spike shop — so it must
# NOT be penalized. Only money ABOVE the cap is truly idle. (Seed Money/Money Tree
# vouchers raise the cap; using the base $25 is mildly conservative.)
INTEREST_CAP = 25


# ============================================================
# Phase Weight Functions
# ============================================================

def _sigmoid_ramp(ante: float, center: float, width: float = 0.8) -> float:
    """Smooth sigmoid transition centered at `center` ante."""
    x = (ante - center) / width
    x = max(min(x, 10.0), -10.0)  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-x))


def _joker_is_xmult(jc: dict) -> bool:
    """True if a joker card is an xMult engine (fixed or scaling).

    Multiplicative jokers are the depth mechanism in Balatro; dec-032 rewards
    their growth at a premium over additive scalers. Lazy imports mirror the
    other reward helpers (avoid a module-load cycle with hand_eval/data)."""
    from data.jokers import JOKERS
    from environment.hand_eval import _api_key_to_name
    n = _api_key_to_name(jc.get("key", ""))
    s = JOKERS.get(n) if n else None
    return bool(s and (s.get("xmult") or s.get("xmult_scaling")))


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
        self._max_hand_score = 0.0  # best single-hand score this run (high-water)
        self._run_reward = 0.0
        self._prev_scaling_values: dict[int, float] = {}  # slot_id → value
        self._prev_joker_ids: set[int] = set()  # track joker IDs to detect sells
        self._prev_joker_contributions: dict[int, float] = {}  # id → normalized contribution
        # Potentials for delta-based shaping (diversity / interest). These
        # bonuses used to be re-paid on EVERY step, accruing +20-40 over a
        # 300-decision run vs +10 for actually winning — the agent was paid
        # more for existing-while-diverse than for winning. Paying only the
        # CHANGE in potential (acquire a category: +once; lose it: -once)
        # keeps the incentive without the runaway accrual.
        self._prev_diversity_potential = 0.0
        self._prev_interest_potential = 0.0

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

        # New best single-hand score this run (high-water mark)
        reward += self._check_hand_high_water(prev_state, new_state)

        # Economy changes (skip if auto-actions changed money this step)
        if not skip_economy:
            reward += self._check_economy(prev_state, new_state)

        # Scaling joker growth (xmult engines paid at a premium — dec-032)
        if scaling_values is not None:
            reward += self._check_scaling_growth(scaling_values, new_state)

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

        # Check if this was a win. The win reward requires getting PAST ante 8
        # (ante > 8), NOT the API 'won' flag: the base game sets G.GAME.won=true
        # the moment you reach the ante-8 boss (state_events.lua end_round),
        # win OR lose — so 'won' is true even when you die on the boss (e.g.
        # 90,592/100,000). Surviving past ante 8 (ante advances to 9 in endless
        # mode) is the only reliable signal that the boss was actually beaten.
        # Genuine wins are also credited via train.py's post-boss win-fallback.
        if ante > 8:
            reward += REWARD_GAME_WIN
            # Phase 2: check for naneinf
            if self.phase == 2:
                reward += self._check_naneinf(new_state)
        else:
            # Loss (includes dying on the ante-8 boss). dec-048: terminal loss must
            # be <= 0 at EVERY ante. dec-047's convex credit `(ante**1.5)*0.3`
            # overpowered the penalty and made dying NET-POSITIVE from ante 5 on
            # (die@8 = +6.79) — a "safe deep death" basin that out-paid the rare win
            # (deep-audit, 4 agents). Keep ONLY the depth-graded penalty: a shallow
            # death is much worse (~-4.4 @a2), a near-win ~neutral (~0 @a8). The
            # per-ante progress credit is already paid DURING the run by
            # _check_ante_cleared, so a terminal depth CREDIT also double-counted.
            shortfall = max(0, 8 - ante)
            reward += REWARD_GAME_LOSS * (shortfall / 8.0)

        return reward

    def terminal_win_reward(self, state: dict) -> float:
        """Win reward for a cleared run.

        Used as a fallback by train.py when the win screen is auto-dismissed
        by the Lua mod and GAME_OVER never fires, so the win reward would
        otherwise never reach the PPO buffer.
        """
        reward = REWARD_GAME_WIN
        if self.phase == 2:
            reward += self._check_naneinf(state)
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

            # Score ratio bonus — how much did we exceed the target? Read the
            # cleared-round score from PREV state (SELECTING_HAND): on the SHOP
            # transition new_state.round.chips has already reset to 0, so the
            # bonus was silently never paid (audit 06-13 RISK).
            score = prev_state.get("round", {}).get("chips", 0)
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

    def _check_hand_high_water(self, prev_state: dict, new_state: dict) -> float:
        """Reward setting a new best single-hand score this run.

        Tracks the largest single hand (chip gain in one play) seen this run
        and pays only the INCREASE in log10 of that best — a potential-based
        delta (like diversity/interest), so the whole run telescopes to a
        bounded coef * log10(best_final) rather than a runaway per-step sum.
        Log scaling means "bigger score = more reward" is paid per order of
        magnitude, never linearly, so a monster hand can't drown the
        win/ante signals.

        Distinct from _check_score_progress (which pays on *every* scoring
        hand): this fires only when the engine's single-hand ceiling rises,
        the quantity that matters for Phase 2 (naneinf is one giant hand) —
        hence the Phase-2 amplification.
        """
        if new_state.get("state") != "SELECTING_HAND":
            return 0.0

        prev_chips = prev_state.get("round", {}).get("chips", 0)
        new_chips = new_state.get("round", {}).get("chips", 0)
        gained = new_chips - prev_chips
        if gained <= self._max_hand_score:
            return 0.0

        old_best = self._max_hand_score
        self._max_hand_score = gained
        phi_delta = math.log10(gained + 1) - math.log10(old_best + 1)
        # Phase 2 (naneinf) ONLY. In Phase 1 (depth) this paid 0.1/decade for
        # single-hand size — a 5x-larger-per-decade chip incentive than
        # score_progress (0.02), pulling the policy back toward comfortable
        # early-blind chip-farming and away from depth. Audit 06-13 CRITICAL.
        phase_mult = 3.0 if self.phase == 2 else 0.0
        return REWARD_HAND_HIGH_WATER * phi_delta * phase_mult

    def _check_economy(self, prev_state: dict, new_state: dict) -> float:
        """Reward/penalty for money changes."""
        prev_money = prev_state.get("money", 0)
        new_money = new_state.get("money", 0)
        delta = new_money - prev_money

        reward = 0.0

        if delta > 0:
            reward += delta * REWARD_MONEY_GAIN
        elif delta < 0:
            # delta is negative (money spent) and REWARD_MONEY_LOSS is
            # negative; use abs(delta) so spending actually incurs a penalty
            # (dollars_spent * -0.01) rather than a positive reward.
            reward += abs(delta) * REWARD_MONEY_LOSS

        # Interest bonus: reward maintaining money in $5 increments
        # Balatro gives $1 interest per $5 held, up to $5 max (at $25)
        # Dampened during Scale phase — stop rewarding gold hoarding.
        # Paid as a POTENTIAL DELTA (crossing a tier up: +, dropping: −),
        # not re-paid per step — see reset() for why.
        interest_tiers = min(new_money // 5, 5) if new_money >= 5 else 0
        ante = new_state.get("ante_num", 1)
        _, w_scale, _ = compute_phase_weights(ante)
        interest_damping = 1.0 - 0.7 * w_scale  # 30% of normal at peak Scale
        interest_potential = (interest_tiers * REWARD_INTEREST_THRESHOLD
                              * interest_damping)
        reward += interest_potential - self._prev_interest_potential
        self._prev_interest_potential = interest_potential

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
            # All jokers gone — settle the potential down to zero
            delta = 0.0 - self._prev_diversity_potential
            self._prev_diversity_potential = 0.0
            return delta

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

        # Pay only the CHANGE in diversity potential (see reset()) — gaining
        # a category pays once, losing one costs once, holding pays nothing.
        delta = reward - self._prev_diversity_potential
        self._prev_diversity_potential = reward
        return delta

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

        # Count xmult jokers ALREADY held (before this acquisition) for the
        # stacking premium — the 2nd/3rd xmult is what builds the compounding
        # engine that reaches ante 7+ (dec-026).
        def _is_xmult(jc):
            n = _api_key_to_name(jc.get("key", ""))
            s = JOKERS.get(n) if n else None
            return bool(s and (s.get("xmult") or s.get("xmult_scaling")))
        prev_xmult = sum(1 for j in prev_jokers if _is_xmult(j))

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

        # Stacking premium: 1st xmult pays base, 2nd ~2x, 3rd ~3x.
        reward *= 1.0 + REWARD_XMULT_STACK_BONUS * prev_xmult

        # Scale by phase — xmult is most valuable in Scale phase, but the FIRST
        # xmult engine is laid in the Stabilize phase (antes 1-2), and the old
        # 0.3 floor half-paid exactly those foundational buys (audit dec-032:
        # "phase-suppressed exactly when foundations are laid"). Lift the floor
        # to 0.7 so buying the first/second xmult early isn't penalized for
        # timing; the Scale-phase peak (~1.0) is unchanged.
        ante = new_state.get("ante_num", 1)
        _, w_scale, _ = compute_phase_weights(ante)
        phase_multiplier = 0.7 + 0.3 * w_scale
        result = reward * phase_multiplier

        # FIRST-ENGINE bootstrap bonus (dec-033): the 0->1 xmult transition is
        # the chicken-and-egg crux — the growth premium can't reward an engine
        # the agent doesn't own yet. Pay a flat, un-phase-scaled one-shot bonus
        # so the very first xmult buy is directly attractive even early. (We're
        # past the reward==0 guard, so an xmult was genuinely acquired here.)
        if prev_xmult == 0:
            result += REWARD_XMULT_FIRST_ENGINE
        return result

    def _check_gold_hoarding(self, new_state: dict) -> float:
        """Penalize ONLY truly-idle gold — money above the interest cap ($25),
        where extra dollars earn no more interest and should be spent on power
        (dec-034 Pillar 3c: save-then-spike). Holding UP TO the cap is OPTIMAL
        (max interest = the war chest a White-Stake player banks toward a
        power-spike shop), so it is NOT penalized. The old $10 buffer actively
        fought that strategy by punishing the very $10-25 accumulation that funds
        the spike (strategy audit's #3 gap)."""
        ante = new_state.get("ante_num", 1)
        _, w_scale, _ = compute_phase_weights(ante)
        if w_scale < 0.1:
            return 0.0  # negligible outside Scale phase

        money = new_state.get("money", 0)
        # Money up to the interest cap earns interest — never penalize the chest.
        excess = money - INTEREST_CAP
        if excess <= 0:
            return 0.0

        # Gradual penalty on idle money ABOVE the cap only, capped.
        penalty = excess * REWARD_GOLD_HOARD_PENALTY * w_scale
        return max(penalty, -0.15)  # hard floor

    def _check_scaling_growth(self, scaling_values: dict[int, float],
                              new_state: Optional[dict] = None) -> float:
        """Reward growth in scaling joker values.

        xMult-engine growth pays REWARD_XMULT_GROWTH_PREMIUM x the additive rate
        (dec-032): multiplicative compounding is the mechanism that breaks the
        exponential ante wall, and this is a DENSE per-firing-hand signal the
        value function can credit — the differentiating xmult reward the rare
        terminal depth payoff could never deliver. The premium targets only the
        slots whose joker is an xmult engine; additive scalers keep the base rate.
        """
        xmult_ids = set()
        if new_state is not None:
            for jc in new_state.get("jokers", {}).get("cards", []):
                jid = jc.get("id")
                if jid is not None and _joker_is_xmult(jc):
                    xmult_ids.add(jid)

        reward = 0.0
        for slot_id, new_val in scaling_values.items():
            old_val = self._prev_scaling_values.get(slot_id, 0.0)
            if new_val > old_val and new_val > 0:
                # Log-scale growth reward
                growth = math.log10(new_val + 1) - math.log10(old_val + 1)
                if growth > 0:
                    rate = REWARD_SCALING_GROWTH
                    if slot_id in xmult_ids:
                        rate *= REWARD_XMULT_GROWTH_PREMIUM
                    reward += growth * rate

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

        # Win requires getting PAST ante 8 (ante > 8), not the 'won' flag, which
        # the base game sets true the moment you reach the ante-8 boss — win or
        # lose (see RewardCalculator._check_terminal).
        if ante > 8:
            reward += w.game_win
            if self.phase == 2:
                reward += self._check_naneinf(new_state)
        else:
            reward += w.game_loss
            reward += ante * w.per_ante_survived

        return reward

    def terminal_win_reward(self, state: dict) -> float:
        reward = self._weights.game_win
        if self.phase == 2:
            reward += self._check_naneinf(state)
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

            # Read cleared score from prev_state (see base class note).
            score = prev_state.get("round", {}).get("chips", 0)
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
            # abs(delta) so spending money is penalized (money_loss is negative)
            reward += abs(delta) * w.money_loss

        # Potential delta, not per-step accrual (see RewardCalculator.reset)
        interest_tiers = min(new_money // 5, 5) if new_money >= 5 else 0
        interest_potential = interest_tiers * w.interest_threshold
        reward += interest_potential - self._prev_interest_potential
        self._prev_interest_potential = interest_potential

        return reward

    def _check_scaling_growth(self, scaling_values: dict[int, float],
                              new_state: dict = None) -> float:
        # dec-043: signature must match the base / the step() call site
        # (scaling_values, new_state) or any sweep using this subclass TypeErrors
        # on the first scaling update. new_state unused here (sweep keeps the
        # simpler pre-dec-026 growth model).
        reward = 0.0
        w = self._weights

        for slot_id, new_val in scaling_values.items():
            old_val = self._prev_scaling_values.get(slot_id, 0.0)
            if new_val > old_val and new_val > 0:
                growth = math.log10(new_val + 1) - math.log10(old_val + 1)
                if growth > 0:
                    reward += growth * w.scaling_growth

        return reward
