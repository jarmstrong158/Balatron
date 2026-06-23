"""
Balatron — Neural Network

Shared trunk + 3 policy heads + 1 value head for PPO.

Architecture:
    Input (838) → Shared Trunk (768 → 768 → 512)
    → Play Head   (512 → 256 → 45)  — SELECTING_HAND
    → Shop Head   (512 → 256 → 45)  — SHOP, SMODS_BOOSTER_OPENED
    → Blind Head  (512 → 128 → 45)  — BLIND_SELECT
    → Value Head  (512 → 256 → 1)   — all states

The game state determines which policy head produces the action logits.
The value head always produces a scalar state-value estimate.
Action masks are applied externally (in PPO loss / action selection).

See NOTES.md for architecture decisions.
"""

import torch
import torch.nn as nn
import numpy as np

from environment.game_state import STATE_VECTOR_SIZE
from environment.action_space import (
    ACTION_HEAD_SIZE,
    TARGET_SHOP_JOKER_OFFSET, SHOP_JOKER_SLOTS,
    TARGET_SHOP_VOUCHER_OFFSET, SHOP_VOUCHER_SLOTS,
    TARGET_SHOP_PACK_OFFSET, SHOP_PACK_SLOTS,
    TARGET_OWNED_JOKER_OFFSET, JOKER_SLOTS,
    TARGET_CONSUMABLE_OFFSET, CONSUMABLE_SLOTS,
    TARGET_PACK_CARD_OFFSET, PACK_CARD_SLOTS,
    NUM_TARGETS,
)


# ============================================================
# Constants
# ============================================================

# Map action_type → valid target range (offset, count)
# Actions not in this dict don't use a target (reroll, select/skip blind, end shop)
ACTION_TARGET_RANGES = {
    2:  (TARGET_SHOP_JOKER_OFFSET, SHOP_JOKER_SLOTS),      # buy_joker → shop jokers 0-2
    3:  (TARGET_SHOP_VOUCHER_OFFSET, SHOP_VOUCHER_SLOTS),   # buy_voucher → vouchers 3-4
    4:  (TARGET_SHOP_PACK_OFFSET, SHOP_PACK_SLOTS),          # buy_pack → packs 5-6
    5:  (TARGET_OWNED_JOKER_OFFSET, JOKER_SLOTS),            # sell_joker → owned jokers 7-11
    6:  (TARGET_CONSUMABLE_OFFSET, CONSUMABLE_SLOTS),        # sell_consumable → consumables 12-13
    8:  (TARGET_CONSUMABLE_OFFSET, CONSUMABLE_SLOTS),        # use_consumable → consumables 12-13
    11: (TARGET_PACK_CARD_OFFSET, PACK_CARD_SLOTS),          # select_pack_card → pack cards 14-18
}

TRUNK_LAYERS = [768, 768, 512]
PLAY_HEAD_HIDDEN = 256
SHOP_HEAD_HIDDEN = 256
BLIND_HEAD_HIDDEN = 128
VALUE_HEAD_HIDDEN = 256

# Game state → head routing
HEAD_PLAY = 0
HEAD_SHOP = 1
HEAD_BLIND = 2

STATE_TO_HEAD = {
    "SELECTING_HAND": HEAD_PLAY,
    "SHOP": HEAD_SHOP,
    "SMODS_BOOSTER_OPENED": HEAD_SHOP,
    "BLIND_SELECT": HEAD_BLIND,
}


# ============================================================
# Network Components
# ============================================================

class TrunkBlock(nn.Module):
    """Single trunk layer: Linear → LayerNorm → ReLU."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.linear(x)))


class PolicyHead(nn.Module):
    """Single policy head: Linear → ReLU → Linear (logits)."""

    def __init__(self, trunk_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, trunk_out: torch.Tensor) -> torch.Tensor:
        return self.net(trunk_out)


class ValueHead(nn.Module):
    """Value head: Linear → ReLU → Linear (scalar)."""

    def __init__(self, trunk_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, trunk_out: torch.Tensor) -> torch.Tensor:
        return self.net(trunk_out).squeeze(-1)


# ============================================================
# Full Network
# ============================================================

class BalatronNetwork(nn.Module):
    """Actor-critic network for Balatron PPO agent.

    Forward pass:
        state_vector (838) → trunk (512) → {policy_head, value_head}

    The policy head is selected by the game state. During training,
    we batch by game state so each sample uses the correct head.
    """

    def __init__(self, input_size: int = STATE_VECTOR_SIZE,
                 action_size: int = ACTION_HEAD_SIZE):
        super().__init__()

        self.input_size = input_size
        self.action_size = action_size

        # Shared trunk
        trunk_modules = []
        prev_size = input_size
        for layer_size in TRUNK_LAYERS:
            trunk_modules.append(TrunkBlock(prev_size, layer_size))
            prev_size = layer_size
        self.trunk = nn.Sequential(*trunk_modules)
        self.trunk_output_size = TRUNK_LAYERS[-1]

        # Policy heads
        self.play_head = PolicyHead(self.trunk_output_size, PLAY_HEAD_HIDDEN, action_size)
        self.shop_head = PolicyHead(self.trunk_output_size, SHOP_HEAD_HIDDEN, action_size)
        self.blind_head = PolicyHead(self.trunk_output_size, BLIND_HEAD_HIDDEN, action_size)

        self._policy_heads = nn.ModuleList([self.play_head, self.shop_head, self.blind_head])

        # DECOUPLED value trunk (dec-029 un-freeze). The value head reads from
        # its OWN trunk, NOT the shared policy trunk. With a shared trunk the
        # value-loss gradient (large — un-normalized returns reach tens of
        # units) dominated the trunk representation, and the tiny (normalized,
        # entropy-counter-pressured) policy gradient rode on top — pinning the
        # policy near-frozen (KL ~0.0045, sustained, every lever null). Separate
        # trunks make value and policy gradients touch DISJOINT parameters, so
        # the policy trunk is shaped ONLY by the policy objective.
        value_trunk_modules = []
        prev_size = input_size
        for layer_size in TRUNK_LAYERS:
            value_trunk_modules.append(TrunkBlock(prev_size, layer_size))
            prev_size = layer_size
        self.value_trunk = nn.Sequential(*value_trunk_modules)

        # Value head (reads the value trunk)
        self.value_head = ValueHead(self.trunk_output_size, VALUE_HEAD_HIDDEN)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """Orthogonal initialization — standard for PPO."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, head_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through trunk + selected policy head + value head.

        Args:
            state: input state vector, shape (batch, 838)
            head_idx: which policy head to use (HEAD_PLAY, HEAD_SHOP, HEAD_BLIND)

        Returns:
            action_logits: shape (batch, 45) — raw logits before masking
            state_value: shape (batch,) — scalar value estimate
        """
        trunk_out = self.trunk(state)
        action_logits = self._policy_heads[head_idx](trunk_out)
        state_value = self.value_head(self.value_trunk(state))  # decoupled trunk
        return action_logits, state_value

    def forward_mixed(self, states: torch.Tensor,
                      head_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mixed game states in a batch.

        For training batches where different samples need different heads.
        Groups by head index, runs each group, then reassembles.

        Args:
            states: shape (batch, 838)
            head_indices: shape (batch,) — int tensor of head indices per sample

        Returns:
            action_logits: shape (batch, 45)
            state_values: shape (batch,)
        """
        batch_size = states.shape[0]
        trunk_out = self.trunk(states)

        # Value head runs on everything — from its OWN decoupled trunk (dec-029)
        state_values = self.value_head(self.value_trunk(states))

        # Policy heads — group by head index
        action_logits = torch.zeros(batch_size, self.action_size,
                                    device=states.device, dtype=states.dtype)

        for h_idx in range(len(self._policy_heads)):
            mask = head_indices == h_idx
            if mask.any():
                head_out = self._policy_heads[h_idx](trunk_out[mask])
                action_logits[mask] = head_out

        return action_logits, state_values

    def get_action_and_value(self, state: torch.Tensor, head_idx: int,
                             action_mask: torch.Tensor,
                             action: torch.Tensor = None,
                             deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.

        This is the main interface for PPO — combines policy sampling
        with value estimation in a single call.

        Args:
            state: shape (batch, 838)
            head_idx: which policy head
            action_mask: shape (batch, 45) — 1.0 for valid actions
            action: optional pre-selected action for log_prob computation
                    shape depends on action representation
            deterministic: if True, use argmax instead of sampling

        Returns:
            action: selected action tensor
            log_prob: log probability of the action
            entropy: policy entropy
            value: state value estimate
        """
        action_logits, value = self.forward(state, head_idx)

        # Mask handling — CRITICAL (06-13 audit, dec-015). The action_mask
        # carries exp(HAND_BIAS_STRENGTH*k) HEURISTIC BIAS, not just legality.
        # Folding log(mask) into the policy logits injected a ±4-5 nat prior
        # (~57x the head's signal) that saturated the softmax, pinned entropy
        # at a structural floor, and meant the policy never had to LEARN the
        # masked decisions (3 independent audits converged on this). We now
        # use the mask ONLY for hard legality in the distribution the policy
        # samples from / is scored on, and re-home the heuristic guidance as a
        # separate ANNEALING prior-KL term (computed below, applied in PPO).
        neg_inf = torch.tensor(-1e9, dtype=action_logits.dtype,
                               device=action_logits.device)
        legal = action_mask > 0
        masked_logits = torch.where(legal, action_logits, neg_inf)
        # Heuristic prior = the mask's own (biased) distribution over types,
        # kept ONLY to compute the annealing KL pull toward the teacher.
        prior_logits = torch.where(legal, torch.log(action_mask), neg_inf)

        # Split into action type, card selection, target selection
        type_logits = masked_logits[:, :14]
        card_logits = masked_logits[:, 14:26]
        target_logits = masked_logits[:, 26:]

        # Action type — categorical distribution
        type_dist = torch.distributions.Categorical(logits=type_logits)

        # Card selection — independent Bernoulli per slot
        card_probs = torch.sigmoid(card_logits)
        card_dist = torch.distributions.Bernoulli(probs=card_probs)

        if action is None:
            # Sample new action
            if deterministic:
                type_action = type_logits.argmax(dim=-1)
                card_action = (card_probs > 0.5).float()
            else:
                type_action = type_dist.sample()
                card_action = card_dist.sample()

            # Mask target logits based on chosen action type so the NN
            # only picks targets that are valid for the selected action.
            # e.g. buy_pack (4) can only target pack slots 5-6.
            conditioned_target_logits = self._condition_target_on_action(
                target_logits, type_action
            )
            cond_target_dist = torch.distributions.Categorical(logits=conditioned_target_logits)

            if deterministic:
                target_action = conditioned_target_logits.argmax(dim=-1)
            else:
                target_action = cond_target_dist.sample()

            # Pack into single tensor: [type(1), cards(12), target(1)] = 14
            action = torch.cat([
                type_action.unsqueeze(-1).float(),
                card_action,
                target_action.unsqueeze(-1).float(),
            ], dim=-1)

        # Unpack action for log_prob computation
        type_action = action[:, 0].long()
        card_action = action[:, 1:13]
        target_action = action[:, 13].long()

        # For log_prob, use conditioned target dist matching the action type
        conditioned_target_logits = self._condition_target_on_action(
            target_logits, type_action
        )
        cond_target_dist = torch.distributions.Categorical(logits=conditioned_target_logits)

        # The 12 card-selection bits only affect execution for action type 8
        # (use consumable with hand-card targets) — for play/discard the
        # planner chooses the cards and for everything else they're unread.
        # Folding their log-probs/entropy into the totals unconditionally
        # made the PPO ratio churn on dimensions with no causal effect
        # (more clipping), let approx_kl trip target_kl on irrelevant drift,
        # and pointed most of the entropy bonus (12 Bernoullis, up to 8.3
        # nats) at no-op bits instead of the action-type head.
        card_used = (type_action == 8).float()

        # Gate the target dimension the same way as the cards. Action types
        # WITHOUT a real target (play, discard, reroll, blind select/skip,
        # skip pack, end shop) get a constant uniform target dist from
        # _condition_target_on_action — a non-learnable ln(NUM_TARGETS)≈2.9-nat
        # term with ZERO gradient. Folding it into the totals unconditionally
        # pinned total_entropy at a fixed ~2.9 floor (the "entropy flat ~2.6,
        # never converging" symptom) and wasted the entropy budget on a no-op
        # dimension. Gate it off for no-target types so entropy reflects the
        # heads that can actually learn. (audit 06-13 CRITICAL)
        target_used = torch.zeros_like(card_used)
        for _at in ACTION_TARGET_RANGES:
            target_used = target_used + (type_action == _at).float()

        # Log probabilities
        type_log_prob = type_dist.log_prob(type_action)
        card_log_prob = card_dist.log_prob(card_action).sum(dim=-1)
        target_log_prob = cond_target_dist.log_prob(target_action)

        total_log_prob = (type_log_prob + card_used * card_log_prob
                          + target_used * target_log_prob)

        # Entropy — target entropy uses the CONDITIONED dist (the one that
        # is actually sampled from); the unconditional dist pushed mass
        # toward targets that can never be sampled for the chosen type.
        type_entropy = type_dist.entropy()
        card_entropy = card_dist.entropy().sum(dim=-1)
        target_entropy = cond_target_dist.entropy()

        total_entropy = (type_entropy + card_used * card_entropy
                         + target_used * target_entropy)

        # Heuristic prior-KL (annealing teacher). Pull the policy's TYPE
        # distribution toward the heuristic's masked preference. This re-homes
        # the guidance the bias mask used to inject directly into the logits,
        # but as a SEPARATE loss term that PPO anneals to zero — so the policy
        # eventually owns the decision. KL=0 when only one type is legal
        # (forced) or when the policy already matches the heuristic.
        prior_type_dist = torch.distributions.Categorical(logits=prior_logits[:, :14])
        prior_kl = torch.distributions.kl.kl_divergence(prior_type_dist, type_dist)

        return action, total_log_prob, total_entropy, value, prior_kl

    def _condition_target_on_action(
        self,
        target_logits: torch.Tensor,
        type_action: torch.Tensor,
    ) -> torch.Tensor:
        """Mask target logits so only valid targets for the chosen action type remain.

        For example, if action_type=4 (buy_pack), only pack targets (5-6) are valid.
        Actions without a target (reroll, select_blind, etc.) get uniform targets.

        Args:
            target_logits: shape (batch, NUM_TARGETS) — masked logits
            type_action: shape (batch,) — chosen action type indices

        Returns:
            conditioned_logits: shape (batch, NUM_TARGETS)
        """
        batch_size = target_logits.shape[0]
        device = target_logits.device
        conditioned = torch.full_like(target_logits, -1e9)

        for action_type_val, (offset, count) in ACTION_TARGET_RANGES.items():
            # Find which batch elements chose this action type
            batch_mask = (type_action == action_type_val)
            if batch_mask.any():
                # Copy only the valid target range from original logits
                conditioned[batch_mask, offset:offset + count] = \
                    target_logits[batch_mask, offset:offset + count]

        # For action types NOT in ACTION_TARGET_RANGES (play, discard, reroll,
        # select/skip blind, skip pack, end shop), target doesn't matter.
        # Give them uniform logits so they pick anything (it'll be ignored).
        no_target_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for action_type_val in ACTION_TARGET_RANGES:
            no_target_mask &= (type_action != action_type_val)
        if no_target_mask.any():
            conditioned[no_target_mask] = 0.0  # uniform over all targets

        return conditioned

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters by component."""
        counts = {}
        counts["trunk"] = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        counts["play_head"] = sum(p.numel() for p in self.play_head.parameters() if p.requires_grad)
        counts["shop_head"] = sum(p.numel() for p in self.shop_head.parameters() if p.requires_grad)
        counts["blind_head"] = sum(p.numel() for p in self.blind_head.parameters() if p.requires_grad)
        counts["value_head"] = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


# ============================================================
# Convenience
# ============================================================

def get_head_index(game_state: str) -> int:
    """Map BalatroBot game state to policy head index."""
    return STATE_TO_HEAD.get(game_state, HEAD_PLAY)


def create_network(device: str = "cuda") -> BalatronNetwork:
    """Create and move network to device."""
    net = BalatronNetwork()
    if device == "cuda" and torch.cuda.is_available():
        net = net.cuda()
    return net
