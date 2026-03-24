"""
Balatron — PPO (Proximal Policy Optimization)

Implements:
- RolloutBuffer: stores transitions from episode collection
- GAE: Generalized Advantage Estimation
- PPOTrainer: clipped surrogate loss + value loss + entropy bonus

Standard PPO with the following Balatron-specific additions:
- Mixed head indices per transition (different policy heads per game state)
- Action mask storage and replay for valid action enforcement
- Composite action format: [type(1) + cards(12) + target(1)]

See NOTES.md for hyperparameter choices.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent.network import BalatronNetwork, get_head_index
from environment.game_state import STATE_VECTOR_SIZE
from environment.action_space import ACTION_HEAD_SIZE


# ============================================================
# Hyperparameters
# ============================================================

@dataclass
class PPOConfig:
    """All PPO hyperparameters in one place."""

    # Core PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda
    clip_epsilon: float = 0.2        # Surrogate clipping range
    clip_value: float = 0.2          # Value function clipping range
    entropy_coef: float = 0.01       # Entropy bonus weight
    value_coef: float = 0.5          # Value loss weight
    max_grad_norm: float = 0.5       # Gradient clipping

    # Training schedule
    num_epochs: int = 4              # PPO epochs per rollout
    num_minibatches: int = 4         # Minibatches per epoch
    rollout_steps: int = 2048        # Steps per rollout before update

    # Optimization
    adam_eps: float = 1e-5           # Adam epsilon
    weight_decay: float = 0.0       # L2 regularization

    # Annealing
    anneal_lr: bool = True           # Linear LR annealing
    target_kl: Optional[float] = 0.03  # Early stop if KL exceeds this

    # Device
    device: str = "cpu"              # "cpu" or "cuda"

    # Action dimensions (from action_space.py)
    action_size: int = 14            # [type(1) + cards(12) + target(1)]
    action_head_size: int = ACTION_HEAD_SIZE  # 45 (network output)


# ============================================================
# Rollout Buffer
# ============================================================

class RolloutBuffer:
    """Stores transitions from episode rollouts for PPO training.

    Fixed-size buffer that gets filled during collection, then
    consumed during the PPO update. Reset between rollouts.
    """

    def __init__(self, capacity: int, state_dim: int = STATE_VECTOR_SIZE,
                 action_dim: int = 14, mask_dim: int = ACTION_HEAD_SIZE,
                 device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.full = False

        # Pre-allocate numpy arrays (filled during collection on CPU)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, mask_dim), dtype=np.float32)
        self.head_indices = np.zeros(capacity, dtype=np.int64)

        # Computed after rollout
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, log_prob: float,
            reward: float, value: float, done: bool, mask: np.ndarray,
            head_idx: int):
        """Add a single transition to the buffer."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = float(done)
        self.masks[self.pos] = mask
        self.head_indices[self.pos] = head_idx
        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool,
                    gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation.

        Must be called after rollout is complete and before get_batches().

        Args:
            last_value: V(s_{T+1}) — bootstrap value for the last state
            last_done: whether the last state was terminal
            gamma: discount factor
            gae_lambda: GAE lambda
        """
        n = self.pos
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, num_minibatches: int):
        """Yield shuffled minibatches as tensors on the target device.

        Args:
            num_minibatches: number of minibatches to split into

        Yields:
            dict with keys: states, actions, log_probs, advantages,
                           returns, masks, head_indices
        """
        n = self.pos
        indices = np.random.permutation(n)
        batch_size = n // num_minibatches

        for start in range(0, n - batch_size + 1, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield {
                "states": torch.tensor(self.states[batch_idx], device=self.device),
                "actions": torch.tensor(self.actions[batch_idx], device=self.device),
                "log_probs": torch.tensor(self.log_probs[batch_idx], device=self.device),
                "advantages": torch.tensor(self.advantages[batch_idx], device=self.device),
                "returns": torch.tensor(self.returns[batch_idx], device=self.device),
                "masks": torch.tensor(self.masks[batch_idx], device=self.device),
                "head_indices": torch.tensor(self.head_indices[batch_idx],
                                             device=self.device, dtype=torch.long),
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.pos


# ============================================================
# PPO Trainer
# ============================================================

class PPOTrainer:
    """PPO training algorithm.

    Handles:
    - Rollout buffer management
    - GAE advantage computation
    - Clipped surrogate policy loss
    - Clipped value loss
    - Entropy bonus
    - Minibatch SGD with gradient clipping
    - Learning rate annealing
    - KL divergence early stopping
    """

    def __init__(self, network: BalatronNetwork, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.network = network
        self.device = self.config.device

        # Move network to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.network = self.network.cuda()

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            capacity=self.config.rollout_steps,
            device=self.device,
        )

        # Training stats
        self.total_updates = 0
        self.total_steps = 0

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         log_prob: float, reward: float, value: float,
                         done: bool, mask: np.ndarray, game_state: str):
        """Store a transition from environment interaction.

        Args:
            state: state vector (520,)
            action: action array (14,) — [type, cards×12, target]
            log_prob: log probability of the action under current policy
            reward: shaped reward
            value: value estimate V(s)
            done: episode terminated
            mask: action validity mask (45,)
            game_state: BalatroBot state name for head routing
        """
        head_idx = get_head_index(game_state)
        self.buffer.add(state, action, log_prob, reward, value, done, mask, head_idx)
        self.total_steps += 1

    def update(self, last_value: float, last_done: bool) -> dict:
        """Run PPO update on collected rollout.

        Call this after filling the rollout buffer.

        Args:
            last_value: V(s) for the state after the last transition
            last_done: whether the last state was terminal

        Returns:
            dict of training metrics
        """
        cfg = self.config

        # Compute advantages
        self.buffer.compute_gae(last_value, last_done, cfg.gamma, cfg.gae_lambda)

        # Normalize advantages
        n = self.buffer.size
        adv = self.buffer.advantages[:n]
        adv_mean = adv.mean()
        adv_std = adv.std()
        if adv_std > 1e-8:
            self.buffer.advantages[:n] = (adv - adv_mean) / (adv_std + 1e-8)

        # Learning rate annealing
        if cfg.anneal_lr:
            # This is a simple version — caller should pass progress fraction
            pass  # Handled externally via set_learning_rate()

        # PPO epochs
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "num_epochs_run": 0,
        }
        num_batches = 0

        for epoch in range(cfg.num_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for batch in self.buffer.get_batches(cfg.num_minibatches):
                batch_metrics = self._update_batch(batch)

                for k in ["policy_loss", "value_loss", "entropy",
                          "total_loss", "clip_fraction"]:
                    metrics[k] += batch_metrics[k]

                epoch_kl += batch_metrics["approx_kl"]
                epoch_batches += 1
                num_batches += 1

            metrics["num_epochs_run"] += 1

            # KL divergence early stopping
            if cfg.target_kl is not None and epoch_batches > 0:
                mean_kl = epoch_kl / epoch_batches
                if mean_kl > cfg.target_kl:
                    break

        # Average metrics
        if num_batches > 0:
            for k in ["policy_loss", "value_loss", "entropy",
                      "total_loss", "approx_kl", "clip_fraction"]:
                metrics[k] /= num_batches

        # Explained variance
        values = self.buffer.values[:n]
        returns = self.buffer.returns[:n]
        var_returns = np.var(returns)
        if var_returns > 1e-8:
            metrics["explained_variance"] = 1.0 - np.var(returns - values) / var_returns
        else:
            metrics["explained_variance"] = 0.0

        self.total_updates += 1

        # Reset buffer for next rollout
        self.buffer.reset()

        return metrics

    def _update_batch(self, batch: dict) -> dict:
        """Run single minibatch PPO update.

        Returns dict of batch-level metrics.
        """
        cfg = self.config
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        masks = batch["masks"]
        head_indices = batch["head_indices"]

        # Forward pass — need to handle mixed heads
        # Group by head index for efficiency
        batch_size = states.shape[0]
        new_log_probs = torch.zeros(batch_size, device=self.device)
        entropy = torch.zeros(batch_size, device=self.device)
        new_values = torch.zeros(batch_size, device=self.device)

        for h_idx in range(3):
            head_mask = head_indices == h_idx
            if not head_mask.any():
                continue

            h_states = states[head_mask]
            h_actions = actions[head_mask]
            h_masks = masks[head_mask]

            _, h_log_probs, h_entropy, h_values = self.network.get_action_and_value(
                h_states, h_idx, h_masks, action=h_actions
            )

            new_log_probs[head_mask] = h_log_probs
            entropy[head_mask] = h_entropy
            new_values[head_mask] = h_values

        # Policy loss — clipped surrogate
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL divergence
        approx_kl = ((ratio - 1) - log_ratio).mean().item()

        # Clipped surrogate
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss — clipped
        value_loss_unclipped = (new_values - returns) ** 2
        if cfg.clip_value > 0:
            old_values = batch.get("old_values")
            if old_values is not None:
                values_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -cfg.clip_value, cfg.clip_value
                )
                value_loss_clipped = (values_clipped - returns) ** 2
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = value_loss_unclipped.mean()
        else:
            value_loss = value_loss_unclipped.mean()

        # Entropy bonus
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + cfg.value_coef * value_loss
            - cfg.entropy_coef * entropy_loss
        )

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        # Clip fraction (how often clipping was active)
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def set_learning_rate(self, lr: float):
        """Update learning rate (for annealing)."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, path: str):
        """Save network and optimizer state."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
            "total_steps": self.total_steps,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load network and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except RuntimeError as e:
            # State vector size changed — pad or trim mismatched params
            print(f"[WARN] Checkpoint shape mismatch, migrating weights: {e}")
            saved_state = checkpoint["network_state_dict"]
            current_state = self.network.state_dict()
            compatible = {}
            padded_keys = []
            skipped_keys = []
            for k, v in saved_state.items():
                if k not in current_state:
                    skipped_keys.append(k)
                    continue
                target_shape = current_state[k].shape
                if v.shape == target_shape:
                    compatible[k] = v
                elif len(v.shape) == 2 and len(target_shape) == 2 and v.shape[0] == target_shape[0]:
                    # Input dimension grew — zero-pad columns (new state features)
                    import torch
                    padded = torch.zeros(target_shape, dtype=v.dtype, device=v.device)
                    padded[:, :v.shape[1]] = v
                    compatible[k] = padded
                    padded_keys.append(f"{k}: {v.shape} → {target_shape}")
                elif len(v.shape) == 1 and len(target_shape) == 1 and v.shape[0] < target_shape[0]:
                    # Bias grew — zero-pad
                    import torch
                    padded = torch.zeros(target_shape, dtype=v.dtype, device=v.device)
                    padded[:v.shape[0]] = v
                    compatible[k] = padded
                    padded_keys.append(f"{k}: {v.shape} → {target_shape}")
                else:
                    skipped_keys.append(k)
            if padded_keys:
                print(f"[WARN] Zero-padded {len(padded_keys)} params: {padded_keys}")
            if skipped_keys:
                print(f"[WARN] Skipped {len(skipped_keys)} params: {skipped_keys}")
            current_state.update(compatible)
            self.network.load_state_dict(current_state)
            # Don't load optimizer — references old parameter shapes
        self.total_updates = checkpoint.get("total_updates", 0)
        self.total_steps = checkpoint.get("total_steps", 0)

    def get_stats(self) -> dict:
        """Get trainer statistics."""
        return {
            "total_updates": self.total_updates,
            "total_steps": self.total_steps,
            "learning_rate": self.get_learning_rate(),
            "buffer_size": self.buffer.size,
        }
