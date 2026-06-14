"""Top-level training configuration.

Extracted verbatim from training/train.py (06-14 monolith decoupling). Holds
the TrainConfig dataclass and its mapping to the lower-level PPOConfig. Values
HERE win over ppo.py defaults for the forwarded fields.
"""

from dataclasses import dataclass

from agent.ppo import PPOConfig


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Training
    total_timesteps: int = 1_000_000     # Total environment steps
    rollout_steps: int = 2048            # Steps per rollout
    phase: int = 1                       # 1 = general, 2 = naneinf

    # PPO (forwarded to PPOConfig — values HERE win, not ppo.py defaults)
    learning_rate: float = 3e-4
    # 0.995: with 200-400 decisions/run, gamma=0.99 discounted the win
    # reward to ~5% at early-ante decisions (0.99^300) — the shop choices
    # that determine wins could barely feel it. 0.995^300 ~ 22%.
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.04   # 0.01 -> 0.025 -> 0.04 (06-13): the policy is
                                 # over-confident post-BC (real entropy ~0.2,
                                 # PL~0 — barely moving). 0.025 did NOT raise it,
                                 # so escalated to 0.04 to force exploration so
                                 # PPO can surpass the heuristic under Path A.
                                 # Watch entropy toward ~0.5-0.8; if play turns
                                 # erratic (ante craters & stays), dial back.
    value_coef: float = 0.5
    # 8 epochs: rollouts cost ~25 min of live game, the update costs
    # seconds — extract more learning per rollout, target_kl bounds drift.
    num_epochs: int = 8
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

    # Parallel game instances (ports 12346..12346+N-1); env 0 records
    num_envs: int = 1

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
            num_envs=self.num_envs,
        )
