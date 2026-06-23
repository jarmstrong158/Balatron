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
    entropy_coef: float = 0.03   # 0.10 -> 0.03 (06-22, dec-029). With the VALUE
                                 # TRUNK now DECOUPLED (network.py value_trunk), the
                                 # policy gradient is no longer drowned by the value
                                 # gradient on a shared trunk, so the policy can
                                 # finally take a real step — and the 0.10 entropy
                                 # term was actively fighting it toward uniform.
                                 # History: 0.01->0.025->0.04->0.06->0.10 escalation
                                 # was a wrong-diagnosis lever (the limiter was the
                                 # frozen policy gradient, not entropy-starvation).
                                 # Watch: KL should rise from ~0.0045 toward target
                                 # 0.03 now. If play craters & stays, revert.
    value_coef: float = 0.25         # 0.5 -> 0.25 (06-21). AUDIT root-cause fix:
                                     # value & policy share the trunk, and at
                                     # VL~20 the value term (0.5*20=10) swamped
                                     # the policy+entropy gradient (~370x the
                                     # entropy term, ~2000x policy_loss) — which
                                     # is why entropy_coef 0.04->0.10 only moved
                                     # entropy ~0.45->0.48 and ante stayed flat.
                                     # Halving value_coef un-swamps the shared
                                     # trunk so entropy can reach the band AND the
                                     # policy gradient gets real influence. WATCH
                                     # EV: value fn is healthy (~0.7); if EV craters
                                     # (<0.5) we starved it — back off toward 0.4.
    # 8 epochs: rollouts cost ~25 min of live game, the update costs
    # seconds — extract more learning per rollout, target_kl bounds drift.
    num_epochs: int = 8
    num_minibatches: int = 4
    target_kl: float = 0.03

    # Self-imitation (SIL Phase 2): replay the demo buffer's saved winning/
    # high-ante runs through an imitation loss so the policy reinforces its own
    # rare successes instead of forgetting them. Off by default (0.0); flip on
    # once the demo buffer has a corpus. Small coef to avoid mode-collapse.
    sil_coef: float = 0.1            # SIL ACTIVE (06-21): replay banked wins.
    sil_batch_size: int = 256

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

    # Self-imitation demo capture (Phase 1: capture-only, behavior-neutral).
    # Logs the (state, action, mask, head) trajectory of any run that WINS or
    # reaches ante >= demo_min_ante to a persistent buffer, so Phase 2 can
    # replay the agent's own successes through the BC loss. Capture does NOT
    # touch the policy — safe to run alongside other experiments.
    # State-injection CURRICULUM (dec-030, plan A step 2). The policy is
    # near-converged for the current reward because wins are 0.5%-rare and
    # advantages are tiny. Harvest ante-4/5 partial-build states (BalatroBot
    # save endpoint) and start a fraction of rollouts from them (load) so wins
    # become frequent and advantages become real where engines matter. Anneals
    # to 0 so training finishes on full self-play. Self-bootstraps (library
    # starts empty -> fresh runs harvest -> loads kick in).
    curriculum_enabled: bool = True
    curriculum_prob: float = 0.4         # P(load a seed) at update 0
    curriculum_anneal_updates: int = 800  # anneal prob -> 0 over this many updates
    curriculum_min_ante: int = 4         # harvest runs at ante 4..5 with >=1 xmult
    seed_dir: str = "seeds"
    seed_capacity: int = 200             # cap the seed library (FIFO)

    collect_demos: bool = True
    demo_capacity: int = 30000          # transitions (~100-200 full runs)
    demo_min_ante: int = 6              # capture runs reaching >= this ante
    demo_path: str = "demos/win_demos.npz"
    demo_save_every: int = 1            # flush to disk every captured run —
                                        # captures are rare (wins/ante>=6) and
                                        # save() is atomic+cheap, so don't risk
                                        # losing them to the ~20-30min recycle.

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
            sil_coef=self.sil_coef,
            sil_batch_size=self.sil_batch_size,
        )
