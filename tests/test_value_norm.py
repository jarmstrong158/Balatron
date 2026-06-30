"""dec-054: value-target normalization (PopArt-lite).

Locks in the two invariants that make it safe to ship default-off and enable later:
  1. value_norm=False is byte-identical (normalizer stays (0,1); stored value
     is unchanged).
  2. value_norm=on path denormalizes the stored value by the running stats, so
     GAE/returns stay in raw reward space while the head learns normalized values.
"""
import numpy as np
from agent.ppo import PPOTrainer, PPOConfig
from agent.network import BalatronNetwork
from environment.game_state import STATE_VECTOR_SIZE
from environment.action_space import ACTION_HEAD_SIZE


def _store(trainer, value):
    trainer.store_transition(
        state=np.zeros(STATE_VECTOR_SIZE, dtype=np.float32),
        action=np.zeros(14, dtype=np.float32),
        log_prob=0.0, reward=1.0, value=value, done=False,
        mask=np.ones(ACTION_HEAD_SIZE, dtype=np.float32),
        game_state="SELECTING_HAND", env_id=0,
    )
    return trainer.buffers[0].values[trainer.buffers[0].pos - 1]


def test_value_norm_off_is_identity():
    t = PPOTrainer(BalatronNetwork(), PPOConfig(value_norm=False))
    assert t.ret_mean == 0.0 and t.ret_std == 1.0
    assert _store(t, 42.0) == 42.0   # stored unchanged


def test_value_norm_denormalizes_stored_value():
    t = PPOTrainer(BalatronNetwork(), PPOConfig(value_norm=True))
    t.ret_mean, t.ret_std = 30.0, 40.0   # pretend stats have adapted
    # head emits a normalized value of 2.0 -> stored raw = 2*40 + 30 = 110
    assert abs(_store(t, 2.0) - 110.0) < 1e-4
