"""Tests for self-imitation learning (SIL Phase 2) wired into the PPO update.

The core guarantee: with sil_coef>0 and a populated demo buffer, the SIL loss
is a real gradient that pulls the policy TOWARD the saved demo actions
(imitation), and it's a clean no-op when off / empty.
"""
import numpy as np
import torch

from agent.ppo import PPOTrainer, PPOConfig
from agent.network import BalatronNetwork
from demo_buffer import DemoBuffer
from environment.game_state import STATE_VECTOR_SIZE
from environment.action_space import ACTION_HEAD_SIZE

AD = 14


def _trainer(tmp_path, sil_coef=1.0):
    cfg = PPOConfig(num_envs=1, sil_coef=sil_coef, sil_batch_size=8,
                    learning_rate=1e-3)
    tr = PPOTrainer(BalatronNetwork(), cfg)
    return tr


def _demos_from_net(net, n=8, head=0):
    """Build demo transitions whose actions are SAMPLED FROM the net (so they
    are valid, evaluable actions) for random states under an all-legal mask."""
    states = np.random.randn(n, STATE_VECTOR_SIZE).astype(np.float32)
    masks = np.ones((n, ACTION_HEAD_SIZE), dtype=np.float32)
    s_t = torch.tensor(states)
    m_t = torch.tensor(masks)
    actions = []
    with torch.no_grad():
        for i in range(n):
            a, *_ = net.get_action_and_value(s_t[i:i+1], head, m_t[i:i+1])
            actions.append(np.asarray(a[0]).astype(np.float32))
    actions = np.stack(actions)
    heads = np.full(n, head, dtype=np.int64)
    return states, actions, masks, heads


def _buf(tmp_path, n=0):
    return DemoBuffer(100, STATE_VECTOR_SIZE, AD, ACTION_HEAD_SIZE,
                      path=str(tmp_path / "d.npz"))


def test_sil_off_when_coef_zero(tmp_path):
    tr = _trainer(tmp_path, sil_coef=0.0)
    b = _buf(tmp_path)
    b.add_trajectory(*_demos_from_net(tr.network))
    tr.demo_buffer = b
    assert tr._sil_loss() is None  # coef 0 → no SIL


def test_sil_off_when_no_buffer(tmp_path):
    tr = _trainer(tmp_path, sil_coef=1.0)
    tr.demo_buffer = None
    assert tr._sil_loss() is None


def test_sil_off_when_buffer_empty(tmp_path):
    tr = _trainer(tmp_path, sil_coef=1.0)
    tr.demo_buffer = _buf(tmp_path)       # empty
    assert tr._sil_loss() is None


def test_sil_loss_is_finite_tensor_with_grad(tmp_path):
    tr = _trainer(tmp_path, sil_coef=1.0)
    b = _buf(tmp_path)
    b.add_trajectory(*_demos_from_net(tr.network))
    tr.demo_buffer = b
    loss = tr._sil_loss()
    assert loss is not None
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_sil_pulls_policy_toward_demo_actions(tmp_path):
    """Gradient-descending the SIL loss should INCREASE the policy's log-prob
    of the demo actions (i.e. the -logprob loss goes down). This is the whole
    point: imitate your own winning runs."""
    tr = _trainer(tmp_path, sil_coef=1.0)
    states, actions, masks, heads = _demos_from_net(tr.network, n=8, head=0)
    b = _buf(tmp_path)
    b.add_trajectory(states, actions, masks, heads)
    tr.demo_buffer = b

    l0 = tr._sil_loss().item()
    for _ in range(30):
        tr.optimizer.zero_grad()
        loss = tr._sil_loss()
        loss.backward()
        tr.optimizer.step()
    l1 = tr._sil_loss().item()
    assert l1 < l0 - 1e-3, f"SIL did not reduce -logprob: {l0:.4f} -> {l1:.4f}"
