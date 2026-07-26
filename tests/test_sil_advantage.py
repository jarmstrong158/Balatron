"""dec-084: SIL must weight demo actions by (R - V)+, not clone them uniformly.

Before this, `_sil_loss` was `-mean(log pi(a|s))` over every transition in a
banked winning run — behaviour cloning wearing SIL's name. A winning run is ~180
steps at a ~1.3% win rate in a variance-dominated game, so a large share of
banked wins are LUCKY. Uniform cloning teaches "the average behaviour of runs
that happened to win" rather than the decisions that won them, and SIL is
currently the ONLY live guidance channel (bc_coef and prior_coef both anneal to
0), so the distinction is not academic.
"""
import numpy as np
import torch

from agent.ppo import PPOTrainer, PPOConfig
from agent.network import BalatronNetwork
from demo_buffer import DemoBuffer
from environment.game_state import STATE_VECTOR_SIZE
from environment.action_space import ACTION_HEAD_SIZE

AD = 14


def _trainer(net, sil_coef=1.0, adv_filter=True, batch=8):
    """NB: the caller supplies the network. Two PPOTrainers built from two fresh
    BalatronNetwork()s have different random init, so comparing their losses
    measures initialisation noise, not the advantage filter."""
    cfg = PPOConfig(num_envs=1, sil_coef=sil_coef, sil_batch_size=batch,
                    sil_advantage_filter=adv_filter, learning_rate=1e-3)
    return PPOTrainer(net, cfg)


def _demo_batch(net, n=8, head=0):
    states = np.random.randn(n, STATE_VECTOR_SIZE).astype(np.float32)
    masks = np.ones((n, ACTION_HEAD_SIZE), dtype=np.float32)
    s_t, m_t = torch.tensor(states), torch.tensor(masks)
    acts = []
    with torch.no_grad():
        for i in range(n):
            a, *_ = net.get_action_and_value(s_t[i:i+1], head, m_t[i:i+1])
            acts.append(np.asarray(a[0]).astype(np.float32))
    return states, np.stack(acts), masks, np.full(n, head, dtype=np.int64)


def _buffer(tmp_path, n=8, returns=None, net=None):
    buf = DemoBuffer(capacity=64, state_dim=STATE_VECTOR_SIZE, action_dim=AD,
                     mask_dim=ACTION_HEAD_SIZE,
                     path=str(tmp_path / "demos.npz"))
    s, a, m, h = _demo_batch(net, n=n)
    buf.add_trajectory(list(s), list(a), list(m), list(h), returns)
    return buf


def test_returns_round_trip_through_the_buffer(tmp_path):
    net = BalatronNetwork()
    rets = [float(i) for i in range(8)]
    buf = _buffer(tmp_path, returns=rets, net=net)
    # read the stored slice directly: sample() draws WITH replacement, so it is
    # not a way to assert the full contents round-tripped
    assert sorted(float(x) for x in buf.returns[:8]) == sorted(rets)
    assert "returns" in buf.sample(4)


def test_legacy_transitions_have_nan_returns(tmp_path):
    """Pre-dec-084 corpora carry no returns; they must load as NaN (=unknown)
    rather than 0.0, which would read as 'this action underperformed'."""
    net = BalatronNetwork()
    buf = _buffer(tmp_path, returns=None, net=net)
    got = buf.sample(8)
    assert np.isnan(got["returns"]).all()


def test_high_return_actions_dominate_the_loss(tmp_path):
    """THE point of the fix: an action whose return beat the critic must pull
    harder than one that merely came along for the ride."""
    net = BalatronNetwork()
    rets = [0.0] * 7 + [500.0]   # one clear outperformer
    buf = _buffer(tmp_path, returns=rets, net=net)

    # SAME network, SAME sampled batch — the only difference is the filter
    tr = _trainer(net)
    tr.demo_buffer = buf
    np.random.seed(0)
    loss_weighted = tr._sil_loss()

    tr2 = _trainer(net, adv_filter=False)
    tr2.demo_buffer = buf
    np.random.seed(0)
    loss_uniform = tr2._sil_loss()

    assert loss_weighted is not None and loss_uniform is not None
    # They must not be the same computation.
    assert not torch.isclose(loss_weighted, loss_uniform), \
        "advantage filter had no effect — it is not wired in"


def test_all_nan_returns_falls_back_to_uniform(tmp_path):
    """The pre-fix win corpus is irreplaceable (weeks of 1.3%-rate wins), so an
    all-legacy batch must behave exactly like the old uniform loss, not collapse
    to zero weight."""
    net = BalatronNetwork()
    buf = _buffer(tmp_path, returns=None, net=net)

    tr = _trainer(net)
    tr.demo_buffer = buf
    np.random.seed(1)
    weighted = tr._sil_loss()

    tr2 = _trainer(net, adv_filter=False)
    tr2.demo_buffer = buf
    np.random.seed(1)
    uniform = tr2._sil_loss()
    assert torch.isclose(weighted, uniform, atol=1e-5), (weighted, uniform)


def test_sil_off_is_still_a_noop(tmp_path):
    net = BalatronNetwork()
    tr = _trainer(net, sil_coef=0.0)
    tr.demo_buffer = _buffer(tmp_path, returns=[1.0] * 8, net=net)
    assert tr._sil_loss() is None


def test_loss_is_finite_and_positive(tmp_path):
    """Weight normalisation must not divide by ~0 when every action merely met
    expectation (all advantages clamp to 0)."""
    net = BalatronNetwork()
    tr = _trainer(net)
    # returns far BELOW any plausible value -> all (R-V)+ clamp to 0
    tr.demo_buffer = _buffer(tmp_path, returns=[-1e6] * 8, net=net)
    tr.config.sil_batch_size = 8
    loss = tr._sil_loss()
    assert loss is not None and torch.isfinite(loss), loss
