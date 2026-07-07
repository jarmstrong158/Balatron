"""Confidence-gated planner deferral tests (dec-061).

The gate routes EXISTING planner compute by the policy's per-decision confidence,
on the INFERENCE/EVAL path only. These tests pin the four behaviors the feature
promises:
  1. the gate fires when confidence is BELOW threshold and NOT when above;
  2. the planner path is taken only on low-confidence decisions;
  3. feature-off (and the extreme threshold) reproduces current play behavior;
  4. the TRAINING path never activates the gate (eval_mode is the choke point).

Run:  pytest tests/test_confidence_gate.py
"""

import math

import numpy as np
import torch

from agent.confidence_gate import (
    ConfidenceGate, gate_is_active, SIGNAL_ENTROPY, SIGNAL_TOP1,
)
from agent.network import BalatronNetwork
from environment.game_state import STATE_VECTOR_SIZE
from environment.action_space import ACTION_HEAD_SIZE, ACTION_BUY_JOKER
from training.action_executor import ActionExecutor


# ── confidence scoring ───────────────────────────────────────────────────────

def test_top1_confidence_is_the_top_probability():
    g = ConfidenceGate(enabled=True, signal=SIGNAL_TOP1)
    assert g.confidence(entropy=0.0, top1=0.9, n_legal=3) == 0.9
    assert g.confidence(entropy=0.0, top1=0.3, n_legal=5) == 0.3


def test_entropy_confidence_is_normalized_certainty():
    g = ConfidenceGate(enabled=True, signal=SIGNAL_ENTROPY)
    # uniform over the legal set -> H == log(n_legal) -> certainty 0.
    n = 4
    assert g.confidence(entropy=math.log(n), top1=1 / n, n_legal=n) == 0.0
    # a near-deterministic type dist -> H ~ 0 -> certainty ~ 1.
    assert g.confidence(entropy=0.0, top1=0.99, n_legal=n) == 1.0


def test_forced_single_legal_action_is_fully_certain():
    # nothing to route when only one action type is legal.
    for sig in (SIGNAL_ENTROPY, SIGNAL_TOP1):
        g = ConfidenceGate(enabled=True, signal=sig)
        assert g.confidence(entropy=0.0, top1=1.0, n_legal=1) == 1.0
        assert g.should_defer(g.confidence(entropy=0.0, top1=1.0, n_legal=1)) is False


# ── deferral decision (fires below, not above) ───────────────────────────────

def test_gate_fires_below_threshold_not_above():
    g = ConfidenceGate(enabled=True, signal=SIGNAL_TOP1, threshold=0.5)
    assert g.should_defer(0.4) is True     # below -> defer
    assert g.should_defer(0.5) is False    # at threshold -> keep policy sample
    assert g.should_defer(0.6) is False    # above -> keep policy sample


def test_threshold_zero_gates_nothing_superset_invariant():
    # The default extreme: even ENABLED, threshold 0.0 defers nothing, so the
    # feature is a provable superset of current behavior.
    g = ConfidenceGate(enabled=True, signal=SIGNAL_TOP1, threshold=0.0)
    for c in (0.0, 0.01, 0.5, 1.0):
        assert g.should_defer(c) is False


def test_threshold_one_gates_every_real_choice():
    g = ConfidenceGate(enabled=True, signal=SIGNAL_TOP1, threshold=1.0)
    assert g.should_defer(0.999) is True   # any real (multi-legal) choice
    # but a forced decision is still 1.0 -> not deferred
    assert g.should_defer(g.confidence(entropy=0.0, top1=1.0, n_legal=1)) is False


def test_disabled_gate_never_defers():
    g = ConfidenceGate(enabled=False, signal=SIGNAL_TOP1, threshold=1.0)
    assert g.should_defer(0.0) is False


# ── telemetry (deferral rate + confidence distribution) ──────────────────────

def test_stats_track_deferral_rate_and_distribution():
    g = ConfidenceGate(enabled=True, signal=SIGNAL_TOP1, threshold=0.5)
    for c in (0.1, 0.2, 0.8, 0.9):
        g.record(c, deferred=g.should_defer(c))
    s = g.stats()
    assert s["decisions"] == 4
    assert s["deferred"] == 2                 # 0.1 and 0.2 are below 0.5
    assert abs(s["deferral_rate"] - 0.5) < 1e-9
    assert abs(s["confidence_mean"] - 0.5) < 1e-9
    assert sum(s["confidence_hist"]) == 4


# ── the training-untouched invariant ─────────────────────────────────────────

def test_gate_inactive_on_training_path_even_when_enabled():
    """CRITICAL: eval_mode is the single choke point. With eval_mode=False (the
    training rollout collection path) the gate is inert no matter how it is
    configured — planner deferral can never enter the on-policy PPO data."""
    g = ConfidenceGate(enabled=True, signal=SIGNAL_TOP1, threshold=1.0)
    assert gate_is_active(g, eval_mode=False) is False   # training
    assert gate_is_active(g, eval_mode=True) is True      # eval


def test_gate_inactive_when_disabled_on_either_path():
    g = ConfidenceGate(enabled=False)
    assert gate_is_active(g, eval_mode=True) is False
    assert gate_is_active(g, eval_mode=False) is False


# ── planner recommendation (the deferral target) ─────────────────────────────

def _mask(*legal_types):
    m = np.zeros(ACTION_HEAD_SIZE, dtype=np.float32)
    for t in legal_types:
        m[t] = 1.0
    return m


def test_planner_recommended_action_is_buy_joker_in_shop_when_legal():
    ex = ActionExecutor(policy_authority=True)
    rec = ex.planner_recommended_action({"state": "SHOP"}, _mask(ACTION_BUY_JOKER, 7))
    assert rec is not None
    assert int(rec[0]) == ACTION_BUY_JOKER    # routes WHICH-joker to the planner


def test_planner_abstains_off_the_shop():
    ex = ActionExecutor(policy_authority=True)
    # play-time decisions (play/discard) are not the planner's domain
    assert ex.planner_recommended_action(
        {"state": "SELECTING_HAND"}, _mask(0, 1)) is None
    assert ex.planner_recommended_action(
        {"state": "BLIND_SELECT"}, _mask(9, 10)) is None


def test_planner_abstains_when_buy_joker_illegal():
    ex = ActionExecutor(policy_authority=True)
    # shop, but no affordable buyable joker -> buy_joker masked out
    assert ex.planner_recommended_action({"state": "SHOP"}, _mask(7, 13)) is None


def test_planner_abstains_without_policy_authority():
    ex = ActionExecutor(policy_authority=False)
    assert ex.planner_recommended_action(
        {"state": "SHOP"}, _mask(ACTION_BUY_JOKER)) is None


# ── network confidence exposure (no extra forward pass, no training change) ───

def _net_inputs():
    net = BalatronNetwork()
    net.eval()
    state = torch.zeros(1, STATE_VECTOR_SIZE)
    mask = torch.zeros(1, ACTION_HEAD_SIZE)
    for t in (0, 1, 2, 7):            # a few legal action types
        mask[0, t] = 1.0
    return net, state, mask


def test_return_confidence_default_keeps_the_training_5_tuple():
    net, state, mask = _net_inputs()
    with torch.no_grad():
        out = net.get_action_and_value(state, 0, mask)
    assert len(out) == 5             # exactly what PPO / training unpack


def test_return_confidence_adds_entropy_and_top1():
    net, state, mask = _net_inputs()
    with torch.no_grad():
        out = net.get_action_and_value(state, 0, mask, return_confidence=True)
    assert len(out) == 7
    type_entropy, top1 = out[5], out[6]
    assert float(type_entropy[0]) >= 0.0
    assert 0.0 <= float(top1[0]) <= 1.0


def test_confidence_flag_does_not_change_the_sampled_action():
    """The confidence signal is read off the SAME distribution AFTER sampling, so
    turning it on must not perturb the sample (feature-off == feature-on sample)."""
    net, state, mask = _net_inputs()
    with torch.no_grad():
        torch.manual_seed(1234)
        a_plain = net.get_action_and_value(state, 0, mask)[0]
        torch.manual_seed(1234)
        a_conf = net.get_action_and_value(state, 0, mask, return_confidence=True)[0]
    assert torch.equal(a_plain, a_conf)
