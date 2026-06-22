"""Unit tests for the modules extracted from train.py in the 06-14 decoupling.

Covers the parts that are pure/offline-testable:
  - training/config.py          TrainConfig.to_ppo_config field forwarding
  - training/episode_tracker.py EpisodeTracker accumulate/reset/isolation/stats
  - training/joker_order_logger.py round logging + active-state gating
  - training/env_session.py     per-run defaults + per-env isolation
  - training/action_executor.py ActionExecutor._encode_executed_action mapping
                                and _find_weakest_sellable_joker guards

The async game-driving methods (_action_to_api_call, _auto_*) need a live
BalatroBot env and are covered by the live smoke in the supervisor, not here.

Run:  pytest tests/test_decoupling.py   (or: python tests/test_decoupling.py)
"""

import json

import numpy as np
import pytest

from agent.ppo import PPOConfig
from recorder import NullRecorder
from training.config import TrainConfig
from training.episode_tracker import EpisodeTracker
from training.joker_order_logger import JokerOrderLogger
from training.env_session import EnvSession
from training.action_executor import ActionExecutor, _find_weakest_sellable_joker


# --------------------------------------------------------------------------
# config.py
# --------------------------------------------------------------------------

def test_trainconfig_forwards_fields_to_ppo_config():
    cfg = TrainConfig(learning_rate=1e-3, gamma=0.97, entropy_coef=0.07,
                      num_envs=3, num_epochs=5, num_minibatches=2)
    pc = cfg.to_ppo_config()
    assert isinstance(pc, PPOConfig)
    assert pc.learning_rate == 1e-3
    assert pc.gamma == 0.97
    assert pc.entropy_coef == 0.07
    assert pc.num_envs == 3
    assert pc.num_epochs == 5
    assert pc.num_minibatches == 2
    # pass-through of unset fields
    assert pc.gae_lambda == cfg.gae_lambda
    assert pc.target_kl == cfg.target_kl
    assert pc.clip_epsilon == cfg.clip_epsilon


# --------------------------------------------------------------------------
# episode_tracker.py
# --------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path, monkeypatch):
    # Isolate disk writes so tests never touch real logs/lifetime_stats.json.
    monkeypatch.setattr(EpisodeTracker, "STATS_FILE", str(tmp_path / "stats.json"))
    monkeypatch.setattr(EpisodeTracker, "WIN_LOG_FILE", str(tmp_path / "wins.json"))
    return EpisodeTracker()


def test_episode_accumulates_and_resets_per_episode(tracker):
    tracker.step(1.0, ante=2, env_id=0)
    tracker.step(2.0, ante=3, env_id=0)
    assert tracker.episode_length(0) == 2
    tracker.end_episode(won=False, env_id=0)
    assert tracker.total_rewards == [3.0]
    assert tracker.max_antes == [3]
    # The reset bug this guards: the accumulator must be DISCARDED on
    # end_episode so the next episode's R is its own, not a running sum.
    assert tracker.episode_length(0) == 0
    tracker.step(5.0, ante=4, env_id=0)
    tracker.end_episode(won=False, env_id=0)
    assert tracker.total_rewards == [3.0, 5.0]   # not [3.0, 8.0]


def test_episode_per_env_isolation(tracker):
    # Multi-instance: a stale accumulator from env 0 must not bleed into env 1.
    tracker.step(1.0, ante=2, env_id=0)
    tracker.step(9.0, ante=7, env_id=1)
    tracker.end_episode(won=False, env_id=0)
    assert tracker.total_rewards == [1.0]        # only env 0's reward
    assert tracker.max_antes == [2]
    assert tracker.episode_length(1) == 1         # env 1 untouched


def test_episode_recent_stats_and_win_rate(tracker):
    for ante in [3, 5, 9, 4]:                      # exactly one win (ante > 8)
        tracker.step(1.0, ante=ante, env_id=0)
        tracker.end_episode(won=(ante > 8), env_id=0)
    s = tracker.get_recent_stats()
    assert s["episodes"] == 4
    assert s["max_ante"] == 9
    assert s["mean_ante"] == pytest.approx((3 + 5 + 9 + 4) / 4)
    assert s["win_rate"] == pytest.approx(0.25)


def test_episode_highest_hand_from_chip_delta(tracker):
    tracker.step(0.0, ante=1, raw_state={"round": {"chips": 100}}, env_id=0)
    tracker.step(0.0, ante=1, raw_state={"round": {"chips": 600}}, env_id=0)  # +500
    assert tracker.session_highest_score == 500


def test_episode_empty_stats(tracker):
    s = tracker.get_recent_stats()
    assert s["episodes"] == 0 and s["mean_ante"] == 0.0 and s["win_rate"] == 0.0


# --------------------------------------------------------------------------
# joker_order_logger.py
# --------------------------------------------------------------------------

def test_joker_logger_writes_one_entry(tmp_path):
    log = JokerOrderLogger(log_dir=str(tmp_path))
    log.round_start(ante=2, round_num=1, blind_name="Small", blind_score=300, joker_keys=[])
    log.log_play("Pair", ["AS", "AH"], None, None, None, None)
    log.round_end()
    path = tmp_path / "joker_order_log.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["ante"] == 2
    assert entry["total_plays"] == 1


def test_joker_logger_ignores_when_inactive(tmp_path):
    log = JokerOrderLogger(log_dir=str(tmp_path))
    # No round_start -> not active -> log_play is a no-op, round_end writes nothing.
    log.log_play("Pair", ["AS"], None, None, None, None)
    log.round_end()
    assert not (tmp_path / "joker_order_log.jsonl").exists()


# --------------------------------------------------------------------------
# env_session.py
# --------------------------------------------------------------------------

def test_env_session_defaults_and_null_recorder():
    env = EnvSession(env_id=1, port=12347, phase=1)
    assert env.env_id == 1 and env.port == 12347
    assert env.win_recorded is False
    assert env.win_reward_stored is False
    assert env.shop_rerolls == 0
    assert env.current_ante == 1
    assert env.last_action_succeeded is True
    assert isinstance(env.recorder, NullRecorder)


def test_env_session_instances_are_isolated():
    e0 = EnvSession(env_id=0, port=12346, phase=1)
    e1 = EnvSession(env_id=1, port=12347, phase=1)
    e0.win_recorded = True
    e0.shop_rerolls = 3
    assert e1.win_recorded is False         # no singleton bleed
    assert e1.shop_rerolls == 0
    assert e0.game is not e1.game           # separate game clients
    assert e0.reward_calc is not e1.reward_calc


# --------------------------------------------------------------------------
# action_executor.py
# --------------------------------------------------------------------------

def test_action_executor_init():
    ax = ActionExecutor(policy_authority=True)
    assert ax.policy_authority is True
    assert ax._shop_block_count == 0


def _enc(method, params=None):
    ax = ActionExecutor()
    return ax._encode_executed_action(method, params, np.zeros(14, dtype=np.float32))


def test_encode_play_and_discard():
    assert _enc("play", {"cards": [0, 1]})[0] == 0
    assert _enc("discard", {"cards": [2]})[0] == 1


def test_encode_buy_targets_and_out_of_range():
    a = _enc("buy", {"card": 1}); assert a[0] == 2 and a[13] == 1
    a = _enc("buy", {"voucher": 1}); assert a[0] == 3 and a[13] == 4
    a = _enc("buy", {"pack": 0}); assert a[0] == 4 and a[13] == 5
    assert _enc("buy", {"card": 3}) is None      # shop joker slot out of head range
    assert _enc("buy", {"voucher": 2}) is None
    assert _enc("buy", {}) is None               # nothing to buy


def test_encode_sell_reroll_use():
    a = _enc("sell", {"joker": 2}); assert a[0] == 5 and a[13] == 9
    assert _enc("sell", {"joker": 9}) is None
    assert _enc("reroll")[0] == 7
    # 'use' sets the card bits from the real targets (gated dim for type 8)
    a = _enc("use", {"consumable": 0, "cards": [3, 5]})
    assert a[0] == 8 and a[13] == 12
    assert a[4] == 1.0 and a[6] == 1.0 and a[1] == 0.0
    assert _enc("use", {"consumable": 2}) is None


def test_encode_blind_and_noop():
    assert _enc("select")[0] == 9
    assert _enc("skip")[0] == 10
    assert _enc("next_round")[0] == 13
    assert _enc("gamestate") is None             # no action-tensor equivalent


def test_encode_preserves_other_action_slots():
    # A no-target action only writes a[0]; the sampled target/cards are kept.
    sampled = np.arange(14, dtype=np.float32)
    a = ActionExecutor()._encode_executed_action("reroll", None, sampled)
    assert a[0] == 7
    assert a[13] == sampled[13]                   # target untouched


def test_find_weakest_sellable_joker_guards():
    # No jokers -> nothing sellable.
    idx, val = _find_weakest_sellable_joker([], {})
    assert idx == -1
    # An eternal joker is never sellable -> skipped -> still nothing found.
    jokers = [{"modifier": {"eternal": True}, "key": "j_joker"}]
    assert _find_weakest_sellable_joker(jokers, {})[0] == -1
    # A negative-edition joker is likewise protected.
    jokers = [{"modifier": {"edition": "NEGATIVE"}, "key": "j_joker"}]
    assert _find_weakest_sellable_joker(jokers, {})[0] == -1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_weakest_sellable_prefers_nonxmult_engine_protection():
    """dec-027 engine protection: _find_weakest_sellable_joker must prefer a
    non-xmult joker so a freshly-stacked xmult scaler isn't churned away as
    'weakest' (its value is future compounding, undervalued now)."""
    from training.action_executor import _find_weakest_sellable_joker
    jokers = [{"id": 1, "key": "j_cavendish", "label": "Cavendish"},  # xmult
              {"id": 2, "key": "j_bull", "label": "Bull"}]            # additive
    raw = {"jokers": {"cards": jokers}, "round": {}}
    idx, _ = _find_weakest_sellable_joker(jokers, raw)
    assert idx == 1, "should sell the non-xmult (Bull), not the xmult engine"

    # If the WHOLE roster is xmult, it may fall back to selling one.
    all_x = [{"id": 1, "key": "j_cavendish", "label": "Cavendish"},
             {"id": 2, "key": "j_hologram", "label": "Hologram"}]
    idx2, _ = _find_weakest_sellable_joker(all_x, {"jokers": {"cards": all_x}, "round": {}})
    assert idx2 in (0, 1)   # no non-xmult available -> picks weakest xmult
