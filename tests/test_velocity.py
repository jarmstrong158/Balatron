"""Tests for the per-joker growth-velocity observation feature (06-18).

Covers three concerns:
  1. ScalingTracker.get_velocity() behavior — grow→positive, stall→0,
     new joker→0, decay→negative, and window clamping.
  2. State-vector size consistency — STATE_VECTOR_SIZE grew by JOKER_SLOTS
     and the assembled vector still satisfies the offset assertion.
  3. Checkpoint migration — a network sized to the OLD (pre-velocity) state
     dim loads into a network sized to the NEW dim by zero-padding the
     appended columns, leaving every previously-trained weight aligned.
"""
import numpy as np
import pytest
import torch

from environment.game_state import (
    ScalingTracker,
    GameStateManager,
    STATE_VECTOR_SIZE,
    JOKER_SLOTS,
    JOKER_VELOCITY_SIZE,
)


# A simple per-hand flat-addition scaling joker: Green Joker = +1 mult/hand
# (and -1/discard). Ice Cream = pure decay, starts at 100 chips, -5 per hand.
GREEN = "j_green_joker"
ICE_CREAM = "j_ice_cream"


def _joker(slot_id, key=GREEN):
    return {"id": slot_id, "key": key}


def _play_hand(tracker):
    """Drive one 'hand played' update with no discards/face cards."""
    tracker.update(
        events=["any_hand_played"],
        event_counts={"any_hand_played": 1},
        action_context={"face_cards_scored": False, "hand_type_played": "High Card"},
    )


# ─────────────────────────── ScalingTracker.get_velocity ───────────────────

def test_velocity_new_joker_is_zero():
    """A freshly-acquired joker has <2 snapshots → no velocity yet."""
    t = ScalingTracker()
    t.on_jokers_changed([_joker(101)])
    assert t.get_velocity(101) == 0.0
    # Even after a single hand (one snapshot), still need two to measure Δ.
    _play_hand(t)
    assert t.get_velocity(101) == 0.0


def test_velocity_growing_joker_is_positive():
    """A compounding joker reports positive velocity over recent hands."""
    t = ScalingTracker()
    t.on_jokers_changed([_joker(101)])
    for _ in range(4):           # 4 hands → value climbs 0→4, 4 snapshots
        _play_hand(t)
    assert t.get_value(101) == pytest.approx(4.0)
    # oldest snapshot in window was 1.0 (after hand 1); current 4.0 → +3.0
    assert t.get_velocity(101) == pytest.approx(3.0)


def test_velocity_stalled_joker_decays_to_zero():
    """Once growth stops, velocity falls to 0 as the window fills with the
    plateau value (no longer growing = no longer 'valuable to keep scaling')."""
    t = ScalingTracker()
    t.on_jokers_changed([_joker(101)])
    for _ in range(4):
        _play_hand(t)
    assert t.get_velocity(101) > 0.0

    # Pin the value flat: Green Joker is +1/hand and -1/discard, so a hand with
    # a discard nets 0. After VELOCITY_WINDOW such net-zero hands the whole
    # window holds the same value → velocity reads 0.
    for _ in range(ScalingTracker.VELOCITY_WINDOW):
        t.update(
            events=["any_hand_played", "on_discard"],
            event_counts={"any_hand_played": 1, "on_discard": 1},
            action_context={"face_cards_scored": False},
        )
    assert t.get_velocity(101) == pytest.approx(0.0)


def test_velocity_window_clamps_to_recent_hands():
    """Velocity reflects only the last VELOCITY_WINDOW hands, not lifetime."""
    t = ScalingTracker()
    t.on_jokers_changed([_joker(101)])
    w = ScalingTracker.VELOCITY_WINDOW
    for _ in range(w + 10):      # play far more than the window
        _play_hand(t)
    # value has climbed by (w+10), but velocity sees only the last `w` hands:
    # current - oldest-in-window = (w-1) increments of +1.
    assert t.get_velocity(101) == pytest.approx(float(w - 1))


def test_velocity_new_slot_resets_after_swap():
    """Selling a joker and acquiring a new one in the same slot resets history."""
    t = ScalingTracker()
    t.on_jokers_changed([_joker(101)])
    for _ in range(4):
        _play_hand(t)
    assert t.get_velocity(101) > 0.0
    # Swap slot 101's joker for a different card id → old history is gone.
    t.on_jokers_changed([_joker(202)])
    assert t.get_velocity(101) == 0.0   # 101 no longer tracked
    assert t.get_velocity(202) == 0.0   # 202 brand new


def test_velocity_decaying_joker_is_negative():
    """A pure-decay joker (Ice Cream: 100 chips, -5/hand) reports negative
    velocity, flagging dead weight that's bleeding out."""
    t = ScalingTracker()
    t.on_jokers_changed([_joker(101, ICE_CREAM)])
    assert t.get_value(101) == pytest.approx(100.0)
    for _ in range(4):           # each hand sheds 5 chips: 100→95→90→85→80
        _play_hand(t)
    assert t.get_value(101) == pytest.approx(80.0)
    # oldest snapshot in window was 95 (after hand 1); current 80 → -15
    assert t.get_velocity(101) == pytest.approx(-15.0)


# ─────────────────────────── State-vector size consistency ──────────────────

def test_state_vector_size_grew_by_joker_slots():
    assert JOKER_VELOCITY_SIZE == JOKER_SLOTS


def test_encode_assembles_to_full_size():
    """A minimal raw gamestate encodes to exactly STATE_VECTOR_SIZE (the
    in-encoder offset assertion would raise otherwise)."""
    mgr = GameStateManager()
    raw = {
        "jokers": {"cards": [_joker(101)]},
        "cards": {"cards": []},
    }
    vec = mgr._build_state_vector(raw)
    assert vec.shape[0] == STATE_VECTOR_SIZE


def test_velocity_section_populated_in_vector():
    """After growth, the appended velocity dim for the owned joker is non-zero
    and the empty slots stay zero-padded."""
    mgr = GameStateManager()
    raw = {"jokers": {"cards": [_joker(101)]}, "cards": {"cards": []}}
    # Drive the manager's own tracker through several hands.
    mgr._scaling_tracker.on_jokers_changed([_joker(101)])
    for _ in range(4):
        _play_hand(mgr._scaling_tracker)
    vec = mgr._build_state_vector(raw)
    # velocity is no longer the LAST block — the xmult-stack (4) and plan-features
    # (8) blocks were appended after it, so slice velocity by its actual position.
    from environment.game_state import XMULT_STACK_SIZE, PLAN_FEATURES_SIZE
    v_end = STATE_VECTOR_SIZE - XMULT_STACK_SIZE - PLAN_FEATURES_SIZE
    velo = vec[v_end - JOKER_VELOCITY_SIZE:v_end]
    assert velo[0] > 0.0                      # owned, growing joker
    assert np.all(velo[1:] == 0.0)            # empty slots zero-padded


# ─────────────────────────── xmult-stack perception block ───────────────────

def test_xmult_stack_block(tmp_path=None):
    """The appended xmult-stack block exposes owned-xmult COUNT (always) and
    shop-xmult/stack-ready (SHOP only) — the dec-026 perception fix."""
    from environment.game_state import XMULT_STACK_SIZE, PLAN_FEATURES_SIZE
    mgr = GameStateManager()
    # own 1 xmult (Cavendish); shop has an affordable xmult (The Duo)
    raw = {"state": "SHOP", "money": 10, "ante_num": 3,
           "jokers": {"cards": [{"id": 1, "key": "j_cavendish"}], "limit": 5},
           "shop": {"cards": [{"id": 9, "key": "j_duo", "cost": {"buy": 5}}]},
           "round": {"reroll_cost": 5}, "cards": {"cards": []}}
    # xmult-stack is no longer the LAST block — plan-features (8) follow it.
    blk = mgr._build_state_vector(raw)[-(XMULT_STACK_SIZE + PLAN_FEATURES_SIZE):-PLAN_FEATURES_SIZE]
    assert blk[0] == pytest.approx(1 / 3)   # owned count 1/3
    assert blk[1] == 1.0                     # shop xmult available
    assert blk[2] > 0.0                      # compounding delta
    assert blk[3] == 1.0                     # stack-ready (own>=1 + affordable)

    # Outside SHOP: owned-count still on, shop features zero
    raw["state"] = "SELECTING_HAND"
    blk2 = mgr._build_state_vector(raw)[-(XMULT_STACK_SIZE + PLAN_FEATURES_SIZE):-PLAN_FEATURES_SIZE]
    assert blk2[0] == pytest.approx(1 / 3)   # count always on
    assert blk2[1] == 0.0 and blk2[3] == 0.0  # shop features gated off


def test_plan_features_block():
    """dec-042: the appended plan block exposes the committed hand (one-hot over
    the 7 committable hands) so the policy can see the build it's meant to execute."""
    from environment.game_state import PLAN_FEATURES_SIZE
    mgr = GameStateManager()
    raw = {"state": "SHOP", "money": 5, "ante_num": 2,
           "jokers": {"cards": [], "limit": 5},
           "hands": {"Flush": {"chips": 300, "mult": 30}},  # leveled -> committed
           "cards": {"cards": []}, "round": {}}
    blk = mgr._build_state_vector(raw)[-PLAN_FEATURES_SIZE:]
    assert blk[:7].sum() == pytest.approx(1.0)   # exactly one committed-hand bit set


# ─────────────────────────── Checkpoint migration ───────────────────────────

def test_checkpoint_migration_zero_pads_appended_columns():
    """A first-layer weight matrix trained at the OLD state dim loads into the
    NEW dim by zero-padding the 5 appended columns, preserving trained weights."""
    old_dim = STATE_VECTOR_SIZE - JOKER_VELOCITY_SIZE
    new_dim = STATE_VECTOR_SIZE
    hidden = 64

    # Simulate a trained first layer at the old input dim.
    saved_w = torch.randn(hidden, old_dim)
    saved_b = torch.randn(hidden)

    # The migration logic from ppo.load_checkpoint: 2D input-dim grew.
    target_w_shape = (hidden, new_dim)
    padded_w = torch.zeros(target_w_shape, dtype=saved_w.dtype)
    padded_w[:, :saved_w.shape[1]] = saved_w

    # Trained columns are byte-identical; appended columns are zero.
    assert torch.equal(padded_w[:, :old_dim], saved_w)
    assert torch.all(padded_w[:, old_dim:] == 0.0)
    assert padded_w.shape == target_w_shape
    # A new feature with zero input weight contributes nothing on first load.
    assert padded_w[:, old_dim:].sum().item() == 0.0
