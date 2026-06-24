"""Tests for the relational/attention encoder graft (dec-031, plan C).

The encoder is added to a working agent, so the load-bearing guarantees are:
  1. REGRESSION-FREE INIT — at construction the zero-init output projections
     make the encoder a no-op: forward() must equal the pure trunk+head path.
     (An old checkpoint loads with ZERO behavior change.)
  2. NaN-SAFE on empty sets — early game has zero jokers (all-padded set) and
     SHOP/BLIND states have zero hand cards; neither may produce NaN.
  3. GRADIENT BOOTSTRAP — the zero-init projection receives gradient at step 0
     (so it lifts off zero), and once it is non-zero the encoder itself trains.
  4. OLD-CHECKPOINT COMPAT — a state_dict lacking the encoder keys loads via
     strict=False, leaving the encoder at fresh init.

Run:  pytest tests/test_set_encoder.py
"""

import torch
import pytest

from environment.game_state import STATE_VECTOR_SIZE
from agent.network import BalatronNetwork, HEAD_PLAY, HEAD_SHOP, HEAD_BLIND
from agent.set_encoder import (
    RelationalEncoder, ENCODER_OUT_DIM,
    OWNED_JOKER_OFFSET, HAND_CARD_OFFSET, SHOP_JOKER_OFFSET,
    JOKER_SLOTS, JOKER_SLOT_SIZE, HAND_CARD_SLOTS, HAND_CARD_SIZE,
    SHOP_JOKER_SLOTS, SHOP_JOKER_SIZE,
)


def _state(n_owned=2, n_shop=3, n_cards=5, seed=0):
    """Build a state vector with the given fill pattern (rest zero-padded)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(STATE_VECTOR_SIZE)
    # fill some global scalars
    s[:45] = torch.rand(45, generator=g)
    for i in range(n_owned):
        o = OWNED_JOKER_OFFSET + i * JOKER_SLOT_SIZE
        s[o:o + JOKER_SLOT_SIZE] = torch.rand(JOKER_SLOT_SIZE, generator=g) + 0.1
    for i in range(n_shop):
        o = SHOP_JOKER_OFFSET + i * SHOP_JOKER_SIZE
        s[o:o + SHOP_JOKER_SIZE] = torch.rand(SHOP_JOKER_SIZE, generator=g) + 0.1
    for i in range(n_cards):
        o = HAND_CARD_OFFSET + i * HAND_CARD_SIZE
        s[o:o + HAND_CARD_SIZE] = torch.rand(HAND_CARD_SIZE, generator=g) + 0.1
    return s


def test_offsets_match_state_size():
    # Sanity: the shop-joker block must end before the tail blocks (hand-eval,
    # context, velocity, xmult) — i.e. offsets track the real layout.
    shop_end = SHOP_JOKER_OFFSET + SHOP_JOKER_SLOTS * SHOP_JOKER_SIZE
    assert OWNED_JOKER_OFFSET < HAND_CARD_OFFSET < SHOP_JOKER_OFFSET < shop_end <= STATE_VECTOR_SIZE


def test_encoder_output_shape_and_finite():
    enc = RelationalEncoder()
    batch = torch.stack([_state(2, 3, 5, 1), _state(0, 0, 0, 2), _state(5, 3, 12, 3)])
    out = enc(batch)
    assert out.shape == (3, ENCODER_OUT_DIM)
    assert torch.isfinite(out).all(), "encoder produced NaN/Inf"


def test_nan_safe_no_jokers_and_no_cards():
    # No jokers at all (all-padded joker set) and no hand cards (shop state).
    enc = RelationalEncoder()
    s_nojoker = _state(n_owned=0, n_shop=0, n_cards=5)
    s_shop = _state(n_owned=2, n_shop=3, n_cards=0)   # cards empty in shop
    out = enc(torch.stack([s_nojoker, s_shop]))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("head", [HEAD_PLAY, HEAD_SHOP, HEAD_BLIND])
def test_regression_free_init(head):
    """At init the zero-init projections make the encoder a no-op: forward()
    must equal the pure trunk+head path (so loading an old checkpoint is exact)."""
    net = BalatronNetwork()
    net.eval()
    state = _state(3, 3, 8).unsqueeze(0)
    with torch.no_grad():
        logits, value = net.forward(state, head)
        ref_logits = net._policy_heads[head](net.trunk(state))
        ref_value = net.value_head(net.value_trunk(state))
    assert torch.allclose(logits, ref_logits, atol=1e-6)
    assert torch.allclose(value, ref_value, atol=1e-6)


def test_regression_free_forward_mixed():
    net = BalatronNetwork()
    net.eval()
    states = torch.stack([_state(2, 3, 8, 1), _state(4, 2, 0, 2)])
    heads = torch.tensor([HEAD_PLAY, HEAD_SHOP])
    with torch.no_grad():
        logits, values = net.forward_mixed(states, heads)
        # value path is encoder-free at init -> equals pure value trunk
        ref_values = net.value_head(net.value_trunk(states))
    assert torch.allclose(values, ref_values, atol=1e-6)
    assert logits.shape == (2, net.action_size)


def test_gradient_bootstrap():
    """Step 0: the zero-init projection must receive gradient (so it lifts off
    zero). Once the projection is non-zero, the encoder itself must train."""
    net = BalatronNetwork()
    state = _state(3, 3, 8).unsqueeze(0)

    logits, value = net.forward(state, HEAD_SHOP)
    (logits.sum() + value.sum()).backward()

    # Projection gets non-zero gradient even though its weights start at zero.
    assert net.attn_to_policy.weight.grad is not None
    assert net.attn_to_policy.weight.grad.abs().sum() > 0, "projection got no gradient"

    # Encoder gradient is ZERO at step 0 (projection weights are zero) — that's
    # the documented one-step bootstrap. After the projection is non-zero, the
    # encoder must receive gradient.
    net.zero_grad()
    with torch.no_grad():
        net.attn_to_policy.weight.add_(0.01)   # simulate the projection lifting off zero
    logits, value = net.forward(state, HEAD_SHOP)
    (logits.sum() + value.sum()).backward()
    g = net.encoder.joker_proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "encoder never receives gradient"


def test_old_checkpoint_loads_without_encoder_keys():
    """A checkpoint saved before the encoder existed lacks encoder/attn keys.
    strict=False load must fill the rest and leave the encoder at fresh init,
    and the loaded net must still be regression-free (encoder = no-op)."""
    net = BalatronNetwork()
    full = net.state_dict()
    old = {k: v for k, v in full.items()
           if not (k.startswith("encoder.") or k.startswith("attn_to_"))}

    fresh = BalatronNetwork()
    missing, unexpected = fresh.load_state_dict(old, strict=False)
    assert unexpected == [], f"unexpected keys: {unexpected}"
    # every missing key belongs to the new encoder graft
    assert all(k.startswith("encoder.") or k.startswith("attn_to_") for k in missing)
    assert any(k.startswith("encoder.") for k in missing)

    # Projections must still be zero (fresh init) -> regression-free.
    assert fresh.attn_to_policy.weight.abs().sum() == 0
    assert fresh.attn_to_value.weight.abs().sum() == 0
    state = _state(3, 3, 8).unsqueeze(0)
    with torch.no_grad():
        logits, value = fresh.forward(state, HEAD_SHOP)
        ref = fresh._policy_heads[HEAD_SHOP](fresh.trunk(state))
    assert torch.allclose(logits, ref, atol=1e-6)


def test_param_count_grew_only_by_encoder():
    net = BalatronNetwork()
    counts = net.count_parameters()
    assert counts["encoder"] > 0
    assert counts["attn_proj"] == 2 * ENCODER_OUT_DIM * net.trunk_output_size + 2 * net.trunk_output_size


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
