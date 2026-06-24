"""Relational / attention encoder over the joker and card token sets (plan C).

WHY (dec-031): every reward / policy / curriculum lever nulled against the
ante ~3.7 plateau. The deep audits converged on a representational ceiling:
the policy ALREADY chooses which joker to buy (action_executor policy_authority),
but it does so from a FLAT 842-dim vector where jokers live in fixed,
zero-padded slots. An MLP over that layout cannot easily REASON about the one
thing that gates run depth in Balatro — pairwise joker synergy and xmult
STACKING ("I own one xmult engine, so a second xmult that compounds is worth
far more than its standalone value"). The heuristic `upgrade_delta` feature
spoon-feeds a scalar hint, but the network can't relate joker-to-joker itself.

WHAT: slice the joker tokens (5 owned + 3 shop) and card tokens (12 hand) back
out of the flat vector and run self-attention over the JOINT joker set, so the
network can directly attend "this shop joker" against "what I already own".
A learned CLS token summarizes the set (and guarantees no fully-masked row →
no softmax NaN when the player owns zero jokers, which is common early game).
Cards get a cheap DeepSets pool (mean+max) — secondary, mostly for the play head.

INTEGRATION (see network.py): the encoder's contribution is added to the trunk
outputs through ZERO-INITIALIZED projections, so a freshly-grafted encoder
contributes exactly zero at load — the loaded policy behaves identically, then
the encoder learns from the first update. Checkpoint-safe: STATE_VECTOR_SIZE is
unchanged; the new modules are simply fresh-init (the loader's migration path
keeps fresh weights for keys absent from old checkpoints).
"""

import torch
import torch.nn as nn

from environment.game_state import (
    GAME_META_SIZE, HAND_LEVELS_SIZE, DECK_COMP_SIZE, VOUCHER_SIZE,
    JOKER_SLOTS, JOKER_SLOT_SIZE,
    HAND_CARD_SLOTS, HAND_CARD_SIZE,
    CONSUMABLE_SLOTS, CONSUMABLE_SIZE,
    SHOP_JOKER_SLOTS, SHOP_JOKER_SIZE,
)

# ------------------------------------------------------------------
# Cumulative offsets into the flat state vector. These MUST mirror the
# assembly order in game_state._build_state_vector():
#   GAME_META, HAND_LEVELS, DECK_COMP, VOUCHER, OWNED_JOKERS, HAND_CARDS,
#   CONSUMABLES, SHOP_JOKERS, ...
# Derived from the same size constants so they track layout changes.
# ------------------------------------------------------------------
OWNED_JOKER_OFFSET = GAME_META_SIZE + HAND_LEVELS_SIZE + DECK_COMP_SIZE + VOUCHER_SIZE
HAND_CARD_OFFSET = OWNED_JOKER_OFFSET + JOKER_SLOTS * JOKER_SLOT_SIZE
_CONSUMABLE_OFFSET = HAND_CARD_OFFSET + HAND_CARD_SLOTS * HAND_CARD_SIZE
SHOP_JOKER_OFFSET = _CONSUMABLE_OFFSET + CONSUMABLE_SLOTS * CONSUMABLE_SIZE

# Owned and shop joker slots share the same 54-d width (51 fingerprint + 3
# meta); the trailing 3 dims differ in meaning (owned: synergy/guard/sell-delta;
# shop: cost/affordable/upgrade-delta), which the network disambiguates via the
# 2-d owned/shop type flag appended to each token.
N_JOKER_TOKENS = JOKER_SLOTS + SHOP_JOKER_SLOTS          # 8
JOKER_TOKEN_FEATS = JOKER_SLOT_SIZE + 2                  # +2 owned/shop one-hot

# Encoder hyperparameters (kept small — training is CPU-bound and the user is
# throughput-sensitive; 8-9 tokens make this cheap regardless).
D_MODEL = 128
N_HEAD = 4
N_LAYERS = 2
D_CARD = 64
FFN = 256

# Output embedding width: CLS joker summary (D_MODEL) + card mean + card max.
ENCODER_OUT_DIM = D_MODEL + 2 * D_CARD


class RelationalEncoder(nn.Module):
    """Self-attention over the joint joker set + DeepSets pool over hand cards.

    forward(state) -> (batch, ENCODER_OUT_DIM)
    Pure function of the flat state vector; holds no game logic.
    """

    def __init__(self):
        super().__init__()
        self.joker_proj = nn.Linear(JOKER_TOKEN_FEATS, D_MODEL)
        # Learned CLS token: always-valid summary slot (also prevents a fully
        # masked attention row when the player owns no jokers).
        self.cls = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FFN,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.joker_attn = nn.TransformerEncoder(
            enc_layer, num_layers=N_LAYERS, enable_nested_tensor=False)

        # Cards: cheap per-card projection, pooled (mean+max). No attention —
        # cards are zero in SHOP/BLIND states and secondary to the buy decision.
        self.card_proj = nn.Sequential(nn.Linear(HAND_CARD_SIZE, D_CARD), nn.GELU())

        self.out_dim = ENCODER_OUT_DIM

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]

        # --- Joker set: 5 owned + 3 shop, joint self-attention ---
        owned = state[:, OWNED_JOKER_OFFSET:OWNED_JOKER_OFFSET + JOKER_SLOTS * JOKER_SLOT_SIZE]
        owned = owned.view(B, JOKER_SLOTS, JOKER_SLOT_SIZE)
        shop = state[:, SHOP_JOKER_OFFSET:SHOP_JOKER_OFFSET + SHOP_JOKER_SLOTS * SHOP_JOKER_SIZE]
        shop = shop.view(B, SHOP_JOKER_SLOTS, SHOP_JOKER_SIZE)

        owned_flag = state.new_zeros(B, JOKER_SLOTS, 2)
        owned_flag[..., 0] = 1.0
        shop_flag = state.new_zeros(B, SHOP_JOKER_SLOTS, 2)
        shop_flag[..., 1] = 1.0
        tokens = torch.cat(
            [torch.cat([owned, owned_flag], dim=-1),
             torch.cat([shop, shop_flag], dim=-1)], dim=1
        )  # (B, 8, 56)

        # A slot is present iff its fingerprint is non-zero (zero-padded = empty).
        valid = tokens[..., :JOKER_SLOT_SIZE].abs().sum(-1) > 1e-6  # (B, 8)

        proj = self.joker_proj(tokens)                              # (B, 8, D_MODEL)
        cls = self.cls.expand(B, 1, D_MODEL)
        seq = torch.cat([cls, proj], dim=1)                        # (B, 9, D_MODEL)
        cls_valid = torch.zeros(B, 1, dtype=torch.bool, device=state.device)
        key_padding_mask = torch.cat([cls_valid, ~valid], dim=1)    # True = ignore
        attended = self.joker_attn(seq, src_key_padding_mask=key_padding_mask)
        joker_summary = attended[:, 0]                              # CLS output (B, D_MODEL)

        # --- Card set: DeepSets mean + max pool ---
        cards = state[:, HAND_CARD_OFFSET:HAND_CARD_OFFSET + HAND_CARD_SLOTS * HAND_CARD_SIZE]
        cards = cards.view(B, HAND_CARD_SLOTS, HAND_CARD_SIZE)
        cvalid = cards.abs().sum(-1) > 1e-6                         # (B, 12)
        cproj = self.card_proj(cards)                              # (B, 12, D_CARD)
        cm = cvalid.unsqueeze(-1).float()
        card_mean = (cproj * cm).sum(1) / cm.sum(1).clamp_min(1.0)  # (B, D_CARD)
        card_max = cproj.masked_fill(~cvalid.unsqueeze(-1), float("-inf")).max(1).values
        card_max = torch.where(torch.isinf(card_max), torch.zeros_like(card_max), card_max)

        return torch.cat([joker_summary, card_mean, card_max], dim=-1)  # (B, ENCODER_OUT_DIM)
