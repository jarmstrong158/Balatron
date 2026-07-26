"""dec-081: with slots FULL, a shop-joker buy is a LEGALITY question, not a
heuristic value veto (con-011).

The old branch demanded a >=1.1x gain in IMMEDIATE single-hand score, considering
only the heuristically-weakest sell candidate. Everything else fell through to
`continue`, leaving the mask at its np.zeros default -> buy ILLEGAL ->
`any_buyable_joker` False -> ACTION_BUY_JOKER hard-blocked -> the agent leaves the
shop. Since slots fill by ante ~3 that froze the build: 289 swaps in 31,185
full-slot shops (0.93%) while the planner wanted one in ~55%.

These tests pin BOTH arms, because the A/B's validity depends on the control being
byte-identical to the old behaviour.
"""
import importlib
import os
import sys



def _reload(flag: str):
    """Re-import action_space with BALATRON_SWAP_LEGALITY set to `flag`."""
    prev = os.environ.get("BALATRON_SWAP_LEGALITY")
    os.environ["BALATRON_SWAP_LEGALITY"] = flag
    sys.modules.pop("environment.action_space", None)
    mod = importlib.import_module("environment.action_space")
    return mod, prev


def _restore(prev):
    if prev is None:
        os.environ.pop("BALATRON_SWAP_LEGALITY", None)
    else:
        os.environ["BALATRON_SWAP_LEGALITY"] = prev
    sys.modules.pop("environment.action_space", None)
    importlib.import_module("environment.action_space")


def _owned(key, sell=3):
    return {"key": key, "joker_key": key, "label": key,
            "cost": {"buy": sell * 2, "sell": sell}, "modifier": {}}


def _shop(name, cost):
    key = "j_" + name.lower().replace(" ", "_")
    return {"key": key, "joker_key": key, "label": name, "set": "JOKER",
            "cost": {"buy": cost, "sell": max(1, cost // 2)}, "modifier": {}}


def _state(money, offers):
    """Full 5-slot build of cheap flat jokers — the real plateau shape."""
    owned = [_owned(k) for k in ("j_joker", "j_greedy_joker", "j_jolly",
                                 "j_zany_joker", "j_mad")]
    return {
        "ante_num": 4, "ante": 4, "state": "SHOP", "money": money,
        "jokers": {"cards": owned},
        "shop": {"cards": offers},
        "consumables": {"cards": [], "limit": 2},
        "hands": {"Flush": {"chips": 35, "mult": 4, "level": 1}},
        "blinds": {}, "round": {"reroll_cost": 5}, "used_vouchers": [],
    }


def _buy_reachable(mod, state, n_offers):
    mask = mod.build_action_mask(state)
    off = mod.NUM_ACTION_TYPES + mod.HAND_CARD_SLOTS
    type_ok = mask[mod.ACTION_BUY_JOKER] > 0
    target_ok = any(
        mask[off + mod.TARGET_SHOP_JOKER_OFFSET + i] > 0 for i in range(n_offers))
    return bool(type_ok and target_ok)


# A weak flat joker: does NOT clear the legacy 1.1x immediate-score bar.
MARGINAL = [_shop("Cloud 9", 5)]


def test_control_blocks_marginal_full_slot_buy():
    """Control arm must reproduce the OLD behaviour exactly: a joker that fails
    the 1.1x immediate-score test is unreachable when slots are full."""
    mod, prev = _reload("0")
    try:
        assert mod.SWAP_LEGALITY is False
        assert _buy_reachable(mod, _state(20, MARGINAL), 1) is False
    finally:
        _restore(prev)


def test_treatment_allows_affordable_full_slot_buy():
    """Treatment: the same buy is LEGAL because a sellable joker frees the slot
    and funds it. Whether it's worth it is the planner's call, not the mask's."""
    mod, prev = _reload("1")
    try:
        assert mod.SWAP_LEGALITY is True
        assert _buy_reachable(mod, _state(20, MARGINAL), 1) is True
    finally:
        _restore(prev)


def test_treatment_still_blocks_unaffordable():
    """Legality is a real gate, not a rubber stamp: a joker that cannot be paid
    for even after selling the best sell candidate stays blocked."""
    mod, prev = _reload("1")
    try:
        # $0 on hand, sell values are 3 -> a $40 joker is unaffordable
        assert _buy_reachable(mod, _state(0, [_shop("Cloud 9", 40)]), 1) is False
    finally:
        _restore(prev)


def test_treatment_leaves_open_slot_path_untouched():
    """dec-081 only touches the FULL-slot branch; with a free slot both arms
    must agree, so the A/B isolates the swap gate alone."""
    states = []
    for flag in ("0", "1"):
        mod, prev = _reload(flag)
        try:
            st = _state(20, MARGINAL)
            st["jokers"]["cards"] = st["jokers"]["cards"][:2]   # open slots
            states.append(_buy_reachable(mod, st, 1))
        finally:
            _restore(prev)
    assert states[0] == states[1]


def test_default_is_control():
    """Ship-safety: with no env var set the binary must behave as control, so a
    trainer restart can never silently deploy the treatment (dec-079 pattern)."""
    prev = os.environ.pop("BALATRON_SWAP_LEGALITY", None)
    try:
        sys.modules.pop("environment.action_space", None)
        mod = importlib.import_module("environment.action_space")
        assert mod.SWAP_LEGALITY is False
    finally:
        _restore(prev)
