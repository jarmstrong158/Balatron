"""dec-060: save->spike economy — spend the war chest before a hard boss.

The reroll gate protects an interest reserve normally, but relaxes it before a
hard (dec-059) boss so the agent converts money into power at the gate it saved
for."""
from training.action_executor import ActionExecutor


class _Env:
    shop_rerolls = 0


def _state(boss=None, ante=6, money_ctx=True):
    gs = {"ante_num": ante, "round": {"reroll_cost": 5}, "used_vouchers": []}
    if boss:
        gs["blinds"] = {"boss": {"status": "UPCOMING", "name": boss}}
    return gs


def test_reroll_gate_holds_reserve_without_hard_boss():
    ax = ActionExecutor(policy_authority=True)
    # ante 6 floor is $25; with $28 and a $5 reroll, money-reroll=$23 < $25 -> blocked
    assert ax._planner_reroll_ok(_Env(), _state(boss=None), 28) is False


def test_reroll_gate_spikes_before_hard_boss():
    ax = ActionExecutor(policy_authority=True)
    # same $28, but a Wall upcoming relaxes the floor to $10 -> $23 >= $10 -> allowed
    assert ax._planner_reroll_ok(_Env(), _state(boss="The Wall"), 28) is True
    # a benign/unknown boss does NOT relax it (difficulty 1.0)
    assert ax._planner_reroll_ok(_Env(), _state(boss="The Mark"), 28) is False


def test_reroll_gate_still_caps_and_needs_buy_headroom():
    ax = ActionExecutor(policy_authority=True)
    # even before a hard boss, too little to buy after rerolling -> blocked
    assert ax._planner_reroll_ok(_Env(), _state(boss="The Wall"), 6) is False
    e = _Env(); e.shop_rerolls = 3   # per-shop cap still applies
    assert ax._planner_reroll_ok(e, _state(boss="The Wall"), 100) is False
