"""dec-075: the build ESCAPE HATCH.

The agent locks into a flat 5-joker build by ante 3 and cannot get out:
  * slots run 4.74-4.94/5 full by ante 4-5 (the modal death antes),
  * 48.7% of ante-4 builds own ZERO xmult; 36.6% of ante-4-6 deaths never got one,
  * and BOTH routes to an engine were closed:
      1. action_executor: the reroll-to-hunt block lives in the OPEN-slot branch,
         so at full slots a non-improving swap SILENTLY NO-OPed (4,960 zero-effect
         shop steps + 605 forced random pack-buys in one log).
      2. action_space: the mask HARD-ZEROED a legal ACTION_REROLL on heuristic
         grounds, making rerolling illegal in 96.3% of shops. con-011 forbids this:
         the mask is a legality gate; opinions belong in the annealing prior-KL.
"""
import numpy as np

from environment.action_space import build_action_mask, ACTION_REROLL


def _shop(buyable_joker=True, slots_full=False, money=12, reroll_cost=5):
    jok = [{"label": "Joker", "key": "j_joker", "cost": {"buy": 4, "sell": 2}}
           for _ in range(5 if slots_full else 2)]
    shop = ([{"key": "j_droll", "joker_key": "j_droll", "label": "Droll Joker",
              "set": "JOKER", "cost": {"buy": 4, "sell": 2}}] if buyable_joker else
            [{"key": "c_jupiter", "set": "PLANET", "label": "Jupiter",
              "cost": {"buy": 3, "sell": 1}}])
    return {"state": "SHOP", "money": money,
            "jokers": {"cards": jok, "limit": 5},
            "shop": {"cards": shop, "limit": 2},
            "packs": {"cards": []}, "consumables": {"cards": [], "limit": 2},
            "hand": {"cards": []}, "cards": {"cards": [{} for _ in range(40)]},
            "hands": {"Pair": {"chips": 10, "mult": 2, "level": 1, "played": 3}},
            "round": {"reroll_cost": reroll_cost, "hands_left": 4, "discards_left": 3},
            "ante_num": 4, "blinds": {}, "used_vouchers": []}


# ── con-011: the mask is a LEGALITY gate, not a heuristic veto ───────────────

def test_reroll_is_legal_when_a_joker_is_buyable():
    """Was hard-zeroed ('don't reroll past it') => illegal in 96.3% of shops,
    structurally preventing the agent from spinning for an xmult engine."""
    m = build_action_mask(_shop(buyable_joker=True))
    assert m[ACTION_REROLL] > 0


def test_reroll_is_legal_at_full_slots():
    """The ante-4-5 death state: slots full, build flat. Must be able to hunt."""
    m = build_action_mask(_shop(buyable_joker=True, slots_full=True))
    assert m[ACTION_REROLL] > 0


def test_reroll_is_legal_at_full_slots_with_thin_money():
    m = build_action_mask(_shop(buyable_joker=False, slots_full=True, money=9))
    assert m[ACTION_REROLL] > 0


def test_reroll_illegal_only_when_genuinely_unaffordable():
    """The ONE valid 0.0: real legality. This must still block."""
    m = build_action_mask(_shop(buyable_joker=True, money=6, reroll_cost=5))
    assert m[ACTION_REROLL] == 0.0


def test_affordable_planet_no_longer_vetoes_reroll():
    """Regression: `any_buyable_joker` is overloaded — dec-016 set it for PLANETS
    to keep BUY reachable, which silently vetoed reroll too."""
    m = build_action_mask(_shop(buyable_joker=False, money=40))
    assert m[ACTION_REROLL] > 0


# ── the executor's full-slot escape hatch ───────────────────────────────────

class _Env:
    env_id = 0
    shop_rerolls = 0
    pending_upgrade_buy = None
    auto_action_this_step = False

    def __getattr__(self, name):
        return None


def test_full_slots_no_improving_swap_rerolls_instead_of_no_op(monkeypatch):
    """The hole: at full slots with no improving swap the executor returned
    ('gamestate', None) — a wasted step, forever. It must now try to reroll."""
    from training.action_executor import ActionExecutor
    ae = ActionExecutor()
    monkeypatch.setattr(ae, "_planner_pick_swap", lambda *a, **k: None)
    monkeypatch.setattr(ae, "_already_clearing", lambda *a, **k: False)
    monkeypatch.setattr(ae, "_planner_reroll_ok", lambda *a, **k: True)
    action = np.zeros(14)
    action[0] = 2          # buy_joker
    action[13] = 0
    method, _ = ae._action_to_api_call(_Env(), action, _shop(slots_full=True))
    assert method == "reroll"


def test_full_slots_still_no_ops_when_already_clearing(monkeypatch):
    """dec-068's save-gate must still win: if we already clear the ante, don't
    burn money hunting."""
    from training.action_executor import ActionExecutor
    ae = ActionExecutor()
    monkeypatch.setattr(ae, "_planner_pick_swap", lambda *a, **k: None)
    monkeypatch.setattr(ae, "_already_clearing", lambda *a, **k: True)
    monkeypatch.setattr(ae, "_planner_reroll_ok", lambda *a, **k: True)
    action = np.zeros(14)
    action[0] = 2
    action[13] = 0
    method, _ = ae._action_to_api_call(_Env(), action, _shop(slots_full=True))
    assert method == "gamestate"
