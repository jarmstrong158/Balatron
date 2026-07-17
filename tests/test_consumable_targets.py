"""dec-072: never fire a TARGETED consumable without targets.

The policy's action_type==8 branch used to fall through to a bare
use{consumable} for ANY consumable. The API rejects a targeted one outright
("requires card selection..."), and it then sits in a slot getting retried
forever (Trance 30x, Death 22x in a single log), clogging the 2 consumable slots
so planets can't be held — which stalls leveling. Same class as con-005: never
fall through into a guaranteed reject.
"""
import numpy as np

from training.action_executor import ActionExecutor
from environment.hand_eval import CONSUMABLE_NEEDS_TARGET


class _Env:
    """Permissive stand-in — the decode path reads assorted env fields before
    reaching the consumable branch; anything unset reads as None."""
    env_id = 0
    shop_rerolls = 0
    pending_upgrade_buy = None
    auto_action_this_step = False

    def __getattr__(self, name):
        return None


def _action(consumable_slot=0, cards=None):
    """[type(1), cards(12), target(1)]; type 8 = use consumable, target 12+idx."""
    a = np.zeros(14)
    a[0] = 8
    if cards:
        for i in cards:
            a[1 + i] = 1.0
    a[13] = 12 + consumable_slot
    return a


def _state(cons_key, state="SHOP", hand_n=8):
    return {
        "state": state,
        "consumables": {"cards": [{"key": cons_key}], "limit": 2},
        "hand": {"cards": [{"value": {"rank": "Ace", "suit": "Spades"}}
                           for _ in range(hand_n)]},
        "jokers": {"cards": []},
        "cards": {"cards": []},
        "hands": {}, "round": {"hands_left": 4}, "money": 10, "ante_num": 3,
    }


def test_targeted_consumable_outside_selecting_hand_does_not_fire_bare():
    """c_trance (blue seal) in the SHOP: must NOT emit a bare use — that's the
    guaranteed reject that clogged the slot."""
    ae = ActionExecutor()
    method, params = ae._action_to_api_call(_Env(), _action(), _state("c_trance"))
    if method == "use":
        assert params.get("cards"), "targeted consumable fired without targets"
    else:
        assert method == "gamestate"   # no-op is the correct alternative


def test_untargeted_consumable_still_fires():
    """A planet needs no targets — it must still be used normally."""
    ae = ActionExecutor()
    method, params = ae._action_to_api_call(_Env(), _action(), _state("c_jupiter"))
    assert method == "use"
    assert params["consumable"] == 0


def test_targeted_consumable_with_selected_cards_passes_them():
    ae = ActionExecutor()
    method, params = ae._action_to_api_call(
        _Env(), _action(cards=[0, 1]), _state("c_death", state="SELECTING_HAND"))
    assert method == "use"
    assert params.get("cards") == [0, 1]


def test_needs_target_set_covers_the_observed_failures():
    """Every consumable seen failing in the logs must be in the set."""
    for k in ("c_trance", "c_death", "c_strength", "c_empress", "c_moon",
              "c_justice", "c_chariot", "c_tower", "c_sun", "c_star",
              "c_world", "c_magician"):
        assert k in CONSUMABLE_NEEDS_TARGET, k
