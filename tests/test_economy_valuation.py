"""dec-065: non-scoring joker valuation + seal acquisition.

build_survivability was blind to 62/150 jokers (no `score_effect` → hard
d-surv=0.00), so the planner rerolled past 41% of the pool. These tests pin the
four levers (economy A-model, tier prior C, hand-upgrade #1, boss-nullify #2) and
the seal-acquisition changes, and — critically — the BOUNDS that keep economy
from ever displacing real power (dec-038: money is not the binding constraint).
"""
from environment.planner import (
    build_value, build_survivability, _score_survivability,
    _economic_survivability_bonus, ECON_SURV_CAP, PRIOR_SURV_CAP,
)
from environment.hand_eval import (
    evaluate_pack_standard, score_playing_card_pickup, evaluate_pack_spectral,
)

REROLL_BAR = 0.25  # action_executor PLANNER_REROLL_THRESHOLD


def _jk(key):
    return {"key": key}


def _healthy():
    """A build that already projects deep — reach-to-spend ~1.0."""
    return {"ante_num": 3, "money": 20,
            "hands": {"Flush": {"chips": 200, "mult": 40}},
            "cards": {"cards": [{} for _ in range(40)]},
            "round": {"hands_left": 4}, "blinds": {}}


def _dying():
    """A build that dies ~this ante — reach-to-spend ~0."""
    return {"ante_num": 3, "money": 20,
            "hands": {"Pair": {"chips": 10, "mult": 2}},
            "cards": {"cards": [{} for _ in range(40)]},
            "round": {"hands_left": 4}, "blinds": {}}


# ── #3 economy (A), bounded ──────────────────────────────────────────────────

def test_strong_economy_clears_reroll_bar_on_healthy_build():
    cur = [_jk("j_joker")]
    gs = _healthy()
    for key in ("j_golden", "j_mail", "j_satellite", "j_business"):
        assert build_value(_jk(key), cur, gs) >= REROLL_BAR, key


def test_weak_economy_stays_below_reroll_bar():
    cur = [_jk("j_joker")]
    gs = _healthy()
    # Credit Card ($0.5/round) should NOT out-value hunting a real engine.
    assert build_value(_jk("j_credit_card"), cur, gs) < REROLL_BAR


def test_economy_collapses_on_dying_build():
    """The safety belt: money you die before spending is worth ~nothing, so a
    dying build must not be lured off buying power."""
    cur = [_jk("j_joker")]
    dying = _dying()
    assert build_value(_jk("j_business"), cur, dying) < 0.02
    # ...and a real scaling engine outranks economy exactly where power matters.
    assert build_value(_jk("j_ride_the_bus"), cur, dying) > \
        build_value(_jk("j_business"), cur, dying)


def test_second_economy_joker_not_attractive():
    cur = [_jk("j_joker")]
    gs = _healthy()
    owns_one = cur + [_jk("j_business")]
    assert build_value(_jk("j_golden"), owns_one, gs) < 0.05


def test_economy_bonus_hard_capped():
    cur = [_jk("j_joker"), _jk("j_golden"), _jk("j_business"), _jk("j_mail")]
    gs = _healthy()
    base = _score_survivability(cur, gs)
    assert _economic_survivability_bonus(cur, gs, base) <= ECON_SURV_CAP + 1e-9


def test_rocket_ramp_beats_its_flat_yield():
    """Rocket ($1/round, +$2 per boss defeated) is a SCALING economy joker. Using
    its flat money_per_round=$1 alone under-credits the ramp below the reroll bar;
    projecting the ramp must lift it to the cap on an ahead-of-curve build."""
    cur = [_jk("j_joker")]
    strong = {"ante_num": 6, "money": 20,
              "hands": {"Flush": {"chips": 800, "mult": 120}},
              "cards": {"cards": [{} for _ in range(40)]},
              "round": {"hands_left": 4}, "blinds": {}}
    v = build_value(_jk("j_rocket"), cur, strong)
    assert v >= REROLL_BAR                     # ramp clears the bar (flat $1 would not)
    # and it is at least as valued as a flat $4/round economy joker
    assert v >= build_value(_jk("j_golden"), cur, strong) - 1e-9


def test_economy_worthless_in_final_antes():
    """No future shop to spend in → no economy credit."""
    late = _healthy()
    late["ante_num"] = 12
    assert _economic_survivability_bonus([_jk("j_golden")], late, 12.0) == 0.0


# ── #4 tier prior (C) ────────────────────────────────────────────────────────

def test_scoreless_utility_joker_is_rankable_but_below_econ():
    """8 Ball (card-creation, no score_effect) must be > 0 (rankable) yet small
    enough it never stops an engine hunt."""
    cur = [_jk("j_joker")]
    gs = _healthy()
    v = build_value(_jk("j_8_ball"), cur, gs)
    assert 0.0 < v < REROLL_BAR
    assert v <= PRIOR_SURV_CAP + 1e-9


# ── #1 hand-upgrade jokers ───────────────────────────────────────────────────

def test_hand_upgrade_joker_raises_survivability():
    cur = [_jk("j_joker")]
    gs = _healthy()
    assert build_value(_jk("j_space"), cur, gs) > 0.0  # Space Joker levels hands


# ── #2 boss-nullifier ────────────────────────────────────────────────────────

def test_chicot_reduces_boss_difficulty():
    wall = {"ante_num": 6, "money": 10,
            "hands": {"Flush": {"chips": 120, "mult": 20}},
            "cards": {"cards": [{} for _ in range(40)]},
            "round": {"hands_left": 4},
            "blinds": {"boss": {"status": "CURRENT", "name": "The Wall"}}}
    cur = [_jk("j_joker")]
    assert build_survivability(cur + [_jk("j_chicot")], wall) >= \
        build_survivability(cur, wall)


# ── seals: spectral valuation ────────────────────────────────────────────────

def _spec(key):
    return {"key": key, "set": "SPECTRAL"}


def test_blue_seal_spectral_outranks_generic():
    """Trance (blue seal = leveling) should now beat a filler spectral."""
    hand = [{"value": {"rank": "Ace"}, "modifier": {}}]
    # Trance vs Sigil (mild): Trance must be picked.
    pack = [_spec("c_sigil"), _spec("c_trance")]
    idx, _ = evaluate_pack_spectral(pack, hand, [], {"money": 10})
    assert pack[idx]["key"] == "c_trance"


# ── seals: standard pack acquisition ─────────────────────────────────────────

def _card(rank="9", seal="", enh="", ed=""):
    m = {}
    if seal:
        m["seal"] = seal
    if enh:
        m["enhancement"] = enh
    if ed:
        m["edition"] = ed
    return {"value": {"rank": rank}, "modifier": m}


def test_standard_pack_prefers_seal():
    pack = [_card("9"), _card("King", seal="BLUE"), _card("Ace", enh="GLASS")]
    idx, targets = evaluate_pack_standard(pack, [], [], {})
    assert idx == 1 and targets == []


def test_standard_pack_skips_pure_dilution():
    vanilla = [_card("2"), _card("7"), _card("9")]
    assert evaluate_pack_standard(vanilla, [], [], {}) is None


def test_blue_seal_beats_glass_enhancement():
    assert score_playing_card_pickup(_card("King", seal="BLUE"), [], {}) > \
        score_playing_card_pickup(_card("Ace", enh="GLASS"), [], {})
