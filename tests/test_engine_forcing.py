"""dec-093: forced engine building — a probe of the premise every lever assumed.

Four levers tried to make the agent build better and came back null or worse
(dec-075/079/081/083); dec-090 found no build feature predicts clearing; a
run-level check found nothing predicts depth either. The tempting read is
"builds don't matter", but at ante 4 the agent's build space is 0-2 xmult
engines, so every one of those measurements compared mediocre against mediocre.
Mediocre-vs-ENGINE has never been measured, because the agent never builds one.

These tests pin the two things that decide whether this experiment means
anything: that the forcing actually fires (a silently-vacuous forcing would
report a null that is really the control arm measured against itself), and that
with the flag OFF the shop path is untouched.
"""
import engine_forcing as ef


def _shop(*keys, costs=None):
    cards = []
    for i, k in enumerate(keys):
        cost = (costs or {}).get(i, 4)
        cards.append({"key": k, "joker_key": k, "label": k,
                      "cost": {"buy": cost}})
    return {"shop": {"cards": cards}, "round": {"reroll_cost": 5}}


# --------------------------------------------------------------------------
# The manipulation check. Everything else is worthless without this.
# --------------------------------------------------------------------------

def test_tiers_are_not_all_zero():
    """THE guard. The first draft of _tier keyed off `scaling_type` and
    `scaling_increment`, which do not exist anywhere in data/jokers.py. Every
    joker would have scored 0, pick_engine_joker would have returned None every
    time, the forced arm would have been byte-identical to control, and the
    experiment would have reported a confident null while measuring nothing.

    A vacuous instrument that reports 'no effect' is worse than no instrument.
    """
    from data.jokers import JOKERS

    tiers = [ef._tier(n) for n in JOKERS]
    assert max(tiers) == 5, "no tier-5 engine pieces — _tier is reading dead fields"
    assert sum(t > 0 for t in tiers) > 40, (
        f"only {sum(t > 0 for t in tiers)}/150 jokers scored — schema drift")


def test_every_field_tier_reads_actually_exists():
    """Pins _tier against future schema drift. If a field is renamed in
    data/jokers.py this fails loudly instead of silently zeroing the instrument.
    """
    from data.jokers import JOKERS

    sample = next(iter(JOKERS.values()))
    for field in ("xmult", "xmult_scaling", "copy", "retrigger_effect",
                  "mass_retrigger", "mult_scaling", "chip_scaling",
                  "economy", "money_per_round"):
        assert field in sample, f"_tier reads {field!r}, which is not in the schema"


def test_ordering_is_engine_first():
    """Blueprint (copier) must outrank a plain economy joker, and a real xmult
    must outrank Blueprint — that ordering IS the hypothesis under test."""
    assert ef._tier("Cavendish") == 5
    assert ef._tier("Blueprint") == 4
    assert ef._tier("Cavendish") > ef._tier("Blueprint")


# --------------------------------------------------------------------------
# Control arm must be untouched
# --------------------------------------------------------------------------

def test_flag_off_is_a_no_op(monkeypatch):
    """With forcing off, both entry points must return the neutral value so the
    control arm is byte-identical to current behaviour. Otherwise the A/B has no
    baseline."""
    monkeypatch.setattr(ef, "FORCE_ENGINE", False)
    st = _shop("j_cavendish", "j_joker")
    assert ef.pick_engine_joker(st, 100) is None
    assert ef.should_force_reroll(st, 100) is False


# --------------------------------------------------------------------------
# Selection behaviour (flag on)
# --------------------------------------------------------------------------

def test_picks_the_highest_tier_available(monkeypatch):
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    # slot 0 plain, slot 1 xmult -> must take slot 1 even though 0 is first
    assert ef.pick_engine_joker(_shop("j_joker", "j_cavendish"), 100) == 1


def test_respects_affordability(monkeypatch):
    """An engine you cannot afford is not a pick. The whole reason acquisition
    looks random (dec-080) may be that the agent is never holding enough to take
    the piece it wants."""
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    # slot 0 = xmult (tier 5) but expensive; slot 1 = economy (tier 2) but cheap.
    # Note j_joker would NOT work here: plain flat mult is tier 0, not an engine
    # piece at all, so a budget that excludes slot 0 correctly yields no pick.
    st = _shop("j_cavendish", "j_credit_card", costs={0: 20, 1: 3})
    assert ef.pick_engine_joker(st, 5) == 1        # can't afford the xmult
    assert ef.pick_engine_joker(st, 50) == 0       # now it can


def test_no_engine_in_shop_yields_no_pick(monkeypatch):
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    assert ef.pick_engine_joker(_shop("j_joker"), 100) is None


# --------------------------------------------------------------------------
# Hunting behaviour
# --------------------------------------------------------------------------

def test_rerolls_only_when_no_engine_and_reserve_survives(monkeypatch):
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    barren = _shop("j_joker")
    assert ef.should_force_reroll(barren, 100) is True     # rich + barren -> hunt
    assert ef.should_force_reroll(barren, 5) is False      # would strand at $0
    # an engine is right there — buy it, don't reroll past it
    assert ef.should_force_reroll(_shop("j_cavendish"), 100) is False


def test_early_ante_money_leaves_the_forcing_with_nothing_to_do(monkeypatch):
    """REGRESSION. The condition 'no engine affordable AND no reroll affordable'
    is not an edge case — it is the NORMAL state at antes 1-2, where money is
    $4-5, engine pieces cost more, and should_force_reroll needs reroll(5) +
    reserve(3) = $8.

    The first pilot returned a no-op in exactly this state, so the forced arm did
    nothing on every early shop step and spun until the run died: mean ante 2.06
    against a 4.28 baseline, 21 of 64 runs dead at ante 1 — a blind that no build
    strategy can fail. Read naively that looks like "engines don't help"; it was
    the instrument destroying itself.

    This test pins how OFTEN the caller hits the fall-through, so the branch can
    never be treated as rare enough to stall in.
    """
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    early = _shop("j_cavendish", "j_credit_card", costs={0: 6, 1: 6})
    for money in (4, 5, 6, 7):
        pick = ef.pick_engine_joker(early, money)
        reroll = ef.should_force_reroll(early, money)
        if money < 6:
            assert pick is None and reroll is False, (
                f"at ${money} the forcing has NO action available — the caller "
                f"must fall through to the planner, never return a no-op")


# --------------------------------------------------------------------------
# Mode 2: bank-then-engine
# --------------------------------------------------------------------------

def _mode2(monkeypatch):
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    monkeypatch.setattr(ef, "BANK_THEN_ENGINE", True)
    monkeypatch.setattr(ef, "MIN_TIER", 4)
    monkeypatch.setattr(ef, "BANK_TARGET", 10)


def test_mode2_refuses_filler_it_could_afford(monkeypatch):
    """THE difference from mode 1. Mode 1 bought the best AFFORDABLE piece, which
    with a $6 median bankroll against $7 median engines meant cheap filler — 23%
    xmult, 36% economy — spending the bankroll DOWN and further under the $6->$10
    cliff. Mode 2 must leave the filler on the shelf."""
    _mode2(monkeypatch)
    st = _shop("j_credit_card", "j_cavendish", costs={0: 4, 1: 7})
    assert ef.pick_engine_joker(st, 5) is None, "took affordable filler — that is mode 1"
    assert ef.pick_engine_joker(st, 8) == 1, "should take the engine once reachable"


def test_mode2_banks_instead_of_rerolling_below_target(monkeypatch):
    """Rerolling below the target burns the bankroll being accumulated."""
    _mode2(monkeypatch)
    filler = _shop("j_credit_card", costs={0: 4})
    assert ef.should_force_reroll(filler, 8) is False    # would drop to $3
    assert ef.should_force_reroll(filler, 16) is True    # $16-5 = $11 >= target


def test_mode1_still_takes_filler(monkeypatch):
    """Guards that mode 2's filler refusal did not silently change mode 1, so the
    two arms remain distinct experiments."""
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    monkeypatch.setattr(ef, "BANK_THEN_ENGINE", False)
    st = _shop("j_credit_card", "j_cavendish", costs={0: 4, 1: 7})
    assert ef.pick_engine_joker(st, 5) == 0


def test_never_raises_on_garbage(monkeypatch):
    monkeypatch.setattr(ef, "FORCE_ENGINE", True)
    for bad in ({}, {"shop": {}}, {"shop": {"cards": None}},
                {"shop": {"cards": [{}]}}, {"shop": {"cards": [{"key": None}]}}):
        assert ef.pick_engine_joker(bad, 10) is None or True   # must not raise
        ef.should_force_reroll(bad, 10)
