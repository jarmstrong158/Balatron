"""dec-096: reconstructing a HUMAN's actions from consecutive game states.

The agent knows what it did because it chose it. A human does not announce
actions, so they must be inferred from state deltas — and a mis-inferred action
teaches the agent the WRONG lesson from a good run, which is worse than having no
data at all. That makes `infer_action` the piece worth testing hard.

The governing rule these tests enforce: when a transition is ambiguous, return
None so the caller DROPS it. Never guess.
"""
import human_record as hr
from environment.action_space import (
    ACTION_SELECT_PACK_CARD,
    ACTION_SKIP_PACK,
    ACTION_BUY_JOKER,
    ACTION_BUY_PACK,
    ACTION_DISCARD,
    ACTION_END_SHOP,
    ACTION_PLAY,
    ACTION_REROLL,
    ACTION_SELL_JOKER,
    ACTION_USE_CONSUMABLE,
)
from human_record import Recorder, infer_action


def S(state="SELECTING_HAND", money=10, hands=3, discards=3, jokers=0,
      cons=0, shop=2, packs=2, chips=0, ante=4):
    return {
        "state": state, "ante_num": ante, "money": money,
        "round": {"hands_left": hands, "discards_left": discards, "chips": chips},
        "jokers": {"cards": [{"label": f"j{i}"} for i in range(jokers)]},
        "consumables": {"cards": [{} for _ in range(cons)]},
        "shop": {"cards": [{} for _ in range(shop)]},
        "packs": {"cards": [{} for _ in range(packs)]},
    }


def _type(prev, cur):
    got = infer_action(prev, cur)
    return got[0] if got else None


# --------------------------------------------------------------------------
# In-blind
# --------------------------------------------------------------------------

def test_play_and_discard_are_not_confused():
    """Both remove cards from hand; only a PLAY consumes a hand. If these swap,
    the agent learns to discard where a human played."""
    assert _type(S(hands=3, discards=3), S(hands=2, discards=3)) == ACTION_PLAY
    assert _type(S(hands=3, discards=3), S(hands=3, discards=2)) == ACTION_DISCARD


def test_a_play_that_also_changed_chips_is_still_a_play():
    assert _type(S(hands=3, chips=0), S(hands=2, chips=4200)) == ACTION_PLAY


def test_chips_gained_is_captured_for_the_play():
    got = infer_action(S(hands=3, chips=100), S(hands=2, chips=5100))
    assert got[0] == ACTION_PLAY and got[1]["chips_gained"] == 5000


# --------------------------------------------------------------------------
# Shop
# --------------------------------------------------------------------------

def test_buying_a_joker():
    got = infer_action(S(state="SHOP", money=12, jokers=1),
                       S(state="SHOP", money=6, jokers=2))
    assert got[0] == ACTION_BUY_JOKER and got[1]["cost"] == 6


def test_selling_a_joker_is_not_read_as_a_buy():
    """Direction matters: joker count DOWN and money UP. Reading a sale as a
    purchase would teach the agent to buy at the moment a human divested."""
    got = infer_action(S(state="SHOP", money=6, jokers=3),
                       S(state="SHOP", money=9, jokers=2))
    assert got[0] == ACTION_SELL_JOKER and got[1]["gain"] == 3


def test_reroll_is_money_down_with_nothing_acquired():
    got = infer_action(S(state="SHOP", money=15, jokers=2, cons=1),
                       S(state="SHOP", money=10, jokers=2, cons=1))
    assert got[0] == ACTION_REROLL and got[1]["cost"] == 5


def test_buying_a_pack():
    assert _type(S(state="SHOP", money=10, packs=2),
                 S(state="SHOP", money=6, packs=1)) == ACTION_BUY_PACK


def test_using_a_consumable_costs_nothing():
    """A consumable leaving inventory with money UNCHANGED is a use, not a sale."""
    assert _type(S(state="SHOP", money=10, cons=2),
                 S(state="SHOP", money=10, cons=1)) == ACTION_USE_CONSUMABLE


def test_leaving_the_shop():
    assert _type(S(state="SHOP"), S(state="BLIND_SELECT")) == ACTION_END_SHOP


# --------------------------------------------------------------------------
# THE governing rule: never guess
# --------------------------------------------------------------------------

def test_no_change_yields_no_action():
    """Most polls land mid-animation with nothing happening. Those must not
    become actions, or the demo buffer fills with phantom no-ops."""
    assert infer_action(S(), S()) is None


def test_missing_states_yield_no_action():
    assert infer_action(None, S()) is None
    assert infer_action(S(), None) is None
    assert infer_action({}, {}) is None


def test_an_unrecognised_change_is_dropped_not_guessed():
    """A change the encoding cannot express must return None. Silently guessing
    would look like coverage while labelling transitions wrongly."""
    assert infer_action(S(shop=2), S(shop=5)) is None


# --------------------------------------------------------------------------
# Coverage accounting — the number that decides whether the spike is viable
# --------------------------------------------------------------------------

def test_coverage_counts_only_real_changes():
    """Idle polls must not inflate coverage. If they counted as covered, a
    recorder that inferred nothing would still report ~100%."""
    r = Recorder(out_dir="unused")
    r.observe(S(hands=3))
    r.observe(S(hands=3))          # idle
    r.observe(S(hands=3))          # idle
    assert r.changes == 0 and r.coverage == 0.0

    r.observe(S(hands=2))          # a play
    assert r.changes == 1 and r.inferred == 1 and r.coverage == 1.0

    r.observe(S(hands=2, shop=5))  # a change it cannot label
    assert r.changes == 2 and r.inferred == 1 and r.coverage == 0.5


def test_recorder_never_emits_an_unlabelled_event():
    r = Recorder(out_dir="unused")
    for st in (S(hands=3), S(hands=3, shop=5), S(hands=2, shop=5)):
        r.observe(st)
    assert all(e["action"] is not None for e in r.events)
    assert len(r.events) == r.inferred


# --------------------------------------------------------------------------
# dec-096: the transition filter — coverage must measure DECISIONS
# --------------------------------------------------------------------------

def test_engine_transitions_are_not_decisions():
    """The first coverage figure (47%) divided by every state change, putting the
    game's own state machine in the denominator. A diagnostic showed the misses
    were dominated by scoring animations, round eval, draws and counter refills —
    none of them decisions."""
    from human_record import is_decision

    # scoring animation: chips move, nothing else
    assert not is_decision(S(chips=0), S(chips=5000))
    # engine states the game enters by itself
    for st in ("HAND_PLAYED", "DRAW_TO_HAND", "ROUND_EVAL", "GAME_OVER"):
        assert not is_decision(S(state="SELECTING_HAND"), S(state=st))
    # round boundary REFILLS counters — no player action increases them
    assert not is_decision(S(hands=0, discards=0), S(hands=4, discards=3, money=15))


def test_real_choices_are_still_decisions():
    """The filter must not excuse genuine actions — that is how a bad recorder
    would score well. Leaving the shop is a choice even though only `state`
    changes."""
    from human_record import is_decision

    assert is_decision(S(state="SHOP"), S(state="BLIND_SELECT"))
    assert is_decision(S(hands=3), S(hands=2))                      # play
    assert is_decision(S(state="SHOP", money=12, jokers=1),
                       S(state="SHOP", money=6, jokers=2))          # buy


def test_coverage_excludes_engine_noise_but_keeps_real_misses():
    """Coverage = labelled / decisions. Engine transitions must not dilute it,
    and an unlabelled DECISION must still count against it."""
    r = Recorder(out_dir="unused")
    r.observe(S(hands=3))
    r.observe(S(hands=3, chips=4000))            # engine: scoring animation
    r.observe(S(hands=2, chips=4000))            # decision: a play
    assert r.decisions == 1 and r.inferred == 1 and r.coverage == 1.0

    r.observe(S(hands=2, chips=4000, shop=5))    # decision we cannot label
    assert r.decisions == 2 and r.inferred == 1 and r.coverage == 0.5
    assert r.missed and "n_shop" in r.missed[-1]["fields"]


# --------------------------------------------------------------------------
# Fixes from the FIRST REAL HUMAN SESSION (coverage measured 37%, not the
# predicted >60%). Both defects were invisible when validating against the
# agent, because the agent never produces these transitions.
# --------------------------------------------------------------------------

def test_engine_states_are_excluded_as_SOURCE_too():
    """The original test checked the engine state as DESTINATION only. That let
    `DRAW_TO_HAND -> SELECTING_HAND` and `PLAY_TAROT -> SELECTING_HAND` count as
    unlabelled decisions on every single hand — the engine FINISHING a draw or a
    tarot animation. They inflated the denominator with transitions that have no
    action to find, and were a large share of the live misses."""
    from human_record import is_decision

    assert not is_decision(S(state="DRAW_TO_HAND"), S(state="SELECTING_HAND"))
    assert not is_decision(S(state="PLAY_TAROT"), S(state="SELECTING_HAND"))
    assert not is_decision(S(state="ROUND_EVAL"), S(state="SHOP"))
    # ...but a real choice out of a NON-engine state still counts
    assert is_decision(S(state="SHOP"), S(state="BLIND_SELECT"))


def test_booster_packs_are_a_recognised_action_class():
    """A whole class the first version missed. Balatro routes pack contents
    through SMODS_BOOSTER_OPENED, and the agent's action space HAS
    SELECT_PACK_CARD and SKIP_PACK — they were simply never wired, because only
    the agent's own transitions were consulted when this was built."""
    from environment.action_space import ACTION_SELECT_PACK_CARD, ACTION_SKIP_PACK

    # took a card, then left the pack
    assert _type(S(state="SMODS_BOOSTER_OPENED", cons=0),
                 S(state="SHOP", cons=1)) == ACTION_SELECT_PACK_CARD
    # left without taking anything
    assert _type(S(state="SMODS_BOOSTER_OPENED", cons=1),
                 S(state="SHOP", cons=1)) == ACTION_SKIP_PACK
    # a selection made while still inside the pack
    assert _type(S(state="SMODS_BOOSTER_OPENED", money=10),
                 S(state="SMODS_BOOSTER_OPENED", money=8)) == ACTION_SELECT_PACK_CARD


def test_entering_a_pack_is_itself_the_purchase():
    """SUPERSEDES an earlier assertion that this returns None.

    That version assumed "the buy was already recorded; entering the pack is its
    consequence." Live human play disproved it: the shop-side BUY_PACK rule needs
    spent > 0, and at a 0.35s poll the money change and the pack-count change
    routinely land on DIFFERENT ticks — so neither half matched and the purchase
    was lost entirely. It was the largest remaining coverage gap.

    Entering a booster is unambiguous (there is no other way in), so it is now
    treated as the signal, with Recorder._is_duplicate_pack_buy collapsing the
    pair when the money half also fired."""
    from environment.action_space import ACTION_BUY_PACK

    assert _type(S(state="SHOP"),
                 S(state="SMODS_BOOSTER_OPENED")) == ACTION_BUY_PACK


# --------------------------------------------------------------------------
# Pack-buy across poll ticks — the largest remaining live gap
# --------------------------------------------------------------------------

def test_entering_a_booster_is_the_purchase():
    """You cannot reach a booster state any other way, so the transition
    identifies the action on its own — no money delta needed.

    This was `return None` on the assumption the shop-side BUY_PACK rule had
    already caught it. That rule needs spent > 0, but at a 0.35s poll the money
    change and the pack-count change routinely land on DIFFERENT ticks, so
    neither half matched and the buy vanished entirely."""
    from environment.action_space import ACTION_BUY_PACK

    # money already left on an earlier tick — only the state change is visible
    assert _type(S(state="SHOP", money=6, packs=2),
                 S(state="SMODS_BOOSTER_OPENED", money=6, packs=1)) == ACTION_BUY_PACK


def test_a_pack_buy_split_across_ticks_is_counted_once():
    """Both halves are treated as the signal, so the pair must collapse."""
    r = Recorder(out_dir="unused")
    r.observe(S(state="SHOP", money=10, packs=2))
    r.observe(S(state="SHOP", money=6, packs=1))                  # money half
    r.observe(S(state="SMODS_BOOSTER_OPENED", money=6, packs=1))  # state half
    assert r.inferred == 1, f"counted the same purchase {r.inferred} times"


def test_the_dedup_does_not_swallow_legitimate_repeats():
    """Scoped to BUY_PACK on purpose. A blanket 'same action twice is a
    duplicate' rule would silently drop two plays or two rerolls in a row —
    a far worse failure than double-counting a pack."""
    r = Recorder(out_dir="unused")
    r.observe(S(hands=4))
    r.observe(S(hands=3))      # play
    r.observe(S(hands=2))      # play again, immediately
    assert r.inferred == 2


def test_two_genuinely_separate_pack_buys_both_count():
    """The dedup window is a few polls, not the whole session."""
    r = Recorder(out_dir="unused")
    r.observe(S(state="SHOP", money=10, packs=2))
    r.observe(S(state="SMODS_BOOSTER_OPENED", money=10, packs=1))
    for _ in range(6):                       # time passes
        r.observe(S(state="SHOP", money=10, packs=1))
    r.observe(S(state="SMODS_BOOSTER_OPENED", money=10, packs=0))
    assert r.inferred >= 2, "the second pack buy was swallowed as a duplicate"


def test_a_play_is_captured_when_the_state_ALSO_changes():
    """THE bug that produced zero PLAY events across an entire ante.

    Live, playing a hand is `SELECTING_HAND -> HAND_PLAYED` with hands_left
    dropping on the SAME tick. HAND_PLAYED is an engine state, so the
    destination filter rejected the transition before infer_action ever ran, and
    every play and discard vanished as engine noise.

    Every earlier play/discard test holds the state constant on both sides —
    which never happens in the real game. That is exactly why the suite was green
    while the recorder captured no in-blind actions at all."""
    from human_record import is_decision

    assert is_decision(S(state="SELECTING_HAND", hands=3),
                       S(state="HAND_PLAYED", hands=2))
    assert _type(S(state="SELECTING_HAND", hands=3),
                 S(state="HAND_PLAYED", hands=2)) == ACTION_PLAY

    assert is_decision(S(state="SELECTING_HAND", discards=3),
                       S(state="HAND_PLAYED", discards=2))
    assert _type(S(state="SELECTING_HAND", discards=3),
                 S(state="HAND_PLAYED", discards=2)) == ACTION_DISCARD


def test_a_counter_REFILL_is_still_engine_even_mid_state_change():
    """The increase rule must keep winning — a new round refills and that is not
    the player, however the state moves."""
    from human_record import is_decision

    assert not is_decision(S(state="ROUND_EVAL", hands=0, discards=0),
                           S(state="SELECTING_HAND", hands=4, discards=3))


# ---------------------------------------------------------------------------
# CONTENT capture. The recorder logged action TYPES only -- "a pack card was
# taken", never WHICH -- which is unusable as a demonstration, since the agent's
# action is (type, target, card_bits). These pin the identity capture.
# ---------------------------------------------------------------------------

def _card(cid, key="c_x", label="X", rank=None, suit=None, hl=False, **kw):
    c = {"id": cid, "key": key, "set": kw.pop("set", "DEFAULT"), "label": label,
         "modifier": kw.pop("modifier", {}), "cost": {"buy": 0, "sell": 0},
         "state": {"highlight": True} if hl else {}}
    if rank is not None:
        c["value"] = {"rank": rank, "suit": suit}
    c.update(kw)
    return c


def _st(state="SELECTING_HAND", hand=(), jokers=(), cons=(), shop=(),
        vouchers=(), packs=(), pack=(), money=10, hands=3, discards=3,
        ante=1, chips=0):
    return {
        "state": state, "ante_num": ante, "money": money,
        "round": {"hands_left": hands, "discards_left": discards,
                  "chips": chips},
        "hand": {"cards": list(hand)}, "jokers": {"cards": list(jokers)},
        "consumables": {"cards": list(cons)}, "shop": {"cards": list(shop)},
        "vouchers": {"cards": list(vouchers)}, "packs": {"cards": list(packs)},
        "pack": {"cards": list(pack)},
    }


def test_played_cards_are_identified_by_rank_and_suit():
    """A PLAY event must name the five cards, not just say a hand was played."""
    hand = [_card(1, rank="A", suit="S", hl=True),
            _card(2, rank="K", suit="S", hl=True),
            _card(3, rank="4", suit="H")]
    prev = _st(hand=hand, hands=3)
    cur = _st("HAND_PLAYED", hand=[_card(3, rank="4", suit="H")], hands=2)
    c = hr.content_of(prev, cur)
    played = {(x["value"]["rank"], x["value"]["suit"]) for x in c["selected"]}
    assert played == {("A", "S"), ("K", "S")}
    # and the unselected card must NOT be reported as played
    assert ("4", "H") not in played


def test_bought_joker_is_identified_by_key():
    prev = _st("SHOP", jokers=[_card(9, "j_joker", "Joker", set="JOKER")],
               shop=[_card(20, "j_blueprint", "Blueprint", set="JOKER")],
               money=15)
    cur = _st("SHOP", jokers=[_card(9, "j_joker", "Joker", set="JOKER"),
                              _card(20, "j_blueprint", "Blueprint", set="JOKER")],
              shop=[], money=5)
    c = hr.content_of(prev, cur)
    assert [j["key"] for j in c["jokers_arrived"]] == ["j_blueprint"]


def test_pack_card_taken_is_identified_and_alternatives_recorded():
    """'which cards were picked' -- and what was passed over."""
    offered = [_card(31, "c_judgement", "Judgement", set="TAROT"),
               _card(32, "c_fool", "The Fool", set="TAROT")]
    prev = _st("BOOSTER", pack=offered)
    cur = _st("BOOSTER", pack=[offered[1]],
              cons=[_card(31, "c_judgement", "Judgement", set="TAROT")])
    c = hr.content_of(prev, cur)
    assert [x["key"] for x in c["pack_left"]] == ["c_judgement"]
    assert [x["key"] for x in hr.context_of(prev)["pack_open"]] == \
        ["c_judgement", "c_fool"]


def test_tarot_target_cards_are_recorded():
    """'every tarot card used' includes WHICH cards it was applied to."""
    prev = _st(hand=[_card(1, rank="4", suit="H", hl=True)],
               cons=[_card(50, "c_strength", "Strength", set="TAROT")])
    cur = _st("PLAY_TAROT", hand=[_card(1, rank="5", suit="H")], cons=[])
    c = hr.content_of(prev, cur)
    assert [x["key"] for x in c["consumables_left"]] == ["c_strength"]
    assert [x["value"]["rank"] for x in c["selected"]] == ["4"]


def test_enhancement_applied_in_place_is_captured():
    """A tarot that only changes a modifier moves no card between areas."""
    prev = _st(hand=[_card(1, rank="4", suit="H", modifier={})])
    cur = _st(hand=[_card(1, rank="4", suit="H",
                          modifier={"enhancement": "GLASS"})])
    c = hr.content_of(prev, cur)
    assert c["hand_modified"][0]["after"]["modifier"]["enhancement"] == "GLASS"


def test_in_blind_tarot_use_is_a_decision():
    """Regression: PLAY_TAROT is an engine state, so the destination filter
    silently dropped every tarot used from hand -- one of the action classes
    explicitly required."""
    prev = _st("SELECTING_HAND", cons=[_card(50, "c_strength", set="TAROT")])
    cur = _st("PLAY_TAROT", cons=[])
    assert hr.is_decision(prev, cur) is True
    assert hr.infer_action(prev, cur)[0] == ACTION_USE_CONSUMABLE


def test_event_carries_content_and_context():
    """End to end: the written event must contain the choice AND the options."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = hr.Recorder(d)
        r.observe(_st("SHOP", shop=[_card(20, "j_blueprint", set="JOKER")],
                      money=15))
        r.observe(_st("SHOP", jokers=[_card(20, "j_blueprint", set="JOKER")],
                      money=5))
        assert r.events, "buy was not recorded at all"
        ev = r.events[-1]
        assert ev["action_name"] == "BUY_JOKER"
        assert ev["content"]["jokers_arrived"][0]["key"] == "j_blueprint"
        assert ev["context"]["shop"][0]["key"] == "j_blueprint"
        assert ev["context"]["money"] == 15


def test_empty_lua_tables_serialize_as_lists():
    """Live-verified shape: an unmodified playing card arrives as
    {"state": [], "modifier": []} because Lua encodes an empty table as a JSON
    ARRAY. Reading .get on that raises, so every access must tolerate a list."""
    live = {"id": 26, "set": "DEFAULT", "label": "Base Card", "key": "C_T",
            "value": {"rank": "T", "effect": "+10 chips", "suit": "C"},
            "state": [], "modifier": [], "cost": {"buy": 1, "sell": 1}}
    prev, cur = _st(hand=[live]), _st(hand=[live])
    assert hr.content_of(prev, cur) == {}          # no spurious modification
    assert hr.context_of(prev)["hand"][0]["key"] == "C_T"

    hl = dict(live, state={"highlight": True})
    c = hr.content_of(_st(hand=[hl]), _st("HAND_PLAYED", hand=[], hands=2))
    assert c["selected"][0]["value"]["rank"] == "T"


# --- deferred content resolution -------------------------------------------

def _rec(tmp):
    return hr.Recorder(tmp)


def test_play_content_recovered_after_animation_settles(tmp_path):
    """Live-observed: the tick that decrements hands_left often lands BEFORE the
    played cards leave `hand`, so the diff is empty. The content is not lost,
    only late -- it must be recovered once the hand settles."""
    hand = [_card(i, key=f"C_{i}", rank="T", suit="C") for i in range(1, 9)]
    r = _rec(str(tmp_path))
    r.observe(_st(hand=hand, hands=4))
    # hands_left drops but the hand has not visually changed yet
    r.observe(_st("HAND_PLAYED", hand=hand, hands=3))
    assert r.events and r.events[-1]["content"] == {}, "precondition"
    # ...now the played cards are gone and replacements are drawn
    settled = hand[5:] + [_card(90, key="C_9", rank="9", suit="H")]
    r.observe(_st(hand=settled, hands=3))
    ev = r.events[-1]
    assert ev["content_resolved"] == "deferred"
    gone = {c["key"] for c in ev["content"]["hand_left"]}
    assert gone == {"C_1", "C_2", "C_3", "C_4", "C_5"}


def test_pending_event_is_not_lost_when_a_new_action_arrives(tmp_path):
    """A second action while one is pending must not overwrite and drop it."""
    hand = [_card(i, key=f"C_{i}", rank="T", suit="C") for i in range(1, 9)]
    r = _rec(str(tmp_path))
    r.observe(_st(hand=hand, hands=4))
    r.observe(_st("HAND_PLAYED", hand=hand, hands=3))       # play, unresolved
    r.observe(_st(hand=hand[3:], hands=3, discards=2))      # discard arrives
    assert len(r.events) == 2
    names = [e["action_name"] for e in r.events]
    assert names == ["PLAY", "DISCARD"]
    assert all("content_resolved" in e or e["content"] for e in r.events[:1])
    # both must be on disk, not just in memory
    import glob
    import json as _j
    files = glob.glob(str(tmp_path / "*.jsonl"))
    lines = [_j.loads(x) for f in files
             for x in open(f, encoding="utf-8") if x.strip()]
    assert len(lines) == 2, f"an event was lost: {lines}"


def test_unresolvable_content_is_flagged_not_silently_empty(tmp_path):
    """Timeout must record content_resolved=False, so a gap reads as a gap."""
    r = _rec(str(tmp_path))
    r.observe(_st("SHOP", money=10, packs=[_card(7, "p_buffoon", set="BOOSTER")]))
    r.observe(_st("SHOP", money=6, packs=[_card(7, "p_buffoon", set="BOOSTER")]))
    for _ in range(hr.Recorder.PENDING_TIMEOUT + 2):
        r.observe(_st("SHOP", money=6,
                      packs=[_card(7, "p_buffoon", set="BOOSTER")]))
    assert r.events[-1]["content_resolved"] is False


def test_partial_animation_diff_is_superseded_by_the_complete_one(tmp_path):
    """Live regression: cards leave the hand a FEW AT A TIME over the play
    animation. Resolving on the first non-empty diff captured 1 of 5 played
    cards. The newest diff is always the most complete."""
    hand = [_card(i, key=f"C_{i}", rank="T", suit="C") for i in range(1, 9)]
    r = _rec(str(tmp_path))
    r.observe(_st(hand=hand, hands=4))
    r.observe(_st("HAND_PLAYED", hand=hand, hands=3))
    r.observe(_st("HAND_PLAYED", hand=hand[1:], hands=3))     # 1 card gone
    r.observe(_st("DRAW_TO_HAND", hand=hand[5:], hands=3))    # all 5 gone
    settled = hand[5:] + [_card(90, key="C_new", rank="9", suit="H")]
    r.observe(_st(hand=settled, hands=3))
    r.observe(_st(hand=settled, hands=3))                     # idle -> flush
    gone = {c["key"] for c in r.events[-1]["content"]["hand_left"]}
    assert gone == {"C_1", "C_2", "C_3", "C_4", "C_5"}, \
        f"partial capture: {gone}"


def test_round_boundary_does_not_claim_the_whole_hand_was_played(tmp_path):
    """The final hand of a blind empties the hand entirely. Diffing against
    that would report every remaining card as played."""
    hand = [_card(i, key=f"C_{i}", rank="T", suit="C") for i in range(1, 9)]
    r = _rec(str(tmp_path))
    r.observe(_st(hand=hand, hands=1))
    r.observe(_st("HAND_PLAYED", hand=hand[5:], hands=0))   # 5 played
    r.observe(_st("ROUND_EVAL", hand=[], hands=0))          # hand emptied
    r.observe(_st("SHOP", hand=[], hands=0))
    r.flush_pending()
    gone = {c["key"] for c in r.events[-1]["content"].get("hand_left", [])}
    assert gone == {"C_1", "C_2", "C_3", "C_4", "C_5"}, \
        f"round boundary corrupted the capture: {gone}"


def test_abandoning_a_run_is_not_a_discard():
    """Live regression: run teardown zeroes hands/discards, and the
    counter-decrease shortcut (which runs before the engine-state filter) read
    `GAME_OVER -> MENU` as a DISCARD -- naming three real cards as discarded."""
    prev = _st("GAME_OVER", hand=[_card(1, "D_3", rank="3", suit="D")],
               hands=1, discards=1)
    cur = _st("MENU", hand=[], hands=0, discards=0)
    assert hr.is_decision(prev, cur) is False
    assert hr.is_decision(_st("SELECTING_HAND"), _st("MENU", hands=0,
                                                     discards=0)) is False


def test_run_start_from_menu_is_not_an_action():
    assert hr.is_decision(_st("MENU", hands=0, discards=0),
                          _st("BLIND_SELECT")) is False


def test_pack_pick_names_the_card_taken_from_that_pack(tmp_path):
    """Live regression: only 6 of 24 pack picks named the card taken. The rest
    reported the pack OPENING, or a pick from the PREVIOUS pack -- content one
    step out of phase with the offer. The open pack is the only valid baseline."""
    offer = [_card(31, "c_judgement", "Judgement", set="TAROT"),
             _card(32, "c_devil", "The Devil", set="TAROT")]
    r = _rec(str(tmp_path))
    r.observe(_st("BOOSTER", pack=offer, cons=[]))
    r.observe(_st("BOOSTER", pack=[offer[1]],
                  cons=[_card(31, "c_judgement", "Judgement", set="TAROT")]))
    r.observe(_st("SHOP", pack=[]))          # pack closes -> finalize
    names = [e["action_name"] for e in r.events]
    assert names == ["SELECT_PACK_CARD"], f"spurious skip recorded: {names}"
    ev = r.events[-1]
    assert [c["key"] for c in ev["content"]["pack_left"]] == ["c_judgement"]
    assert [c["key"] for c in ev["chosen"]] == ["c_judgement"]


def test_chosen_prefers_the_exact_highlight_over_the_partial_diff():
    """On PLAY, hand_left disagreed with selected in 23 of 28 live events: the
    area diff catches the hand mid-animation, the highlight flag is exact."""
    content = {"hand_left": [{"id": 1, "key": "H_8"}],
               "selected": [{"id": 1, "key": "H_8"}, {"id": 2, "key": "H_2"},
                            {"id": 3, "key": "C_J"}]}
    assert [c["key"] for c in hr.chosen_of(ACTION_PLAY, content)] == \
        ["H_8", "H_2", "C_J"]
    assert [c["key"] for c in hr.chosen_of(ACTION_DISCARD, content)] == \
        ["H_8", "H_2", "C_J"]


def test_chosen_falls_back_when_there_is_no_highlight():
    content = {"hand_left": [{"id": 1, "key": "H_8"}]}
    assert [c["key"] for c in hr.chosen_of(ACTION_PLAY, content)] == ["H_8"]
    assert hr.chosen_of(ACTION_PLAY, {}) == []


def test_real_pack_skip_is_still_recorded(tmp_path):
    """Suppressing the spurious skip must not suppress a genuine one."""
    offer = [_card(31, "c_judgement", set="TAROT")]
    r = _rec(str(tmp_path))
    r.observe(_st("BOOSTER", pack=offer))
    r.observe(_st("SHOP", pack=[]))     # left having taken nothing
    assert [e["action_name"] for e in r.events] == ["SKIP_PACK"]


# --- pack picks that add no consumable ------------------------------------
# On a live winning run these were ALL recorded as skipping the pack, because
# the pick test was "consumables went up or money changed".

def _celestial(levels_before, levels_after):
    a = _st("BOOSTER", pack=[_card(1, "c_pluto", set="PLANET"),
                             _card(2, "c_neptune", set="PLANET")])
    b = _st("BOOSTER", pack=[_card(2, "c_neptune", set="PLANET")])
    a["hands"] = {"Pair": {"level": levels_before}}
    b["hands"] = {"Pair": {"level": levels_after}}
    return a, b


def test_celestial_pick_is_a_pick_not_a_skip():
    """A Celestial pick adds NO card anywhere -- it levels a hand. This is the
    single most important action class for the hand-level hypothesis."""
    a, b = _celestial(1, 2)
    assert hr.infer_action(a, b)[0] == ACTION_SELECT_PACK_CARD


def test_buffoon_pick_is_a_pick_not_a_skip():
    """A Buffoon pick adds a JOKER, never a consumable."""
    a = _st("BOOSTER", pack=[_card(1, "j_blueprint", set="JOKER"),
                             _card(2, "j_runner", set="JOKER")], jokers=[])
    b = _st("BOOSTER", pack=[_card(2, "j_runner", set="JOKER")],
            jokers=[_card(1, "j_blueprint", set="JOKER")])
    assert hr.infer_action(a, b)[0] == ACTION_SELECT_PACK_CARD


def test_standard_pick_is_a_pick_not_a_skip():
    """A Standard pick adds a card to the DECK."""
    a = _st("BOOSTER", pack=[_card(1, "S_A", rank="A", suit="S"),
                             _card(2, "D_A", rank="A", suit="D")])
    b = _st("BOOSTER", pack=[_card(2, "D_A", rank="A", suit="D")])
    a["cards"] = {"cards": [_card(50 + i) for i in range(52)]}
    b["cards"] = {"cards": [_card(50 + i) for i in range(53)]}
    assert hr.infer_action(a, b)[0] == ACTION_SELECT_PACK_CARD


def test_genuinely_untouched_pack_is_still_a_skip():
    a = _st("BOOSTER", pack=[_card(1, "c_pluto", set="PLANET")])
    b = _st("SHOP", pack=[])
    a["hands"] = b["hands"] = {"Pair": {"level": 1}}
    assert hr.infer_action(a, b)[0] == ACTION_SKIP_PACK


def test_highlight_seen_before_the_action_is_used(tmp_path):
    """Audited against the game's own counters: discards captured 119 of 267
    cards (45%). If the player highlights and clicks inside one poll, the
    pre-action snapshot has no highlight and the partial diff was used instead.
    The selection is on screen for seconds beforehand -- remember it."""
    hand = [_card(i, key=f"C_{i}", rank="T", suit="C") for i in range(1, 9)]
    hl = [dict(c, state={"highlight": True}) for c in hand[:4]]
    r = _rec(str(tmp_path))
    r.observe(_st(hand=hl + hand[4:], hands=4, discards=3))   # highlighted
    # ...clicked within the same poll: this snapshot shows no highlight
    r.observe(_st(hand=hand[4:], hands=4, discards=2))
    ev = r.events[-1]
    assert ev["action_name"] == "DISCARD"
    assert {c["key"] for c in ev["content"]["selected"]} == \
        {"C_1", "C_2", "C_3", "C_4"}


def test_remembered_highlight_is_not_reused_by_a_later_action(tmp_path):
    """A stale selection attributed to a second action would be a fabrication."""
    hand = [_card(i, key=f"C_{i}", rank="T", suit="C") for i in range(1, 9)]
    hl = [dict(c, state={"highlight": True}) for c in hand[:2]]
    r = _rec(str(tmp_path))
    r.observe(_st(hand=hl + hand[2:], hands=4, discards=3))
    r.observe(_st(hand=hand[2:], hands=4, discards=2))        # discard consumes
    r.observe(_st(hand=hand[2:], hands=3, discards=2))        # later play
    play = [e for e in r.events if e["action_name"] == "PLAY"][-1]
    assert play["content"].get("selected_from") != "remembered_highlight"
