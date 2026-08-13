"""dec-096: record HUMAN Balatro play into agent-consumable trajectories.

WHY
---
Every build-side result in this investigation is uninterpretable for the same
reason: RANGE RESTRICTION. The agent holds ~0.67 xmult engines per run and never
leaves the 0-2 band (3 engines = 2.9% of the field, 4 = 0.1%), so dec-085,
dec-090 and dec-093 all compared mediocre builds against other mediocre builds
and correctly found nothing. Whether a real 3-5 engine build actually wins has
never been measured, and dec-093 showed no shop policy can reach that range.

Human wins supply the missing arm OBSERVATIONALLY -- runs that genuinely contain
assembled engines -- without the forcing that backfired twice. Secondarily they
feed demo_buffer/SIL, where the agent currently imitates only its own ~1% wins.

THE HARD PART: a human does not announce actions. The trainer knows what it did
because it chose it; here the action must be RECONSTRUCTED from consecutive game
states. `infer_action` does that, and it is the piece worth testing, because a
mis-inferred action teaches the agent the wrong lesson from a good run -- worse
than no data.

Inference is deliberately CONSERVATIVE: anything ambiguous returns None and the
transition is DROPPED rather than guessed. Coverage is reported so the drop rate
is visible instead of silent -- a recorder that quietly guesses would poison the
demo buffer with plausible-looking wrong labels.

USAGE
-----
    python human_record.py --port 12346 --out recordings/human/

Start it, then play normally. It polls the gamestate the mod already exposes and
never sends an action, so it cannot interfere with play.

WHAT IS RECORDED
----------------
Every event carries the action TYPE, the CONTENT (exactly what was chosen), and
the CONTEXT (everything that was available and passed over):

    action_name  PLAY / DISCARD / BUY_JOKER / USE_CONSUMABLE / ...
    content      cards played or discarded (rank, suit, enhancement, edition,
                 seal), joker bought or sold, pack card taken, consumable used,
                 and the hand cards a tarot was applied to
    context      the full hand, joker row, consumables, shop, vouchers, open
                 pack, money, hands/discards left, blind and hand levels

Content is derived by DIFFING each card area on `id` (the game's `sort_id`,
stable per card), plus the `state.highlight` flag that marks the cards a player
has selected. An earlier version logged only the action type -- "a pack card was
taken", never which one -- which is not a demonstration at all, since the agent's
action is (action_type, target_index, card_bits) and two thirds of that was
missing. `content_coverage` reports the fraction of events that captured the
choice; it is the number that decides whether a recording is usable, and it read
0% on the first live test while `coverage` read 100%.

STATUS: records full action content, verified against live play. It does NOT yet
write demo_buffer trajectories -- state-vector encoding and head_indices are the
next step.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from environment.action_space import (
    ACTION_BUY_JOKER,
    ACTION_BUY_PACK,
    ACTION_BUY_VOUCHER,
    ACTION_DISCARD,
    ACTION_END_SHOP,
    ACTION_PLAY,
    ACTION_REROLL,
    ACTION_SELECT_BLIND,
    ACTION_SELECT_PACK_CARD,
    ACTION_SELL_JOKER,
    ACTION_SKIP_BLIND,
    ACTION_SKIP_PACK,
    ACTION_USE_CONSUMABLE,
)

POLL_S = 0.35

# Readable label alongside the numeric action id. The id is what the agent
# consumes; the name is what makes a recording auditable by eye -- the only way
# a mislabelled action gets caught before it reaches the demo buffer.
ACTION_NAMES = {
    ACTION_PLAY: "PLAY",
    ACTION_DISCARD: "DISCARD",
    ACTION_BUY_JOKER: "BUY_JOKER",
    ACTION_BUY_VOUCHER: "BUY_VOUCHER",
    ACTION_BUY_PACK: "BUY_PACK",
    ACTION_SELL_JOKER: "SELL_JOKER",
    ACTION_USE_CONSUMABLE: "USE_CONSUMABLE",
    ACTION_REROLL: "REROLL",
    ACTION_END_SHOP: "END_SHOP",
    ACTION_SELECT_BLIND: "SELECT_BLIND",
    ACTION_SKIP_BLIND: "SKIP_BLIND",
    ACTION_SELECT_PACK_CARD: "SELECT_PACK_CARD",
    ACTION_SKIP_PACK: "SKIP_PACK",
}


def _n(state: dict, *path, default=0):
    cur = state
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def _hand_level_sum(s: dict) -> int:
    """Total of every poker-hand level. A Celestial pack pick shows up ONLY
    here -- it adds no card anywhere."""
    hands = s.get("hands")
    if not isinstance(hands, dict):
        return 0
    total = 0
    for v in hands.values():
        if isinstance(v, dict):
            try:
                total += int(v.get("level") or 0)
            except (TypeError, ValueError):
                pass
    return total


def _counts(s: dict) -> dict:
    """The handful of scalars whose deltas identify an action."""
    return {
        "state": s.get("state") or s.get("game_state") or "",
        "ante": _n(s, "ante_num"),
        "money": float(_n(s, "money", default=0) or 0),
        "hands_left": _n(s, "round", "hands_left", default=-1),
        "discards_left": _n(s, "round", "discards_left", default=-1),
        "n_jokers": len(_n(s, "jokers", "cards", default=[]) or []),
        "n_cons": len(_n(s, "consumables", "cards", default=[]) or []),
        "n_shop": len(_n(s, "shop", "cards", default=[]) or []),
        "n_packs": len(_n(s, "packs", "cards", default=[]) or []),
        "chips": _n(s, "round", "chips", default=0),
        # Acquisition signals for pack picks. A pack pick does NOT always add a
        # consumable: a Celestial pick LEVELS A HAND, a Buffoon pick adds a
        # JOKER, a Standard pick adds a card to the DECK. Counting only
        # consumables missed all three -- see the booster branch of infer_action.
        "deck_n": len(_n(s, "cards", "cards", default=[]) or []),
        "hand_levels": _hand_level_sum(s),
        "n_pack": len(_n(s, "pack", "cards", default=[]) or []),
    }


# States the game passes through ON ITS OWN. Arriving in one of these is the
# engine advancing (animating a played hand, drawing, evaluating the round), not
# the player deciding anything.
ENGINE_STATES = {
    "HAND_PLAYED", "DRAW_TO_HAND", "ROUND_EVAL", "SCREENWIPE",
    "NEW_ROUND", "GAME_OVER", "MENU", "SPLASH", "PLAY_TAROT",
}


def is_decision(prev: dict, cur: dict) -> bool:
    """Could this transition have been caused by a PLAYER choice?

    dec-096. The first coverage number (47%) counted every state change as a
    missed action, which put the game's own state machine in the denominator: a
    diagnostic showed the unlabelled transitions were dominated by `state`-only
    and `chips`+`state` changes — scoring animations, round evaluation, draws and
    round-boundary counter refills. None of those are decisions, so scoring the
    recorder against them measures the wrong thing.

    Conservative in the OPPOSITE direction to infer_action: when unsure, call it a
    decision. That keeps genuinely-missed actions visible in the coverage figure
    instead of being excused as engine noise — the failure mode that would let a
    bad recorder look good.
    """
    if not prev or not cur:
        return False
    a, b = _counts(prev), _counts(cur)
    if a == b:
        return False

    # A round boundary REFILLS hands/discards and pays out. No player action ever
    # increases those counters, so an increase identifies the engine.
    if b["hands_left"] > a["hands_left"] or b["discards_left"] > a["discards_left"]:
        return False

    # Run TEARDOWN zeroes the round counters, which the decrease shortcut below
    # would read as the player spending them. Caught live: abandoning a dead run
    # produced `GAME_OVER -> MENU` labelled DISCARD, with three real cards named
    # as the discarded ones -- a fabricated action with fabricated content, the
    # precise failure this module exists to prevent. These states are never part
    # of a decision, so they are excluded before the shortcut, not after.
    TEARDOWN = {"MENU", "SPLASH", "GAME_OVER"}
    if str(a["state"]) in TEARDOWN or str(b["state"]) in TEARDOWN:
        return False

    # A DECREASE is unambiguously the player, and must be decided BEFORE the
    # engine-state test below.
    #
    # Playing a hand is `SELECTING_HAND -> HAND_PLAYED` with hands_left dropping,
    # and HAND_PLAYED is an engine state — so the destination test rejected every
    # play and discard as engine noise, and the recorder captured ZERO in-blind
    # actions across a whole ante. The unit tests missed it because they hold the
    # state constant at SELECTING_HAND on both sides, which never happens live:
    # the counter and the state change on the same tick.
    if b["hands_left"] < a["hands_left"] or b["discards_left"] < a["discards_left"]:
        return True

    # A CONSUMABLE leaving the slot is likewise the player, and likewise has to
    # be decided before the engine-state test: using a tarot on cards in hand is
    # `SELECTING_HAND -> PLAY_TAROT`, and PLAY_TAROT is an engine state, so the
    # test below rejected every in-blind tarot and spectral use. Same shape as
    # the play/discard bug above — the state and the counter move on one tick,
    # and the engine-state filter saw only the state.
    if b["n_cons"] < a["n_cons"]:
        return True

    # Engine states on EITHER side. Landing in one is the game advancing itself;
    # LEAVING one is the game finishing what it started.
    #
    # dec-096 originally tested the destination only, reasoning that leaving SHOP
    # or BLIND_SELECT is a real choice. True, but SHOP and BLIND_SELECT are not
    # engine states, so the source test does not touch them. What the missing
    # source test DID let through was `DRAW_TO_HAND -> SELECTING_HAND` and
    # `PLAY_TAROT -> SELECTING_HAND` — the engine completing a draw or a tarot
    # animation — counted as unlabelled decisions on every hand. Measured on live
    # human play they were a large share of the misses, depressing coverage with
    # transitions that have no action to find.
    if str(b["state"]) in ENGINE_STATES or str(a["state"]) in ENGINE_STATES:
        return False

    # Chips moving during a scoring animation, with no resource spent and nothing
    # gained or lost, is the engine tallying — not a decision.
    if (b["chips"] != a["chips"] and b["money"] == a["money"]
            and b["n_jokers"] == a["n_jokers"] and b["n_cons"] == a["n_cons"]
            and b["hands_left"] == a["hands_left"]
            and b["discards_left"] == a["discards_left"]):
        return False

    return True


def infer_action(prev: dict, cur: dict):
    """Best-effort (action_type, detail) for the transition prev -> cur.

    Returns None when the transition is ambiguous or is not an action at all
    (idle poll, animation frame). Callers MUST drop None rather than guess: a
    wrong label teaches the agent the wrong lesson from a good run.
    """
    if not prev or not cur:
        return None
    a, b = _counts(prev), _counts(cur)

    # --- booster packs (FIRST: state context dominates) --------------------
    # Ordering matters. This block sat AFTER the shop rules at first, so a
    # money-only change inside a pack matched the reroll rule (money down,
    # nothing else moved) and was labelled ACTION_REROLL — a wrong label, the
    # exact failure this module exists to avoid.
    # A whole action class the first version missed. Balatro routes pack contents
    # through SMODS_BOOSTER_OPENED, and the agent's action space HAS
    # SELECT_PACK_CARD and SKIP_PACK — they were simply never wired here, because
    # the agent's own transitions were the only ones consulted when this was
    # built. On live human play these were a large share of the unlabelled
    # decisions.
    if "BOOSTER" in str(a["state"]):
        # A pick is ANY acquisition, not just a consumable. The original test
        # was `n_cons up or money changed`, which detects Arcana and Spectral
        # picks and nothing else -- so on a live winning run every Celestial
        # pick (hand level up), every Buffoon pick (joker) and every Standard
        # pick (deck card) was recorded as SKIPPING the pack. For the question
        # this recording exists to answer -- do celestial picks track the hands
        # actually played -- that is the exact wrong answer.
        #
        # A card leaving the OPEN pack is the authoritative signal for every
        # pack type; the rest are corroborating.
        took_from_open = (a["n_pack"] > 0 and b["n_pack"] > 0
                          and b["n_pack"] < a["n_pack"])
        gained = (took_from_open
                  or b["n_cons"] > a["n_cons"]
                  or b["n_jokers"] > a["n_jokers"]
                  or b["deck_n"] > a["deck_n"]
                  or b["hand_levels"] > a["hand_levels"]
                  or b["money"] != a["money"])
        if "BOOSTER" not in str(b["state"]):
            # Leaving the pack: took something, or closed it without taking.
            return ((ACTION_SELECT_PACK_CARD, {"from_pack": True}) if gained
                    else (ACTION_SKIP_PACK, {}))
        if gained:
            return (ACTION_SELECT_PACK_CARD, {"from_pack": True})
        return None

    # --- in-blind ---------------------------------------------------------
    # A play and a discard both remove cards; they are told apart by WHICH
    # counter moved. Checking discards first matters: some decks/jokers let a
    # discard also change chips, but only a play consumes a hand.
    if b["hands_left"] < a["hands_left"] and b["hands_left"] >= 0:
        return (ACTION_PLAY, {"chips_gained": b["chips"] - a["chips"]})
    if b["discards_left"] < a["discards_left"] and b["discards_left"] >= 0:
        return (ACTION_DISCARD, {})

    # --- shop -------------------------------------------------------------
    spent = a["money"] - b["money"]
    if b["n_jokers"] > a["n_jokers"] and spent > 0:
        return (ACTION_BUY_JOKER, {"cost": spent})
    if b["n_jokers"] < a["n_jokers"] and b["money"] > a["money"]:
        return (ACTION_SELL_JOKER, {"gain": b["money"] - a["money"]})
    if b["n_packs"] < a["n_packs"] and spent > 0:
        return (ACTION_BUY_PACK, {"cost": spent})
    if b["n_cons"] > a["n_cons"] and spent > 0:
        return (ACTION_BUY_VOUCHER, {"cost": spent})
    if b["n_cons"] < a["n_cons"] and abs(spent) < 1e-9:
        return (ACTION_USE_CONSUMABLE, {})
    # Reroll: shop contents replaced, money down, nothing acquired.
    if (spent > 0 and b["n_jokers"] == a["n_jokers"]
            and b["n_cons"] == a["n_cons"] and b["n_shop"] == a["n_shop"]):
        return (ACTION_REROLL, {"cost": spent})

    # --- transitions ------------------------------------------------------
    if a["state"] != b["state"]:
        # Entering a pack is the consequence of the buy already recorded above,
        # not a separate decision.
        if "BOOSTER" in str(b["state"]):
            # Entering a booster IS the purchase. You cannot reach a booster
            # state any other way, so the transition identifies the action on its
            # own — no money delta required.
            #
            # This was previously `return None`, on the assumption the shop-side
            # BUY_PACK rule had already recorded it. That rule needs `spent > 0`,
            # but the money change and the pack-count change frequently land on
            # DIFFERENT poll ticks at 0.35s, so neither half matched alone and the
            # buy vanished. It was the largest remaining gap on live human play.
            #
            # The caller de-duplicates (see Recorder._is_duplicate_pack_buy) for
            # the case where the money-based rule DID fire a tick earlier.
            return (ACTION_BUY_PACK, {"via": "entered_booster"})
        if a["state"] == "SHOP":
            return (ACTION_END_SHOP, {})
        if "BLIND_SELECT" in str(a["state"]):
            # A skip advances past the blind without playing it; a select enters
            # it. Distinguished by whether a round actually started.
            if b["hands_left"] > 0 and "BLIND" not in str(b["state"]):
                return (ACTION_SELECT_BLIND, {})
            if b["ante"] > a["ante"] or b["state"] == "SHOP":
                return (ACTION_SKIP_BLIND, {})
            return (ACTION_SELECT_BLIND, {})
    return None



# ---------------------------------------------------------------------------
# CONTENT capture (dec-096 follow-up)
#
# The action TYPE alone is not a demonstration. "A pack card was taken" cannot
# be replayed or learned from; the agent's action space is
# (action_type, target_index, card_bits), so the CHOICE has to be recorded, not
# just the fact that a choice happened.
#
# Every card the mod serialises carries `id = card.sort_id` (see
# extract_card in the mod's src/lua/utils/gamestate.lua), which is a stable
# identity. So the content of any action is recoverable by DIFFING the relevant
# area by id across the transition: what left `hand` was played or discarded,
# what arrived in `jokers` was bought, what left `consumables` was used.
#
# Each event also stores the ALTERNATIVES that were available -- the full hand,
# shop, packs and joker row at the moment of the decision. A demonstration needs
# to show what was passed over, not only what was picked.
# ---------------------------------------------------------------------------

CARD_FIELDS = ("id", "key", "set", "label", "value", "modifier", "cost",
               "ability_extra")


def _area(raw: dict, *path) -> list:
    """The card list at e.g. ("hand",) / ("jokers",) / ("shop",)."""
    node = _n(raw, *path, default={}) or {}
    if not isinstance(node, dict):
        return []
    return [c for c in (node.get("cards") or []) if isinstance(c, dict)]


def _as_dict(v) -> dict:
    """Lua serialises an EMPTY table as a JSON array, not an object.

    Verified against a live gamestate: a plain playing card comes back with
    `"state": []` and `"modifier": []`, and only becomes an object once it has a
    key. Every read of those fields must therefore tolerate a list, or the first
    unmodified card blows up the recorder mid-run. hand_eval.py carries the same
    helper for the same reason.
    """
    return v if isinstance(v, dict) else {}


def _slim(card: dict) -> dict:
    """Keep the identifying fields, drop UI noise."""
    return {k: card[k] for k in CARD_FIELDS if k in card}


def _by_id(cards: list) -> dict:
    return {c.get("id"): c for c in cards if c.get("id") is not None}


def _diff(before: list, after: list):
    """(left, arrived) as full card records, matched on the stable sort_id."""
    b, a = _by_id(before), _by_id(after)
    left = [_slim(v) for k, v in b.items() if k not in a]
    arrived = [_slim(v) for k, v in a.items() if k not in b]
    return left, arrived


# Every card area the mod serialises (see the tail of its get_gamestate):
#   jokers, consumables, cards(deck), hand, shop(shop_jokers),
#   vouchers(shop_vouchers), packs(shop_booster), pack(pack_cards)
# `pack` is the CONTENTS of an opened booster and `vouchers` is separate from
# `shop` -- the first draft diffed neither, which is precisely where "which card
# did you pick from the pack" and "which voucher" live. `cards` (the deck) is
# excluded from diffing because it churns on every draw, but it is snapshotted
# by size in the context.
AREAS = (("hand",), ("jokers",), ("consumables",), ("shop",), ("vouchers",),
         ("packs",), ("pack",))


def content_of(prev: dict, cur: dict) -> dict:
    """What was actually chosen, derived from the area diffs.

    Returns e.g. {"played": [...], "bought": [...], "used": [...]} with FULL card
    records, so the decision can be reconstructed exactly.
    """
    out = {}
    for path in AREAS:
        left, arrived = _diff(_area(prev, *path), _area(cur, *path))
        if left:
            out[path[0] + "_left"] = left
        if arrived:
            out[path[0] + "_arrived"] = arrived

    # Cards whose MODIFIER changed but which stayed put -- a tarot applied to
    # cards in hand shows up here and nowhere else, since nothing enters or
    # leaves. Without this, "used a tarot" would record the consumable but never
    # which cards it was aimed at.
    changed = []
    pb, cb = _by_id(_area(prev, "hand")), _by_id(_area(cur, "hand"))
    for cid, before in pb.items():
        after = cb.get(cid)
        if (after is not None
                and _as_dict(before.get("modifier"))
                != _as_dict(after.get("modifier"))):
            changed.append({"id": cid,
                            "before": _slim(before), "after": _slim(after)})
    if changed:
        out["hand_modified"] = changed

    # PLAY and DISCARD are the one case a diff alone can get wrong: the acted-on
    # cards are mid-animation on the transition tick, so they may or may not have
    # left `hand` yet. The game marks selected cards `state.highlight`, so the
    # cards the player picked are exactly those highlighted in the state BEFORE
    # the action -- an authoritative reading that does not depend on animation
    # timing.
    picked = [_slim(c) for c in _area(prev, "hand")
              if _as_dict(c.get("state")).get("highlight")]
    if picked:
        out["selected"] = picked
    return out


# Which content field holds the AUTHORITATIVE answer for each action, so a
# consumer never has to guess. Measured on a live run: on PLAY, `hand_left`
# disagreed with `selected` in 23 of 28 events, because the area diff catches the
# hand mid-animation while `selected` comes from the game's own highlight flag
# and is exact. The diff fields are kept for audit; `chosen` is what to read.
CHOSEN_FIELD = {
    ACTION_PLAY: ("selected", "hand_left"),
    ACTION_DISCARD: ("selected", "hand_left"),
    ACTION_BUY_JOKER: ("jokers_arrived", "shop_left"),
    ACTION_SELL_JOKER: ("jokers_left",),
    ACTION_BUY_VOUCHER: ("vouchers_left", "consumables_arrived", "shop_left"),
    ACTION_BUY_PACK: ("packs_left",),
    ACTION_USE_CONSUMABLE: ("consumables_left",),
    ACTION_SELECT_PACK_CARD: ("pack_left",),
    ACTION_REROLL: ("shop_arrived",),
}


def chosen_of(action: int, content: dict) -> list:
    """The single authoritative list of cards this action acted on."""
    for field in CHOSEN_FIELD.get(action, ()):
        if content.get(field):
            return content[field]
    return []


def context_of(raw: dict) -> dict:
    """The alternatives on the table when the decision was made."""
    rnd = _n(raw, "round", default={}) or {}
    blinds = _n(raw, "blinds", default={}) or {}
    cur_blind = None
    if isinstance(blinds, dict):
        for slot in ("small", "big", "boss"):
            b = blinds.get(slot)
            if isinstance(b, dict) and b.get("status") == "CURRENT":
                cur_blind = {"slot": slot, "name": b.get("name"),
                             "score": b.get("score")}
                break
    return {
        "hand": [_slim(c) for c in _area(raw, "hand")],
        "jokers": [_slim(c) for c in _area(raw, "jokers")],
        "consumables": [_slim(c) for c in _area(raw, "consumables")],
        "shop": [_slim(c) for c in _area(raw, "shop")],
        "packs": [_slim(c) for c in _area(raw, "packs")],
        "vouchers": [_slim(c) for c in _area(raw, "vouchers")],
        "pack_open": [_slim(c) for c in _area(raw, "pack")],
        "deck_n": len(_area(raw, "cards")),
        "money": _n(raw, "money", default=0),
        "hands_left": rnd.get("hands_left"),
        "discards_left": rnd.get("discards_left"),
        "chips": rnd.get("chips"),
        "blind": cur_blind,
        "hand_levels": _n(raw, "hands", default={}),
    }


class Recorder:
    """Polls the live game and reconstructs the action stream. Read-only."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.prev: dict | None = None
        self.events: list[dict] = []
        self.polls = 0
        self.changes = 0       # every state change, engine included
        self.decisions = 0     # changes a player could have caused
        self.inferred = 0      # decisions we successfully labelled
        self.missed: list[dict] = []
        self._path: str | None = None
        self._last_pack_buy_poll: int = -99
        # Deferred content resolution -- see _resolve_pending.
        self._pending: tuple | None = None
        # Most recent NON-EMPTY contents of an open booster. A pack pick has to
        # be resolved against the pack as it stood while open, not against a
        # later state -- see _resolve_pending.
        self._pack_open: list = []
        # Did the player take anything from the pack currently open?
        self._took_from_pack = False
        # Most recent NON-EMPTY set of highlighted hand cards. A human
        # highlights, then clicks -- so the selection is visible for many polls
        # before the action, and remembering it is far more reliable than
        # sampling it on the transition tick.
        self._last_highlight: list = []

    def observe(self, raw: dict) -> None:
        self.polls += 1
        # Track the highlighted cards on EVERY poll.
        #
        # Audited against the game's own end-of-run counters: discards captured
        # 119 of 267 cards (45%) because if the player highlights and clicks
        # inside one 0.35s poll, the pre-action snapshot has no highlight yet and
        # the code fell back to the partial area diff. The selection is on screen
        # for seconds beforehand, so the last non-empty highlight is almost
        # always the right answer.
        hl = [c for c in _area(raw, "hand")
              if _as_dict(c.get("state")).get("highlight")]
        if hl:
            self._last_highlight = [_slim(c) for c in hl]

        pack_now = _area(raw, "pack")
        if pack_now:
            if not self._pack_open:
                # A pack just OPENED -- it starts untouched. The flag must be
                # cleared here, not when the pack closes: clearing on close ran
                # before the exit transition was classified, so the exit no
                # longer knew a card had been taken and the spurious skip came
                # back.
                self._took_from_pack = False
            self._pack_open = pack_now
        elif self._pack_open:
            self._pack_open = []
        if self.prev is not None and _counts(self.prev) == _counts(raw):
            # An IDLE poll is the ideal moment to resolve: nothing is moving, so
            # the hand has certainly settled.
            self._resolve_pending(raw)
        if self.prev is not None and _counts(self.prev) != _counts(raw):
            self.changes += 1
            if not is_decision(self.prev, raw):
                # Resolution MUST happen even here. Everything that follows a
                # play is an engine transition (HAND_PLAYED -> DRAW_TO_HAND ->
                # SELECTING_HAND), so an early return that skipped resolution
                # meant a played hand's cards were never recovered live -- the
                # deferral would only ever have fired on the next decision.
                self._resolve_pending(raw)
                self.prev = raw
                return
            self.decisions += 1
            got = infer_action(self.prev, raw)
            if got is None:
                # Keep what was dropped so a low coverage number can be
                # diagnosed instead of merely reported.
                a, b = _counts(self.prev), _counts(raw)
                self.missed.append(
                    {"fields": sorted(k for k in a if a[k] != b[k]),
                     "from": a["state"], "to": b["state"]})
            elif self._is_spurious_pack_skip(got):
                # Leaving a pack you already took from is not a skip.
                self.decisions -= 1
            elif self._is_duplicate_pack_buy(got):
                # Counted once already; not a miss and not a second action.
                self.decisions -= 1
            else:
                self.inferred += 1
                atype, detail = got
                event = {
                    "t": round(time.time(), 2),
                    "action": atype,
                    "action_name": ACTION_NAMES.get(atype, str(atype)),
                    "detail": detail,
                    "ante": _counts(raw)["ante"],
                    "state": _counts(self.prev)["state"],
                    # WHAT WAS CHOSEN -- full card records, derived by diffing
                    # each area on the stable sort_id. Without this the event
                    # says only that a choice happened, which cannot be replayed
                    # or imitated; the action encoding needs the card selection,
                    # not just the action type.
                    "content": self._content_with_highlight(self.prev, raw,
                                                            atype),
                    # WHAT WAS PASSED OVER -- the hand, shop, packs, jokers and
                    # consumables as they stood when the player decided. A
                    # demonstration is the choice AND the alternatives.
                    "context": context_of(self.prev),
                }
                # A NEW action while one is still pending would otherwise
                # overwrite the pending tuple and lose that event entirely.
                # Resolve it first against self.prev -- the state immediately
                # before this new action, which is by definition settled, and so
                # is the best possible resolution state.
                if self._pending is not None:
                    self._resolve_pending(self.prev, force=True)
                # The state BEFORE the action is kept so the content can be
                # recomputed later against a settled state (see
                # _resolve_pending).
                if atype in (ACTION_PLAY, ACTION_DISCARD):
                    # Consumed -- a stale selection must never be attributed to
                    # a LATER action.
                    self._last_highlight = []
                self._pending = (event, self.prev, self.polls)
                self.events.append(event)
                # Flush AFTER appending. The first version called
                # _flush(self.events[-1]) before the append, so the very first
                # labelled action indexed an empty list and raised IndexError —
                # which, in a module whose whole job is not losing data, would
                # have killed the recorder on its first real observation.
                if event["content"]:
                    event["chosen"] = chosen_of(atype, event["content"])
                    self._flush(event)
                    self._pending = None
        self._resolve_pending(raw)
        self.prev = raw

    # Poll ticks to wait for the game to settle before giving up on recovering
    # an action's content. At POLL_S=0.35 this is ~5s -- longer than any play or
    # buy animation, short enough that an event is never held indefinitely.
    PENDING_TIMEOUT = 14

    def _resolve_pending(self, raw: dict, force: bool = False) -> None:
        """Recover the content of an event whose diff was empty at the time.

        Measured live: at a 0.35s poll the transition that decrements
        hands_left is frequently caught BEFORE the played cards have left
        `hand`, so prev->cur shows no card moving anywhere and the event lands
        with empty content -- three PLAYs in a row recorded 0% content in the
        first live test. The information is not lost, it just has not happened
        yet: a tick or two later the hand has settled and the departed cards are
        plainly visible against the pre-action snapshot.

        So an event with empty content is held, and re-diffed against each new
        state until something shows up. The comparison is always against the
        PRE-ACTION state, never the previous poll, so a slow animation cannot
        make the diff drift.

        On timeout the event is flushed anyway, with content_resolved=False, so
        a gap is recorded honestly instead of the event being dropped or, worse,
        silently written with a content field that means "nothing was chosen".
        """
        if self._pending is None:
            return
        event, before, at_poll = self._pending

        # PACK PICKS resolve against the open pack, not the whole state.
        #
        # Measured on a live run: only 6 of 24 pack picks named the card taken.
        # The rest reported `pack_arrived` (the pack OPENING) or a pick from the
        # PREVIOUS pack -- content one step out of phase with the offer, e.g.
        # offered [Judgement, The Devil] recorded as taking Temperance. The
        # deferred diff was resolving against a state where the pack had already
        # closed and the next one opened.
        #
        # The pack contents while open are the only correct baseline, so they are
        # tracked separately and the pick is whatever left THEM.
        if event["action"] in (ACTION_SELECT_PACK_CARD, ACTION_SKIP_PACK):
            baseline = _area(before, "pack") or self._pack_open
            now = _area(raw, "pack")
            if baseline:
                left, _ = _diff(baseline, now)
                if left:
                    event["content"]["pack_left"] = left
                    event["content_resolved"] = "pack_diff"
                if not now:
                    # Pack closed: this is as resolved as it will ever get.
                    event.setdefault("content_resolved", False)
                    event["chosen"] = chosen_of(event["action"], event["content"])
                    self._flush(event)
                    self._pending = None
                    return

        # Take the LATEST diff, not the first non-empty one.
        #
        # Measured live: resolving on the first non-empty diff captured ONE of
        # five played cards, because the cards leave the hand a few at a time
        # over the animation and the first tick that shows any movement shows
        # only part of it. Cards never come BACK to the hand mid-round, so the
        # diff only grows -- the newest reading is always the most complete.
        #
        # Except across a round boundary, where the hand empties entirely and a
        # diff would claim every remaining card was played. An empty hand is
        # therefore never a resolution state.
        hand_now = _area(raw, "hand")
        settled = str(_counts(raw)["state"]) not in ENGINE_STATES
        if hand_now or settled:
            content = content_of(before, raw)
            if content:
                event["content"] = content
                event["content_resolved"] = "deferred"

        # Flush once the game is idle again (settled state, nothing moving), on
        # timeout, or when a new action forces the issue.
        idle = settled and self.prev is not None and _counts(self.prev) == _counts(raw)
        if event.get("content") and (idle or force):
            event["chosen"] = chosen_of(event["action"], event["content"])
            self._flush(event)
            self._pending = None
        elif force or self.polls - at_poll >= self.PENDING_TIMEOUT:
            event.setdefault("content_resolved", False)
            event["chosen"] = chosen_of(event["action"], event["content"])
            self._flush(event)
            self._pending = None

    def flush_pending(self) -> None:
        """Write any still-unresolved event. Called on shutdown so the last
        action of a session is never lost to the deferral window."""
        if self._pending is not None:
            event, _, _ = self._pending
            event.setdefault("content_resolved", False)
            event["chosen"] = chosen_of(event["action"], event["content"])
            self._flush(event)
            self._pending = None

    def _content_with_highlight(self, prev: dict, cur: dict, atype: int) -> dict:
        """content_of, with the remembered highlight filling in `selected`."""
        content = content_of(prev, cur)
        if atype in (ACTION_PLAY, ACTION_DISCARD) and not content.get("selected"):
            if self._last_highlight:
                content["selected"] = self._last_highlight
                content["selected_from"] = "remembered_highlight"
        return content

    def _is_spurious_pack_skip(self, got) -> bool:
        """True when this SKIP_PACK is just the exit from a pack already used.

        Every pack exit produced a SKIP_PACK, whether or not cards were taken:
        the live run recorded 35 BUY_PACK and 35 SKIP_PACK, exactly one skip per
        pack. That is a fabricated action -- it teaches the agent that a pack it
        took two cards from was skipped, inverting the very decision the
        recording exists to capture.

        A real skip is leaving a pack having taken NOTHING from it.
        """
        atype, _ = got
        if atype != ACTION_SKIP_PACK:
            if atype == ACTION_SELECT_PACK_CARD:
                self._took_from_pack = True
            return False
        took = self._took_from_pack
        self._took_from_pack = False
        return took

    def _is_duplicate_pack_buy(self, got) -> bool:
        """True when this BUY_PACK is the same purchase already recorded.

        A pack buy shows up twice at a 0.35s poll: once as money leaving in the
        SHOP, and once as the state entering the booster. Either half can be the
        one that lands first, and sometimes only one lands at all -- so BOTH are
        treated as the signal and this collapses them.

        Scoped to BUY_PACK deliberately. A blanket "same action twice in a row is
        a duplicate" rule would silently swallow legitimate repeats -- two plays,
        two discards, two rerolls -- which is a far worse failure than
        double-counting a pack.
        """
        if not got or got[0] != ACTION_BUY_PACK:
            return False
        if self.polls - self._last_pack_buy_poll <= 3:
            return True
        self._last_pack_buy_poll = self.polls
        return False

    def _flush(self, event: dict) -> None:
        """Append ONE event to disk immediately.

        The first version accumulated everything in memory and wrote once, at
        exit. On Windows a console process cannot be terminated gracefully
        (taskkill refuses without /F, and /F skips the handler), so the first
        real human session's 176 events were unrecoverable the moment the run
        ended — the recorder held the only copy.

        That is the same single-point-of-loss pattern as the rotating logs that
        destroyed a control arm and a baseline earlier in this project. Append as
        you go: a session can now be interrupted, crash, or be force-killed
        without losing anything already observed.
        """
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            if self._path is None:
                self._path = os.path.join(
                    self.out_dir, f"human_{int(time.time())}.jsonl")
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except Exception:
            pass          # never let logging break a recording session

    @property
    def coverage(self) -> float:
        """Fraction of DECISION transitions that produced a labelled action.

        The denominator is decisions, not raw changes: the first version divided
        by every state change and reported 47%, but the misses were dominated by
        the game's own state machine (scoring animation, round eval, draws,
        counter refills). Those are not actions and scoring against them measured
        the wrong thing.

        Low coverage here is real: it means human play produces DECISIONS the
        encoding cannot express, and feeding those trajectories in would train on
        a fiction.
        """
        return self.inferred / self.decisions if self.decisions else 0.0

    @property
    def content_coverage(self) -> float:
        """Fraction of recorded actions that also captured WHAT was chosen.

        This, not `coverage`, is the number that decides whether a recording is
        usable. `coverage` only asks "did we notice a decision happened" -- it
        read 62% while every single event stored the action type and nothing
        else, so a recording that could not be replayed or imitated looked fine.
        An event with empty content is an action whose CHOICE was lost.
        """
        if not self.events:
            return 0.0
        with_content = sum(1 for e in self.events if e.get("content"))
        return with_content / len(self.events)

    def breakdown(self) -> str:
        """Per-action-type tally of content capture, for the live status line."""
        tally: dict = {}
        for e in self.events:
            name = e.get("action_name", "?")
            got, tot = tally.get(name, (0, 0))
            tally[name] = (got + (1 if e.get("content") else 0), tot + 1)
        return " ".join(f"{k}:{g}/{t}" for k, (g, t) in sorted(tally.items()))

    def save(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"human_{int(time.time())}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e) + "\n")
        return path


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=12346)
    ap.add_argument("--out", default=os.path.join("recordings", "human"))
    ap.add_argument("--seconds", type=int, default=0, help="0 = until Ctrl-C")
    args = ap.parse_args()

    from environment.game_state import GameStateManager

    game = GameStateManager(port=args.port)
    rec = Recorder(args.out)
    print(f"[HUMAN-REC] polling :{args.port} every {POLL_S}s — play normally. "
          f"Ctrl-C to stop.", flush=True)

    t0 = time.time()
    try:
        while True:
            try:
                raw = await game.fetch_gamestate()
                if raw:
                    rec.observe(raw)
            except Exception as e:
                print(f"[HUMAN-REC] poll failed: {e}", flush=True)
            if rec.polls % 60 == 0 and rec.decisions:
                # content% leads, because it is the number that decides whether
                # the recording is usable. The per-type breakdown makes a
                # single broken action class visible immediately instead of
                # being averaged away -- both previous recordings were lost to
                # problems that a live status line would have shown in a minute.
                print(f"[HUMAN-REC] polls={rec.polls} actions={rec.inferred} "
                      f"content={rec.content_coverage:.0%} "
                      f"coverage={rec.coverage:.0%} | {rec.breakdown()}",
                      flush=True)
            if args.seconds and time.time() - t0 > args.seconds:
                break
            await asyncio.sleep(POLL_S)
    except KeyboardInterrupt:
        pass

    rec.flush_pending()
    path = rec.save() if rec.events else "(nothing recorded)"
    print(f"\n[HUMAN-REC] {rec.inferred} actions from {rec.changes} state "
          f"changes — coverage {rec.coverage:.0%}, "
          f"content captured {rec.content_coverage:.0%}"
          f"\n[HUMAN-REC] {rec.breakdown()}\n[HUMAN-REC] wrote {path}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
