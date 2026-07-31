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

STATUS: SPIKE. Validates whether human play maps into the action encoding at all.
It does NOT yet write demo_buffer trajectories -- state-vector encoding and
head_indices are the next step, and are pointless until coverage is known good.
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
    ACTION_SELL_JOKER,
    ACTION_SKIP_BLIND,
    ACTION_USE_CONSUMABLE,
)

POLL_S = 0.35


def _n(state: dict, *path, default=0):
    cur = state
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


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

    # Landing in an engine state is the game advancing itself. Note the reverse —
    # LEAVING one (e.g. ROUND_EVAL -> SHOP) is also engine-driven, but leaving
    # SHOP or BLIND_SELECT is a real choice, so only the destination is tested.
    if str(b["state"]) in ENGINE_STATES:
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

    def observe(self, raw: dict) -> None:
        self.polls += 1
        if self.prev is not None and _counts(self.prev) != _counts(raw):
            self.changes += 1
            if not is_decision(self.prev, raw):
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
            else:
                self.inferred += 1
                atype, detail = got
                self.events.append({
                    "t": round(time.time(), 2),
                    "action": atype,
                    "detail": detail,
                    "ante": _counts(raw)["ante"],
                    "jokers": [j.get("label") or j.get("key")
                               for j in _n(raw, "jokers", "cards", default=[])],
                })
        self.prev = raw

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
                print(f"[HUMAN-REC] polls={rec.polls} changes={rec.changes} "
                      f"decisions={rec.decisions} actions={rec.inferred} "
                      f"coverage={rec.coverage:.0%}", flush=True)
            if args.seconds and time.time() - t0 > args.seconds:
                break
            await asyncio.sleep(POLL_S)
    except KeyboardInterrupt:
        pass

    path = rec.save() if rec.events else "(nothing recorded)"
    print(f"\n[HUMAN-REC] {rec.inferred} actions from {rec.changes} state "
          f"changes — coverage {rec.coverage:.0%}\n[HUMAN-REC] wrote {path}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
