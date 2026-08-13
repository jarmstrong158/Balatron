"""Repair the pack picks in human recordings written before the dec-102 fix.

WHY
---
The pack-pick detector tested `consumables went up or money changed`. That fires
for Arcana and Spectral packs only. A CELESTIAL pick levels a hand, a BUFFOON
pick adds a joker, a STANDARD pick adds a deck card -- so all three were recorded
as SKIP_PACK. On the live winning run that meant every Jupiter taken was written
down as a pack passed over, inverting the single signal the recording exists to
capture: does the hand levelled track the hand played.

The information is not lost. Every event stores the full `hand_levels` dict, the
joker row, and the deck size in its context, so the acquisition is visible by
diffing consecutive events' contexts. And a levelled hand maps deterministically
to a planet, so the specific card is recoverable, not merely the fact of a pick.

VALIDATION
----------
The player stated, before this script existed, that every celestial pack this run
was Jupiter (Telescope voucher + a Flush-heavy build). That is a falsifiable
prediction: if the repaired celestial picks do not come out overwhelmingly
Jupiter, this repair is wrong and its output must not be used. `--check` reports
that distribution rather than assuming it.

Non-destructive: writes <name>.repaired.jsonl and leaves the original alone.

USAGE
-----
    python audit_repair_human_record.py recordings/human/*.jsonl --check
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from environment.action_space import ACTION_SELECT_PACK_CARD

# A levelled hand identifies the planet uniquely -- this is the whole reason a
# celestial pick is recoverable after the fact.
PLANET_FOR_HAND = {
    "High Card": "Pluto",
    "Pair": "Mercury",
    "Two Pair": "Uranus",
    "Three of a Kind": "Venus",
    "Straight": "Saturn",
    "Flush": "Jupiter",
    "Full House": "Earth",
    "Four of a Kind": "Mars",
    "Straight Flush": "Neptune",
    "Five of a Kind": "Planet X",
    "Flush House": "Ceres",
    "Flush Five": "Eris",
}


def _levels(ctx: dict) -> dict:
    hands = ctx.get("hand_levels")
    if not isinstance(hands, dict):
        return {}
    out = {}
    for name, v in hands.items():
        if isinstance(v, dict):
            try:
                out[name] = int(v.get("level") or 0)
            except (TypeError, ValueError):
                pass
    return out


def _joker_ids(ctx: dict) -> set:
    return {j.get("id") for j in ctx.get("jokers") or [] if j.get("id") is not None}


def repair(events: list) -> tuple[list, dict]:
    """Relabel SKIP_PACK events that were really picks. Returns (events, stats)."""
    stats = {"celestial": 0, "buffoon": 0, "standard": 0, "left_as_skip": 0,
             "planets": {}}
    for i, e in enumerate(events):
        if e.get("action_name") != "SKIP_PACK":
            continue
        # The acquisition happens INSIDE the pack, before the exit is logged, so
        # the skip's OWN context already carries the raised level / new joker.
        # The correct baseline is therefore the PRECEDING event (the pack buy),
        # not the following one. Diffing forward recovered 1 celestial pick out
        # of a run full of them; diffing back is what actually finds them.
        prev_ctx = (events[i - 1].get("context") or {}) if i else {}
        own = e.get("context") or {}
        nxt = (events[i + 1].get("context") or {}) if i + 1 < len(events) else {}

        # Try back-diff first, then forward, and keep whichever shows movement.
        for before, after in ((prev_ctx, own), (own, nxt)):
            if not before or not after:
                continue
            if (_levels(after) != _levels(before)
                    or _joker_ids(after) != _joker_ids(before)
                    or (after.get("deck_n") or 0) != (before.get("deck_n") or 0)):
                break
        else:
            stats["left_as_skip"] += 1
            continue

        lb, la = _levels(before), _levels(after)
        levelled = [h for h, v in la.items() if v > lb.get(h, v)]
        new_jokers = _joker_ids(after) - _joker_ids(before)
        deck_grew = (after.get("deck_n") or 0) > (before.get("deck_n") or 0)

        if levelled:
            planet = PLANET_FOR_HAND.get(levelled[0], levelled[0])
            e["action"] = ACTION_SELECT_PACK_CARD
            e["action_name"] = "SELECT_PACK_CARD"
            e["chosen"] = [{"label": planet, "set": "PLANET",
                            "recovered_from": "hand_level_increment",
                            "hand": levelled[0]}]
            e["repaired"] = "celestial"
            stats["celestial"] += 1
            stats["planets"][planet] = stats["planets"].get(planet, 0) + 1
        elif new_jokers:
            picked = [j for j in after.get("jokers") or []
                      if j.get("id") in new_jokers]
            e["action"] = ACTION_SELECT_PACK_CARD
            e["action_name"] = "SELECT_PACK_CARD"
            e["chosen"] = picked
            e["repaired"] = "buffoon"
            stats["buffoon"] += 1
        elif deck_grew:
            # The specific card is NOT recoverable -- only that one of the
            # offered cards was taken. Recorded honestly as such rather than
            # guessed, since a wrong card is worse than a missing one.
            e["action"] = ACTION_SELECT_PACK_CARD
            e["action_name"] = "SELECT_PACK_CARD"
            e["chosen"] = []
            e["chosen_unknown_from"] = before.get("pack_open") or []
            e["repaired"] = "standard_card_unknown"
            stats["standard"] += 1
        else:
            stats["left_as_skip"] += 1
    return events, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true",
                    help="report the recovered planet distribution")
    args = ap.parse_args()

    paths = [p for pat in args.files for p in glob.glob(pat)]
    total = {"celestial": 0, "buffoon": 0, "standard": 0, "left_as_skip": 0,
             "planets": {}}
    for path in sorted(paths):
        if path.endswith(".repaired.jsonl"):
            continue
        events = [json.loads(x) for x in open(path, encoding="utf-8")
                  if x.strip()]
        events, st = repair(events)
        out = path.replace(".jsonl", ".repaired.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        print(f"{os.path.basename(path)}: {len(events)} events -> "
              f"celestial={st['celestial']} buffoon={st['buffoon']} "
              f"standard={st['standard']} still_skip={st['left_as_skip']}")
        for k in ("celestial", "buffoon", "standard", "left_as_skip"):
            total[k] += st[k]
        for p, n in st["planets"].items():
            total["planets"][p] = total["planets"].get(p, 0) + n

    print(f"\nTOTAL recovered picks: celestial={total['celestial']} "
          f"buffoon={total['buffoon']} standard={total['standard']}  "
          f"(genuine skips left alone: {total['left_as_skip']})")
    if args.check:
        print("\nRecovered planet distribution -- the falsifiable check:")
        for p, n in sorted(total["planets"].items(), key=lambda x: -x[1]):
            print(f"  {p:<10} {n}")
        jup = total["planets"].get("Jupiter", 0)
        tot = sum(total["planets"].values())
        if tot:
            print(f"\n  Jupiter share: {jup}/{tot} = {jup / tot:.0%}")
            print("  Prediction was 'every celestial this run was Jupiter'. "
                  "A low share here means the repair is WRONG, not that the "
                  "prediction was.")


if __name__ == "__main__":
    main()
