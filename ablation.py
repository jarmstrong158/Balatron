"""dec-101: turn a shipped component OFF so its contribution can be measured.

WHY
---
The 08-03 three-way audit's central finding was not any single bug. It was that
NOTHING CURRENTLY RUNNING HAS A VALID EFFICACY MEASUREMENT. Roughly 30 shipped
changes, and the project cannot say which are load-bearing, which are inert, and
which are net-negative:

  * unmeasured on deploy and never revisited: dec-059, dec-060, dec-061,
    dec-065, dec-068, dec-070, dec-072, dec-077, dec-084
  * validated against `realized_vs_proj`, which dec-076 later proved is
    1/raw_margin on beaten blinds -- a tautology: dec-052, dec-053, dec-070
  * kept on a non-significant A/B: dec-078
  * measured on 60-600 runs against an ~18,600 runs/arm requirement, i.e.
    non-results (dec-098): dec-075, dec-079, dec-081, dec-083, dec-093

There is no no-op / pure-heuristic reference point anywhere in the record, so the
~4.2 mean ante has never been attributed to anything. dec-100 showed that is not
a theoretical worry: three boss guards had been reading a key the API never
returns, and one of them was INVERTING behaviour at the deadliest boss.

WHY THIS IS NEWLY AFFORDABLE
---------------------------
Ablation used to be unmeasurable. Under win rate a realistic component is worth
maybe +1pp per blind, which needs ~18,600 runs per arm. dec-098 established that
per-blind clear rate is ~24x more sample-efficient -- a 2pp effect needs ~4,500
blinds, about a day, computed from ordinary training logs with no stopped
training and no eval session. So an ablation sweep is now roughly one component
per day.

USAGE
-----
    BALATRON_ABLATE=boss_overrides   # then restart the supervisor

Comma-separated for several at once. The supervisor logs the active set as a
regime boundary (dec-100) and snapshots per-ante clear rates to
baselines/boundaries/ so the control side of the comparison cannot rotate away --
which is exactly how the swap-legality control arm was lost.

Read the result with:
    python audit_blind_clear.py --split-step <step at deploy>

SAFETY: default is EMPTY, so with the variable unset every component behaves
exactly as before and the control arm is byte-identical.
"""
from __future__ import annotations

import os as _os

# Every component that can be ablated. Keeping this explicit means a typo in the
# env var fails LOUDLY at import instead of silently ablating nothing and
# producing a "no effect" result that is really a no-op arm -- the dec-093
# failure mode, where a forcing that never fired nearly became a confident null.
KNOWN = {
    "boss_overrides": (
        "mouth_should_dig / needle_should_dig hard-overriding the policy's PLAY "
        "into a discard (action_executor ~line 419). dec-100 fixed these from "
        "permanently-inert to actually firing, so their real contribution has "
        "never been measured in either state."
    ),
}

_raw = _os.environ.get("BALATRON_ABLATE", "")
ABLATED = {s.strip() for s in _raw.split(",") if s.strip()}

_unknown = ABLATED - set(KNOWN)
if _unknown:
    raise ValueError(
        f"BALATRON_ABLATE names unknown component(s): {sorted(_unknown)}. "
        f"Known: {sorted(KNOWN)}. Refusing to start rather than run an arm that "
        f"silently ablates nothing — a no-op arm reports a false null."
    )


def is_ablated(name: str) -> bool:
    """True when `name` should be disabled for this run.

    Raises on an unknown name for the same reason the env var does: a component
    that is silently never ablated produces a measurement of the control against
    itself.
    """
    if name not in KNOWN:
        raise KeyError(f"unknown ablation component {name!r}; known: {sorted(KNOWN)}")
    return name in ABLATED


def describe() -> str:
    """One line for the supervisor log, so the arm is recoverable from logs."""
    if not ABLATED:
        return "ablations: none (full system)"
    return "ABLATED: " + ", ".join(sorted(ABLATED))
