"""Rate-limited warnings for swallowed exceptions on the decision path.

WHY THIS EXISTS
---------------
An audit found 104 broad `except Exception` handlers, 55 of them silent
(`pass`/`continue`). Most are deliberate and correct — instrumentation and
logging must never take the trainer down mid-run, and that intent is usually
stated in a comment right there.

The dangerous subset is the handful on the DECISION path, where swallowing a
failure does not merely lose a log line, it silently changes what the agent
does:

  * action_space: `find_best_hands` failing leaves `current_score = 0.0`, so the
    mask biases card selection as if the hand were worthless.
  * action_space: `find_best_discard` failing leaves `discard_ev = 0.0`, so
    discarding never looks worth it.
  * game_state: a per-joker valuation failing drops that joker's flags out of
    the state vector the policy reads.

A single transient failure there is genuinely fine to absorb. A PERSISTENT one
is a silent behavioural regression that no metric would obviously attribute —
which is exactly the class of bug that costs weeks. Bare `print` was rejected:
these sit in per-step hot paths, so an unconditional print would emit thousands
of lines a minute and get muted or drowned, which is how you end up ignoring the
signal you added.

So: warn LOUDLY the first time each distinct site fails, then exponentially back
off. Never raises — a diagnostics helper that can break the caller would defeat
its own purpose.
"""
from __future__ import annotations

import threading

# site key -> times seen. Module-level, process-wide, guarded for the
# multi-env trainer (each env runs its own asyncio task in one process).
_SEEN: dict[str, int] = {}
_LOCK = threading.Lock()


def _should_emit(n: int) -> bool:
    """Emit on the 1st, 2nd, 5th, 10th, 100th, 1000th ... occurrence."""
    if n <= 2:
        return True
    if n == 5:
        return True
    x = 10
    while x <= n:
        if n == x:
            return True
        x *= 10
    return False


def warn_once(site: str, exc: BaseException, extra: str = "") -> None:
    """Report a swallowed exception at `site`, rate-limited per site.

    `site` must be a STABLE identifier (module.function/what-failed), not a
    formatted message — the count is keyed on it.
    """
    try:
        with _LOCK:
            n = _SEEN.get(site, 0) + 1
            _SEEN[site] = n
        if not _should_emit(n):
            return
        tail = f" | {extra}" if extra else ""
        print(f"[SWALLOWED x{n}] {site}: {type(exc).__name__}: {exc}{tail}",
              flush=True)
    except Exception:
        # Diagnostics must never be the thing that breaks a run.
        pass


def swallowed_counts() -> dict[str, int]:
    """Snapshot of every site that has swallowed at least one exception.

    Lets a health check or a post-mortem ask "was anything failing silently?"
    instead of hoping someone read the logs at the right moment.
    """
    with _LOCK:
        return dict(_SEEN)


def reset_swallowed() -> None:
    """Test-only: clear the counters."""
    with _LOCK:
        _SEEN.clear()
