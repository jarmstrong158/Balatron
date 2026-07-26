"""Rate-limited reporting for exceptions swallowed on the DECISION path.

An audit found 104 broad `except Exception` handlers, 55 silent. Most are
correct — instrumentation must never take the trainer down. The dangerous few
sit where swallowing changes what the agent DOES (a failed `find_best_hands`
leaves current_score at 0.0, so the mask treats the hand as worthless). A single
failure there is fine; a persistent one is a silent behavioural regression.

Plain `print` was rejected: these are per-step hot paths, so unconditional
logging emits thousands of lines a minute and gets muted — which is how you end
up ignoring the signal you just added.
"""
import diagnostics


def setup_function():
    diagnostics.reset_swallowed()


def test_first_occurrence_is_reported(capsys):
    diagnostics.warn_once("mod.fn/thing", ValueError("boom"))
    out = capsys.readouterr().out
    assert "mod.fn/thing" in out and "ValueError" in out and "boom" in out


def test_backs_off_instead_of_spamming(capsys):
    """The whole point: survive a hot path without drowning the log."""
    for _ in range(200):
        diagnostics.warn_once("hot.path", RuntimeError("x"))
    emitted = capsys.readouterr().out.count("hot.path")
    # 1st, 2nd, 5th, 10th, 100th -> 5 lines out of 200 failures
    assert emitted == 5, emitted


def test_counts_every_occurrence_even_when_silent():
    """Back-off must not lose the COUNT — that is what makes a post-mortem able
    to ask 'was anything failing quietly?' instead of trusting log scrollback."""
    for _ in range(37):
        diagnostics.warn_once("counted.site", KeyError("k"))
    assert diagnostics.swallowed_counts()["counted.site"] == 37


def test_sites_are_tracked_independently(capsys):
    diagnostics.warn_once("site.a", ValueError("a"))
    diagnostics.warn_once("site.b", ValueError("b"))
    counts = diagnostics.swallowed_counts()
    assert counts == {"site.a": 1, "site.b": 1}
    out = capsys.readouterr().out
    assert "site.a" in out and "site.b" in out


def test_never_raises_even_on_a_hostile_exception():
    """A diagnostics helper that can break its caller defeats its own purpose —
    these sit inside `except` blocks on the decision path."""
    class Nasty(Exception):
        def __str__(self):
            raise RuntimeError("cannot stringify me")
    diagnostics.warn_once("hostile.site", Nasty())   # must not propagate
    assert "hostile.site" in diagnostics.swallowed_counts()


def test_clean_run_reports_nothing():
    assert diagnostics.swallowed_counts() == {}
