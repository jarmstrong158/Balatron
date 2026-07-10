"""Fix (livelock breaker): the trainer must commit progress that the supervisor
can actually reload, and it must do so BEFORE the recycle window elapses.

Two regressions are pinned here:

1. Tag/glob contract. The supervisor's newest_checkpoint() only matches
   `checkpoints/balatron_phase1_update*.pt` (supervise.CHECKPOINT_GLOB). The old
   teardown save used tag="final" -> `balatron_phase1_final.pt`, which that glob
   NEVER matches — so even a graceful save was silently unresumable. The teardown
   and safety saves must be UNTAGGED so the glob picks them up.

2. Bookkeeping. Every save refreshes _last_ckpt_update / _last_ckpt_time so the
   wall-clock safety net measures time since the real last save (and is a no-op
   right after a milestone save).
"""
import glob
import os
import re
import tempfile

from training.config import TrainConfig
from training.train import Trainer, SAFETY_CHECKPOINT_S


def _make_trainer(ckpt_dir: str) -> Trainer:
    cfg = TrainConfig(
        phase=1,
        checkpoint_dir=ckpt_dir,
        record_wins=False,     # no ffmpeg recorder
        collect_demos=False,   # no demo buffer file writes
        num_envs=1,
    )
    return Trainer(cfg, checkpoint_path=None)


# The supervisor's contract, copied verbatim so this test fails if train.py's
# filename format ever drifts away from what supervise.py will load.
SUPERVISOR_GLOB = "balatron_phase1_update*.pt"


def test_untagged_save_matches_supervisor_glob():
    d = tempfile.mkdtemp()
    t = _make_trainer(d)
    t.num_updates = 3749
    t._save_checkpoint()  # untagged — the teardown / safety path
    matched = glob.glob(os.path.join(d, SUPERVISOR_GLOB))
    assert matched, "untagged save must match the supervisor's update*.pt glob"
    assert re.search(r"balatron_phase1_update0*3749\.pt$", matched[0])


def test_final_tag_is_not_loadable_by_supervisor():
    """Documents the bug: a 'final'-tagged checkpoint is invisible to the
    supervisor's newest_checkpoint() glob, so it can never break the livelock."""
    d = tempfile.mkdtemp()
    t = _make_trainer(d)
    t.num_updates = 10
    t._save_checkpoint(tag="final")
    assert not glob.glob(os.path.join(d, SUPERVISOR_GLOB)), \
        "a final-tagged save must NOT be what we rely on for resume"
    assert os.path.exists(os.path.join(d, "balatron_phase1_final.pt"))


def test_save_refreshes_safety_bookkeeping():
    d = tempfile.mkdtemp()
    t = _make_trainer(d)
    t._last_ckpt_update = 0
    t._last_ckpt_time = 0.0
    t.num_updates = 42
    t._save_checkpoint()
    assert t._last_ckpt_update == 42
    assert t._last_ckpt_time > 0.0


def test_safety_interval_beats_recycle_window():
    """The net must fire well before the supervisor's earliest rate-floor
    recycle (~5min grace + 12min window). 8min leaves ≥2 saves per window."""
    assert SAFETY_CHECKPOINT_S <= 600
