"""dec-058 follow-up: the LR lock must survive a checkpoint save->load recycle.

dec-058 locked the LR at 1e-4, but the audit found it resetting to the dec-039
damaged value (2.7e-4) on trainer recycle at updates 625 and 2465. Root cause:
`PPOTrainer.load_checkpoint` restores the optimizer `state_dict`, which carries
the LR that was live when the checkpoint was saved. Nothing at load time
neutralized it — only a separate, `anneal_lr`-gated `set_learning_rate` inside
`run()` did, a band-aid far from where the stale LR re-enters.

These tests pin the recycle contract at the load boundary itself.
"""
import os
import tempfile

from agent.ppo import PPOTrainer, PPOConfig
from agent.network import BalatronNetwork

DAMAGED_LR = 2.7e-4   # the dec-039 value the audit kept finding after recycles
LOCK_LR = 1.0e-4      # the dec-058 lock


def _save_damaged_checkpoint() -> str:
    """A checkpoint whose optimizer carries the dec-039 damaged LR."""
    t = PPOTrainer(BalatronNetwork(), PPOConfig(learning_rate=LOCK_LR))
    t.set_learning_rate(DAMAGED_LR)
    path = os.path.join(tempfile.mkdtemp(), "damaged.pt")
    t.save_checkpoint(path)
    return path


def test_stale_lr_is_carried_without_override():
    """Documents the root cause: a plain load resurrects the saved LR."""
    path = _save_damaged_checkpoint()
    # Fresh trainer built at the supervisor's default 3e-4 (no --lr passed).
    t = PPOTrainer(BalatronNetwork(), PPOConfig(learning_rate=3e-4))
    t.load_checkpoint(path)  # no override -> old behavior
    assert abs(t.get_learning_rate() - DAMAGED_LR) < 1e-12


def test_lr_override_locks_on_load():
    """The fix: passing lr_override pins the LR at load time."""
    path = _save_damaged_checkpoint()
    t = PPOTrainer(BalatronNetwork(), PPOConfig(learning_rate=3e-4))
    t.load_checkpoint(path, lr_override=LOCK_LR)
    assert abs(t.get_learning_rate() - LOCK_LR) < 1e-12


def test_locked_lr_survives_a_second_recycle():
    """A checkpoint saved by a locked trainer stays locked through re-save."""
    path = _save_damaged_checkpoint()
    # First recycle applies the lock...
    t1 = PPOTrainer(BalatronNetwork(), PPOConfig(learning_rate=3e-4))
    t1.load_checkpoint(path, lr_override=LOCK_LR)
    path2 = os.path.join(tempfile.mkdtemp(), "recycled.pt")
    t1.save_checkpoint(path2)
    # ...and the next recycle inherits the locked value, not the damaged one.
    t2 = PPOTrainer(BalatronNetwork(), PPOConfig(learning_rate=3e-4))
    t2.load_checkpoint(path2, lr_override=LOCK_LR)
    assert abs(t2.get_learning_rate() - LOCK_LR) < 1e-12
