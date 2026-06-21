"""Balatron — Self-Imitation Demo Buffer (Phase 1: capture + persistence).

Persists the (state, action, mask, head) transitions of WINNING / high-ante
runs so the policy can later replay them through the existing behavior-cloning
loss (self-imitation learning). Vanilla PPO mostly *forgets* a rare win — one
win per ~200 runs barely moves the gradient before it's overwritten — so the
agent re-discovers the same builds over and over. Replaying its own proven
successes makes them stick.

Phase 1 (this) is CAPTURE-ONLY and behavior-neutral: it records trajectories to
disk but does NOT touch the policy or the loss. Phase 2 wires sample() into the
PPO update behind a small sil_coef.

Why it MUST persist to disk: the supervisor recycles the trainer every ~20-30
min, and wins are ~0.5% of runs — without persistence every hard-won trajectory
is lost on the next restart. (That's exactly why we have 74 lifetime wins and
zero usable trajectories today: wins were only ever saved as video.)

Storage is a flat transition-level ring buffer (FIFO eviction). A trajectory
may be split across the ring wrap — fine, because self-imitation samples
individual transitions, not whole episodes.
"""
import os

import numpy as np


class DemoBuffer:
    """Capacity-capped, disk-persisted store of demonstration transitions."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 mask_dim: int, path: str = "demos/win_demos.npz"):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mask_dim = mask_dim
        self.path = path

        self.pos = 0
        self.full = False
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, mask_dim), dtype=np.float32)
        self.head_indices = np.zeros(capacity, dtype=np.int64)

        # Lifetime diagnostics (persisted) — how much we've ever captured.
        self.trajectories_added = 0
        self.transitions_added = 0

        self._load()

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add_trajectory(self, states, actions, masks, head_indices) -> int:
        """Append one episode's real transitions. Ring-buffer: oldest evicted
        once full. Returns the number of transitions added."""
        n = len(states)
        if n == 0:
            return 0
        for i in range(n):
            p = self.pos
            self.states[p] = states[i]
            self.actions[p] = actions[i]
            self.masks[p] = masks[i]
            self.head_indices[p] = head_indices[i]
            self.pos += 1
            if self.pos >= self.capacity:
                self.pos = 0
                self.full = True
        self.trajectories_added += 1
        self.transitions_added += n
        return n

    def sample(self, batch_size: int):
        """Random transition batch for the Phase-2 self-imitation forward pass.
        Returns a dict of numpy arrays, or None if the buffer is empty."""
        n = len(self)
        if n == 0:
            return None
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "masks": self.masks[idx],
            "head_indices": self.head_indices[idx],
        }

    def save(self):
        """Persist the filled portion to disk ATOMICALLY (compressed).

        The supervisor hard-kills the trainer into this write path; a torn
        savez leaves a corrupt npz and _load() would then silently start
        empty, losing the WHOLE irreplaceable win corpus. Write to a temp
        file then os.replace (atomic on the same volume).
        """
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            n = len(self)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    states=self.states[:n],
                    actions=self.actions[:n],
                    masks=self.masks[:n],
                    head_indices=self.head_indices[:n],
                    meta=np.array([self.trajectories_added,
                                   self.transitions_added], dtype=np.int64),
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception as e:  # never let demo persistence break training
            print(f"[DEMO] save failed: {e}", flush=True)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            d = np.load(self.path)
            ds = d["states"]
            n = min(len(ds), self.capacity)
            old_dim = ds.shape[1] if ds.ndim == 2 else self.state_dim
            # State-vector growth migration (mirrors ppo.load_checkpoint's
            # zero-pad). The corpus is IRREPLACEABLE; a dim change must NOT
            # silently throw and start empty (which is what plain assignment
            # would do the next time STATE_VECTOR_SIZE grows).
            if old_dim == self.state_dim:
                self.states[:n] = ds[:n]
            elif old_dim < self.state_dim:
                self.states[:n, :old_dim] = ds[:n]   # zero-pad appended cols
                print(f"[DEMO] migrated demos {old_dim}->{self.state_dim} "
                      f"dims (zero-padded)", flush=True)
            else:
                self.states[:n] = ds[:n, :self.state_dim]  # state shrank: trim
                print(f"[DEMO] migrated demos {old_dim}->{self.state_dim} "
                      f"dims (truncated)", flush=True)
            self.actions[:n] = d["actions"][:n]
            self.masks[:n] = d["masks"][:n]
            self.head_indices[:n] = d["head_indices"][:n]
            self.pos = n % self.capacity
            self.full = n >= self.capacity
            if "meta" in d:
                self.trajectories_added = int(d["meta"][0])
                self.transitions_added = int(d["meta"][1])
            print(f"[DEMO] loaded {n} demo transitions from {self.path} "
                  f"({self.trajectories_added} trajectories lifetime)",
                  flush=True)
        except Exception as e:
            # Do NOT overwrite a possibly-good file with an empty buffer on a
            # transient read error — leave self empty in memory but keep the
            # on-disk corpus intact (save() only runs on a future capture).
            print(f"[DEMO] load failed ({e}) — starting empty IN MEMORY "
                  f"(on-disk file left untouched)", flush=True)
