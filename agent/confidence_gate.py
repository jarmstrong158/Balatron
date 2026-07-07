"""Confidence-gated planner deferral (dec-061) — INFERENCE / EVAL ONLY.

This routes EXISTING planner compute by the policy's per-decision confidence. It
adds no new planner and changes nothing about training.

Today the policy owns the action TYPE and the dec-034 build planner owns
which-joker (in the shop) — a fixed hierarchy. This gate makes that hierarchy
dynamic AT DECISION TIME on the inference/eval path only: when the policy is
CONFIDENT about the type it samples, use the fast policy path (today's behavior);
when it is UNCERTAIN, route THAT single decision to the existing planner instead.

Confidence is read from the action-type distribution the policy already computes
(no extra forward pass):
  - "top1":    confidence = top-1 action probability (in [0, 1]).
  - "entropy": confidence = 1 - H / log(n_legal), the normalized certainty of the
               masked type distribution (1 = one action dominates, 0 = uniform).
The value-head output is also available per decision but is not used to gate
(kept as an available signal; entropy/top-1 are the routing signal).

Deferral fires when confidence < threshold. The threshold is a config parameter
whose EXTREME reproduces today's behavior exactly:
  - threshold = 0.0 -> confidence < 0 is never true -> gates NOTHING (today).
  - threshold = 1.0 -> gates every decision that has a real (multi-legal) choice.
The default is 0.0, and the whole gate is additionally opt-in (`enabled`, default
False) — so with the feature off the play path is byte-for-byte unchanged, and
even switched on at the default threshold it defers nothing. The feature is thus
a provable SUPERSET of current behavior.

CRITICAL: `gate_is_active` requires `eval_mode=True`. Training rollout collection
passes eval_mode=False, so this is the single choke point that keeps deferral out
of the on-policy distribution PPO learns from (overriding actions during
collection would reintroduce off-policy contamination — deliberately avoided).
"""
import json
import math

# Confidence signal choices (config `gate_signal`).
SIGNAL_ENTROPY = "entropy"
SIGNAL_TOP1 = "top1"

_N_BINS = 10  # confidence histogram resolution for telemetry


def gate_is_active(gate, eval_mode: bool) -> bool:
    """The gate is LIVE only on the inference/eval path (eval_mode=True) and only
    when opted in. Training rollout collection passes eval_mode=False, so this
    returns False there no matter how the gate is configured — the invariant that
    keeps planner deferral out of on-policy PPO data."""
    return bool(eval_mode and gate is not None and gate.enabled)


class ConfidenceGate:
    """Per-decision confidence scorer + deferral decision + eval telemetry."""

    def __init__(self, enabled: bool = False, signal: str = SIGNAL_ENTROPY,
                 threshold: float = 0.0):
        if signal not in (SIGNAL_ENTROPY, SIGNAL_TOP1):
            raise ValueError(
                f"gate_signal must be '{SIGNAL_ENTROPY}' or '{SIGNAL_TOP1}', "
                f"got {signal!r}")
        self.enabled = bool(enabled)
        self.signal = signal
        self.threshold = float(threshold)
        self.reset_stats()

    # ── confidence ──────────────────────────────────────────────────────────
    def confidence(self, *, entropy: float, top1: float, n_legal: int) -> float:
        """Confidence in [0, 1] from the policy's action-type distribution.
        HIGH = certain. `entropy` is the type distribution's entropy in nats,
        `top1` its top-1 probability, `n_legal` the number of legal action types.

        A forced decision (n_legal <= 1) is fully certain (1.0) under either
        signal — there is nothing to route."""
        if n_legal <= 1:
            return 1.0
        if self.signal == SIGNAL_TOP1:
            return max(0.0, min(1.0, top1))
        # entropy: normalize by the max entropy of a uniform over the legal set.
        h_max = math.log(n_legal)
        if h_max <= 0.0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - entropy / h_max))

    def should_defer(self, confidence: float) -> bool:
        """Defer this decision to the planner iff enabled and below threshold.
        With threshold=0.0 this is always False (gates nothing)."""
        return self.enabled and confidence < self.threshold

    # ── telemetry ───────────────────────────────────────────────────────────
    def reset_stats(self):
        self.n_decisions = 0
        self.n_deferred = 0
        self._conf_sum = 0.0
        self._conf_min = 1.0
        self._conf_max = 0.0
        self._hist = [0] * _N_BINS

    def record(self, confidence: float, deferred: bool):
        """Accumulate one decision's confidence + whether it was routed to the
        planner, for per-eval-run deferral-rate and confidence-distribution
        reporting."""
        self.n_decisions += 1
        if deferred:
            self.n_deferred += 1
        c = max(0.0, min(1.0, confidence))
        self._conf_sum += c
        self._conf_min = min(self._conf_min, c)
        self._conf_max = max(self._conf_max, c)
        b = min(_N_BINS - 1, int(c * _N_BINS))
        self._hist[b] += 1

    def deferral_rate(self) -> float:
        return self.n_deferred / self.n_decisions if self.n_decisions else 0.0

    def stats(self) -> dict:
        n = self.n_decisions
        return {
            "enabled": self.enabled,
            "signal": self.signal,
            "threshold": self.threshold,
            "decisions": n,
            "deferred": self.n_deferred,
            "deferral_rate": self.deferral_rate(),
            "confidence_mean": (self._conf_sum / n) if n else 0.0,
            "confidence_min": self._conf_min if n else 0.0,
            "confidence_max": self._conf_max if n else 0.0,
            # histogram bins over [0,1], bin i covers [i/10, (i+1)/10)
            "confidence_hist": list(self._hist),
        }

    def summary_line(self) -> str:
        s = self.stats()
        return (f"[GATE] signal={s['signal']} threshold={s['threshold']:.3f} "
                f"deferral_rate={100*s['deferral_rate']:.1f}% "
                f"({s['deferred']}/{s['decisions']}) "
                f"conf_mean={s['confidence_mean']:.3f} "
                f"[{s['confidence_min']:.3f},{s['confidence_max']:.3f}]")

    def dump(self, eval_out_path: str | None) -> str | None:
        """Print the summary and write the full stats JSON to a sidecar file
        (`<eval_out>.gate.json`) so an ON/OFF comparison can read the deferral
        (planner-call) count and confidence distribution. Returns the path
        written, or None if no eval_out_path."""
        print(self.summary_line(), flush=True)
        if not eval_out_path:
            return None
        path = eval_out_path + ".gate.json"
        try:
            with open(path, "w") as f:
                json.dump(self.stats(), f, indent=2)
            print(f"[GATE] telemetry -> {path}", flush=True)
        except OSError:
            return None
        return path
