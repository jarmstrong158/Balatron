# Boss Robustness

Why this exists: the agent builds a **single-committed-hand** build, and the bosses
that kill the most deep runs are precisely the ones designed to punish that. Deep
(ante 5+) boss kill rates from `blind_results.jsonl` (dec-051 instrumentation):

| boss | kill % | mechanic | attacks |
|------|-------:|----------|---------|
| The Mouth   | 74% | locks the round to the first hand TYPE played | single-hand build, played before assembled |
| The Wall    | 67% | 2× target | raw power |
| The Psychic | 65% | must play 5 cards | Pair/Two-Pair builds (score on 2–4 cards) |
| The Needle  | 63% | only 1 hand for the whole blind | draw variance |
| The Eye     | 62% | no repeating a hand type | mono-hand build |
| The Water   | 61% | start with 0 discards | the dig-to-assemble strategy |

Calibration context (dec-051): deep-boss clear **plateaus at ~62% even at 4–8×
projected margin** — build power can't buy past these mechanics. Boss-robustness,
not more power, is the lever. The planner/scorer currently **detect** boss effects
(scoring) but do **not** strategize around them.

## Two layers

**Layer 1 — play-side tactical overrides** (deterministic, surgical, A/B-able now).
**Layer 2 — make the planner boss-aware** (`build_survivability`/`target_hand_type`
read the upcoming boss — already in state — and bias the build).

## Status

- [x] **The Mouth (Layer 1)** — `mouth_should_dig` (hand_eval.py) + override in
      `action_executor` PLAY branch (dec-052). Before the first play locks a type,
      if the strong committed hand isn't assembled and discards remain, dig instead
      of locking a weak type. Verify via The Mouth's kill rate in `blind_results`.
- [ ] **The Psychic (Layer 2)** — bias the build to score on a 5-card hand when
      Psychic is upcoming (structural: the agent commits to Pair too often).
- [ ] **The Water (Layer 2)** — value low-variance / no-dig-needed builds when Water
      is upcoming.
- [x] **The Needle (Layer 1)** — `needle_should_dig` (hand_eval.py) + the same
      `action_executor` PLAY override (dec-053). The Needle gives only 1 hand all
      blind; dig with all discards to maximize that single hand instead of playing
      a weak one immediately. Dig while the best current hand can't clear the target.
- [ ] **The Eye (Layer 2)** — prefer builds that score across multiple hand types
      (no-repeat punishes the mono-hand build).
- [ ] **Plumb the upcoming boss name into the planner** (prerequisite for Layer 2).

## Verification

Each fix is A/B'd through the eval harness on its **boss-specific kill rate** (the
blind name is logged per outcome), not the global win rate — so each is
independently verifiable instead of waiting on the ~0.5% win signal.
