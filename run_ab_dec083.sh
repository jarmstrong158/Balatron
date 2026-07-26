#!/usr/bin/env bash
# dec-083 paired A/B: Monte-Carlo P(clear) leaf vs the analytical point estimate.
# Both arms off ONE binary via BALATRON_ROLLOUT, same shared 300-seed set.
set -u
CKPT="$1"
echo "=== dec-083 A/B on $CKPT ==="
echo "--- ARM A: control (rollout=0, analytical leaf) ---"
python -u eval_session.py --checkpoint "$CKPT" --rollout 0 \
    --out logs/eval_dec083_control.jsonl --no-restart
echo "--- ARM B: treatment (rollout=1, Monte-Carlo P(clear)) ---"
python -u eval_session.py --checkpoint "$CKPT" --rollout 1 \
    --out logs/eval_dec083_treat.jsonl
echo "=== A/B COMPLETE - training restored ==="
