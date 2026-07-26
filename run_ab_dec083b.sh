#!/usr/bin/env bash
# dec-083 BATCH 2: 300 FRESH seeds, same checkpoint + same binary as batch 1,
# so the two batches POOL into one 600-seed paired comparison.
# Batch 1 was directionally positive but underpowered (mean ante +0.107,
# CI [-0.070,+0.283]); 600 paired seeds should resolve it either way.
set -u
CKPT="$1"
echo "=== dec-083 A/B batch 2 on $CKPT (fresh seeds) ==="
echo "--- ARM A: control (rollout=0) ---"
python -u eval_session.py --checkpoint "$CKPT" --rollout 0 \
    --seeds eval_seeds_batch2.txt \
    --out logs/eval_dec083b_control.jsonl --no-restart
echo "--- ARM B: treatment (rollout=1) ---"
python -u eval_session.py --checkpoint "$CKPT" --rollout 1 \
    --seeds eval_seeds_batch2.txt \
    --out logs/eval_dec083b_treat.jsonl
echo "=== BATCH 2 COMPLETE - training restored ==="
