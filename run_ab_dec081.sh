#!/usr/bin/env bash
# dec-081 paired A/B: full-slot buy as LEGALITY gate vs legacy heuristic veto.
# Both arms run off ONE binary via BALATRON_SWAP_LEGALITY, same shared seed set.
set -u
CKPT="$1"
echo "=== dec-081 A/B on $CKPT ==="
echo "--- ARM A: control (swap_legality=0) ---"
python -u eval_session.py --checkpoint "$CKPT" --swap-legality 0 \
    --out logs/eval_dec081_control.jsonl --no-restart
echo "--- ARM B: treatment (swap_legality=1) ---"
python -u eval_session.py --checkpoint "$CKPT" --swap-legality 1 \
    --out logs/eval_dec081_treat.jsonl
echo "=== A/B COMPLETE — training restored ==="
