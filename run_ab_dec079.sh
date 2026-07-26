#!/usr/bin/env bash
# dec-079 paired A/B: SAVE-gate headroom (AHEAD_BUFFER_EARLY_BONUS).
#   arm A (control)   bonus=0.0 -> provably byte-identical dec-068
#   arm B (treatment) bonus=1.0 -> ante-scaled headroom, antes 4+ unchanged
# Same checkpoint, same 300 seeds -> paired McNemar at the reach-ante-6 gate.
# Arm A runs --no-restart so the stack stays down between arms (no port theft);
# arm B restores the supervisor in its finally block.
set -u
CKPT="${1:-checkpoints/balatron_phase1_update006268.pt}"
cd "$(dirname "$0")" || exit 1

echo "=== dec-079 A/B on $CKPT ==="
echo "--- ARM A: control (bonus=0.0) ---"
python -u eval_session.py --checkpoint "$CKPT" \
    --ahead-early-bonus 0.0 --no-restart \
    --out logs/eval_dec079_control.jsonl
echo "--- ARM B: treatment (bonus=1.0) ---"
python -u eval_session.py --checkpoint "$CKPT" \
    --ahead-early-bonus 1.0 \
    --out logs/eval_dec079_treat.jsonl
echo "=== A/B COMPLETE — training restored ==="
