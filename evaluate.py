"""Held-out evaluation runner (dec-045; resumable dec-055) — the eval RUN-LOOP.

Plays a fixed seed bank with a FROZEN checkpoint and NO learning. Reuses the
Trainer's real play path (run_eval -> _collect_rollout), so eval behaviour matches
training.

RESUMABLE (dec-055): each finished run is written to a dedicated results file
(default logs/eval_<checkpoint>.jsonl). On startup, seeds already present there are
SKIPPED, so a crash / session teardown mid-eval costs only a restart — just run the
same command again and it continues where it left off. The results file is isolated
from training's game_history (no seed-filtering needed for analysis).

    python evaluate.py --checkpoint checkpoints/balatron_phase1_updateNNNNNN.pt \
                       --seeds eval_seeds.txt [--num-envs 3] [--limit 0] [--out PATH]
    # ...if it dies, just run the exact same line again — it resumes.
    python eval_report.py logs/eval_balatron_phase1_updateNNNNNN.jsonl   # analyze

OPERATIONAL NOTE: the BalatroBot game server(s) on ports 12346..12346+num_envs-1
must be up and NOT in use by a training trainer (pause training first). An eval over
300 seeds is a multi-hour run.
"""
import argparse
import asyncio
import json
import os


def _done_seeds(path):
    """Seeds already completed in the results file (for resume)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line).get("seed")
                    if s:
                        done.add(s)
                except json.JSONDecodeError:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="eval_seeds.txt")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = use all seeds")
    ap.add_argument("--out", default=None,
                    help="results file (default logs/eval_<checkpoint>.jsonl)")
    # Confidence-gated planner deferral (dec-061) — INFERENCE/EVAL ONLY. Off by
    # default so this reproduces prior eval behavior exactly. Turn on to route
    # LOW-confidence decisions to the existing planner; compare the two runs with
    # eval_report.py (advance rate) and the <out>.gate.json (planner-call count).
    ap.add_argument("--gate", action="store_true",
                    help="enable confidence-gated planner deferral (eval only)")
    ap.add_argument("--gate-signal", default="entropy", choices=["entropy", "top1"],
                    help="confidence signal: 'entropy' or 'top1' (default entropy)")
    ap.add_argument("--gate-threshold", type=float, default=0.0,
                    help="defer when confidence < threshold; 0.0 gates nothing "
                         "(reproduces gate-off), 1.0 gates every real choice")
    args = ap.parse_args()

    from training.config import TrainConfig
    from training.train import Trainer

    with open(args.seeds) as f:
        seeds = [s.strip() for s in f if s.strip()]
    if args.limit:
        seeds = seeds[:args.limit]

    out_path = args.out or os.path.join(
        "logs", "eval_" + os.path.splitext(os.path.basename(args.checkpoint))[0] + ".jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # RESUME: drop seeds already recorded in the results file.
    done = _done_seeds(out_path)
    remaining = [s for s in seeds if s not in done]
    print(f"[EVAL] checkpoint={args.checkpoint} seeds={len(seeds)} "
          f"done={len(done)} remaining={len(remaining)} -> {out_path}")
    if not remaining:
        print(f"[EVAL] all seeds already evaluated. Analyze: "
              f"python eval_report.py {out_path}")
        return

    n = max(1, args.num_envs)
    cfg = TrainConfig()
    cfg.num_envs = n
    cfg.curriculum_enabled = False
    cfg.record_wins = False
    cfg.device = "cpu"
    # dec-061 confidence-gated planner deferral (inference/eval only)
    cfg.gate_enabled = args.gate
    cfg.gate_signal = args.gate_signal
    cfg.gate_threshold = args.gate_threshold
    if args.gate:
        print(f"[EVAL] confidence gate ON: signal={args.gate_signal} "
              f"threshold={args.gate_threshold}", flush=True)

    trainer = Trainer(cfg, checkpoint_path=args.checkpoint)
    trainer.eval_out_path = out_path

    # Distribute the REMAINING seeds round-robin across envs (ports 12346..).
    for i, env in enumerate(trainer.sessions):
        env.forced_seeds = list(remaining[i::n])
        env.eval_finished = False

    asyncio.run(trainer.run_eval())
    print(f"[EVAL] analyze: python eval_report.py {out_path}")


if __name__ == "__main__":
    main()
