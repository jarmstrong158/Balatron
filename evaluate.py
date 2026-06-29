"""Held-out evaluation runner (dec-045) — the eval RUN-LOOP.

Plays a fixed seed bank with a FROZEN checkpoint and NO learning, tagging each
run's outcome into logs/game_history.jsonl by seed. Reuses the Trainer's real
play path (run_eval -> _collect_rollout), so eval behaviour matches training.

Analyze the result with the measurement instrument:
    python eval_report.py logs/game_history.jsonl
Or A/B two checkpoints run on the SAME seeds (paired, removes seed luck):
    python eval_report.py eval_A.jsonl eval_B.jsonl

Usage:
    python evaluate.py --checkpoint checkpoints/balatron_phase1_updateNNNNNN.pt \
                       --seeds eval_seeds.txt [--num-envs 3] [--limit 300]

OPERATIONAL NOTE: the BalatroBot game server(s) on ports 12346..12346+num_envs-1
must be UP and NOT in use by a training trainer. Pause training first (stop the
trainer; leave the Balatro game processes running), then run this. An eval over
300 seeds is a multi-hour run — consider running it in the background.
"""
import argparse
import asyncio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="eval_seeds.txt")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = use all seeds")
    args = ap.parse_args()

    from training.config import TrainConfig
    from training.train import Trainer

    with open(args.seeds) as f:
        seeds = [s.strip() for s in f if s.strip()]
    if args.limit:
        seeds = seeds[:args.limit]
    n = max(1, args.num_envs)

    # Eval config: no curriculum (fresh starts), no win recorder, CPU.
    cfg = TrainConfig()
    cfg.num_envs = n
    cfg.curriculum_enabled = False
    cfg.record_wins = False
    cfg.device = "cpu"

    trainer = Trainer(cfg, checkpoint_path=args.checkpoint)

    # Distribute the seed bank round-robin across envs (ports 12346..).
    for i, env in enumerate(trainer.sessions):
        env.forced_seeds = list(seeds[i::n])
        env.eval_finished = False

    print(f"[EVAL] checkpoint={args.checkpoint} seeds={len(seeds)} envs={n}")
    asyncio.run(trainer.run_eval())


if __name__ == "__main__":
    main()
