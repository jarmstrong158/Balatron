"""Run a held-out eval WITHOUT the supervisor stealing the game servers back.

WHY THIS EXISTS (dec-074) — the reason ZERO evals have ever completed:
`evaluate.py` needs the BalatroBot servers to itself (its docstring says "pause
training first"), but the supervisor's existence layer (con-010) relaunches the
trainer within ~30s of it dying. So an eval's ports get taken back mid-run and it
desyncs — exactly how logs/eval_baseline.out died on 06-30 (INVALID_STATE on
next_round/play). Nobody automated the interplay, so the eval was always "later",
and every A/B since dec-045 has instead been an eyeball on a confounded live
trainer with 3-8 concurrent uncontrolled changes.

The dance, done safely:
  1. touch SUPERVISOR_STOP   -> supervisor exits cleanly, LEAVING THE GAMES UP
  2. kill the trainer        -> nothing relaunches it; ports 12346.. are free
  3. run evaluate.py         -> resumable (dec-055); re-run to continue
  4. ALWAYS: remove the stop file + relaunch the supervisor (training resumes)

Step 4 is in a finally block, so training comes back even on Ctrl-C, crash, or a
failed eval. Verify with: tail logs/supervisor_stdout.log

    python eval_session.py --checkpoint checkpoints/balatron_phase1_updateNNNN.pt
    python eval_session.py --checkpoint ... --limit 120      # shorter first pass
    python eval_report.py logs/eval_<checkpoint>.jsonl       # analyze

NOTE: this PAUSES TRAINING for the duration (a 300-seed eval is multi-hour).
That is the intended trade: an unmeasured trainer produces unvalidated changes.
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
STOP_FILE = os.path.join(REPO, "SUPERVISOR_STOP")
PY = sys.executable

try:
    import psutil
except ImportError:
    psutil = None


def _procs(pattern: str):
    """python.exe processes whose cmdline contains `pattern` (token-ish match).
    con-012(4): match real args, not a substring of some `python -c` blob."""
    out = []
    if psutil is None:
        return out
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = p.info.get("cmdline") or []
            if not cl:
                continue
            if any(pattern in str(tok) for tok in cl) and "python" in (p.info.get("name") or "").lower():
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def stop_stack(timeout: float = 120.0) -> None:
    """Stop the supervisor (cleanly, games survive) then the trainer."""
    open(STOP_FILE, "w").write("eval in progress\n")
    print(f"[EVAL-SESSION] wrote {STOP_FILE}; waiting for supervisor to exit "
          f"(it polls every ~30s)...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _procs("supervise.py"):
            print("[EVAL-SESSION] supervisor exited (games left up).", flush=True)
            break
        time.sleep(3)
    else:
        print("[EVAL-SESSION] WARNING: supervisor still alive; killing it.", flush=True)
        for p in _procs("supervise.py"):
            try:
                p.kill()
            except Exception:
                pass
    for p in _procs("training.train"):
        print(f"[EVAL-SESSION] killing trainer pid {p.pid} (frees the game ports)", flush=True)
        try:
            p.kill()
        except Exception:
            pass
    time.sleep(3)


def restart_stack() -> None:
    """Remove the stop file and bring the supervisor back (it rebuilds trainer)."""
    try:
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
            print("[EVAL-SESSION] removed SUPERVISOR_STOP", flush=True)
    except OSError as e:
        print(f"[EVAL-SESSION] WARNING: could not remove {STOP_FILE}: {e}", flush=True)
    if _procs("supervise.py"):
        print("[EVAL-SESSION] supervisor already running.", flush=True)
        return
    log_out = os.path.join(REPO, "logs", "supervisor_stdout.log")
    log_err = os.path.join(REPO, "logs", "supervisor_stderr.log")
    with open(log_out, "a") as so, open(log_err, "a") as se:
        subprocess.Popen([PY, "-u", "supervise.py"], cwd=REPO, stdout=so, stderr=se,
                         creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    print("[EVAL-SESSION] supervisor relaunched — training resumes.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", default="eval_seeds.txt")
    ap.add_argument("--num-envs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all seeds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-restart", action="store_true",
                    help="leave training stopped afterwards (for back-to-back evals)")
    ap.add_argument("--rf", default=None,
                    help="override REALIZATION_FACTOR for this eval (dec-078 A/B)")
    args = ap.parse_args()

    if psutil is None:
        print("FATAL: psutil unavailable — cannot manage the stack.", file=sys.stderr)
        return 2

    cmd = [PY, "-u", "evaluate.py", "--checkpoint", args.checkpoint,
           "--seeds", args.seeds, "--num-envs", str(args.num_envs)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.out:
        cmd += ["--out", args.out]

    # THE reason evals have always died (gotcha 6): with stdout redirected to a
    # file, Windows Python encodes as cp1252 and the first non-ASCII log line
    # (e.g. "[SHOP] REDIRECT pack buy → joker buy") raises UnicodeEncodeError and
    # kills the process. supervise.py already fixes this for the TRAINER
    # (supervise.py:567 `env = dict(os.environ, PYTHONUTF8="1")`) — evaluate.py
    # never got the same treatment, so every eval crashed on its first redirect.
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if args.rf is not None:
        env["BALATRON_RF"] = str(args.rf)   # dec-078: pin RF for this arm

    rc = 1
    try:
        stop_stack()
        print(f"[EVAL-SESSION] running: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        print(f"[EVAL-SESSION] evaluate.py exited rc={rc}", flush=True)
    except KeyboardInterrupt:
        print("[EVAL-SESSION] interrupted.", flush=True)
    finally:
        if args.no_restart:
            print("[EVAL-SESSION] --no-restart: training left stopped. "
                  "Remove SUPERVISOR_STOP and run supervise.py to resume.", flush=True)
        else:
            restart_stack()
    return rc


if __name__ == "__main__":
    sys.exit(main())
