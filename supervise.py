"""Supervisor: keeps the full Balatron training stack alive.

Run this ONE detached process instead of launching the server and trainer
by hand. Every cycle it ensures:

  1. The BalatroBot server + game are up (port 12346 listening) —
     otherwise it kills stray Balatro processes and relaunches the server.
  2. The trainer is running — otherwise it relaunches from the NEWEST
     checkpoint with PYTHONUTF8=1, logging to logs/trainer_<ts>.log.

Why this exists: the trainer's internal watchdog only revives the GAME
when it hangs; nothing revived the TRAINER, and twice the whole stack
(server + trainer + monitoring shells) died simultaneously with no crash
trace, sitting idle until a human noticed. The supervisor is detached
from any terminal/session, so it survives whatever kills its children.

Launch (detached, hidden):
  powershell -Command "Start-Process -WindowStyle Hidden python -ArgumentList '-u','supervise.py' -WorkingDirectory 'C:\\Users\\jarms\\repos\\balatron'"

Stop:
  Create a file named SUPERVISOR_STOP in the repo root (checked each
  cycle), or kill the python process running supervise.py.
"""

import datetime
import glob
import os
import re
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO, "logs")
LOG_PATH = os.path.join(LOG_DIR, "supervisor.log")
STOP_FILE = os.path.join(REPO, "SUPERVISOR_STOP")
UVX = r"C:\Users\jarms\.local\bin\uvx.exe"
# Parallel game instances: one server+game per port. Must match the
# trainer's --num-envs (ports 12346..12346+N-1).
NUM_ENVS = 3
PORTS = [12346 + i for i in range(NUM_ENVS)]
PORT = PORTS[0]  # legacy references

CHECK_INTERVAL_S = 30
SERVER_BOOT_TIMEOUT_S = 120
TRAINER_GRACE_S = 60  # after starting the trainer, wait before re-checking
HEARTBEAT_PATH = os.path.join(LOG_DIR, "heartbeat")
HEARTBEAT_STALE_S = 300  # no env step in 5 min = trainer FROZEN

# Throughput stall detection, two complementary signals:
#
# 1. Step rate (hard crawl): steps/min over a rolling window. The floor
#    is deliberately LOW — legitimate deep runs (ante 5+ boss fights,
#    long scoring animations) can run ~13-20 steps/min while perfectly
#    healthy; the first deployment's 25/min floor killed exactly such a
#    run 24 min after going live. 10/min for 40 min = genuinely dead.
# 2. Checkpoint age (chronic churn): wedge/restart cycles can hold
#    15-20 steps/min — above any safe rate floor — while updates crawl
#    (overnight 06-11: ~190 min/checkpoint vs ~70-90 normal for 9
#    hours). If the trainer has been up long enough and no checkpoint
#    has landed, the stack is churning, not training.
STALL_WINDOW_S = 2400            # 40 min of samples before judging rate
STALL_MIN_STEPS_PER_MIN = 10.0   # below this = hard crawl, not deep play
CHECKPOINT_STALL_S = 9000        # 150 min without a checkpoint = churn
CHECKPOINT_GLOB = os.path.join(REPO, "checkpoints",
                               "balatron_phase1_update*.pt")

# Degradation recycle: the trainer's per-update FPS (printed in its log)
# decays ~1/n over the process's multi-hour lifetime — a long-lived-process
# accumulation that is NOT memory, orphan launchers, or external CPU (all
# ruled out 06-12); root cause still open. Left alone it crawls from its
# fresh peak to ~1/100th over ~10h, dragging update cadence from ~5 min to
# ~70 min.
#
# The threshold is an ABSOLUTE FPS floor, NOT relative to the run's peak.
# Peak-relative was tried (0.40 * peak) and was WRONG: the fresh FPS is a
# first-update BURST (~1800-2060) that settles to a healthy ~250 within
# ~30 min, so "13% of peak" flagged a perfectly fast trainer and recycled
# the whole stack every 30 min — a churn loop that looked like a crash. The
# burst peak is not a sustainable baseline, so absolute FPS is what actually
# signals "slow": the genuine degradation that prompted this runs the rate
# down to ~15-40 over hours (update cadence ~40-70 min). Require the floor to
# be breached for SUSTAIN consecutive updates so a single deep-run dip (one
# long ante-8 boss update) doesn't trip it — real degradation is monotonic
# and stays under. The 1817->250 settle never touches 40, so no early trip.
FPS_RECYCLE_FLOOR = 40           # recycle when FPS is genuinely low (<=)
FPS_RECYCLE_SUSTAIN = 3          # ... for this many consecutive updates
RECYCLE_MIN_AGE_S = 1800         # never recycle a trainer younger than 30min
# Hard age backstop. The FPS-floor recycle proved too slow/brittle overnight
# 06-13: the trainer's fresh peak was unusually low (~334, not ~2000), it
# decayed through the floor with a value landing exactly on 40 (breaking the
# "consecutive < 40" streak), and once FPS is low each update takes ~an hour,
# so the streak completes hours late. The trainer crawled 7.5h at FPS ~27.
# This backstop recycles ANY trainer older than the limit regardless of FPS,
# so it can never crawl all night. Resume-from-checkpoint is cheap.
MAX_TRAINER_AGE_S = 10800        # 3h hard cap on a trainer's lifetime


def log(msg: str):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_listening(port: int = PORT) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def kill_port_owner(port: int):
    """Kill only the process tree owning a specific port (multi-instance:
    a blanket taskkill /IM would murder every game at once)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
         f"-ErrorAction SilentlyContinue).OwningProcess | Select-Object -First 1"],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    if out:
        subprocess.run(["taskkill", "/F", "/PID", out, "/T"],
                       capture_output=True, timeout=15)


def trainer_pid() -> str:
    """PID of a python process running training.train, or empty string."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object {$_.CommandLine -match 'training[.]train'} | "
         "Select-Object -First 1 -ExpandProperty ProcessId"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    return out


def newest_checkpoint() -> str:
    pattern = os.path.join(REPO, "checkpoints", "balatron_phase1_update*.pt")
    cps = glob.glob(pattern)
    return max(cps, key=os.path.getmtime) if cps else ""


def kill_strays():
    for image in ("Balatro.exe", "balatrobot.exe"):
        subprocess.run(["taskkill", "/F", "/IM", image],
                       capture_output=True, timeout=15)


def reap_orphan_launchers() -> int:
    """Kill 'balatrobot serve' launchers that no longer have a live Balatro.exe
    child.

    The process tree is uvx.exe -> python.exe (balatrobot serve) -> Balatro.exe.
    On a restart we kill the port owner (the game) plus the tracked uvx handle,
    but uvx has already exited after bootstrapping, so the middle python
    launcher gets reparented and survives. ~1 orphan leaks per restart; over a
    night that's 100+ idle-but-resident processes (~40MB each) that thrash the
    scheduler and page the trainer out, collapsing FPS on a clean 1/n curve.

    A launcher is an orphan iff NO live Balatro.exe is among its descendants.
    The live tree is balatrobot.exe -> python(serve) -> python(serve) ->
    Balatro.exe — TWO nested serve launchers — so a direct-parent test would
    wrongly flag the outer launcher (a live grandparent) and /T-kill the game
    under it. Instead we protect the whole ancestor chain of every live game
    and reap only launchers outside it. Port-agnostic, safe with N parallel
    games. The >180s age guard avoids reaping a launcher that booted seconds
    ago and hasn't spawned its game yet (boot is well under that)."""
    ps = (
        "$now=Get-Date; $all=Get-CimInstance Win32_Process; "
        "$byId=@{}; foreach($p in $all){ $byId[[int]$p.ProcessId]=$p }; "
        "$prot=New-Object System.Collections.Generic.HashSet[int]; "
        "foreach($b in ($all|Where-Object {$_.Name -eq 'Balatro.exe'})){ "
        "$cur=[int]$b.ParentProcessId; $g=0; "
        "while($cur -and $byId.ContainsKey($cur) -and $g -lt 20){ "
        "[void]$prot.Add($cur); $cur=[int]$byId[$cur].ParentProcessId; $g++ } }; "
        "$o=$all|Where-Object { $_.Name -eq 'python.exe' -and "
        "$_.CommandLine -match 'balatrobot' -and $_.CommandLine -match 'serve' "
        "-and -not $prot.Contains([int]$_.ProcessId) "
        "-and ($now-$_.CreationDate).TotalSeconds -gt 180 }; "
        "$o | ForEach-Object { taskkill /F /T /PID $_.ProcessId 2>$null "
        "| Out-Null }; ($o | Measure-Object).Count"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return 0


def heartbeat_age() -> float:
    """Seconds since the trainer last made a real environment step.
    Returns 0 if the heartbeat file doesn't exist yet (fresh trainer
    gets the grace period instead)."""
    try:
        return time.time() - os.path.getmtime(HEARTBEAT_PATH)
    except OSError:
        return 0.0


def heartbeat_step():
    """Trainer's global step counter from the heartbeat file, or None if
    unavailable (old single-field format, missing file, partial write)."""
    try:
        with open(HEARTBEAT_PATH) as f:
            parts = f.read().split()
        return int(float(parts[1])) if len(parts) >= 2 else None
    except (OSError, ValueError, IndexError):
        return None


def kill_trainer(pid: str):
    subprocess.run(["taskkill", "/F", "/PID", pid],
                   capture_output=True, timeout=15)


def newest_checkpoint_age() -> float:
    """Seconds since the newest checkpoint was written, or 0 if none."""
    cps = glob.glob(CHECKPOINT_GLOB)
    if not cps:
        return 0.0
    return time.time() - max(os.path.getmtime(p) for p in cps)


def trainer_age_s() -> float:
    """Seconds since the current trainer started, parsed from its log
    filename (trainer_<YYYYMMDDTHHMMSS>.log). Robust across supervisor
    restarts (trainer_seen_at resets when the supervisor restarts, which
    would hide a long-lived trainer's true age). 0 if unknown."""
    logs = glob.glob(os.path.join(LOG_DIR, "trainer_*.log"))
    if not logs:
        return 0.0
    newest = max(logs, key=os.path.getmtime)
    m = re.search(r"trainer_(\d{8}T\d{6})\.log", os.path.basename(newest))
    if not m:
        return 0.0
    try:
        start = datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        return (datetime.datetime.now() - start).total_seconds()
    except ValueError:
        return 0.0


def trainer_recent_fps(n: int):
    """The last n per-update FPS values from the newest trainer log (newest
    last). Fewer than n if the trainer hasn't printed that many updates yet;
    empty if no log or no FPS line — the caller treats too-few as 'not
    degraded' so a just-started trainer is never recycled on thin data.
    """
    logs = glob.glob(os.path.join(LOG_DIR, "trainer_*.log"))
    if not logs:
        return []
    newest = max(logs, key=os.path.getmtime)
    try:
        with open(newest, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return []
    vals = [int(x) for x in re.findall(r"\| FPS\s+(\d+) \|", txt)]
    return vals[-n:]


def start_server(port: int = PORT) -> bool:
    """Launch one BalatroBot server + game on a port; wait for it."""
    kill_port_owner(port)
    time.sleep(3)
    env = dict(os.environ,
               # 8x. Was dropped to 4x during the 06-11 crash wave on the
               # theory the nil-races were speed-bound — disproven (4x
               # crashed at the same cadence). The real fix is the
               # double-fire protection (endpoint lock guard + trainer
               # transition debounce), so full speed is back. Never raise
               # above 8 (DECISIONS gotcha 7).
               BALATROBOT_GAMESPEED="8",
               BALATROBOT_ANIMATION_FPS="120",
               BALATROBOT_PORT=str(port))
    server_log = open(os.path.join(LOG_DIR, "server.log"), "a")
    subprocess.Popen(
        [UVX, "balatrobot", "serve", "--fast"],
        env=env, cwd=REPO,
        stdout=server_log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    deadline = time.time() + SERVER_BOOT_TIMEOUT_S
    while time.time() < deadline:
        if port_listening(port):
            log(f"server up, port {port} listening")
            return True
        time.sleep(2)
    log(f"ERROR: server on port {port} failed to come up within timeout")
    return False


def start_trainer() -> bool:
    cp = newest_checkpoint()
    if not cp:
        log("ERROR: no checkpoint found — refusing to start untrained")
        return False
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    trainer_log_path = os.path.join(LOG_DIR, f"trainer_{ts}.log")
    env = dict(os.environ, PYTHONUTF8="1")  # cp1252 emoji crash (gotcha 6)
    trainer_log = open(trainer_log_path, "a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-u", "-m", "training.train",
         "--total-timesteps", "1500000",
         "--device", "cpu",
         "--checkpoint-interval", "2",
         "--num-envs", str(NUM_ENVS),
         "--checkpoint", cp],
        env=env, cwd=REPO,
        stdout=trainer_log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    log(f"trainer started from {os.path.basename(cp)} -> {trainer_log_path}")
    return True


def main():
    log(f"supervisor started (pid {os.getpid()}), checking every "
        f"{CHECK_INTERVAL_S}s")
    port_down_checks = {p: 0 for p in PORTS}
    # (timestamp, global_step) samples for throughput stall detection
    step_samples: list = []
    # When the current trainer was (first seen) running — anchors the
    # checkpoint-age check so an old checkpoint inherited at startup
    # doesn't trigger an immediate kill.
    trainer_seen_at = None
    while True:
        if os.path.exists(STOP_FILE):
            log("SUPERVISOR_STOP file found — exiting (stack left as-is)")
            return

        try:
            # Reap orphaned balatrobot launchers every cycle — each restart
            # leaks one, and left unchecked they pile up overnight and choke
            # the machine (see reap_orphan_launchers docstring).
            reaped = reap_orphan_launchers()
            if reaped:
                log(f"reaped {reaped} orphan balatrobot launcher(s)")

            # Per-port health with debounce: the TRAINER's internal
            # watchdog also restarts its own game and is faster (~45s);
            # acting on the first down check raced it. 3 checks (~90s)
            # gives the trainer's recovery room to land first.
            server_ok = True
            for p in PORTS:
                if port_listening(p):
                    port_down_checks[p] = 0
                    continue
                port_down_checks[p] += 1
                if port_down_checks[p] >= 3:
                    log(f"port {p} down {port_down_checks[p]} checks — "
                        f"(re)starting its server")
                    if not start_server(p):
                        server_ok = False
                    port_down_checks[p] = 0
                else:
                    log(f"port {p} down (check {port_down_checks[p]}/3) — "
                        f"waiting for trainer's own recovery")
                    server_ok = False

            if server_ok:
                pid = trainer_pid()
                if not pid:
                    trainer_seen_at = None
                    log("trainer not running — (re)starting")
                    if start_trainer():
                        # Fresh trainer: stamp the heartbeat so the stale
                        # check measures from launch, not from the
                        # previous trainer's last step.
                        try:
                            with open(HEARTBEAT_PATH, "w") as f:
                                f.write(str(time.time()))
                        except OSError:
                            pass
                        time.sleep(TRAINER_GRACE_S)
                else:
                    if trainer_seen_at is None:
                        trainer_seen_at = time.time()

                    # Degradation recycle: clean full-stack restart (same
                    # mechanism as the FROZEN path) to reset the ~1/n FPS
                    # decay. Two triggers:
                    #   (a) AGE BACKSTOP — any trainer older than the hard cap,
                    #       regardless of FPS. The reliable safety net (the FPS
                    #       trigger was too slow/brittle overnight 06-13).
                    #   (b) FPS FLOOR — FPS <= floor for SUSTAIN consecutive
                    #       updates: a faster catch for fast degradation. Uses
                    #       <= so a value landing exactly on the floor still
                    #       counts (a strict < let the streak reset and the
                    #       trainer crawled all night).
                    t_age = trainer_age_s()
                    recent = trainer_recent_fps(FPS_RECYCLE_SUSTAIN)
                    fps_degraded = (
                        t_age > RECYCLE_MIN_AGE_S
                        and len(recent) >= FPS_RECYCLE_SUSTAIN
                        and all(v <= FPS_RECYCLE_FLOOR for v in recent)
                    )
                    if t_age > MAX_TRAINER_AGE_S or fps_degraded:
                        why = (f"age {t_age/3600:.1f}h > {MAX_TRAINER_AGE_S/3600:.0f}h cap"
                               if t_age > MAX_TRAINER_AGE_S
                               else f"FPS {recent} <= floor {FPS_RECYCLE_FLOOR}")
                        log(f"trainer pid {pid} {why} — recycling stack")
                        kill_trainer(pid)
                        kill_strays()
                        step_samples.clear()
                        trainer_seen_at = None
                        time.sleep(3)
                        continue

                    # Liveness: a trainer that exists but has made zero
                    # environment steps in HEARTBEAT_STALE_S is wedged
                    # (frozen fetch, hung start endpoint, boot-splash
                    # zombie game...). Kill BOTH trainer and game — a
                    # wedged trainer almost always means a wedged game —
                    # and let the next cycles rebuild the stack.
                    age = heartbeat_age()
                    if age > HEARTBEAT_STALE_S:
                        log(f"trainer pid {pid} alive but heartbeat is "
                            f"{age:.0f}s stale — killing FROZEN stack")
                        kill_trainer(pid)
                        kill_strays()
                        step_samples.clear()
                    else:
                        # Throughput: stepping but too slowly = stall
                        # (e.g. chronic wedge/restart churn). Judge only
                        # on a full window of samples.
                        now = time.time()
                        step = heartbeat_step()
                        if step is not None:
                            if step_samples and step < step_samples[-1][1]:
                                # step went backwards (resumed from an
                                # older checkpoint) — restart the window
                                step_samples.clear()
                            step_samples.append((now, step))
                            # keep only the window
                            cutoff = now - STALL_WINDOW_S
                            step_samples[:] = [s for s in step_samples
                                               if s[0] >= cutoff]
                            span = (step_samples[-1][0] - step_samples[0][0]
                                    if len(step_samples) >= 2 else 0)
                            if span >= STALL_WINDOW_S * 0.9:
                                delta = step_samples[-1][1] - step_samples[0][1]
                                rate = delta / (span / 60.0)
                                if rate < STALL_MIN_STEPS_PER_MIN:
                                    log(f"trainer pid {pid} STALLED: "
                                        f"{rate:.0f} steps/min over "
                                        f"{span/60:.0f} min (floor "
                                        f"{STALL_MIN_STEPS_PER_MIN:.0f}) — "
                                        f"killing degraded stack")
                                    kill_trainer(pid)
                                    kill_strays()
                                    step_samples.clear()
                                    trainer_seen_at = None

                        # Checkpoint cadence: chronic wedge churn can hold
                        # a step rate above any safe floor while updates
                        # crawl. If this trainer has been up for the full
                        # threshold and the newest checkpoint is older
                        # than it, the stack is churning, not training.
                        # (trainer_seen_at is None right after a rate-kill
                        # above — skip; the stack is already going down.)
                        up_for = (time.time() - trainer_seen_at
                                  if trainer_seen_at is not None else 0)
                        cp_age = newest_checkpoint_age()
                        if (up_for > CHECKPOINT_STALL_S
                                and cp_age > CHECKPOINT_STALL_S):
                            log(f"trainer pid {pid} CHURNING: up "
                                f"{up_for/60:.0f} min, newest checkpoint "
                                f"{cp_age/60:.0f} min old (limit "
                                f"{CHECKPOINT_STALL_S/60:.0f}) — killing "
                                f"churning stack")
                            kill_trainer(pid)
                            kill_strays()
                            step_samples.clear()
                            trainer_seen_at = None
        except Exception as e:  # never let one bad cycle kill the supervisor
            log(f"ERROR in supervise cycle: {e}")

        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
