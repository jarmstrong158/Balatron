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
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO, "logs")
LOG_PATH = os.path.join(LOG_DIR, "supervisor.log")
STOP_FILE = os.path.join(REPO, "SUPERVISOR_STOP")
UVX = r"C:\Users\jarms\.local\bin\uvx.exe"
PORT = 12346

CHECK_INTERVAL_S = 30
SERVER_BOOT_TIMEOUT_S = 120
TRAINER_GRACE_S = 60  # after starting the trainer, wait before re-checking
HEARTBEAT_PATH = os.path.join(LOG_DIR, "heartbeat")
HEARTBEAT_STALE_S = 300  # no env step in 5 min = trainer wedged


def log(msg: str):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_listening() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        return s.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        s.close()


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


def heartbeat_age() -> float:
    """Seconds since the trainer last made a real environment step.
    Returns 0 if the heartbeat file doesn't exist yet (fresh trainer
    gets the grace period instead)."""
    try:
        return time.time() - os.path.getmtime(HEARTBEAT_PATH)
    except OSError:
        return 0.0


def kill_trainer(pid: str):
    subprocess.run(["taskkill", "/F", "/PID", pid],
                   capture_output=True, timeout=15)


def start_server() -> bool:
    """Launch the BalatroBot server + game; wait for the port."""
    kill_strays()
    time.sleep(3)
    env = dict(os.environ,
               BALATROBOT_GAMESPEED="8",   # 8x — higher destabilizes (DECISIONS gotcha 7)
               BALATROBOT_ANIMATION_FPS="120")
    server_log = open(os.path.join(LOG_DIR, "server.log"), "a")
    subprocess.Popen(
        [UVX, "balatrobot", "serve", "--fast"],
        env=env, cwd=REPO,
        stdout=server_log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    deadline = time.time() + SERVER_BOOT_TIMEOUT_S
    while time.time() < deadline:
        if port_listening():
            log("server up, port listening")
            return True
        time.sleep(2)
    log("ERROR: server failed to come up within timeout")
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
    port_down_checks = 0
    while True:
        if os.path.exists(STOP_FILE):
            log("SUPERVISOR_STOP file found — exiting (stack left as-is)")
            return

        try:
            server_ok = port_listening()
            if not server_ok:
                # Debounce: the TRAINER's internal watchdog also restarts
                # the game and is faster (~45s). Acting on the first down
                # check raced it — both spawned servers, kill_strays killed
                # the trainer's fresh instance, and two wrappers fought
                # over the port. Require 3 consecutive down checks (~90s)
                # so the trainer's own recovery gets to land first.
                port_down_checks += 1
                if port_down_checks >= 3:
                    log(f"port 12346 down {port_down_checks} checks — "
                        f"(re)starting server")
                    server_ok = start_server()
                    port_down_checks = 0
                else:
                    log(f"port 12346 down (check {port_down_checks}/3) — "
                        f"waiting for trainer's own recovery")
            else:
                port_down_checks = 0

            if server_ok:
                pid = trainer_pid()
                if not pid:
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
                    # Liveness: a trainer that exists but has made zero
                    # environment steps in HEARTBEAT_STALE_S is wedged
                    # (frozen fetch, hung start endpoint, boot-splash
                    # zombie game...). Kill BOTH trainer and game — a
                    # wedged trainer almost always means a wedged game —
                    # and let the next cycles rebuild the stack.
                    age = heartbeat_age()
                    if age > HEARTBEAT_STALE_S:
                        log(f"trainer pid {pid} alive but heartbeat is "
                            f"{age:.0f}s stale — killing wedged stack")
                        kill_trainer(pid)
                        kill_strays()
        except Exception as e:  # never let one bad cycle kill the supervisor
            log(f"ERROR in supervise cycle: {e}")

        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
