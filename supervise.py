"""Supervisor: keeps the full Balatron training stack alive AND FAST.

Run this ONE detached process instead of launching the server and trainer
by hand. Every cycle it ensures:

  1. The BalatroBot servers + games are up (ports 12346..N listening).
  2. Exactly ONE trainer is running, resumed from the newest checkpoint.
  3. The trainer is making FAST progress — not crawling, frozen, or churning.
  4. The machine is not being starved of RAM by an external memory leak.

WHY THE 06-14 REWRITE (read before touching thresholds):
The recurring "I come back after 7-8h and Balatron is slow" was misdiagnosed
three times as an internal trainer FPS decay. The real cause, found 06-14:
*external memory starvation*. Steam's `steamwebhelper.exe` leaks to 10-14 GB
over hours; with the user's other apps the machine hits ~94% RAM, Windows
pages aggressively, and Balatron (only ~4 GB itself) crawls to ~12 steps/min
because its working set keeps getting paged out. The old detectors couldn't
catch the resulting crawl — heartbeat still ticked, the rate stayed just above
the 10/min floor, and the checkpoint-stall net waited 150 MINUTES. So a slow
crawl ran invisibly for hours and the user always caught it mid-degradation.

The rebuild makes the supervisor bulletproof for long unattended runs:
  - Detection is FAST and uses the RELIABLE signal (heartbeat steps/min over a
    short window), not the log's cumulative FPS (which is a misleading average).
  - Kills CASCADE: every recycle kills ALL trainers, ALL games, ALL orphan
    launchers — the old single-PID kill let duplicates/orphans pile up until
    RAM was exhausted.
  - The supervisor is a SINGLETON (kills rival supervise.py on startup) — two
    supervisors each spawned a trainer, doubling the load.
  - A MEMORY GUARDIAN watches the external hog and (optionally) reclaims it,
    and always logs the true cause so a human knows it's Steam, not Balatron.
  - Logs are pruned so the 2.9 GB log pile stops adding IO pressure.

Launch (detached, hidden):
  powershell -Command "Start-Process -WindowStyle Hidden python -ArgumentList '-u','supervise.py' -WorkingDirectory 'C:\\Users\\jarms\\repos\\balatron'"

Stop:
  Create a file named SUPERVISOR_STOP in the repo root (checked each cycle),
  or kill the python process running supervise.py.
"""

import datetime
import glob
import os
import re
import socket
import subprocess
import sys
import time

try:
    import psutil
except ImportError:  # psutil is required for the rewrite; fail loud.
    psutil = None

REPO = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO, "logs")
LOG_PATH = os.path.join(LOG_DIR, "supervisor.log")
STOP_FILE = os.path.join(REPO, "SUPERVISOR_STOP")
STATUS_PATH = os.path.join(LOG_DIR, "supervisor_status.txt")
UVX = r"C:\Users\jarms\.local\bin\uvx.exe"

# Parallel game instances: one server+game per port. Must match the trainer's
# --num-envs (ports 12346..12346+N-1).
NUM_ENVS = 3
PORTS = [12346 + i for i in range(NUM_ENVS)]
PORT = PORTS[0]

CHECK_INTERVAL_S = 30
SERVER_BOOT_TIMEOUT_S = 120
TRAINER_GRACE_S = 60        # after starting a trainer, wait before re-checking
HEARTBEAT_PATH = os.path.join(LOG_DIR, "heartbeat")

# --- Liveness / throughput thresholds (06-14: ALL tightened drastically) -----
# The trainer writes (timestamp, global_step) to the heartbeat every real env
# step. steps/min from that file is the ONE reliable speed signal. Healthy is
# ~400-1000 steps/min; churn/crawl is ~10-130. The old detectors waited hours;
# these act within minutes.
HEARTBEAT_STALE_S = 240        # no env step in 4 min = FROZEN (hard hang)
RATE_WINDOW_S = 720            # 12 min of heartbeat samples before judging rate
RATE_FLOOR_PER_MIN = 80.0      # < this over a full window = crawl (1/5 healthy)
GRACE_AFTER_START_S = 300      # a fresh trainer gets 5 min before rate/ckpt nets
CHECKPOINT_STALL_S = 2400      # 40 min with no new checkpoint = churn (was 150m)
MAX_TRAINER_AGE_S = 5400       # 90 min proactive recycle: never let it get old
                               # enough to bloat/page. Resume is cheap.

CHECKPOINT_GLOB = os.path.join(REPO, "checkpoints", "balatron_phase1_update*.pt")

# --- Memory guardian (the actual root cause) ---------------------------------
# When system RAM is critically high, Balatron is starved no matter how fresh
# it is. The guardian (a) recycles Balatron to reclaim its own paged/bloated
# working set, and (b) optionally restarts the worst EXTERNAL leaker so the
# machine isn't saturated. steamwebhelper is the confirmed culprit (leaks to
# 10-14 GB; normal < 1 GB) and is safe to restart — Steam respawns a fresh
# lighter helper and a running game is unaffected.
MEM_CRITICAL_PCT = 90.0        # system RAM at/above this = pressure
MEM_GUARDIAN_ENABLED = True    # restart the external hog when it's clearly leaked
# name -> (GB threshold above which it's "clearly leaked" and safe to restart)
MEM_RECLAIM_TARGETS = {"steamwebhelper.exe": 4.0}
# Don't churn Balatron under sustained external pressure recycling can't fix:
# after this many recycles inside the window, stop recycling for RAM and just
# keep the stack alive (still logs the diagnosis).
RECYCLE_BURST_LIMIT = 3
RECYCLE_BURST_WINDOW_S = 3600

# --- Log pruning -------------------------------------------------------------
LOG_PRUNE_KEEP = 6             # keep this many newest trainer logs
LOG_PRUNE_MAX_AGE_S = 86400    # delete trainer logs older than 24 h


def log(msg: str):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def write_status(state: str, detail: str = ""):
    """One-line machine + human readable status the user (or dashboard) can
    glance at to know WHY the stack is in whatever state it's in."""
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} "
                    f"| {state} | {detail}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Process helpers (psutil-based: fast, exact, no per-cycle PowerShell spawns)
# ---------------------------------------------------------------------------

def _iter_procs(attrs):
    if psutil is None:
        return
    for p in psutil.process_iter(attrs):
        try:
            yield p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _cmdline(p) -> str:
    try:
        return " ".join(p.info.get("cmdline") or [])
    except Exception:
        return ""


def trainer_pids() -> list:
    """ALL pids running training.train (excludes this supervisor and transient
    `python -c` helpers). Returns a list so duplicates can be cleaned up.

    Matches on cmdline TOKENS, not the joined string: the real launch is
    `python -u -m training.train ...` (so 'training.train' and '-m' are their
    own tokens), whereas a `python -c "...training.train..."` diagnostic has it
    only as a SUBSTRING of the single -c code token. Substring matching here
    would mis-see diagnostics as duplicate trainers and trigger a spurious
    recycle — token matching is immune to that."""
    pids = []
    me = os.getpid()
    for p in _iter_procs(["pid", "name", "cmdline"]):
        if p.info["pid"] == me:
            continue
        toks = p.info.get("cmdline") or []
        if "training.train" in toks and "-m" in toks:
            pids.append(p.info["pid"])
    return pids


def supervisor_rivals() -> list:
    """Other live supervise.py processes (not me). Token match (a token ending
    in supervise.py) so `python -c "import supervise..."` never counts."""
    me = os.getpid()
    out = []
    for p in _iter_procs(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        toks = p.info.get("cmdline") or []
        if any(t.endswith("supervise.py") for t in toks):
            out.append(p.info["pid"])
    return out


def kill_pids(pids, tree: bool = True):
    """Kill pids (optionally whole trees) and wait briefly for them to die."""
    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except Exception:
            pass
    for p in procs:
        try:
            if tree:
                for c in p.children(recursive=True):
                    try:
                        c.kill()
                    except Exception:
                        pass
            p.kill()
        except Exception:
            pass
    if procs:
        try:
            psutil.wait_procs(procs, timeout=8)
        except Exception:
            pass


def kill_all_balatro():
    """Kill every Balatro.exe and every balatrobot 'serve' launcher tree.
    Used on recycle so NOTHING leaks across restarts."""
    victims = []
    for p in _iter_procs(["pid", "name", "cmdline"]):
        name = (p.info.get("name") or "")
        cl = _cmdline(p)
        if name == "Balatro.exe":
            victims.append(p.info["pid"])
        elif name == "python.exe" and "balatrobot" in cl and "serve" in cl:
            victims.append(p.info["pid"])
        elif name in ("balatrobot.exe",):
            victims.append(p.info["pid"])
    kill_pids(victims, tree=True)
    return len(victims)


def reap_orphan_launchers() -> int:
    """Kill 'balatrobot serve' launchers that have NO live Balatro.exe in their
    descendant tree. ~1 leaks per restart; unchecked they pile up overnight,
    thrash the scheduler and page the trainer out. psutil version of the old
    PowerShell reaper — faster and can't time out mid-night.

    The live tree is balatrobot.exe -> python(serve) -> python(serve) ->
    Balatro.exe (TWO nested launchers), so we protect the whole ancestor chain
    of every live game and reap only serve-launchers outside it. The >180s age
    guard avoids reaping a launcher that just booted and hasn't spawned its
    game yet."""
    if psutil is None:
        return 0
    procs = {}
    for p in _iter_procs(["pid", "name", "cmdline", "ppid", "create_time"]):
        procs[p.info["pid"]] = p
    # Protect the ancestor chain of every live Balatro.exe.
    protected = set()
    for pid, p in procs.items():
        if (p.info.get("name") or "") != "Balatro.exe":
            continue
        cur = p.info.get("ppid")
        guard = 0
        while cur and cur in procs and guard < 20:
            protected.add(cur)
            cur = procs[cur].info.get("ppid")
            guard += 1
    now = time.time()
    orphans = []
    for pid, p in procs.items():
        if (p.info.get("name") or "") != "python.exe":
            continue
        cl = _cmdline(p)
        if "balatrobot" not in cl or "serve" not in cl:
            continue
        if pid in protected:
            continue
        if now - (p.info.get("create_time") or now) < 180:
            continue
        orphans.append(pid)
    kill_pids(orphans, tree=True)
    return len(orphans)


def port_listening(port: int = PORT) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def kill_port_owner(port: int):
    """Kill the process tree owning a specific port (psutil; no PowerShell)."""
    if psutil is None:
        return
    try:
        for c in psutil.net_connections(kind="inet"):
            if (c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN
                    and c.pid):
                kill_pids([c.pid], tree=True)
                return
    except Exception:
        pass


# ---------------------------------------------------------------------------
# State / health readers
# ---------------------------------------------------------------------------

def newest_checkpoint() -> str:
    cps = glob.glob(CHECKPOINT_GLOB)
    return max(cps, key=os.path.getmtime) if cps else ""


def newest_checkpoint_age() -> float:
    cps = glob.glob(CHECKPOINT_GLOB)
    if not cps:
        return 0.0
    return time.time() - max(os.path.getmtime(p) for p in cps)


def heartbeat_age() -> float:
    try:
        return time.time() - os.path.getmtime(HEARTBEAT_PATH)
    except OSError:
        return 0.0


def heartbeat_step():
    try:
        with open(HEARTBEAT_PATH) as f:
            parts = f.read().split()
        return int(float(parts[1])) if len(parts) >= 2 else None
    except (OSError, ValueError, IndexError):
        return None


def trainer_age_s() -> float:
    """Seconds since the current trainer started, from its log filename."""
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


def system_ram_pct() -> float:
    if psutil is None:
        return 0.0
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def top_memory_hog():
    """(name, pid, gb) of the single biggest-RSS process, for diagnosis."""
    if psutil is None:
        return ("?", 0, 0.0)
    best = ("?", 0, 0.0)
    for p in _iter_procs(["pid", "name", "memory_info"]):
        try:
            rss = p.info["memory_info"].rss / 1e9
        except Exception:
            continue
        if rss > best[2]:
            best = (p.info.get("name") or "?", p.info["pid"], rss)
    return best


def reclaim_external_hog() -> str:
    """Restart any configured external leaker that is clearly bloated. Returns
    a human description of what (if anything) was reclaimed."""
    if psutil is None or not MEM_GUARDIAN_ENABLED:
        return ""
    reclaimed = []
    for p in _iter_procs(["pid", "name", "memory_info"]):
        name = (p.info.get("name") or "")
        thresh = MEM_RECLAIM_TARGETS.get(name)
        if thresh is None:
            continue
        try:
            rss = p.info["memory_info"].rss / 1e9
        except Exception:
            continue
        if rss >= thresh:
            kill_pids([p.info["pid"]], tree=False)
            reclaimed.append(f"{name}({rss:.1f}GB)")
    return ", ".join(reclaimed)


METRICS_HISTORY = os.path.join(LOG_DIR, "metrics_history.log")
_UPDATE_LINE_RE = re.compile(r"Update\s+(\d+) \|.*\| LR ")


def _preserve_update_lines(path: str):
    """Append a log's PPO 'Update ...' lines to the persistent metrics history
    before the log is deleted, so the dashboard's learning curve survives
    pruning. Dedup by update number (latest wins)."""
    try:
        existing = {}
        if os.path.exists(METRICS_HISTORY):
            with open(METRICS_HISTORY, encoding="utf-8", errors="replace") as f:
                for ln in f:
                    m = _UPDATE_LINE_RE.search(ln)
                    if m:
                        existing[int(m.group(1))] = ln.rstrip("\n")
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = _UPDATE_LINE_RE.search(ln)
                if m:
                    existing[int(m.group(1))] = ln.rstrip("\n")
        with open(METRICS_HISTORY, "w", encoding="utf-8") as out:
            for k in sorted(existing):
                out.write(existing[k] + "\n")
    except OSError:
        pass


def prune_logs():
    """Delete old trainer logs so the log pile stops adding IO pressure —
    preserving their PPO update lines to the persistent metrics history first."""
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "trainer_*.log")),
                  key=os.path.getmtime, reverse=True)
    now = time.time()
    for i, path in enumerate(logs):
        too_old = (now - os.path.getmtime(path)) > LOG_PRUNE_MAX_AGE_S
        if i >= LOG_PRUNE_KEEP or too_old:
            _preserve_update_lines(path)
            try:
                os.remove(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Stack lifecycle
# ---------------------------------------------------------------------------

def start_server(port: int = PORT) -> bool:
    kill_port_owner(port)
    time.sleep(3)
    env = dict(os.environ,
               BALATROBOT_GAMESPEED="8",       # never raise above 8 (gotcha 7)
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
    # Stamp the heartbeat so liveness measures from launch, not the old run.
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(f"{time.time()} 0")
    except OSError:
        pass
    return True


def recycle_stack(reason: str):
    """Clean full-stack teardown: kill ALL trainers, ALL games, ALL orphan
    launchers. The main loop rebuilds from scratch next cycle. Cascading by
    design — the old single-PID kill is what let duplicates/orphans accumulate
    until RAM was exhausted."""
    log(f"RECYCLE ({reason}) — tearing down full stack")
    write_status("RECYCLING", reason)
    kill_pids(trainer_pids(), tree=True)
    killed = kill_all_balatro()
    reap_orphan_launchers()
    log(f"  killed trainers + {killed} game/launcher processes")
    time.sleep(3)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    if psutil is None:
        log("FATAL: psutil not available to the supervisor interpreter — "
            "install it (pip install psutil). Exiting.")
        return

    # Singleton: a second supervisor would spawn a second trainer and double
    # the load (a confirmed cause of the overnight process pileup).
    rivals = supervisor_rivals()
    if rivals:
        kill_pids(rivals, tree=False)
        log(f"killed {len(rivals)} rival supervisor(s): {rivals}")

    log(f"supervisor started (pid {os.getpid()}), checking every "
        f"{CHECK_INTERVAL_S}s | rate floor {RATE_FLOOR_PER_MIN:.0f}/min over "
        f"{RATE_WINDOW_S//60}min, ckpt-stall {CHECKPOINT_STALL_S//60}min, "
        f"age cap {MAX_TRAINER_AGE_S//60}min, mem-crit {MEM_CRITICAL_PCT:.0f}%")

    port_down_checks = {p: 0 for p in PORTS}
    step_samples = []            # (timestamp, global_step) rolling window
    recycle_times = []           # timestamps of recent recycles (burst guard)
    last_prune = 0.0

    while True:
        if os.path.exists(STOP_FILE):
            log("SUPERVISOR_STOP file found — exiting (stack left as-is)")
            return

        try:
            # Prune logs occasionally (hourly) — keep IO pressure down.
            if time.time() - last_prune > 3600:
                prune_logs()
                last_prune = time.time()

            # Reap orphan launchers every cycle.
            reaped = reap_orphan_launchers()
            if reaped:
                log(f"reaped {reaped} orphan balatrobot launcher(s)")

            # Collapse duplicate trainers immediately (keep the youngest by
            # leaving the newest log's process; simplest safe rule: if >1, a
            # recycle is the clean resolution).
            tpids = trainer_pids()
            if len(tpids) > 1:
                log(f"{len(tpids)} trainers running (expect 1) — collapsing")
                recycle_stack("duplicate trainers")
                step_samples.clear()
                recycle_times.append(time.time())
                time.sleep(3)
                continue

            # --- Memory guardian (the real root cause) ----------------------
            ram = system_ram_pct()
            if ram >= MEM_CRITICAL_PCT:
                hog = top_memory_hog()
                reclaimed = reclaim_external_hog()
                # Burst guard: under sustained external pressure recycling
                # can't fix, stop churning Balatron — just keep it alive and
                # keep shouting the real cause.
                now = time.time()
                recycle_times[:] = [t for t in recycle_times
                                    if now - t < RECYCLE_BURST_WINDOW_S]
                if reclaimed:
                    log(f"MEM {ram:.0f}% critical — reclaimed external hog: "
                        f"{reclaimed} (top was {hog[0]} {hog[2]:.1f}GB)")
                    write_status("MEM_RECLAIMED",
                                 f"RAM {ram:.0f}%, restarted {reclaimed}")
                else:
                    msg = (f"RAM {ram:.0f}% critical; biggest consumer is "
                           f"{hog[0]} ({hog[2]:.1f}GB). Balatron is being "
                           f"STARVED by external memory — close it for "
                           f"sustained speed.")
                    if len(recycle_times) < RECYCLE_BURST_LIMIT:
                        log(f"MEM pressure — {msg} Recycling Balatron to "
                            f"reclaim its working set.")
                        recycle_stack(f"RAM {ram:.0f}% (external hog {hog[0]})")
                        recycle_times.append(now)
                        step_samples.clear()
                        time.sleep(3)
                        continue
                    else:
                        log(f"MEM pressure persists ({len(recycle_times)} "
                            f"recycles/h) — backing off. {msg}")
                        write_status("MEM_STARVED", msg)

            # --- Per-port server health (debounced: trainer's own watchdog
            #     restarts its game in ~45s; act on the 3rd down check). ------
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

            if not server_ok:
                time.sleep(CHECK_INTERVAL_S)
                continue

            # --- Trainer presence -------------------------------------------
            tpids = trainer_pids()
            if not tpids:
                log("trainer not running — (re)starting")
                if start_trainer():
                    step_samples.clear()
                    time.sleep(TRAINER_GRACE_S)
                time.sleep(CHECK_INTERVAL_S)
                continue

            pid = tpids[0]
            t_age = trainer_age_s()

            # --- (1) Proactive age recycle: never let it bloat/page ---------
            if t_age > MAX_TRAINER_AGE_S:
                recycle_stack(f"age {t_age/60:.0f}min > {MAX_TRAINER_AGE_S//60}min cap")
                step_samples.clear()
                recycle_times.append(time.time())
                continue

            # --- (2) FROZEN: heartbeat not advancing ------------------------
            hb_age = heartbeat_age()
            if hb_age > HEARTBEAT_STALE_S:
                recycle_stack(f"FROZEN: heartbeat {hb_age:.0f}s stale")
                step_samples.clear()
                recycle_times.append(time.time())
                continue

            # Fresh trainers get a grace period before rate/checkpoint nets so
            # the first rollout (which hasn't checkpointed yet) isn't flagged.
            if t_age < GRACE_AFTER_START_S:
                write_status("WARMUP", f"trainer {t_age:.0f}s old")
                time.sleep(CHECK_INTERVAL_S)
                continue

            # --- (3) CRAWL: steps/min below floor over a full window --------
            now = time.time()
            step = heartbeat_step()
            if step is not None:
                if step_samples and step < step_samples[-1][1]:
                    step_samples.clear()         # resumed from older ckpt
                step_samples.append((now, step))
                cutoff = now - RATE_WINDOW_S
                step_samples[:] = [s for s in step_samples if s[0] >= cutoff]
                span = (step_samples[-1][0] - step_samples[0][0]
                        if len(step_samples) >= 2 else 0)
                if span >= RATE_WINDOW_S * 0.9:
                    delta = step_samples[-1][1] - step_samples[0][1]
                    rate = delta / (span / 60.0)
                    if rate < RATE_FLOOR_PER_MIN:
                        recycle_stack(f"CRAWL: {rate:.0f} steps/min over "
                                      f"{span/60:.0f}min (floor "
                                      f"{RATE_FLOOR_PER_MIN:.0f})")
                        step_samples.clear()
                        recycle_times.append(now)
                        continue
                    write_status("HEALTHY",
                                 f"{rate:.0f} steps/min, age {t_age/60:.0f}min, "
                                 f"RAM {ram:.0f}%")

            # --- (4) CHURN: up long enough but checkpoint gone stale --------
            cp_age = newest_checkpoint_age()
            if t_age > CHECKPOINT_STALL_S and cp_age > CHECKPOINT_STALL_S:
                recycle_stack(f"CHURN: up {t_age/60:.0f}min, newest checkpoint "
                              f"{cp_age/60:.0f}min old (limit "
                              f"{CHECKPOINT_STALL_S//60}min)")
                step_samples.clear()
                recycle_times.append(time.time())
                continue

        except Exception as e:  # never let one bad cycle kill the supervisor
            log(f"ERROR in supervise cycle: {type(e).__name__}: {e}")

        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
