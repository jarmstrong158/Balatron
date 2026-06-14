"""Watchdog-for-the-watchdog: guarantee exactly one supervise.py is alive.

Run by a Windows Scheduled Task every ~10 min (and at logon). The supervisor
itself is robust (per-cycle try/except, 9h+ observed uptime), but for truly
unattended 7-8h runs nothing should be a single point of failure — if the
supervisor is ever killed (external kill, crash, a stray reboot), this brings
it back within ~10 min.

CRITICAL: only starts a supervisor when NONE is running. The supervisor is a
singleton that kills rival supervise.py on startup, so blindly launching one
every 10 min would churn (each new one kills the old). Check first, start only
if absent — never disturb a healthy supervisor.
"""
import os
import subprocess
import sys

try:
    import psutil
except ImportError:
    psutil = None

REPO = os.path.dirname(os.path.abspath(__file__))
SUP = os.path.join(REPO, "supervise.py")


def supervisor_alive() -> bool:
    if psutil is None:
        return False
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            toks = p.info["cmdline"] or []
            if any(str(t).endswith("supervise.py") for t in toks):
                return True
        except Exception:
            continue
    return False


def main():
    if supervisor_alive():
        print("supervisor already alive — nothing to do")
        return
    flags = subprocess.CREATE_NEW_PROCESS_GROUP
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags |= subprocess.DETACHED_PROCESS
    subprocess.Popen(
        [sys.executable, "-u", SUP],
        cwd=REPO, creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("supervisor was down — started a new one")


if __name__ == "__main__":
    main()
