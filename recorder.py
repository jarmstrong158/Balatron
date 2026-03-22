"""
Balatron — Automated Win Recording System

Captures every run via ffmpeg screen recording, discards losses,
and permanently saves wins. The agent self-documents its own best
performance without any manual intervention.
"""

import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


class RunRecorder:
    """Records Balatro runs, keeping wins and discarding losses."""

    TEMP_DIR = os.path.join("recordings", "temp")
    WINS_DIR = os.path.join("recordings", "wins")
    WINS_LOG = os.path.join("recordings", "wins.log")
    MAX_TEMP_SIZE_MB = 500

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._process: subprocess.Popen | None = None
        self._temp_path: str | None = None
        self._run_start_time: float = 0.0

        if not self.enabled:
            return

        # Check ffmpeg availability
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("[RECORDER] ffmpeg not found — recording disabled", flush=True)
            self.enabled = False
            return

        # Create directories
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.WINS_DIR, exist_ok=True)

        # Clean up leftover temp files from crashed sessions
        self._cleanup_temp()

    def _cleanup_temp(self):
        """Remove any leftover temp files from previous sessions."""
        try:
            for f in Path(self.TEMP_DIR).glob("temp_*.mp4"):
                f.unlink(missing_ok=True)
                print(f"[RECORDER] Cleaned up stale temp: {f.name}", flush=True)
        except OSError:
            pass

    def start_run(self):
        """Start recording a new run."""
        if not self.enabled:
            return

        # Kill any lingering process from a previous run
        self._stop_ffmpeg()
        self._delete_temp()

        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]
        self._temp_path = os.path.join(self.TEMP_DIR, f"temp_{ts}.mp4")
        self._run_start_time = time.monotonic()

        try:
            # Full desktop capture, no audio, ultrafast preset
            cmd = [
                "ffmpeg",
                "-f", "gdigrab",
                "-framerate", "30",
                "-i", "desktop",
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "ultrafast",
                "-an",  # no audio
                "-y",   # overwrite
                self._temp_path,
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.PIPE,  # needed to send 'q' for clean stop
            )
        except (FileNotFoundError, OSError) as e:
            print(f"[RECORDER] Failed to start ffmpeg: {e}", flush=True)
            self._process = None
            self._temp_path = None

    def end_run(
        self,
        won: bool,
        ante_reached: int,
        final_score: int = 0,
        checkpoint_path: str = "",
        total_steps: int = 0,
    ):
        """End the current recording. Save if won, discard if lost."""
        if not self.enabled or self._process is None:
            return

        self._stop_ffmpeg()

        if not self._temp_path or not os.path.exists(self._temp_path):
            self._temp_path = None
            return

        if won:
            self._save_win(ante_reached, final_score, checkpoint_path, total_steps)
        else:
            print(
                f"[RECORDER] Run ended (Ante {ante_reached}) — recording discarded",
                flush=True,
            )
            self._delete_temp()

    def check_file_size(self):
        """Check if temp file exceeds size limit. Call periodically during long runs."""
        if not self.enabled or not self._temp_path:
            return
        try:
            size_mb = os.path.getsize(self._temp_path) / (1024 * 1024)
            if size_mb > self.MAX_TEMP_SIZE_MB:
                print(
                    f"[RECORDER] Temp file exceeds {self.MAX_TEMP_SIZE_MB}MB "
                    f"({size_mb:.0f}MB) — restarting recording",
                    flush=True,
                )
                self._stop_ffmpeg()
                self._delete_temp()
                self.start_run()
        except OSError:
            pass

    def _save_win(
        self,
        ante: int,
        score: int,
        checkpoint_path: str,
        total_steps: int,
    ):
        """Move temp recording to wins directory with structured naming."""
        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]
        win_name = f"win_ante{ante}_score{score}_{ts}.mp4"
        win_path = os.path.join(self.WINS_DIR, win_name)

        try:
            shutil.move(self._temp_path, win_path)
        except OSError as e:
            print(f"[RECORDER] Failed to save win recording: {e}", flush=True)
            self._delete_temp()
            return

        self._temp_path = None

        # Append to wins.log
        try:
            log_ts = datetime.now().isoformat(timespec="seconds")
            log_line = (
                f"{log_ts} | ante={ante} | score={score} | "
                f"checkpoint={checkpoint_path} | steps={total_steps}\n"
            )
            with open(self.WINS_LOG, "a", encoding="utf-8") as f:
                f.write(log_line)
        except OSError:
            pass

        print(
            f"[RECORDER] WIN SAVED — ante{ante} — {win_path}",
            flush=True,
        )

    def _stop_ffmpeg(self):
        """Cleanly stop the ffmpeg process."""
        if self._process is None:
            return

        try:
            # Send 'q' to stdin for graceful shutdown (ffmpeg's quit command)
            if self._process.stdin:
                try:
                    self._process.stdin.write(b"q")
                    self._process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

            # Wait for graceful exit
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
        except OSError:
            pass
        finally:
            self._process = None

    def _delete_temp(self):
        """Delete the current temp file."""
        if self._temp_path:
            try:
                Path(self._temp_path).unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_path = None

    def cleanup(self):
        """Final cleanup — call on training shutdown."""
        self._stop_ffmpeg()
        self._delete_temp()
