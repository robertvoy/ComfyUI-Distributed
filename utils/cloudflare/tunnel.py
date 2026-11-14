"""Cloudflare tunnel lifecycle manager."""

import asyncio
import os
import shutil
import subprocess
import time

try:
    import psutil
except ImportError:  # pragma: no cover - psutil ships with ComfyUI installs
    psutil = None

from ..constants import TUNNEL_START_TIMEOUT
from ..logging import debug_log
from ..network import get_server_port, normalize_host
from ..process import get_process_create_time, open_verified_process, terminate_process
from .binary import ensure_binary
from .process_reader import ProcessReader
from .state import clear_tunnel_state, load_tunnel_state, persist_tunnel_state, resolve_restore_master_host


def _is_cloudflared_name(name):
    return "cloudflared" in str(name or "").lower()


def _terminate_restored(restored, pid, timeout=5):
    """Stop a verified restored cloudflared (a psutil handle), escalating to kill
    like the subprocess path does."""
    try:
        restored.terminate()
        try:
            restored.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            debug_log(f"cloudflared pid {pid} ignored terminate; killing it")
            restored.kill()
            restored.wait(timeout=timeout)
    except psutil.NoSuchProcess:
        pass
    except Exception as exc:
        debug_log(f"Error stopping cloudflared pid {pid}: {exc}")


def _open_tunnel_process(pid, create_time):
    """Return a psutil handle for a saved cloudflared PID only if it is still that process.

    A saved creation time proves identity; state saved by older versions falls
    back to the executable name. Returns None when the PID is gone, reused by
    another program, or cannot be verified (for example without psutil).
    """
    if not pid:
        return None
    if create_time is not None:
        return open_verified_process(pid, expected_create_time=create_time)
    return open_verified_process(pid, name_check=_is_cloudflared_name)


class CloudflareTunnelManager:
    def __init__(self):
        self.process = None
        self.pid = None
        self.pid_create_time = None
        self.public_url = None
        self.last_error = None
        self.log_file = None
        self.status = "stopped"
        self.previous_master_host = None

        self._lock = asyncio.Lock()
        self._reader = ProcessReader()
        self.binary_path = None

        self._restore_state()

    @property
    def base_dir(self):
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def _restore_state(self):
        state = load_tunnel_state()

        self.public_url = state.get("public_url") or None
        self.previous_master_host = state.get("previous_master_host")
        self.log_file = state.get("log_file")
        pid = state.get("pid")
        create_time = state.get("pid_create_time")

        restored = _open_tunnel_process(pid, create_time)
        if restored is not None:
            self.pid = pid
            self.pid_create_time = create_time if create_time is not None else restored.create_time()
            self.status = state.get("status") or "running"
            debug_log(f"Detected existing cloudflared process (pid={pid})")
        else:
            # Gone, or the PID now belongs to another process: forget it, never adopt it.
            # Only the master owns the shared tunnel state; a local worker (same
            # config file) must never rewrite the master's tunnel record.
            if not os.environ.get("COMFYUI_IS_WORKER"):
                clear_tunnel_state(log_file=self.log_file, previous_host=self.previous_master_host)
            self.status = "stopped"
            self.pid = None
            self.pid_create_time = None

    async def start_tunnel(self):
        async with self._lock:
            if self.process and self.process.poll() is None:
                return {
                    "status": self.status,
                    "public_url": self.public_url,
                    "pid": self.process.pid,
                    "log_file": self.log_file,
                }

            if self.pid:
                debug_log(f"Stopping stale cloudflared pid {self.pid} before starting a new one")
                # Already holding the lock: use the lock-free variant (the lock is not reentrant).
                self._stop_locked()

            binary = await asyncio.to_thread(ensure_binary)
            self.binary_path = binary
            port = get_server_port()
            self.status = "starting"
            self.last_error = None
            self.public_url = None

            state = load_tunnel_state()
            master_host = state.get("master_host") or ""
            if state.get("previous_master_host"):
                self.previous_master_host = state.get("previous_master_host")
            else:
                self.previous_master_host = master_host

            os.makedirs(os.path.join(self.base_dir, "logs"), exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file = os.path.join(self.base_dir, "logs", f"cloudflare-{timestamp}.log")

            cmd = [
                binary,
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://127.0.0.1:{port}",
            ]

            debug_log(f"Starting cloudflared: {' '.join(cmd)}")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                self.status = "error"
                raise RuntimeError("cloudflared binary not found")
            except Exception as exc:
                self.status = "error"
                raise RuntimeError(f"Failed to start cloudflared: {exc}") from exc

            self.pid = self.process.pid
            self.pid_create_time = get_process_create_time(self.pid)
            try:
                persist_tunnel_state(
                    status="starting",
                    pid=self.pid,
                    pid_create_time=self.pid_create_time,
                    log_file=self.log_file,
                    previous_host=self.previous_master_host,
                )

                loop = asyncio.get_running_loop()
                self._reader.set_log_file(self.log_file)
                self._reader.start(self.process, loop)

                try:
                    await self._reader.wait_for_url(timeout=TUNNEL_START_TIMEOUT)
                except asyncio.TimeoutError:
                    failure = "Timed out waiting for Cloudflare to assign a URL"
                else:
                    public_url = self._reader.get_url()
                    failure = None if public_url else (
                        self._reader.get_last_error() or "Cloudflare tunnel failed to start"
                    )
            except BaseException:
                # Any other failure or cancellation must not leave the spawned
                # cloudflared running untracked.
                self._stop_locked()
                raise

            if failure:
                # Terminate the cloudflared this start spawned, but keep the error visible.
                self._stop_locked()
                self.last_error = failure
                raise RuntimeError(failure)

            self.public_url = public_url
            self.status = "running"
            debug_log(f"Cloudflare tunnel ready at {self.public_url}")

            persist_tunnel_state(
                status="running",
                public_url=self.public_url,
                pid=self.pid,
                pid_create_time=self.pid_create_time,
                log_file=self.log_file,
                previous_host=self.previous_master_host or "",
                master_host=normalize_host(self.public_url),
            )
            return {
                "status": self.status,
                "public_url": self.public_url,
                "pid": self.pid,
                "log_file": self.log_file,
            }

    async def stop_tunnel(self):
        async with self._lock:
            return self._stop_locked()

    def _stop_locked(self):
        """Stop the tunnel; the caller must hold ``self._lock`` (or own the manager exclusively)."""
        pid = self.process.pid if self.process else self.pid
        if not pid:
            clear_tunnel_state(log_file=self.log_file, previous_host=self.previous_master_host)
            self.status = "stopped"
            return {"status": "stopped"}

        debug_log(f"Stopping cloudflared (pid={pid})")
        if self.process:
            terminate_process(self.process, timeout=5)
        else:
            # A restored PID: kill it only if it is still the same cloudflared
            # process. psutil will not signal the handle if the PID is reused.
            restored = _open_tunnel_process(pid, self.pid_create_time)
            if restored is None:
                debug_log(f"PID {pid} is no longer the cloudflared process; forgetting it without killing")
            else:
                _terminate_restored(restored, pid)

        restore_host = resolve_restore_master_host(self.previous_master_host)

        self.status = "stopped"
        self.public_url = None
        self.pid = None
        self.pid_create_time = None
        self.process = None
        self.last_error = None
        self._reader.stop()

        clear_tunnel_state(
            log_file=self.log_file,
            previous_host=self.previous_master_host,
            master_host=restore_host,
        )
        return {"status": "stopped"}

    def _is_tunnel_process_alive(self, pid):
        if self.process and self.process.pid == pid:
            return self.process.poll() is None
        return _open_tunnel_process(pid, self.pid_create_time) is not None

    def get_status(self):
        alive = False
        pid = self.process.pid if self.process else self.pid
        if pid:
            alive = self._is_tunnel_process_alive(pid)
            if not alive and self.status == "running":
                self.status = "stopped"

        return {
            "status": self.status,
            "public_url": self.public_url,
            "pid": pid,
            "log_file": self.log_file,
            "last_error": self.last_error or self._reader.get_last_error(),
            "binary_path": self.binary_path or shutil.which("cloudflared"),
            "recent_logs": self._reader.get_recent_logs()[-20:],
            "previous_master_host": self.previous_master_host,
        }
