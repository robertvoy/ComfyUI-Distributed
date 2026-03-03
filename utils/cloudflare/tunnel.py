"""Cloudflare tunnel lifecycle manager."""

import asyncio
import os
import shutil
import signal
import time
from typing import Any

from ..constants import PROCESS_TERMINATION_TIMEOUT, TUNNEL_START_TIMEOUT
from ..logging import debug_log
from ..network import get_server_port, normalize_host
from ..process import is_process_alive, terminate_process
from .binary import ensure_binary
from .process_reader import ProcessReader
from .state import (
    TunnelStateUpdate,
    clear_tunnel_state,
    load_tunnel_state,
    persist_tunnel_state,
    resolve_restore_master_host,
)


class CloudflareTunnelManager:
    def __init__(self):
        self.process = None
        self.pid = None
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

    @staticmethod
    def _build_local_tunnel_url(port: int) -> str:
        scheme = (os.environ.get("CLOUDFLARE_LOCAL_BIND_SCHEME") or "http").strip() or "http"
        host = (os.environ.get("CLOUDFLARE_LOCAL_BIND_HOST") or "127.0.0.1").strip() or "127.0.0.1"
        return f"{scheme}://{host}:{port}"

    def _restore_state(self):
        state = load_tunnel_state()

        self.public_url = state.get("public_url") or None
        self.previous_master_host = state.get("previous_master_host")
        self.log_file = state.get("log_file")
        pid = state.get("pid")

        if pid and is_process_alive(pid):
            self.pid = pid
            self.status = state.get("status") or "running"
            debug_log(f"Detected existing cloudflared process (pid={pid})")
        else:
            clear_tunnel_state(log_file=self.log_file, previous_host=self.previous_master_host)
            self.status = "stopped"
            self.pid = None

    @staticmethod
    def _process_is_running(process):
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
        return getattr(process, "returncode", None) is None

    @staticmethod
    async def _launch_process_with_timeout(command, timeout_seconds, **create_kwargs):
        """Launch subprocess with bounded startup wait."""
        try:
            return await asyncio.wait_for(
                asyncio.create_subprocess_exec(*command, **create_kwargs),
                timeout=float(timeout_seconds),
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Timed out launching cloudflared after {timeout_seconds} seconds"
            ) from exc

    @staticmethod
    async def _terminate_process_with_timeout(process, timeout_seconds):
        """Terminate process object with timeout for both sync and async process types."""
        poll = getattr(process, "poll", None)
        if callable(poll):
            terminate_process(process, timeout=timeout_seconds)
            return

        if getattr(process, "returncode", None) is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=float(timeout_seconds))
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def start_tunnel(self) -> dict[str, Any]:
        async with self._lock:
            if self.process and self._process_is_running(self.process):
                return {
                    "status": self.status,
                    "public_url": self.public_url,
                    "pid": self.process.pid,
                    "log_file": self.log_file,
                }

            if self.pid and is_process_alive(self.pid):
                debug_log(f"Stopping stale cloudflared pid {self.pid} before starting a new one")
                await self.stop_tunnel()

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
                self._build_local_tunnel_url(port),
            ]

            debug_log(f"Starting cloudflared: {' '.join(cmd)}")
            try:
                self.process = await self._launch_process_with_timeout(
                    cmd,
                    timeout_seconds=10.0,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except FileNotFoundError:
                self.status = "error"
                raise RuntimeError("cloudflared binary not found")
            except Exception as exc:
                self.status = "error"
                raise RuntimeError(f"Failed to start cloudflared: {exc}") from exc

            self.pid = self.process.pid
            persist_tunnel_state(
                TunnelStateUpdate(
                    status="starting",
                    pid=self.pid,
                    log_file=self.log_file,
                    previous_host=self.previous_master_host,
                )
            )

            loop = asyncio.get_running_loop()
            self._reader.set_log_file(self.log_file)
            self._reader.start(self.process, loop)

            try:
                await self._reader.wait_for_url(timeout=TUNNEL_START_TIMEOUT)
            except asyncio.TimeoutError:
                self.last_error = "Timed out waiting for Cloudflare to assign a URL"
                await self.stop_tunnel()
                raise RuntimeError(self.last_error)

            public_url = self._reader.get_url()
            if not public_url:
                self.last_error = self._reader.get_last_error() or "Cloudflare tunnel failed to start"
                await self.stop_tunnel()
                raise RuntimeError(self.last_error)

            self.public_url = public_url
            self.status = "running"
            debug_log(f"Cloudflare tunnel ready at {self.public_url}")

            persist_tunnel_state(
                TunnelStateUpdate(
                    status="running",
                    public_url=self.public_url,
                    pid=self.pid,
                    log_file=self.log_file,
                    previous_host=self.previous_master_host or "",
                    master_host=normalize_host(self.public_url),
                )
            )
            return {
                "status": self.status,
                "public_url": self.public_url,
                "pid": self.pid,
                "log_file": self.log_file,
            }

    async def stop_tunnel(self) -> dict[str, Any]:
        async with self._lock:
            pid = self.process.pid if self.process else self.pid
            if not pid:
                clear_tunnel_state(log_file=self.log_file, previous_host=self.previous_master_host)
                self.status = "stopped"
                return {"status": "stopped"}

            debug_log(f"Stopping cloudflared (pid={pid})")
            if self.process:
                await self._terminate_process_with_timeout(
                    self.process,
                    timeout_seconds=PROCESS_TERMINATION_TIMEOUT,
                )
            else:
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                except Exception as exc:  # pragma: no cover
                    debug_log(f"Error stopping cloudflared pid {pid}: {exc}")

            restore_host = resolve_restore_master_host(self.previous_master_host)

            self.status = "stopped"
            self.public_url = None
            self.pid = None
            self.process = None
            self.last_error = None
            self._reader.stop()

            clear_tunnel_state(
                log_file=self.log_file,
                previous_host=self.previous_master_host,
                master_host=restore_host,
            )
            return {"status": "stopped"}

    def get_status(self) -> dict[str, Any]:
        alive = False
        pid = self.process.pid if self.process else self.pid
        if pid:
            alive = is_process_alive(pid)
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
