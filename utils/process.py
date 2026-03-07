"""
Process management utilities for ComfyUI-Distributed.
"""
import asyncio
import os
import platform
import signal
import subprocess
import threading
from collections.abc import Sequence
from typing import Any


def _run_coroutine_blocking(coro):
    """Run a coroutine in blocking contexts, even if an event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome = {}

    def _runner():
        try:
            outcome["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - thin bridge
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    error = outcome.get("error")
    if error is not None:
        raise error
    return outcome.get("value")


class AsyncProcessAdapter:
    """Sync-friendly adapter around asyncio subprocess.Process."""

    def __init__(self, process):
        self._process = process
        self.pid = process.pid

    @property
    def returncode(self):
        return self._process.returncode

    @property
    def stdout(self):
        return self._process.stdout

    def poll(self):
        return self._process.returncode

    def terminate(self):
        self._process.terminate()

    def kill(self):
        self._process.kill()

    def wait(self, timeout=None):
        async def _wait():
            if timeout is None:
                return await self._process.wait()
            return await asyncio.wait_for(self._process.wait(), timeout=float(timeout))

        return _run_coroutine_blocking(_wait())


def _normalize_stdio_value(value):
    if value is subprocess.PIPE:
        return asyncio.subprocess.PIPE
    if value is subprocess.STDOUT:
        return asyncio.subprocess.STDOUT
    if value is subprocess.DEVNULL:
        return asyncio.subprocess.DEVNULL
    return value


def launch_process_with_timeout(
    command: Sequence[str | bytes | os.PathLike[str]],
    timeout_seconds: float = 10.0,
    **create_kwargs: Any,
) -> AsyncProcessAdapter:
    """Launch a child process with startup timeout and sync-friendly lifecycle methods."""
    if isinstance(command, (str, bytes)):
        raise TypeError("Command must be a sequence, not a string")

    normalized_kwargs = dict(create_kwargs)
    normalized_kwargs["stdout"] = _normalize_stdio_value(normalized_kwargs.get("stdout"))
    normalized_kwargs["stderr"] = _normalize_stdio_value(normalized_kwargs.get("stderr"))

    async def _launch():
        return await asyncio.wait_for(
            asyncio.create_subprocess_exec(*command, **normalized_kwargs),
            timeout=float(timeout_seconds),
        )

    try:
        process = _run_coroutine_blocking(_launch())
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Timed out launching process after {timeout_seconds} seconds") from exc

    return AsyncProcessAdapter(process)

def is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is still alive."""
    try:
        if platform.system() == "Windows":
            # Windows: use tasklist
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return str(pid) in result.stdout
        else:
            # Unix: send signal 0
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False

def terminate_process(process, timeout=5):
    """Gracefully terminate a process with timeout."""
    wait_timeout = max(float(timeout), 0.1)
    if process.poll() is not None:
        return True

    process.terminate()
    try:
        process.wait(timeout=wait_timeout)
        return True
    except (subprocess.TimeoutExpired, TimeoutError):
        process.kill()
        try:
            process.wait(timeout=wait_timeout)
            return True
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise TimeoutError(
                f"Process did not terminate within {wait_timeout} seconds after kill."
            ) from exc

def get_python_executable():
    """Get the Python executable path."""
    import sys
    return sys.executable
