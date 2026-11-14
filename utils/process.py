"""
Process management utilities for ComfyUI-Distributed.
"""
import os
import subprocess
import platform
import signal

try:
    import psutil
except ImportError:  # pragma: no cover - psutil ships with ComfyUI installs
    psutil = None

# Two readings of one process's creation time can differ slightly: Linux derives
# it from the boot time in whole seconds, which can move by a second between
# readings (or when the clock is adjusted), and the value round-trips through
# JSON. A process that reused the PID started after the saved one exited, so
# it only falls inside this window if the saved process died within seconds.
CREATE_TIME_TOLERANCE_SECONDS = 2.0


def is_process_alive(pid):
    """Check if a process with given PID is still alive."""
    try:
        if platform.system() == "Windows":
            # Windows: use tasklist
            result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                                  capture_output=True, text=True)
            return str(pid) in result.stdout
        else:
            # Unix: send signal 0
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False


def get_process_create_time(pid):
    """Return the OS creation time of ``pid``, or None when it cannot be read.

    Together with the PID this identifies one specific process: a later
    process that reuses the PID has a different creation time.
    """
    if psutil is None or not pid:
        return None
    try:
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def create_times_match(actual, expected):
    """True when two process creation times describe the same process."""
    try:
        return abs(float(actual) - float(expected)) <= CREATE_TIME_TOLERANCE_SECONDS
    except (TypeError, ValueError):
        return False


def open_verified_process(pid, expected_create_time=None, cmdline_check=None, name_check=None):
    """Return a ``psutil.Process`` for ``pid`` only if it is provably the expected process.

    Identity is proven by a matching creation time when one was saved;
    otherwise by ``cmdline_check(cmdline)`` or ``name_check(name)``. Returns
    None when the process is gone, psutil is unavailable, or nothing proves
    the identity, so callers never act on a PID that may have been reused.
    The returned object is safe to signal: psutil refuses to signal it if the
    PID is reused afterwards.
    """
    if psutil is None or not pid:
        return None
    try:
        proc = psutil.Process(int(pid))
        if expected_create_time is not None:
            return proc if create_times_match(proc.create_time(), expected_create_time) else None
        if cmdline_check is not None and cmdline_check(list(proc.cmdline() or [])):
            return proc
        if name_check is not None and name_check(proc.name() or ""):
            return proc
    except Exception:
        return None
    return None


def terminate_process(process, timeout=5):
    """Gracefully terminate a process with timeout."""
    if process.poll() is None:  # Still running
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

def get_python_executable():
    """Get the Python executable path."""
    import sys
    return sys.executable
