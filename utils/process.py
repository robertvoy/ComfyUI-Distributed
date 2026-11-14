"""
Process management utilities for ComfyUI-Distributed.
"""
import os
import subprocess
import platform
import signal
import sys

try:
    import psutil
except ImportError:  # pragma: no cover - psutil ships with ComfyUI installs
    psutil = None

# A PID plus its creation time identifies one process. Windows and macOS report
# the creation time directly. Linux (psutil) adds ticks since boot to a boot time
# read from /proc/stat, which moves whenever the clock is adjusted (WSL2 resyncs
# with the host; 3-13 s seen), so two readings of one process can be seconds
# apart. Linux therefore also saves a start id that the clock cannot move (see
# get_process_start_id) and checks that first. The tolerance covers rounding
# and the JSON round trip.
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


def _parse_start_ticks(stat):
    """Start time in clock ticks since boot: field 22 of /proc/<pid>/stat."""
    # Field 2 (the command name, in parentheses) may hold spaces and ')'.
    fields = stat[stat.rindex(b")") + 1:].split()
    return int(fields[19])  # fields[0] is field 3


def get_process_start_id(pid):
    """Return "<boot id>:<start ticks>" for ``pid`` on Linux, else None.

    Unlike psutil's creation time, this does not depend on the wall clock, so
    it stays identical across clock adjustments and between processes, and
    it never repeats for another process (the boot id changes on reboot).
    """
    if not sys.platform.startswith("linux") or not pid:
        return None
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as handle:
            boot_id = handle.read().strip()
        with open(f"/proc/{int(pid)}/stat", "rb") as handle:
            ticks = _parse_start_ticks(handle.read())
    except (OSError, ValueError, IndexError):
        return None
    return f"{boot_id}:{ticks}" if boot_id else None


def open_verified_process(pid, expected_create_time=None, cmdline_check=None, name_check=None,
                          expected_start_id=None):
    """Return a ``psutil.Process`` for ``pid`` only if it is provably the expected process.

    Identity is proven by a saved start id (exact, Linux; checked instead of
    the creation time, and failing closed when it cannot be read) or a saved
    creation time; a saved identity that does not match is final, because
    that is what a reused PID looks like. Only entries saved with neither fall back to
    ``cmdline_check(cmdline)`` or ``name_check(name)``. Returns None when the
    process is gone, psutil is unavailable, or nothing proves the identity,
    so callers never act on a PID that may have been reused. The returned
    object is safe to signal: psutil refuses to signal it if the PID is
    reused afterwards.
    """
    if psutil is None or not pid:
        return None
    try:
        proc = psutil.Process(int(pid))
        if expected_start_id is not None:
            # Exact. Read after opening ``proc``, so a PID reused before this read
            # does not match; a start id that cannot be read here proves nothing.
            return proc if get_process_start_id(pid) == expected_start_id else None
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
