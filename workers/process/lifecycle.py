import os
import platform
import signal
import subprocess
import time

from ...utils.config import load_config, save_config
from ...utils.constants import PROCESS_TERMINATION_TIMEOUT, PROCESS_WAIT_TIMEOUT, WORKER_CHECK_INTERVAL
from ...utils.logging import debug_log, log
from ...utils.network import get_server_port
from ...utils.process import (
    get_process_start_id,
    get_python_executable,
    is_process_alive,
    terminate_process,
)
from ..ports import validate_worker_port

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    log("psutil not available, using fallback process management")
    PSUTIL_AVAILABLE = False

# Same rules as utils.process: creation times can be seconds apart on Linux, so
# Linux entries also carry a clock-independent start id that is checked first.
_CREATE_TIME_TOLERANCE_SECONDS = 2.0


def get_process_create_time(pid):
    """Return the OS creation time of ``pid`` (identifies it with the PID), or None."""
    if not PSUTIL_AVAILABLE or not pid:
        return None
    try:
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def create_times_match(actual, expected):
    try:
        return abs(float(actual) - float(expected)) <= _CREATE_TIME_TOLERANCE_SECONDS
    except (TypeError, ValueError):
        return False


def open_verified_process(pid, expected_create_time=None, cmdline_check=None, expected_start_id=None):
    """Return a psutil handle for ``pid`` only if it is provably the expected process.

    A saved start id (Linux) or creation time proves identity, and a mismatch
    is final. ``cmdline_check`` is only for entries saved with neither.
    psutil also refuses to signal the returned handle if the PID is reused later.
    """
    if not PSUTIL_AVAILABLE or not pid:
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
    except Exception:
        return None
    return None


def worker_log_dir(comfy_root):
    """Directory that holds the log files of UI-launched workers."""
    return os.path.join(comfy_root, "logs", "workers")


def _cmdline_port(cmdline):
    """Return the last --port value in a command line (argparse semantics)."""
    value = None
    for index, arg in enumerate(cmdline):
        if arg == "--port" and index + 1 < len(cmdline):
            value = cmdline[index + 1]
        elif isinstance(arg, str) and arg.startswith("--port="):
            value = arg.split("=", 1)[1]
    return value


def _runs_comfy_main(cmdline):
    """True when the command line launches ComfyUI's main.py (directly or via worker_monitor.py).

    Used with the port for entries saved before creation times were recorded,
    so an unrelated program that reused the PID and happens to take a
    ``--port`` argument is not mistaken for the worker.
    """
    return any(
        isinstance(arg, str) and os.path.basename(arg.replace("\\", "/")).lower() == "main.py"
        for arg in cmdline
    )


def _expected_port(proc_info):
    config = proc_info.get("config")
    if not isinstance(config, dict) or config.get("port") in (None, ""):
        return None
    return str(config.get("port")).strip()


class ProcessLifecycle:
    """Worker process lifecycle operations operating on manager-owned state."""

    def __init__(self, manager):
        self._manager = manager

    def _open_restored_process(self, proc_info):
        """Return a psutil handle for a restored worker PID only if it is still our process.

        Restored entries have no subprocess handle, so their PID may have been
        reused by an unrelated process (after a reboot or crash). The saved
        start id (Linux) or creation time proves identity, and a mismatch is
        final; entries saved by older versions with neither fall back to the
        worker's ``--port`` on the command line. Anything that cannot be
        verified is treated as not ours.
        """
        pid = proc_info.get("pid")
        if not pid:
            return None
        start_id = proc_info.get("start_id")
        create_time = proc_info.get("create_time")
        if start_id is not None or create_time is not None:
            verified = open_verified_process(pid, expected_create_time=create_time, expected_start_id=start_id)
        else:
            port = _expected_port(proc_info)
            if port is None:
                return None
            verified = open_verified_process(
                pid,
                cmdline_check=lambda cmdline: (
                    _runs_comfy_main(cmdline)
                    and str(_cmdline_port(cmdline) or "").strip() == port
                ),
            )
        if verified is not None and start_id is None:
            # Verified another way: record the clock-independent identity (Linux)
            # from now on, but only if the handle still names the verified process
            # after the read (the PID was not reused in between).
            current_id = get_process_start_id(pid)
            if not verified.is_running():
                return None
            proc_info["start_id"] = current_id
        return verified

    def _drop_process(self, worker_id):
        if self._manager.processes.pop(worker_id, None) is not None:
            self._manager.save_processes()

    def launch_worker(self, worker_config, show_window=False):
        """Launch a worker process with logging."""
        _ = show_window  # Kept for API compatibility.
        comfy_root = self._manager.find_comfy_root()
        config = load_config()
        try:
            master_port = get_server_port()
        except AttributeError:
            master_port = None
        if master_port is None:
            # The auto-launch timer can fire during custom-node imports, before
            # PromptServer assigns its port. Still reserve the configured port.
            from comfy.cli_args import args
            master_port = args.port
        validate_worker_port(worker_config, master_port, config.get("workers", []))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_config.get("cuda_device", 0))
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["COMFYUI_MASTER_PID"] = str(os.getpid())
        env["COMFYUI_IS_WORKER"] = "1"

        cmd = self._manager.build_launch_command(worker_config, comfy_root)
        cwd = comfy_root

        log_dir = worker_log_dir(comfy_root)
        os.makedirs(log_dir, exist_ok=True)

        date_stamp = time.strftime("%Y%m%d")
        worker_name = worker_config.get("name", f"Worker{worker_config['id']}")
        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in worker_name)
        log_file = os.path.join(log_dir, f"{safe_name}_{date_stamp}.log")

        with open(log_file, "a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n\n{'=' * 50}\n")
            log_handle.write("=== ComfyUI Worker Session Started ===\n")
            log_handle.write(f"Worker: {worker_name}\n")
            log_handle.write(f"Port: {worker_config['port']}\n")
            log_handle.write(f"CUDA Device: {worker_config.get('cuda_device', 0)}\n")
            log_handle.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write(f"Command: {' '.join(cmd)}\n")

            config = load_config()
            stop_on_master_exit = config.get("settings", {}).get("stop_workers_on_master_exit", True)
            if stop_on_master_exit:
                log_handle.write("Note: Worker will stop when master shuts down\n")
            else:
                log_handle.write("Note: Worker will continue running after master shuts down\n")
            log_handle.write("=" * 30 + "\n\n")
            log_handle.flush()

            if stop_on_master_exit and env.get("COMFYUI_MASTER_PID"):
                monitor_script = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "worker_monitor.py",
                )
                monitored_cmd = [get_python_executable(), monitor_script] + cmd
                log_handle.write(f"[Worker Monitor] Monitoring master PID: {env['COMFYUI_MASTER_PID']}\n")
                log_handle.flush()
            else:
                monitored_cmd = cmd

            if platform.system() == "Windows":
                create_no_window = 0x08000000
                process = subprocess.Popen(
                    monitored_cmd,
                    env=env,
                    cwd=cwd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=create_no_window,
                )
            else:
                process = subprocess.Popen(
                    monitored_cmd,
                    env=env,
                    cwd=cwd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

        worker_id = str(worker_config["id"])
        self._manager.processes[worker_id] = {
            "pid": process.pid,
            "process": process,
            # Saved with the PID so a restored entry can prove it is still this process.
            "create_time": get_process_create_time(process.pid),
            "start_id": get_process_start_id(process.pid),
            "started_at": time.time(),
            "config": worker_config,
            "log_file": log_file,
            "is_monitor": stop_on_master_exit and env.get("COMFYUI_MASTER_PID"),
            "launching": True,
        }

        self._manager.save_processes()

        if stop_on_master_exit and env.get("COMFYUI_MASTER_PID"):
            debug_log(f"Launched worker {worker_name} via monitor (Monitor PID: {process.pid})")
        else:
            log(f"Launched worker {worker_name} directly (PID: {process.pid})")
        debug_log(f"Log file: {log_file}")
        return process.pid

    def stop_worker(self, worker_id):
        """Stop a worker process."""
        worker_id = str(worker_id)
        if worker_id not in self._manager.processes:
            return False, "Worker not managed by UI"

        proc_info = self._manager.processes[worker_id]
        process = proc_info.get("process")
        pid = proc_info["pid"]
        debug_log(f"Attempting to stop worker {worker_id} (PID: {pid})")

        if not process:
            try:
                debug_log("[Distributed] Stopping restored process (no subprocess object)")
                # Re-verify right before killing: a restored PID may now belong to
                # an unrelated process, which must never be killed.
                verified = self._open_restored_process(proc_info)
                if verified is None:
                    log(
                        f"[Distributed] Worker {worker_id} (PID: {pid}) is no longer the process "
                        "launched by the UI; forgetting it without killing anything"
                    )
                    self._drop_process(worker_id)
                    return False, "Worker already stopped"
                # Kill through the handle just verified: psutil refuses to signal it
                # (or list its children) once the PID belongs to another process.
                if self._kill_process_tree(pid, expected_create_time=verified.create_time(), handle=verified):
                    del self._manager.processes[worker_id]
                    self._manager.save_processes()
                    debug_log(f"Successfully stopped worker {worker_id} and all child processes")
                    return True, "Worker stopped"
                return False, "Failed to stop worker process"
            except Exception as exc:
                log(f"[Distributed] Exception during stop: {exc}")
                return False, f"Error stopping worker: {str(exc)}"

        if process.poll() is not None:
            log(f"[Distributed] Worker {worker_id} already stopped")
            del self._manager.processes[worker_id]
            self._manager.save_processes()
            return False, "Worker already stopped"

        try:
            debug_log(f"Using process tree kill for worker {worker_id}")
            # The live subprocess handle pins this PID, so no identity check is needed.
            if self._kill_process_tree(pid):
                del self._manager.processes[worker_id]
                self._manager.save_processes()
                debug_log(f"Successfully stopped worker {worker_id} and all child processes")
                return True, "Worker stopped"

            log("[Distributed] Process tree kill failed, trying normal termination")
            terminate_process(process, timeout=PROCESS_TERMINATION_TIMEOUT)
            del self._manager.processes[worker_id]
            self._manager.save_processes()
            return True, "Worker stopped (fallback)"
        except Exception as exc:
            log(f"[Distributed] Exception during stop: {exc}")
            return False, f"Error stopping worker: {str(exc)}"

    def get_managed_workers(self):
        """Get list of workers managed by this process."""
        managed = {}
        dropped = False
        for worker_id, proc_info in list(self._manager.processes.items()):
            is_running, _ = self._check_worker_process(worker_id, proc_info)
            if is_running:
                managed[worker_id] = {
                    "pid": proc_info["pid"],
                    "started_at": proc_info["started_at"],
                    "log_file": proc_info.get("log_file"),
                    "launching": proc_info.get("launching", False),
                }
            else:
                self._manager.processes.pop(worker_id, None)
                dropped = True

        if dropped:
            # Persist so a stale PID cannot be restored (and reused) after a restart.
            self._manager.save_processes()
        return managed

    def cleanup_all(self):
        """Stop all managed workers (called on shutdown)."""
        for worker_id in list(self._manager.processes.keys()):
            try:
                self.stop_worker(worker_id)
            except Exception as exc:
                log(f"[Distributed] Error stopping worker {worker_id}: {exc}")

        config = load_config()
        config["managed_processes"] = {}
        save_config(config)

    def _is_process_running(self, pid):
        """Check if a process with given PID is running."""
        return is_process_alive(pid)

    def _check_worker_process(self, worker_id, proc_info):
        """Check if a worker process is still running and return status."""
        _ = worker_id  # Signature retained for compatibility with existing callers.
        process = proc_info.get("process")
        pid = proc_info.get("pid")

        if process:
            return process.poll() is None, True
        if pid:
            # A bare PID only counts when it is provably still the launched process.
            return self._open_restored_process(proc_info) is not None, False
        return False, False

    def _kill_process_tree(self, pid, expected_create_time=None, handle=None):
        """Kill a process and all its children.

        With ``expected_create_time`` the tree is killed only if ``pid`` still
        belongs to the process created at that time (not a reused PID). A
        verified psutil ``handle`` is used instead of looking the PID up again.
        """
        if PSUTIL_AVAILABLE:
            try:
                parent = handle if handle is not None else psutil.Process(pid)
                # psutil refuses to signal ``parent`` if the PID is reused after this check.
                if expected_create_time is not None and not create_times_match(
                    parent.create_time(), expected_create_time
                ):
                    log(f"[Distributed] PID {pid} no longer belongs to the worker process; not killing it")
                    return False
                children = parent.children(recursive=True)

                debug_log(f"Killing process tree for PID {pid} ({parent.name()})")
                for child in children:
                    debug_log(f"  - Child PID {child.pid} ({child.name()})")

                for child in children:
                    try:
                        debug_log(f"Terminating child {child.pid}")
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                _, alive = psutil.wait_procs(children, timeout=PROCESS_WAIT_TIMEOUT)
                for child in alive:
                    try:
                        debug_log(f"Force killing child {child.pid}")
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

                try:
                    debug_log(f"Terminating parent {pid}")
                    parent.terminate()
                    parent.wait(timeout=PROCESS_WAIT_TIMEOUT)
                except psutil.TimeoutExpired:
                    debug_log(f"Force killing parent {pid}")
                    parent.kill()
                except psutil.NoSuchProcess:
                    debug_log(f"Parent process {pid} already gone")
                return True
            except psutil.NoSuchProcess:
                # It exited on its own (for example between the identity check
                # and this kill): the worker is stopped.
                debug_log(f"Process {pid} does not exist")
                return True
            except Exception as exc:
                debug_log(f"Error killing process tree: {exc}")
                if expected_create_time is not None:
                    # The OS commands below cannot verify the PID's identity.
                    return False
        elif expected_create_time is not None:
            # Identity cannot be checked without psutil; never kill a possibly reused PID.
            return False

        debug_log("[Distributed] Using OS commands to kill process tree")
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "process", "where", f"ParentProcessId={pid}", "get", "ProcessId"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")[1:]
                    child_pids = [line.strip() for line in lines if line.strip().isdigit()]
                    debug_log(f"[Distributed] Found child processes: {child_pids}")
                    for child_pid in child_pids:
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", child_pid],
                                capture_output=True,
                                check=False,
                            )
                        except (FileNotFoundError, OSError) as exc:
                            debug_log(f"[Distributed] Warning: taskkill failed for PID {child_pid}: {exc}")

                result = subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid), "/T"],
                    capture_output=True,
                    text=True,
                )
                debug_log(f"[Distributed] Taskkill result: {result.stdout.strip()}")
                return result.returncode == 0
            except Exception as exc:
                log(f"[Distributed] Error with taskkill: {exc}")
                return False

        try:
            subprocess.run(["pkill", "-TERM", "-P", str(pid)], check=False)
            time.sleep(WORKER_CHECK_INTERVAL)
            subprocess.run(["pkill", "-KILL", "-P", str(pid)], check=False)
            os.kill(pid, signal.SIGKILL)
            return True
        except Exception as exc:
            log(f"[Distributed] Error killing process tree for PID {pid}: {exc}")
            return False
