import os
import platform
import signal
import subprocess
import time
from typing import Any

from ...utils.config import load_config, save_config
from ...utils.constants import PROCESS_TERMINATION_TIMEOUT, PROCESS_WAIT_TIMEOUT, WORKER_CHECK_INTERVAL
from ...utils.logging import debug_log, log
from ...utils.process import (
    get_python_executable,
    is_process_alive,
    launch_process_with_timeout,
    terminate_process,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    log("psutil not available, using fallback process management")
    PSUTIL_AVAILABLE = False


class ProcessLifecycle:
    """Worker process lifecycle operations operating on manager-owned state."""

    def __init__(self, manager):
        self._manager = manager

    def launch_worker(self, worker_config: dict[str, Any], show_window: bool = False) -> int:
        """Launch a worker process with logging."""
        _ = show_window  # Kept for API compatibility.
        comfy_root = self._manager.find_comfy_root()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_config.get("cuda_device", 0))
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["COMFYUI_MASTER_PID"] = str(os.getpid())
        env["COMFYUI_IS_WORKER"] = "1"

        cmd = self._manager.build_launch_command(worker_config, comfy_root)
        cwd = comfy_root

        log_dir = os.path.join(comfy_root, "logs", "workers")
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
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    f"{project_root}{os.pathsep}{existing_pythonpath}"
                    if existing_pythonpath
                    else project_root
                )
                monitored_cmd = [get_python_executable(), monitor_script] + cmd
                log_handle.write(f"[Worker Monitor] Monitoring master PID: {env['COMFYUI_MASTER_PID']}\n")
                log_handle.flush()
            else:
                monitored_cmd = cmd

            popen_kwargs = {
                "env": env,
                "cwd": cwd,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
            }
            if platform.system() == "Windows":
                popen_kwargs["creationflags"] = 0x08000000
            else:
                popen_kwargs["start_new_session"] = True

            process = launch_process_with_timeout(
                monitored_cmd,
                timeout_seconds=PROCESS_WAIT_TIMEOUT,
                **popen_kwargs,
            )

        worker_id = str(worker_config["id"])
        self._manager.processes[worker_id] = {
            "pid": process.pid,
            "process": process,
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

    def stop_worker(self, worker_id: str | int) -> tuple[bool, str]:
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
                if self._kill_process_tree(pid):
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

    def get_managed_workers(self) -> dict[str, dict[str, Any]]:
        """Get list of workers managed by this process."""
        managed = {}
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
                del self._manager.processes[worker_id]

        return managed

    def cleanup_all(self) -> None:
        """Stop all managed workers (called on shutdown)."""
        for worker_id in list(self._manager.processes.keys()):
            try:
                self.stop_worker(worker_id)
            except Exception as exc:
                log(f"[Distributed] Error stopping worker {worker_id}: {exc}")

        config = load_config()
        config["managed_processes"] = {}
        save_config(config)

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        return is_process_alive(pid)

    def _check_worker_process(self, worker_id: str, proc_info: dict[str, Any]) -> tuple[bool, bool]:
        """Check if a worker process is still running and return status."""
        _ = worker_id  # Signature retained for compatibility with existing callers.
        process = proc_info.get("process")
        pid = proc_info.get("pid")

        if process:
            return process.poll() is None, True
        if pid:
            return self._is_process_running(pid), False
        return False, False

    def _kill_process_tree(self, pid):
        """Kill a process and all its children."""
        if PSUTIL_AVAILABLE:
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)

                debug_log(f"Killing process tree for PID {pid} ({parent.name()})")
                for child in children:
                    debug_log(f"  - Child PID {child.pid} ({child.name()})")

                for child in children:
                    try:
                        debug_log(f"Terminating child {child.pid}")
                        child.terminate()
                    except psutil.NoSuchProcess:
                        debug_log(f"Child process {child.pid} already exited before terminate")

                _, alive = psutil.wait_procs(children, timeout=PROCESS_WAIT_TIMEOUT)
                for child in alive:
                    try:
                        debug_log(f"Force killing child {child.pid}")
                        child.kill()
                    except psutil.NoSuchProcess:
                        debug_log(f"Child process {child.pid} already exited before kill")

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
                debug_log(f"Process {pid} does not exist")
                return False
            except Exception as exc:
                debug_log(f"Error killing process tree: {exc}")

        debug_log("[Distributed] Using OS commands to kill process tree")
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "process", "where", f"ParentProcessId={pid}", "get", "ProcessId"],
                    capture_output=True,
                    text=True,
                    timeout=PROCESS_WAIT_TIMEOUT,
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
                                timeout=PROCESS_WAIT_TIMEOUT,
                            )
                        except (FileNotFoundError, OSError) as exc:
                            debug_log(f"[Distributed] Warning: taskkill failed for PID {child_pid}: {exc}")

                result = subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid), "/T"],
                    capture_output=True,
                    text=True,
                    timeout=PROCESS_WAIT_TIMEOUT,
                )
                debug_log(f"[Distributed] Taskkill result: {result.stdout.strip()}")
                return result.returncode == 0
            except Exception as exc:
                log(f"[Distributed] Error with taskkill: {exc}")
                return False

        try:
            subprocess.run(
                ["pkill", "-TERM", "-P", str(pid)],
                check=False,
                timeout=PROCESS_WAIT_TIMEOUT,
            )
            time.sleep(WORKER_CHECK_INTERVAL)
            subprocess.run(
                ["pkill", "-KILL", "-P", str(pid)],
                check=False,
                timeout=PROCESS_WAIT_TIMEOUT,
            )
            os.kill(pid, signal.SIGKILL)
            return True
        except Exception as exc:
            log(f"[Distributed] Error killing process tree for PID {pid}: {exc}")
            return False
