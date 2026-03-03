from ...utils.config import load_config, save_config
from ...utils.logging import debug_log


class ProcessPersistence:
    """Persist and restore manager-owned worker process metadata."""

    def __init__(self, manager):
        self._manager = manager

    def load_processes(self) -> None:
        """Load persisted process information from config."""
        config = load_config()
        managed_processes = config.get("managed_processes", {})

        for worker_id, proc_info in managed_processes.items():
            pid = proc_info.get("pid")
            if pid and self._manager._is_process_running(pid):
                self._manager.processes[worker_id] = {
                    "pid": pid,
                    "process": None,
                    "started_at": proc_info.get("started_at"),
                    "config": proc_info.get("config"),
                    "log_file": proc_info.get("log_file"),
                }
                debug_log(f"[Distributed] Restored worker {worker_id} (PID: {pid})")
            elif pid:
                debug_log(f"[Distributed] Worker {worker_id} (PID: {pid}) is no longer running")

    def save_processes(self) -> None:
        """Save process information to config."""
        config = load_config()
        managed_processes = {}

        for worker_id, proc_info in self._manager.processes.items():
            is_running, _ = self._manager._check_worker_process(worker_id, proc_info)
            if not is_running:
                continue
            managed_processes[worker_id] = {
                "pid": proc_info["pid"],
                "started_at": proc_info["started_at"],
                "config": proc_info["config"],
                "log_file": proc_info.get("log_file"),
                "launching": proc_info.get("launching", False),
            }

        config["managed_processes"] = managed_processes
        save_config(config)
