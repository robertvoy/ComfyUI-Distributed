from ...utils.config import load_config, save_config
from ...utils.logging import debug_log


class ProcessPersistence:
    """Persist and restore manager-owned worker process metadata."""

    def __init__(self, manager):
        self._manager = manager

    def load_processes(self):
        """Load persisted process information from config.

        A saved PID is restored only when it still belongs to the worker that
        was launched (same creation time, or the worker's --port on the command
        line for entries saved before creation times were recorded). Entries
        that are gone or cannot be verified are dropped and the config is
        rewritten, so a reused PID is never adopted (and later killed).
        """
        config = load_config()
        managed_processes = config.get("managed_processes", {})
        if not isinstance(managed_processes, dict):
            managed_processes = {}

        dropped = False
        for worker_id, proc_info in managed_processes.items():
            if not isinstance(proc_info, dict):
                dropped = True
                continue
            pid = proc_info.get("pid")
            candidate = {
                "pid": pid,
                "process": None,
                "create_time": proc_info.get("create_time"),
                "started_at": proc_info.get("started_at"),
                "config": proc_info.get("config"),
                "log_file": proc_info.get("log_file"),
            }
            is_running = False
            if pid:
                is_running, _ = self._manager._check_worker_process(worker_id, candidate)
            if is_running:
                self._manager.processes[worker_id] = candidate
                debug_log(f"[Distributed] Restored worker {worker_id} (PID: {pid})")
            else:
                dropped = True
                if pid:
                    debug_log(
                        f"[Distributed] Worker {worker_id} (PID: {pid}) is no longer running "
                        "or could not be verified; forgetting it"
                    )

        if dropped:
            self.save_processes()

    def save_processes(self):
        """Save process information to config."""
        config = load_config()
        managed_processes = {}

        for worker_id, proc_info in list(self._manager.processes.items()):
            is_running, _ = self._manager._check_worker_process(worker_id, proc_info)
            if not is_running:
                continue
            managed_processes[worker_id] = {
                "pid": proc_info["pid"],
                "create_time": proc_info.get("create_time"),
                "started_at": proc_info["started_at"],
                "config": proc_info["config"],
                "log_file": proc_info.get("log_file"),
                "launching": proc_info.get("launching", False),
            }

        config["managed_processes"] = managed_processes
        save_config(config)
