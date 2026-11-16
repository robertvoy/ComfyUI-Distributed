from ...utils.config import config_update, load_config
from ...utils.logging import debug_log


class ProcessPersistence:
    """Persist and restore manager-owned worker process metadata."""

    def __init__(self, manager):
        self._manager = manager

    def load_processes(self):
        """Load persisted process information from config.

        A saved PID is restored only when it still belongs to the worker that
        was launched (same start id on Linux or creation time, or the worker's
        --port on the command line for entries saved before either was recorded). Entries
        that are gone or cannot be verified are dropped and the config is
        rewritten, so a reused PID is never adopted (and later killed).
        """
        config = load_config()
        managed_processes = config.get("managed_processes", {})
        if not isinstance(managed_processes, dict):
            managed_processes = {}

        dropped = False
        gained_identity = False
        for worker_id, proc_info in managed_processes.items():
            if not isinstance(proc_info, dict):
                dropped = True
                continue
            pid = proc_info.get("pid")
            candidate = {
                "pid": pid,
                "process": None,
                "create_time": proc_info.get("create_time"),
                "start_id": proc_info.get("start_id"),
                "started_at": proc_info.get("started_at"),
                "config": proc_info.get("config"),
                "log_file": proc_info.get("log_file"),
            }
            is_running = False
            if pid:
                is_running, _ = self._manager._check_worker_process(worker_id, candidate)
            if is_running:
                self._manager.processes[worker_id] = candidate
                # Verification may have filled in the start id (Linux): save it now,
                # so the next restart can use it even if this master is killed.
                gained_identity |= candidate.get("start_id") is not None and proc_info.get("start_id") is None
                debug_log(f"[Distributed] Restored worker {worker_id} (PID: {pid})")
            else:
                dropped = True
                if pid:
                    debug_log(
                        f"[Distributed] Worker {worker_id} (PID: {pid}) is no longer running "
                        "or could not be verified; forgetting it"
                    )

        if dropped or gained_identity:
            self.save_processes()

    def save_processes(self):
        """Save process information to config."""
        managed_processes = {}

        for worker_id, proc_info in list(self._manager.processes.items()):
            is_running, _ = self._manager._check_worker_process(worker_id, proc_info)
            if not is_running:
                continue
            managed_processes[worker_id] = {
                "pid": proc_info["pid"],
                "create_time": proc_info.get("create_time"),
                "start_id": proc_info.get("start_id"),
                "started_at": proc_info["started_at"],
                "config": proc_info["config"],
                "log_file": proc_info.get("log_file"),
                "launching": proc_info.get("launching", False),
            }

        # Process checks above are slow; only the load -> change -> save holds the lock.
        config_update(lambda config: config.__setitem__("managed_processes", managed_processes))
