from .process.launch_builder import LaunchCommandBuilder
from .process.lifecycle import ProcessLifecycle
from .process.persistence import ProcessPersistence
from .process.root_discovery import ComfyRootDiscovery


class WorkerProcessManager:
    """Thin composition wrapper around worker process subsystems."""

    def __init__(self):
        self.processes = {}
        self._root_discovery = ComfyRootDiscovery()
        self._launch_builder = LaunchCommandBuilder()
        self._lifecycle = ProcessLifecycle(self)
        self._persistence = ProcessPersistence(self)
        self.load_processes()

    def find_comfy_root(self):
        return self._root_discovery.find_comfy_root()

    def _find_windows_terminal(self):
        return self._launch_builder._find_windows_terminal()

    def build_launch_command(self, worker_config, comfy_root):
        return self._launch_builder.build_launch_command(worker_config, comfy_root)

    def launch_worker(self, worker_config, show_window=False):
        return self._lifecycle.launch_worker(worker_config, show_window=show_window)

    def stop_worker(self, worker_id):
        return self._lifecycle.stop_worker(worker_id)

    def get_managed_workers(self):
        return self._lifecycle.get_managed_workers()

    def cleanup_all(self):
        return self._lifecycle.cleanup_all()

    def load_processes(self):
        return self._persistence.load_processes()

    def save_processes(self):
        return self._persistence.save_processes()

    def _is_process_running(self, pid):
        return self._lifecycle._is_process_running(pid)

    def _check_worker_process(self, worker_id, proc_info):
        return self._lifecycle._check_worker_process(worker_id, proc_info)

    def _kill_process_tree(self, pid):
        return self._lifecycle._kill_process_tree(pid)
