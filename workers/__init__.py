from .process_manager import WorkerProcessManager
from functools import lru_cache

# Keep monitor module importable via workers package for subprocess integration.
from . import worker_monitor as worker_monitor  # noqa: F401


@lru_cache(maxsize=1)
def _worker_manager() -> WorkerProcessManager:
    manager = WorkerProcessManager()
    manager.queues = {}
    return manager

def get_worker_manager() -> WorkerProcessManager:
    return _worker_manager()
