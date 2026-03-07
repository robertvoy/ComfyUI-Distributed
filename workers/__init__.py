from .process_manager import WorkerProcessManager
from functools import lru_cache

@lru_cache(maxsize=1)
def _worker_manager() -> WorkerProcessManager:
    return WorkerProcessManager()

def get_worker_manager() -> WorkerProcessManager:
    return _worker_manager()
