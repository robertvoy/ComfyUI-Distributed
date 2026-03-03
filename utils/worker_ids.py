import json
from typing import Any


def parse_enabled_worker_ids(enabled_worker_ids: str | list[Any] | None) -> list[str]:
    """Parse and de-duplicate enabled worker IDs from a JSON list payload."""
    try:
        raw = json.loads(enabled_worker_ids) if isinstance(enabled_worker_ids, str) else enabled_worker_ids
    except Exception:
        raw = []

    workers = []
    seen = set()
    for worker_id in raw if isinstance(raw, list) else []:
        worker_id_str = str(worker_id)
        if worker_id_str in seen:
            continue
        seen.add(worker_id_str)
        workers.append(worker_id_str)
    return workers
