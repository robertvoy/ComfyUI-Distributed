from typing import Any

from ..utils.parsing import coerce_bool
from ..utils.worker_ids import coerce_enabled_worker_ids


def _parse_bool(value: Any, default: bool = False) -> bool:
    return coerce_bool(value, default=default)


def _parse_enabled_workers(value: Any) -> list[str]:
    if value is None:
        return []
    return coerce_enabled_worker_ids(value)


def parse_distributed_hidden_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse common distributed hidden inputs from node kwargs."""
    return {
        "multi_job_id": str(kwargs.get("multi_job_id", "") or ""),
        "is_worker": _parse_bool(kwargs.get("is_worker", False)),
        "master_url": str(kwargs.get("master_url", "") or ""),
        "enabled_worker_ids": _parse_enabled_workers(kwargs.get("enabled_worker_ids", "[]")),
        "worker_id": str(kwargs.get("worker_id", "") or ""),
        "delegate_only": _parse_bool(kwargs.get("delegate_only", False)),
    }
