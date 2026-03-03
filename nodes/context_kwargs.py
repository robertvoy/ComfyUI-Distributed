from typing import Any


def parse_distributed_hidden_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse common distributed hidden inputs from node kwargs."""
    return {
        "multi_job_id": str(kwargs.get("multi_job_id", "")),
        "is_worker": bool(kwargs.get("is_worker", False)),
        "master_url": str(kwargs.get("master_url", "")),
        "enabled_worker_ids": str(kwargs.get("enabled_worker_ids", "[]")),
        "worker_id": str(kwargs.get("worker_id", "")),
        "delegate_only": bool(kwargs.get("delegate_only", False)),
    }
