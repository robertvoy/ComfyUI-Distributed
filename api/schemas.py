from __future__ import annotations

from typing import Any

from ..utils.auth import (
    AUTH_HEADER_NAME,
    distributed_auth_headers,
    is_authorized_request,
)
from ..utils.parsing import coerce_bool, require_bool_literal

__all__ = [
    "AUTH_HEADER_NAME",
    "coerce_bool",
    "coerce_positive_float",
    "coerce_positive_int",
    "distributed_auth_headers",
    "is_authorized_request",
    "require_bool_literal",
    "require_fields",
    "require_worker_id",
    "require_positive_int",
    "validate_worker_id",
]


def require_fields(data: dict, *fields) -> list[str]:
    """Return field names that are missing or empty in a JSON object."""
    if not isinstance(data, dict):
        return list(fields)

    missing = []
    for field in fields:
        if field not in data:
            missing.append(field)
            continue
        value = data.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)

    return missing


def validate_worker_id(worker_id: str, config: dict[str, Any]) -> bool:
    """Return True when worker_id exists in config['workers']."""
    try:
        require_worker_id(worker_id, config)
        return True
    except ValueError:
        return False


def require_worker_id(worker_id: str, config: dict[str, Any], field_name: str = "worker_id") -> str:
    """Strict worker-id validator for request boundaries."""
    worker_id_str = str(worker_id).strip()
    if not worker_id_str:
        raise ValueError(f"Field '{field_name}' must be a non-empty worker id.")

    workers = (config or {}).get("workers", [])
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        if str(worker.get("id")).strip() == worker_id_str:
            return worker_id_str
    raise ValueError(f"Worker {worker_id_str} not found.")


def require_positive_int(value, field_name: str = "value") -> int:
    """Strict positive-int parser for request boundary validation."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"Field '{field_name}' must be a positive integer.")
    return parsed


def coerce_positive_int(value, default: int) -> int:
    """Permissive positive-int coercion used for non-boundary defaults."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, parsed)


def coerce_positive_float(value, default: float) -> float:
    """Permissive positive-float coercion used for non-boundary defaults."""
    fallback = float(default)
    if fallback <= 0.0:
        fallback = 0.1
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0.0 else fallback
