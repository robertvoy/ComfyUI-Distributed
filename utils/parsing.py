from __future__ import annotations

from typing import Any


def require_bool_literal(value: Any, *, field_name: str = "value") -> bool:
    """Strict boolean parser for API boundaries."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Field '{field_name}' must be a boolean literal ('true' or 'false').")


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Permissive boolean parser for internal/runtime values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return default
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default
