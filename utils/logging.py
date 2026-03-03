"""
Shared logging utilities for ComfyUI-Distributed.
"""
import os
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from .constants import GPU_CONFIG_FILE

_DEBUG_TTL: float = 5.0


@dataclass
class _DebugState:
    enabled: bool | None = None
    updated_at: float = 0.0


@lru_cache(maxsize=1)
def _debug_state() -> _DebugState:
    return _DebugState()

def is_debug_enabled() -> bool:
    """Check if debug is enabled."""
    state = _debug_state()

    now = time.monotonic()
    if state.enabled is not None and (now - state.updated_at) < _DEBUG_TTL:
        return state.enabled

    enabled = state.enabled if state.enabled is not None else False
    if os.path.exists(GPU_CONFIG_FILE):
        try:
            with open(GPU_CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                enabled = bool(config.get("settings", {}).get("debug", False))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            # Fall back to the previous cached value to avoid flapping behavior.
            print(
                "[Distributed] Failed to read debug flag from config; "
                f"using cached value {enabled}: {exc}"
                )

    state.enabled = enabled
    state.updated_at = now
    return enabled

def debug_log(message: str) -> None:
    """Log debug messages only if debug is enabled in config."""
    if is_debug_enabled():
        print(f"[Distributed] {message}")

def log(message: str) -> None:
    """Always log important messages."""
    print(f"[Distributed] {message}")
