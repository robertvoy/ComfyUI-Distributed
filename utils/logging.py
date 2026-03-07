"""
Shared logging utilities for ComfyUI-Distributed.
"""
from .config import get_debug_enabled


def is_debug_enabled() -> bool:
    """Check whether debug logging is currently enabled."""
    try:
        return get_debug_enabled(default=False)
    except Exception:
        return False

def debug_log(message: str) -> None:
    """Log debug messages only if debug is enabled in config."""
    if is_debug_enabled():
        print(f"[Distributed] {message}")

def log(message: str) -> None:
    """Always log important messages."""
    print(f"[Distributed] {message}")
