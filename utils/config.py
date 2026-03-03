"""
Configuration management for ComfyUI-Distributed.
"""
import asyncio
import os
import json
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any
from .logging import log

# Import defaults for timeout fallbacks
from .constants import GPU_CONFIG_FILE, HEARTBEAT_TIMEOUT

CONFIG_FILE = GPU_CONFIG_FILE


@dataclass
class _ConfigState:
    cache: dict[str, Any] | None = None
    mtime: float = 0.0


@lru_cache(maxsize=1)
def _config_state() -> _ConfigState:
    return _ConfigState()


@lru_cache(maxsize=1)
def _config_lock() -> asyncio.Lock:
    return asyncio.Lock()


def _config_path() -> str:
    return CONFIG_FILE

def get_default_config() -> dict[str, Any]:
    """Returns the default configuration dictionary. Single source of truth."""
    return {
        "master": {"host": ""},
        "workers": [],
        "settings": {
            "debug": False,
            "auto_launch_workers": False,
            "stop_workers_on_master_exit": True,
            "master_delegate_only": False,
            "websocket_orchestration": True,
            "worker_probe_concurrency": 8,
            "worker_prep_concurrency": 4,
            "media_sync_concurrency": 2,
            "media_sync_timeout_seconds": 120
        },
        "tunnel": {
            "status": "stopped",
            "public_url": "",
            "pid": None,
            "log_file": "",
            "previous_master_host": ""
        }
    }

def _merge_with_defaults(data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge loaded config data with default keys."""
    if not isinstance(data, dict):
        return defaults

    merged = {}
    for key, default_value in defaults.items():
        loaded_value = data.get(key, default_value)
        if isinstance(default_value, dict) and isinstance(loaded_value, dict):
            merged[key] = _merge_with_defaults(loaded_value, default_value)
        else:
            merged[key] = loaded_value

    # Preserve unknown keys for forward compatibility.
    for key, value in data.items():
        if key not in merged:
            merged[key] = value

    return merged


def invalidate_config_cache() -> None:
    """Invalidate in-memory config cache so next load reads from disk."""
    state = _config_state()
    state.cache = None
    state.mtime = 0.0


def load_config() -> dict[str, Any]:
    """Loads the config, falling back to defaults if the file is missing or invalid."""
    state = _config_state()
    path = _config_path()

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        if state.cache is None:
            state.cache = get_default_config()
        return state.cache

    if state.cache is None or mtime != state.mtime:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            state.cache = _merge_with_defaults(loaded, get_default_config())
        except Exception as e:
            log(f"Error loading config, using defaults: {e}")
            state.cache = get_default_config()
        state.mtime = mtime

    return state.cache

def save_config(config: dict[str, Any]) -> bool:
    """Saves the configuration to file."""
    tmp_path = f"{_config_path()}.tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, _config_path())
        invalidate_config_cache()
        return True
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_error:
            log(f"Error removing temporary config file '{tmp_path}': {cleanup_error}")
        log(f"Error saving config: {e}")
        return False


@asynccontextmanager
async def config_transaction() -> AsyncIterator[dict[str, Any]]:
    """Acquire config lock, yield loaded config, and save if changed."""
    async with _config_lock():
        config = load_config()
        original_snapshot = json.dumps(config, sort_keys=True)
        yield config
        updated_snapshot = json.dumps(config, sort_keys=True)
        if updated_snapshot != original_snapshot:
            if not save_config(config):
                raise RuntimeError("Failed to save config")

def ensure_config_exists() -> None:
    """Creates default config file if it doesn't exist. Used by __init__.py"""
    if not os.path.exists(_config_path()):
        default_config = get_default_config()
        if save_config(default_config):
            from .logging import debug_log
            debug_log("Created default config file")
        else:
            log("Could not create default config file")

def get_worker_timeout_seconds(default: int = HEARTBEAT_TIMEOUT) -> int:
    """Return the unified worker timeout (seconds).

    Priority:
    1) UI-configured setting `settings.worker_timeout_seconds`
    2) Fallback to provided `default` (defaults to HEARTBEAT_TIMEOUT which itself
       can be overridden via the COMFYUI_HEARTBEAT_TIMEOUT env var)

    This value should be used anywhere we consider a worker "timed out" from the
    master's perspective (e.g., collector waits, upscaler result collection).
    """
    try:
        cfg = load_config()
        val = int(cfg.get('settings', {}).get('worker_timeout_seconds', default))
        return max(1, val)
    except Exception:
        return max(1, int(default))


def is_master_delegate_only() -> bool:
    """Returns True when master should skip local workload and act as orchestrator only."""
    try:
        cfg = load_config()
        return bool(cfg.get('settings', {}).get('master_delegate_only', False))
    except Exception:
        return False
