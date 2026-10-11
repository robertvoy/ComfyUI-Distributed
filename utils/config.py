"""
Configuration management for ComfyUI-Distributed.
"""
import asyncio
import copy
import os
import json
from contextlib import asynccontextmanager
from .logging import log

# Import defaults for timeout fallbacks
from .constants import HEARTBEAT_TIMEOUT

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpu_config.json")
_config_cache = None
_config_mtime = 0.0
_config_lock = asyncio.Lock()

_SERVERLESS_OBSOLETE_WORKER_FIELDS = {"host", "port", "listen_port", "cuda_device", "extra_args"}


def _contains_serverless_secret(value):
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and _is_secret_field_name(key))
            or _contains_serverless_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_serverless_secret(item) for item in value)
    return False


def _is_secret_field_name(key):
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return normalized in {"apikey", "clearapikey"} or "credential" in normalized


def _normalize_serverless_workers(config):
    """Remove obsolete local-worker fields without adding local defaults."""
    if not isinstance(config, dict):
        return config
    workers = config.get("workers")
    if isinstance(workers, list):
        for worker in workers:
            if isinstance(worker, dict) and worker.get("type") == "serverless":
                for field in _SERVERLESS_OBSOLETE_WORKER_FIELDS:
                    worker.pop(field, None)
    return config


def _validated_config_for_save(config):
    """Return a safe normalized copy ready for gpu_config.json."""
    if _contains_serverless_secret(config):
        raise ValueError("API keys must be stored outside the config file")
    saved = _normalize_serverless_workers(copy.deepcopy(config))
    workers = saved.get("workers") if isinstance(saved, dict) else None
    if isinstance(workers, list) and any(
        isinstance(worker, dict) and worker.get("type") == "serverless" for worker in workers
    ):
        from .serverless_config import validate_serverless_worker
        for index, worker in enumerate(workers):
            if isinstance(worker, dict) and worker.get("type") == "serverless":
                saved_worker = validate_serverless_worker(worker)
                for field in _SERVERLESS_OBSOLETE_WORKER_FIELDS:
                    saved_worker.pop(field, None)
                workers[index] = saved_worker
    return saved


def _config_path():
    return CONFIG_FILE

def get_default_config():
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

def _merge_with_defaults(data, defaults):
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

    return _normalize_serverless_workers(merged)


def invalidate_config_cache():
    """Invalidate in-memory config cache so next load reads from disk."""
    global _config_cache, _config_mtime
    _config_cache = None
    _config_mtime = 0.0


def load_config():
    """Loads the config, falling back to defaults if the file is missing or invalid."""
    global _config_cache, _config_mtime
    path = _config_path()

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        if _config_cache is None:
            _config_cache = get_default_config()
        return _config_cache

    if _config_cache is None or mtime != _config_mtime:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            _config_cache = _merge_with_defaults(loaded, get_default_config())
        except Exception as e:
            log(f"Error loading config, using defaults: {e}")
            _config_cache = get_default_config()
        _config_mtime = mtime

    return _config_cache

def save_config(config):
    """Saves normalized configuration without allowing API keys in the config file."""
    tmp_path = f"{_config_path()}.tmp"
    try:
        saved_config = _validated_config_for_save(config)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(saved_config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, _config_path())
        invalidate_config_cache()
        return True
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        log(f"Error saving config: {e}")
        return False


@asynccontextmanager
async def config_transaction():
    """Acquire config lock, yield loaded config, and save if changed."""
    async with _config_lock:
        config = load_config()
        original_config = copy.deepcopy(config)
        original_snapshot = json.dumps(config, sort_keys=True)
        try:
            yield config
        except BaseException:
            if isinstance(config, dict):
                config.clear()
                config.update(original_config)
            raise
        updated_snapshot = json.dumps(config, sort_keys=True)
        if updated_snapshot != original_snapshot and not save_config(config):
            if isinstance(config, dict):
                config.clear()
                config.update(original_config)
            raise RuntimeError("Failed to save config")

def ensure_config_exists():
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
