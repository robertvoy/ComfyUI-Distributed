"""
Configuration management for ComfyUI-Distributed.
"""
import asyncio
import copy
import os
import json
import math
import stat
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from .logging import log
from .secret_fields import is_secret_field_name as _is_secret_field_name

# Import defaults for timeout fallbacks
from .constants import HEARTBEAT_TIMEOUT

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpu_config.json")
_config_cache = None
_config_mtime = 0.0
_config_lock = asyncio.Lock()
# Serialises every config file read/write in this process, across threads (route
# handlers on the event loop, executor threads, timer threads) and the async
# config_transaction. Re-entrant so a transaction (or a signal handler running on
# a thread that already holds it) can call load_config/save_config.
_config_io_lock = threading.RLock()

# os.replace on Windows fails while another handle has the target open
# (ERROR_ACCESS_DENIED=5, ERROR_SHARING_VIOLATION=32), e.g. another process
# reading gpu_config.json. Retry briefly before giving up. Saves often run on
# the event loop thread while holding _config_io_lock, so the whole retry
# budget stays short (about a third of a second) to bound how long a stuck
# file can stall the server.
_REPLACE_RETRY_WINERRORS = (5, 32)
_REPLACE_RETRY_DELAYS = (0.01, 0.02, 0.05, 0.1, 0.15)
_IO_LOCK_POLL_INTERVAL = 0.005


def _read_umask():
    # os.umask can only be read by setting it; done once, at import.
    mask = os.umask(0)
    os.umask(mask)
    return mask


# Permissions open() would give a newly created file.
_NEW_FILE_MODE = 0o666 & ~_read_umask()

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
            "media_sync_timeout_seconds": 120,
            "image_list_job_timeout_seconds": 0.0
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

    with _config_io_lock:
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


def _replace_with_retry(src, dst):
    """os.replace, retrying briefly on transient Windows sharing/access errors."""
    for delay in _REPLACE_RETRY_DELAYS:
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            if getattr(e, "winerror", None) not in _REPLACE_RETRY_WINERRORS:
                raise
            time.sleep(delay)
    os.replace(src, dst)


def _write_config_file(path, saved_config):
    """Atomically write JSON via a unique temp file in the target directory."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(saved_config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates 0600 files: keep the existing file's permissions,
        # or give a new file the usual ones.
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            mode = _NEW_FILE_MODE
        try:
            os.chmod(tmp_path, mode)
        except OSError:
            pass
        _replace_with_retry(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_config(config):
    """Saves normalized configuration without allowing API keys in the config file."""
    with _config_io_lock:
        try:
            saved_config = _validated_config_for_save(config)
            _write_config_file(_config_path(), saved_config)
            invalidate_config_cache()
            return True
        except Exception as e:
            log(f"Error saving config: {e}")
            return False


async def _acquire_config_io_lock():
    """Take the thread lock without blocking the event loop while a thread holds it."""
    while not _config_io_lock.acquire(blocking=False):
        await asyncio.sleep(_IO_LOCK_POLL_INTERVAL)


@asynccontextmanager
async def config_transaction():
    """Acquire config lock, yield loaded config, and save if changed.

    The load -> mutate -> save sequence holds both the asyncio lock (other
    coroutines) and the module thread lock (executor/timer threads calling
    load_config/save_config), so no other save can land in between. The body
    must not await work on another thread that itself loads or saves config.
    """
    async with _config_lock:
        await _acquire_config_io_lock()
        try:
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
        finally:
            _config_io_lock.release()

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


def get_image_list_job_timeout_seconds(config=None) -> float:
    """Return the validated image-list total-job timeout in seconds.

    A value of zero disables the total-job deadline. Missing settings default
    to zero; configured values must be finite, non-negative numbers.
    """
    setting_key = "settings.image_list_job_timeout_seconds"
    cfg = load_config() if config is None else config
    if not isinstance(cfg, dict):
        raise ValueError(f"{setting_key} must be a finite non-negative number")

    settings = cfg.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError(f"{setting_key} must be a finite non-negative number")

    value = settings.get("image_list_job_timeout_seconds", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{setting_key} must be a finite non-negative number")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{setting_key} must be a finite non-negative number")

    try:
        timeout = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{setting_key} must be a finite non-negative number") from None

    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(f"{setting_key} must be a finite non-negative number")
    return timeout


def is_master_delegate_only() -> bool:
    """Returns True when master should skip local workload and act as orchestrator only."""
    try:
        cfg = load_config()
        return bool(cfg.get('settings', {}).get('master_delegate_only', False))
    except Exception:
        return False
