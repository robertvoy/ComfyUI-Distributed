from __future__ import annotations

import json
from contextlib import asynccontextmanager

from aiohttp import web
import server

try:
    from ..utils.config import config_transaction, load_config, save_config
except ImportError:
    from ..utils.config import load_config

    try:
        from ..utils.config import save_config
    except ImportError:
        def save_config(_config):
            return True

    @asynccontextmanager
    async def config_transaction():
        config = load_config()
        original_snapshot = json.dumps(config, sort_keys=True)
        yield config
        if json.dumps(config, sort_keys=True) != original_snapshot:
            save_config(config)
from ..utils.logging import debug_log, log
from ..utils.network import handle_api_error, normalize_host


def _positive_int(value: int) -> bool:
    return value > 0


CONFIG_SCHEMA = {
    "workers": (list, None),
    "master": (dict, None),
    "settings": (dict, None),
    "tunnel": (dict, None),
    "managed_processes": (dict, None),
    "worker_timeout_seconds": (int, _positive_int),
    "debug": (bool, None),
    "auto_launch_workers": (bool, None),
    "stop_workers_on_master_exit": (bool, None),
    "master_delegate_only": (bool, None),
    "websocket_orchestration": (bool, None),
    "has_auto_populated_workers": (bool, None),
}

_SETTINGS_FIELDS = {
    "worker_timeout_seconds",
    "debug",
    "auto_launch_workers",
    "stop_workers_on_master_exit",
    "master_delegate_only",
    "websocket_orchestration",
    "has_auto_populated_workers",
}

_WORKER_FIELDS = [
    ("enabled", None, False),
    ("name", None, False),
    ("port", None, False),
    ("host", normalize_host, True),
    ("cuda_device", None, True),
    ("extra_args", None, True),
    ("type", None, False),
]

_MASTER_FIELDS = [
    ("name", None, False),
    ("host", normalize_host, True),
    ("port", None, False),
    ("cuda_device", None, False),
    ("extra_args", None, False),
]


def _apply_field_patch(target: dict, data: dict, field_rules: list) -> None:
    """Apply a partial update to a target dict based on field rules."""
    for key, normalizer, remove_on_none in field_rules:
        if key not in data:
            continue
        value = data[key]
        if value is None and remove_on_none:
            target.pop(key, None)
        else:
            target[key] = normalizer(value) if (normalizer and value is not None) else value


@server.PromptServer.instance.routes.get("/distributed/config")
async def get_config_endpoint(request: web.Request) -> web.StreamResponse:
    config = load_config()
    return web.json_response(config)


@server.PromptServer.instance.routes.post("/distributed/config")
async def update_config_endpoint(request: web.Request) -> web.StreamResponse:
    """Bulk config update with schema validation."""
    try:
        data = await request.json()
    except Exception as e:
        return await handle_api_error(request, f"Invalid JSON payload: {e}", 400)

    if not isinstance(data, dict):
        return await handle_api_error(request, "Config payload must be an object", 400)

    validated_settings = {}
    validated_root = {}
    errors = []

    for key, value in data.items():
        if key not in CONFIG_SCHEMA:
            errors.append(f"Unknown field: {key}")
            continue

        expected_type, validator = CONFIG_SCHEMA[key]
        if not isinstance(value, expected_type):
            errors.append(f"{key}: expected {expected_type.__name__}")
            continue

        if validator and not validator(value):
            errors.append(f"{key}: value {value!r} failed validation")
            continue

        if key in _SETTINGS_FIELDS:
            validated_settings[key] = value
        else:
            validated_root[key] = value

    if errors:
        return web.json_response({
            "status": "error",
            "error": errors,
            "message": "; ".join(errors),
        }, status=400)

    try:
        async with config_transaction() as config:
            settings = config.setdefault("settings", {})
            settings.update(validated_settings)
            for key, value in validated_root.items():
                config[key] = value
            return web.json_response({"status": "success", "config": config})
    except Exception as e:
        return await handle_api_error(request, e)


@server.PromptServer.instance.routes.get("/distributed/queue_status/{job_id}")
async def queue_status_endpoint(request: web.Request) -> web.StreamResponse:
    """Check if a job queue is initialized."""
    try:
        job_id = request.match_info['job_id']
        
        # Import to ensure initialization
        from ..upscale.job_store import ensure_tile_jobs_initialized
        prompt_server = ensure_tile_jobs_initialized()
        
        async with prompt_server.distributed_tile_jobs_lock:
            exists = job_id in prompt_server.distributed_pending_tile_jobs
        
        debug_log(f"Queue status check for job {job_id}: {'exists' if exists else 'not found'}")
        return web.json_response({"exists": exists, "job_id": job_id})
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/config/update_worker")
async def update_worker_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        
        if worker_id is None:
            return await handle_api_error(request, "Missing worker_id", 400)

        async with config_transaction() as config:
            worker_found = False
            workers = config.setdefault("workers", [])

            for worker in workers:
                if worker["id"] == worker_id:
                    _apply_field_patch(worker, data, _WORKER_FIELDS)
                    worker_found = True
                    break

            if not worker_found:
                # If worker not found and all required fields are provided, create new worker
                if all(key in data for key in ["name", "port", "cuda_device"]):
                    new_worker = {
                        "id": worker_id,
                        "name": data["name"],
                        "host": normalize_host(data.get("host", "localhost")),
                        "port": data["port"],
                        "cuda_device": data["cuda_device"],
                        "enabled": data.get("enabled", False),
                        "extra_args": data.get("extra_args", ""),
                        "type": data.get("type", "local")
                    }
                    workers.append(new_worker)
                else:
                    return await handle_api_error(
                        request,
                        f"Worker {worker_id} not found and missing required fields for creation",
                        404,
                    )

            return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/config/delete_worker")
async def delete_worker_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        
        if worker_id is None:
            return await handle_api_error(request, "Missing worker_id", 400)
            
        async with config_transaction() as config:
            workers = config.get("workers", [])

            # Find and remove the worker
            worker_index = -1
            for i, worker in enumerate(workers):
                if worker["id"] == worker_id:
                    worker_index = i
                    break

            if worker_index == -1:
                return await handle_api_error(request, f"Worker {worker_id} not found", 404)

            # Remove the worker
            removed_worker = workers.pop(worker_index)

            return web.json_response({
                "status": "success",
                "message": f"Worker {removed_worker.get('name', worker_id)} deleted"
            })
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/config/update_setting")
async def update_setting_endpoint(request: web.Request) -> web.StreamResponse:
    """Updates a specific key in the settings object."""
    try:
        data = await request.json()
        key = data.get("key")
        value = data.get("value")

        if not key or value is None:
            return await handle_api_error(request, "Missing 'key' or 'value' in request", 400)
        if key not in _SETTINGS_FIELDS:
            return await handle_api_error(request, f"Unknown setting: {key}", 400)

        async with config_transaction() as config:
            if 'settings' not in config:
                config['settings'] = {}

            config['settings'][key] = value

            return web.json_response({"status": "success", "message": f"Setting '{key}' updated."})
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/distributed/config/update_master")
async def update_master_endpoint(request: web.Request) -> web.StreamResponse:
    """Updates master configuration."""
    try:
        data = await request.json()
        
        async with config_transaction() as config:
            if 'master' not in config:
                config['master'] = {}
            _apply_field_patch(config['master'], data, _MASTER_FIELDS)

            return web.json_response({"status": "success", "message": "Master configuration updated."})
    except Exception as e:
        return await handle_api_error(request, e, 400)
