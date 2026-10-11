import json
from contextlib import asynccontextmanager
from typing import Any

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
from ..utils.secret_fields import is_secret_field_name
from ..utils.serverless_config import (
    ServerlessSecretChanges,
    public_config,
    validate_serverless_worker,
)


def _positive_int(value):
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


_SECRET_FIELDS = {"api_key", "clear_api_key"}
_SERVERLESS_LOCAL_FIELDS = {"host", "port", "listen_port", "cuda_device", "extra_args"}
_SERVERLESS_ONLY_FIELDS = {
    "provider", "endpoint_id", "jobs_per_run", "concurrency", "timeout_seconds", "api_key_configured",
}
_SERVERLESS_CONFIG_FIELDS = {
    "worker_id", "type", "provider", "endpoint_id", "jobs_per_run", "concurrency",
    "timeout_seconds", "enabled", "name", "api_key", "clear_api_key",
}


class _ActivePoolCheckError(RuntimeError):
    pass


def _error_response(message, status):
    return web.json_response({"status": "error", "message": message}, status=status)


def _contains_secret_fields(value, *, allow_root_serverless_secrets=False, depth=0):
    if isinstance(value, dict):
        for key, item in value.items():
            field_name = key.casefold() if isinstance(key, str) else ""
            if is_secret_field_name(field_name):
                if not (allow_root_serverless_secrets and depth == 0 and key in _SECRET_FIELDS):
                    return True
            if _contains_secret_fields(
                item, allow_root_serverless_secrets=allow_root_serverless_secrets, depth=depth + 1
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(
            _contains_secret_fields(
                item, allow_root_serverless_secrets=allow_root_serverless_secrets, depth=depth + 1
            )
            for item in value
        )
    return False


def _serverless_secret_store():
    from .orchestration.serverless_runtime import get_secret_store
    return get_secret_store()


def _serverless_pool_is_active(worker_id):
    from .orchestration.serverless_runtime import get_manager
    return get_manager().pool_is_active(worker_id)


def _ensure_pool_idle(worker_id):
    try:
        active = _serverless_pool_is_active(worker_id)
    except Exception:
        raise _ActivePoolCheckError from None
    if active:
        raise _ActivePoolCheckError


def _worker_by_id(workers, worker_id):
    for worker in workers:
        if isinstance(worker, dict) and worker.get("id") == worker_id:
            return worker
    return None


def _worker_type(worker):
    return worker.get("type", "local") if isinstance(worker, dict) else None


def _strip_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_sensitive_fields(item)
            for key, item in value.items()
            if not is_secret_field_name(key)
        }
    if isinstance(value, list):
        return [_strip_sensitive_fields(item) for item in value]
    return value


def _clean_serverless_worker(worker):
    clean = _strip_sensitive_fields(dict(worker))
    for field in _SERVERLESS_LOCAL_FIELDS | {"api_key_configured"}:
        clean.pop(field, None)
    return clean


def _validate_bulk_workers(workers):
    validated = []
    for worker in workers:
        if isinstance(worker, dict) and worker.get("type") == "serverless":
            clean = _clean_serverless_worker(worker)
            validated.append(validate_serverless_worker(clean))
        else:
            validated.append(worker)
    return validated


def _clean_bulk_type_conversions(old_workers, new_workers):
    old_by_id = {str(w.get("id")): w for w in old_workers if isinstance(w, dict) and w.get("id") is not None}
    for index, worker in enumerate(new_workers):
        if not isinstance(worker, dict) or worker.get("id") is None:
            continue
        old = old_by_id.get(str(worker["id"]))
        if old is None or old.get("type") != "serverless" or worker.get("type") == "serverless":
            continue
        converted = dict(worker)
        for field in _SERVERLESS_ONLY_FIELDS:
            converted.pop(field, None)
        new_workers[index] = converted
    return new_workers


def _bulk_serverless_changes(old_workers, new_workers):
    old_by_id = {str(w.get("id")): w for w in old_workers if isinstance(w, dict) and w.get("id") is not None}
    new_by_id = {str(w.get("id")): w for w in new_workers if isinstance(w, dict) and w.get("id") is not None}
    secret_ids = set()
    for worker_id, old in old_by_id.items():
        new = new_by_id.get(worker_id)
        if old.get("type") == "serverless":
            if new is None or new.get("type") != "serverless":
                _ensure_pool_idle(worker_id)
                secret_ids.add(worker_id)
            elif (old.get("endpoint_id") != new.get("endpoint_id")
                  or old.get("concurrency", 1) != new.get("concurrency", 1)):
                _ensure_pool_idle(worker_id)
        elif new is not None and new.get("type") == "serverless":
            secret_ids.add(worker_id)
    return secret_ids


@server.PromptServer.instance.routes.get("/distributed/config")
async def get_config_endpoint(request):
    try:
        config = load_config()
        return web.json_response(public_config(config, _serverless_secret_store()))
    except Exception:
        return _error_response("Unable to read config", 500)


@server.PromptServer.instance.routes.post("/distributed/config")
async def update_config_endpoint(request):
    """Bulk config update with schema validation and credential redaction."""
    try:
        data = await request.json()
    except Exception:
        return _error_response("Invalid JSON payload", 400)

    if not isinstance(data, dict):
        return _error_response("Config payload must be an object", 400)
    if _contains_secret_fields(data):
        return _error_response("API keys cannot be set through bulk config", 400)

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
            errors.append(f"{key}: failed validation")
            continue

        if key == "workers":
            try:
                value = _validate_bulk_workers(value)
            except (TypeError, ValueError):
                return _error_response("Invalid serverless worker configuration", 400)

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

    store = None
    response_config = None
    try:
        with ServerlessSecretChanges() as secret_changes:
            async with config_transaction() as config:
                if "workers" in validated_root:
                    old_workers = config.get("workers", [])
                    validated_root["workers"] = _clean_bulk_type_conversions(
                        old_workers, validated_root["workers"]
                    )
                    changed_ids = _bulk_serverless_changes(old_workers, validated_root["workers"])
                    for worker in validated_root["workers"]:
                        if isinstance(worker, dict) and worker.get("type") == "serverless" and worker.get("enabled", False):
                            store = store or _serverless_secret_store()
                            if worker["id"] in changed_ids or not store.has(worker["id"]):
                                return _error_response("Save an API key before enabling this pool", 400)
                    if changed_ids:
                        store = store or _serverless_secret_store()
                        for worker_id in sorted(changed_ids):
                            prior_secret = secret_changes.capture(store, worker_id)
                            if prior_secret is not None:
                                secret_changes.delete(store, worker_id)

                settings = config.setdefault("settings", {})
                settings.update(validated_settings)
                for key, value in validated_root.items():
                    config[key] = value
                response_config = public_config(config, store or _serverless_secret_store())
            response = web.json_response({"status": "success", "config": response_config})
            secret_changes.commit()
            return response
    except _ActivePoolCheckError:
        return _error_response("Finish or reconcile unresolved serverless jobs before changing this pool", 409)
    except Exception:
        return _error_response("Unable to update config", 500)


@server.PromptServer.instance.routes.get("/distributed/queue_status/{job_id}")
async def queue_status_endpoint(request):
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
async def update_worker_endpoint(request):
    return await _run_with_serverless_secret_changes(_update_worker_endpoint, request)


async def _run_with_serverless_secret_changes(endpoint, request):
    with ServerlessSecretChanges() as secret_changes:
        response = await endpoint(request, secret_changes)
        if response.status < 400:
            secret_changes.commit()
        return response


async def _update_worker_endpoint(request, secret_changes):
    store = None
    worker_id = None
    try:
        try:
            data = await request.json()
        except Exception:
            return _error_response("Invalid JSON payload", 400)
        if not isinstance(data, dict):
            return _error_response("Worker payload must be an object", 400)

        worker_id = data.get("worker_id")
        if worker_id is None:
            return _error_response("Missing worker_id", 400)

        if _contains_secret_fields(data, allow_root_serverless_secrets=True):
            return _error_response("Invalid worker credentials", 400)

        result = None
        async with config_transaction() as config:
            workers = config.setdefault("workers", [])
            existing = _worker_by_id(workers, worker_id)
            current_type = _worker_type(existing)
            target_type = data.get("type", current_type or "local")

            if target_type == "serverless":
                unknown = set(data) - _SERVERLESS_CONFIG_FIELDS - _SERVERLESS_LOCAL_FIELDS
                if unknown:
                    return _error_response("Unknown serverless worker field", 400)
                if not isinstance(worker_id, str) or not worker_id.strip():
                    return _error_response("worker_id must be a non-empty string", 400)
                if "api_key" in data and not isinstance(data["api_key"], str):
                    return _error_response("Invalid worker credentials", 400)
                if "clear_api_key" in data and type(data["clear_api_key"]) is not bool:
                    return _error_response("Invalid worker credentials", 400)
                supplied_key = data.get("api_key", "")
                clear_key = data.get("clear_api_key", False)
                if supplied_key and clear_key:
                    return _error_response("Choose either api_key or clear_api_key", 400)
                if "enabled" in data and type(data["enabled"]) is not bool:
                    return _error_response("enabled must be a boolean", 400)
                if "name" in data and not isinstance(data["name"], str):
                    return _error_response("name must be a string", 400)

                candidate = dict(existing or {})
                candidate["id"] = worker_id
                candidate["type"] = "serverless"
                candidate.setdefault("provider", "runpod")
                candidate.setdefault("enabled", False)
                for field in ("provider", "endpoint_id", "jobs_per_run", "concurrency", "timeout_seconds", "enabled", "name"):
                    if field in data:
                        candidate[field] = data[field]
                candidate = _clean_serverless_worker(candidate)
                validated = validate_serverless_worker(candidate)
                if "enabled" in validated and type(validated["enabled"]) is not bool:
                    return _error_response("enabled must be a boolean", 400)
                if "name" in validated and not isinstance(validated["name"], str):
                    return _error_response("name must be a string", 400)

                store = _serverless_secret_store()
                prior_secret = secret_changes.capture(store, worker_id)
                converting_to_serverless = current_type != "serverless"
                next_secret = None if converting_to_serverless else prior_secret
                if supplied_key:
                    next_secret = supplied_key
                elif clear_key:
                    next_secret = None

                endpoint_changed = (
                    current_type == "serverless"
                    and existing.get("endpoint_id") != validated.get("endpoint_id")
                )
                concurrency_changed = (
                    current_type == "serverless"
                    and existing.get("concurrency", 1) != validated.get("concurrency", 1)
                )
                if current_type == "serverless" and (
                    endpoint_changed or concurrency_changed or next_secret != prior_secret
                ):
                    _ensure_pool_idle(worker_id)

                if validated.get("enabled", False) and next_secret is None:
                    return _error_response("Save an API key before enabling this pool; disable it before clearing the key", 400)

                if next_secret != prior_secret:
                    if next_secret is None:
                        secret_changes.delete(store, worker_id)
                    else:
                        secret_changes.set(store, worker_id, next_secret)

                if existing is None:
                    workers.append(validated)
                else:
                    existing.clear()
                    existing.update(validated)
                result = {"status": "success", "api_key_configured": next_secret is not None}

            elif existing is not None:
                cleaned_existing = _strip_sensitive_fields(existing)
                existing.clear()
                existing.update(cleaned_existing)
                if current_type == "serverless":
                    if _contains_secret_fields(data):
                        return _error_response("Invalid worker credentials", 400)
                    _ensure_pool_idle(worker_id)
                    store = _serverless_secret_store()
                    prior_secret = secret_changes.capture(store, worker_id)
                    if prior_secret is not None:
                        secret_changes.delete(store, worker_id)
                    for field in _SERVERLESS_ONLY_FIELDS:
                        existing.pop(field, None)
                elif _contains_secret_fields(data):
                    return _error_response("API keys are only accepted for serverless workers", 400)
                _apply_field_patch(existing, data, _WORKER_FIELDS)
                result = {"status": "success"}
            elif all(key in data for key in ["name", "port", "cuda_device"]):
                if _contains_secret_fields(data):
                    return _error_response("API keys are only accepted for serverless workers", 400)
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
                result = {"status": "success"}
            else:
                return _error_response(
                    "Worker not found and missing required fields for creation", 404
                )

        return web.json_response(result)
    except _ActivePoolCheckError:
        return _error_response("Finish or reconcile unresolved serverless jobs before changing this pool", 409)
    except ValueError:
        return _error_response("Invalid worker configuration", 400)
    except Exception:
        return _error_response("Unable to update worker configuration", 500)


@server.PromptServer.instance.routes.post("/distributed/config/delete_worker")
async def delete_worker_endpoint(request):
    return await _run_with_serverless_secret_changes(_delete_worker_endpoint, request)


async def _delete_worker_endpoint(request, secret_changes):
    store = None
    worker_id = None
    try:
        try:
            data = await request.json()
        except Exception:
            return _error_response("Invalid JSON payload", 400)
        if not isinstance(data, dict):
            return _error_response("Worker payload must be an object", 400)
        worker_id = data.get("worker_id")
        if worker_id is None:
            return _error_response("Missing worker_id", 400)

        result = None
        async with config_transaction() as config:
            workers = config.get("workers", [])
            removed = _worker_by_id(workers, worker_id)
            if removed is None:
                return _error_response("Worker not found", 404)
            if _worker_type(removed) == "serverless":
                _ensure_pool_idle(worker_id)
                store = _serverless_secret_store()
                secret_worker_id = str(worker_id)
                prior_secret = secret_changes.capture(store, secret_worker_id)
                if prior_secret is not None:
                    secret_changes.delete(store, secret_worker_id)
            workers.remove(removed)
            result = {
                "status": "success",
                "message": "Worker deleted" if _worker_type(removed) == "serverless"
                else f"Worker {removed.get('name', worker_id)} deleted",
            }

        return web.json_response(result)
    except _ActivePoolCheckError:
        return _error_response("Finish or reconcile unresolved serverless jobs before changing this pool", 409)
    except Exception:
        return _error_response("Unable to delete worker", 500)

@server.PromptServer.instance.routes.post("/distributed/config/update_setting")
async def update_setting_endpoint(request):
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
async def update_master_endpoint(request):
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
