from __future__ import annotations

import json
import asyncio
import io
import os
import base64
import binascii
import time
from typing import Any

from aiohttp import web
import server
import torch
from PIL import Image

from ..utils.logging import debug_log
from ..utils.image import pil_to_tensor, ensure_contiguous
from ..utils.network import handle_api_error
from ..utils.constants import JOB_INIT_GRACE_PERIOD, MEMORY_CLEAR_DELAY
from ..utils.runtime_state import ensure_distributed_runtime_state
from .request_guards import authorization_error_or_none
from .schemas import require_bool_literal
from .queue_orchestration import orchestrate_distributed_execution
from .queue_request import parse_queue_request_payload

# Canonical worker result envelope accepted by POST /distributed/job_complete:
# { "job_id": str, "worker_id": str, "batch_idx": int, "image": <base64 PNG>, "is_last": bool }


def _runtime_state():
    return ensure_distributed_runtime_state()


def _decode_image_sync(image_path: str) -> dict[str, Any]:
    """Decode image/video file and compute hash in a threadpool worker."""
    import base64
    import hashlib
    import folder_paths

    full_path = folder_paths.get_annotated_filepath(image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(image_path)

    hash_md5 = hashlib.md5()
    with open(full_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_ext = os.path.splitext(full_path)[1].lower()

    if file_ext in video_extensions:
        with open(full_path, 'rb') as f:
            file_data = f.read()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        mime_type = mime_types.get(file_ext, 'video/mp4')
        image_data = f"data:{mime_type};base64,{base64.b64encode(file_data).decode('utf-8')}"
    else:
        with Image.open(full_path) as img:
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG', compress_level=1)
            image_data = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    return {
        "status": "success",
        "image_data": image_data,
        "hash": file_hash,
    }


def _check_file_sync(filename: str, expected_hash: str) -> dict[str, Any]:
    """Check file presence and hash in a threadpool worker."""
    import hashlib
    import folder_paths

    full_path = folder_paths.get_annotated_filepath(filename)
    if not os.path.exists(full_path):
        return {
            "status": "success",
            "exists": False,
        }

    hash_md5 = hashlib.md5()
    with open(full_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()

    return {
        "status": "success",
        "exists": True,
        "hash_matches": file_hash == expected_hash,
    }


def _decode_canonical_png_tensor(image_payload: str) -> torch.Tensor:
    """Decode canonical base64 PNG payload into a contiguous IMAGE tensor."""
    if not isinstance(image_payload, str) or not image_payload.strip():
        raise ValueError("Field 'image' must be a non-empty base64 PNG string.")

    encoded = image_payload.strip()
    if encoded.startswith("data:"):
        header, sep, data_part = encoded.partition(",")
        if not sep:
            raise ValueError("Field 'image' data URL is malformed.")
        if not header.lower().startswith("data:image/png;base64"):
            raise ValueError("Field 'image' must be a PNG data URL when using data:* format.")
        encoded = data_part

    try:
        png_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Field 'image' is not valid base64 PNG data.") from exc

    if not png_bytes:
        raise ValueError("Field 'image' decoded to empty PNG data.")

    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            img = img.convert("RGB")
            tensor = pil_to_tensor(img)
        return ensure_contiguous(tensor)
    except Exception as exc:
        raise ValueError(f"Failed to decode PNG image payload: {exc}") from exc


def _decode_audio_payload(audio_payload: dict[str, Any]) -> dict[str, Any]:
    """Decode canonical audio payload into an AUDIO dict."""
    from ..utils.audio_payload import decode_audio_payload

    return decode_audio_payload(audio_payload)


@server.PromptServer.instance.routes.post("/distributed/prepare_job")
async def prepare_job_endpoint(request: web.Request) -> web.StreamResponse:
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        data = await request.json()
        multi_job_id = data.get('multi_job_id')
        if not multi_job_id:
            return await handle_api_error(request, "Missing multi_job_id", 400)

        runtime_state = _runtime_state()
        async with runtime_state.distributed_jobs_lock:
            if multi_job_id not in runtime_state.distributed_pending_jobs:
                runtime_state.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
        
        debug_log(f"Prepared queue for job {multi_job_id}")
        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e)

@server.PromptServer.instance.routes.post("/distributed/clear_memory")
async def clear_memory_endpoint(request: web.Request) -> web.StreamResponse:
    debug_log("Received request to clear VRAM.")
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        warnings: list[str] = []
        # Use ComfyUI's prompt server queue system like the /free endpoint does
        if hasattr(server.PromptServer.instance, 'prompt_queue'):
            server.PromptServer.instance.prompt_queue.set_flag("unload_models", True)
            server.PromptServer.instance.prompt_queue.set_flag("free_memory", True)
            debug_log("Set queue flags for memory clearing.")
        
        # Wait a bit for the queue to process
        await asyncio.sleep(MEMORY_CLEAR_DELAY)
        
        # Also do direct cleanup as backup, but with error handling
        import gc
        import comfy.model_management as mm
        
        try:
            mm.unload_all_models()
        except AttributeError as e:
            warning = f"Model unload warning: {e}"
            warnings.append(warning)
            debug_log(warning)
        
        try:
            mm.soft_empty_cache()
        except Exception as e:
            warning = f"Cache clear warning: {e}"
            warnings.append(warning)
            debug_log(warning)
        
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        if warnings:
            debug_log("VRAM cleared with warnings.")
            return web.json_response(
                {
                    "status": "partial",
                    "message": "GPU memory cleared with warnings.",
                    "warnings": warnings,
                }
            )

        debug_log("VRAM cleared successfully.")
        return web.json_response({"status": "success", "message": "GPU memory cleared."})
    except Exception as e:
        # Even if there's an error, try to do basic cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        debug_log(f"VRAM clear failed: {e}")
        return await handle_api_error(request, f"GPU memory clear failed: {e}", 500)


@server.PromptServer.instance.routes.post("/distributed/queue")
async def distributed_queue_endpoint(request: web.Request) -> web.StreamResponse:
    """Queue a distributed workflow, mirroring the UI orchestration pipeline."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        raw_payload = await request.json()
    except Exception as exc:
        return await handle_api_error(request, f"Invalid JSON payload: {exc}", 400)

    try:
        payload = parse_queue_request_payload(raw_payload)
    except ValueError as exc:
        return await handle_api_error(request, exc, 400)

    try:
        prompt_id, worker_count = await orchestrate_distributed_execution(
            payload.prompt,
            payload.workflow_meta,
            payload.client_id,
            enabled_worker_ids=payload.enabled_worker_ids,
            delegate_master=payload.delegate_master,
            trace_execution_id=payload.trace_execution_id,
        )
        return web.json_response({
            "prompt_id": prompt_id,
            "worker_count": worker_count,
        })
    except Exception as exc:
        return await handle_api_error(request, exc, 500)

@server.PromptServer.instance.routes.post("/distributed/load_image")
async def load_image_endpoint(request: web.Request) -> web.StreamResponse:
    """Load an image or video file and return it as base64 data with hash."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        data = await request.json()
        image_path = data.get("image_path")
        
        if not image_path:
            return await handle_api_error(request, "Missing image_path", 400)
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, _decode_image_sync, image_path)
        return web.json_response(payload)
    except FileNotFoundError:
        return await handle_api_error(request, f"File not found: {image_path}", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/check_file")
async def check_file_endpoint(request: web.Request) -> web.StreamResponse:
    """Check if a file exists and matches the given hash."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        data = await request.json()
        filename = data.get("filename")
        expected_hash = data.get("hash")
        
        if not filename or not expected_hash:
            return await handle_api_error(request, "Missing filename or hash", 400)
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, _check_file_sync, filename, expected_hash)
        return web.json_response(payload)
        
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.post("/distributed/job_complete")
async def job_complete_endpoint(request: web.Request) -> web.StreamResponse:
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        data = await request.json()
    except Exception as exc:
        return await handle_api_error(request, f"Invalid JSON payload: {exc}", 400)

    if not isinstance(data, dict):
        return await handle_api_error(request, "Expected a JSON object body", 400)

    try:
        job_id = data.get("job_id")
        worker_id = data.get("worker_id")
        batch_idx = data.get("batch_idx")
        image_payload = data.get("image")
        audio_payload = data.get("audio")
        is_last_raw = data.get("is_last")

        errors = []
        if not isinstance(job_id, str) or not job_id.strip():
            errors.append("job_id: expected non-empty string")
        if not isinstance(worker_id, str) or not worker_id.strip():
            errors.append("worker_id: expected non-empty string")
        if not isinstance(batch_idx, int) or batch_idx < 0:
            errors.append("batch_idx: expected non-negative integer")
        if not isinstance(image_payload, str) or not image_payload.strip():
            errors.append("image: expected non-empty base64 PNG string")
        if audio_payload is not None and not isinstance(audio_payload, dict):
            errors.append("audio: expected object when provided")

        try:
            is_last = require_bool_literal(is_last_raw, field_name="is_last")
        except ValueError as exc:
            errors.append(str(exc))
            is_last = False
        if errors:
            return await handle_api_error(request, errors, 400)

        tensor = _decode_canonical_png_tensor(image_payload)
        decoded_audio = _decode_audio_payload(audio_payload) if audio_payload is not None else None
        multi_job_id = job_id.strip()
        worker_id = worker_id.strip()

        runtime_state = _runtime_state()
        allowed_workers = runtime_state.distributed_job_allowed_workers.get(multi_job_id)
        if allowed_workers is not None and worker_id not in allowed_workers:
            return await handle_api_error(
                request,
                f"Unauthorized worker_id for job {multi_job_id}",
                403,
            )

        pending = None
        queue_size = 0
        deadline = time.monotonic() + float(JOB_INIT_GRACE_PERIOD)
        while pending is None:
            async with runtime_state.distributed_jobs_lock:
                pending = runtime_state.distributed_pending_jobs.get(multi_job_id)
                if pending is not None:
                    queue_item = {
                        "tensor": tensor,
                        "worker_id": worker_id,
                        "image_index": int(batch_idx),
                        "is_last": is_last,
                    }
                    if decoded_audio is not None:
                        queue_item["audio"] = decoded_audio
                    await pending.put(
                        queue_item
                    )
                    queue_size = pending.qsize()
                    break

            if time.monotonic() > deadline:
                return await handle_api_error(request, "job not initialized", 404)
            await asyncio.sleep(0.05)

        debug_log(
            f"job_complete received canonical envelope - job_id: {multi_job_id}, "
            f"worker: {worker_id}, batch_idx: {batch_idx}, is_last: {is_last}, "
            f"queue_size: {queue_size}"
        )

        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e)
