from __future__ import annotations

import asyncio
import io
import time

from aiohttp import web
from PIL import Image
import server

from .request_guards import authorization_error_or_none
from .schemas import require_bool_literal
from ..upscale.job_models import BaseJobState, ImageJobState, TileJobState
from ..upscale.job_store import ensure_tile_jobs_initialized, init_dynamic_job
from ..upscale.payload_parsers import parse_tiles_from_form
from ..utils.logging import debug_log
from ..utils.network import handle_api_error
from ..utils.usdu_management import MAX_PAYLOAD_SIZE


def _parse_int_field(value, field_name: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return parsed


def _parse_bool_field(
    value,
    field_name: str,
    *,
    default: bool = False,
) -> bool:
    if value is None:
        return default
    return require_bool_literal(value, field_name=field_name)


def _is_worker_allowed(job_data: BaseJobState, worker_id: str) -> bool:
    known_workers = {str(wid) for wid in getattr(job_data, "worker_status", {}).keys()}
    known_workers.update(str(wid) for wid in getattr(job_data, "assigned_to_workers", {}).keys())
    if not known_workers:
        return True
    return str(worker_id) in known_workers


@server.PromptServer.instance.routes.post("/distributed/heartbeat")
async def heartbeat_endpoint(request: web.Request) -> web.StreamResponse:
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        data = await request.json()
        worker_id = data.get('worker_id')
        multi_job_id = data.get('multi_job_id')

        if not worker_id or not multi_job_id:
            return await handle_api_error(request, "Missing worker_id or multi_job_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if isinstance(job_data, BaseJobState):
                    if not _is_worker_allowed(job_data, worker_id):
                        return await handle_api_error(request, "Unauthorized worker_id", 403)
                    job_data.worker_status[worker_id] = time.time()
                    debug_log(f"Heartbeat from worker {worker_id}")
                    return web.json_response({"status": "success"})
                return await handle_api_error(request, "Worker status tracking not available", 400)
            return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.post("/distributed/submit_tiles")
async def submit_tiles_endpoint(request: web.Request) -> web.StreamResponse:
    """Endpoint for workers to submit processed tiles in static mode."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                payload_size = int(content_length)
            except ValueError:
                return await handle_api_error(request, "Invalid content-length header", 400)
            if payload_size > MAX_PAYLOAD_SIZE:
                return await handle_api_error(request, f"Payload too large: {content_length} bytes", 413)

        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        try:
            is_last = _parse_bool_field(data.get('is_last', False), "is_last", default=False)
        except ValueError as e:
            return await handle_api_error(request, str(e), 400)

        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()

        try:
            batch_size = _parse_int_field(data.get('batch_size', 0), "batch_size", minimum=0)
        except ValueError as e:
            return await handle_api_error(request, str(e), 400)

        # Handle completion signal
        if batch_size == 0 and is_last:
            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if not isinstance(job_data, TileJobState):
                        return await handle_api_error(request, "Job not configured for tile submissions", 400)
                    if not _is_worker_allowed(job_data, worker_id):
                        return await handle_api_error(request, "Unauthorized worker_id", 403)
                    await job_data.queue.put({
                        'worker_id': worker_id,
                        'is_last': True,
                        'tiles': [],
                    })
                    debug_log(f"Received completion signal from worker {worker_id}")
                    return web.json_response({"status": "success"})

        try:
            tiles = parse_tiles_from_form(data)
        except ValueError as e:
            return await handle_api_error(request, str(e), 400)

        # Submit tiles to queue
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if not isinstance(job_data, TileJobState):
                    return await handle_api_error(request, "Job not configured for tile submissions", 400)
                if not _is_worker_allowed(job_data, worker_id):
                    return await handle_api_error(request, "Unauthorized worker_id", 403)

                q = job_data.queue
                if batch_size > 0 or len(tiles) > 0:
                    await q.put({
                        'worker_id': worker_id,
                        'tiles': tiles,
                        'is_last': is_last,
                    })
                    debug_log(f"Received {len(tiles)} tiles from worker {worker_id} (is_last={is_last})")
                else:
                    await q.put({
                        'worker_id': worker_id,
                        'is_last': True,
                        'tiles': [],
                    })

                return web.json_response({"status": "success"})
            return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.post("/distributed/submit_image")
async def submit_image_endpoint(request: web.Request) -> web.StreamResponse:
    """Endpoint for workers to submit processed images in dynamic mode."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                payload_size = int(content_length)
            except ValueError:
                return await handle_api_error(request, "Invalid content-length header", 400)
            if payload_size > MAX_PAYLOAD_SIZE:
                return await handle_api_error(request, f"Payload too large: {content_length} bytes", 413)

        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        try:
            is_last = _parse_bool_field(data.get('is_last', False), "is_last", default=False)
        except ValueError as e:
            return await handle_api_error(request, str(e), 400)

        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()

        # Handle image submission
        if 'full_image' in data and 'image_idx' in data:
            try:
                image_idx = _parse_int_field(data.get('image_idx'), "image_idx", minimum=0)
            except ValueError as e:
                return await handle_api_error(request, str(e), 400)
            img_data = data['full_image'].file.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            debug_log(f"Received full image {image_idx} from worker {worker_id}")

            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if not isinstance(job_data, ImageJobState):
                        return await handle_api_error(request, "Job not configured for image submissions", 400)
                    if not _is_worker_allowed(job_data, worker_id):
                        return await handle_api_error(request, "Unauthorized worker_id", 403)
                    await job_data.queue.put({
                        'worker_id': worker_id,
                        'image_idx': image_idx,
                        'image': img,
                        'is_last': is_last,
                    })
                    return web.json_response({"status": "success"})

        # Handle completion signal (no image data)
        elif is_last:
            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if not isinstance(job_data, ImageJobState):
                        return await handle_api_error(request, "Job not configured for image submissions", 400)
                    if not _is_worker_allowed(job_data, worker_id):
                        return await handle_api_error(request, "Unauthorized worker_id", 403)
                    await job_data.queue.put({
                        'worker_id': worker_id,
                        'is_last': True,
                    })
                    debug_log(f"Received completion signal from worker {worker_id}")
                    return web.json_response({"status": "success"})
        else:
            return await handle_api_error(request, "Missing image data or invalid request", 400)

        return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.post("/distributed/request_image")
async def request_image_endpoint(request: web.Request) -> web.StreamResponse:
    """Endpoint for workers to request tasks (images in dynamic mode, tiles in static mode)."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        data = await request.json()
        worker_id = data.get('worker_id')
        multi_job_id = data.get('multi_job_id')

        if not worker_id or not multi_job_id:
            return await handle_api_error(request, "Missing worker_id or multi_job_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if not isinstance(job_data, BaseJobState):
                    return await handle_api_error(request, "Invalid job data structure", 500)

                mode = job_data.mode
                if isinstance(job_data, ImageJobState):
                    pending_queue = job_data.pending_images
                elif isinstance(job_data, TileJobState):
                    pending_queue = job_data.pending_tasks
                else:
                    return await handle_api_error(request, "Invalid job configuration", 400)
                if not _is_worker_allowed(job_data, worker_id):
                    return await handle_api_error(request, "Unauthorized worker_id", 403)

                try:
                    task_idx = await asyncio.wait_for(pending_queue.get(), timeout=0.1)
                    job_data.assigned_to_workers.setdefault(worker_id, []).append(task_idx)
                    job_data.worker_status[worker_id] = time.time()
                    remaining = pending_queue.qsize()

                    if mode == 'dynamic':
                        debug_log(f"UltimateSDUpscale API - Assigned image {task_idx} to worker {worker_id}")
                        return web.json_response(
                            {
                                "kind": "image",
                                "task_idx": task_idx,
                                "estimated_remaining": remaining,
                            }
                        )
                    debug_log(f"UltimateSDUpscale API - Assigned tile {task_idx} to worker {worker_id}")
                    return web.json_response({
                        "kind": "tile",
                        "task_idx": task_idx,
                        "estimated_remaining": remaining,
                        "batched_static": job_data.batched_static,
                    })
                except asyncio.TimeoutError:
                    return web.json_response({"kind": "none", "task_idx": None, "estimated_remaining": 0})
            return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/job_status")
async def job_status_endpoint(request: web.Request) -> web.StreamResponse:
    """Endpoint to check if a job is ready."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    multi_job_id = request.query.get('multi_job_id')
    if not multi_job_id:
        return web.json_response({"ready": False})
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        ready = bool(isinstance(job_data, BaseJobState) and job_data.queue is not None)
        return web.json_response({"ready": ready})
