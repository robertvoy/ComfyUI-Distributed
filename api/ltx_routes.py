"""HTTP routes for distributed LTX latent tile results."""

from __future__ import annotations

import time

from aiohttp import web
import server

from ..upscale.job_models import TileJobState
from ..upscale.job_store import MAX_PAYLOAD_SIZE, ensure_tile_jobs_initialized
from ..utils.logging import debug_log
from ..utils.network import handle_api_error


@server.PromptServer.instance.routes.post("/distributed/ltx/submit_tile")
async def submit_ltx_tile_endpoint(request):
    """Accept one processed LTX latent tile from a worker.

    Body is JSON:
    - multi_job_id: distributed job id
    - worker_id: worker id
    - tile_index: spatial tile id, or -1 for completion-only messages
    - payload: base64 safetensors payload produced by nodes.ltx_tiled_sampler
    - is_last: optional worker completion flag
    """
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
            return await handle_api_error(request, f"Payload too large: {content_length} bytes", 413)

        data = await request.json()
        multi_job_id = data.get("multi_job_id")
        worker_id = str(data.get("worker_id") or "")
        is_last = bool(data.get("is_last", False))
        payload = data.get("payload")
        tile_index = data.get("tile_index")

        if not multi_job_id or not worker_id:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if not isinstance(job_data, TileJobState):
                return await handle_api_error(request, "Job not configured for tile submissions", 404)

            job_data.worker_status[worker_id] = time.time()

            if is_last and (payload is None or int(tile_index or -1) < 0):
                await job_data.queue.put({
                    "worker_id": worker_id,
                    "is_last": True,
                    "tiles": [],
                })
                debug_log(f"Received LTX completion signal from worker {worker_id}")
                return web.json_response({"status": "success"})

            if payload is None or tile_index is None:
                return await handle_api_error(request, "Missing tile_index or payload", 400)

            tile_index = int(tile_index)
            await job_data.queue.put({
                "worker_id": worker_id,
                "is_last": is_last,
                "tiles": [{
                    "tile_idx": tile_index,
                    "global_idx": tile_index,
                    "payload": payload,
                    "worker_id": worker_id,
                }],
            })
            debug_log(f"Received LTX tile {tile_index} from worker {worker_id} (is_last={is_last})")
            return web.json_response({"status": "success"})
    except Exception as exc:
        return await handle_api_error(request, exc, 500)
