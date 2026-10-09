import json
import asyncio
import io
import os
import base64
import binascii
import time

from aiohttp import web
import server
import torch
from PIL import Image

from ..utils.logging import debug_log
from ..utils.image import pil_to_tensor, ensure_contiguous
from ..utils.network import handle_api_error
from ..utils.constants import JOB_INIT_GRACE_PERIOD, MEMORY_CLEAR_DELAY
from ..utils import transfer_codec
try:
    from .queue_orchestration import ensure_distributed_state, orchestrate_distributed_execution
except ImportError:
    from .queue_orchestration import orchestrate_distributed_execution

    def ensure_distributed_state():
        return None
from .queue_request import parse_queue_request_payload

prompt_server = server.PromptServer.instance

# Worker result envelopes accepted by POST /distributed/job_complete:
#
# application/json (legacy, one frame per request):
#   { "job_id": str, "worker_id": str, "batch_idx": int, "image": <base64 PNG>, "is_last": bool,
#     "audio"?: {...} }
#
# multipart/form-data (compressed, many frames per request):
#   envelope: JSON { "job_id", "worker_id", "format", "start_index", "count", "width", "height",
#                    "is_last", "audio"?: {...} }
#   media:    one part per frame for still formats, or one video file for video formats


def _decode_image_sync(image_path):
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


def _check_file_sync(filename, expected_hash):
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


def _decode_canonical_png_tensor(image_payload):
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


def _decode_audio_payload(audio_payload):
    """Decode canonical audio payload into an AUDIO dict."""
    from ..utils.audio_payload import decode_audio_payload

    return decode_audio_payload(audio_payload)


@server.PromptServer.instance.routes.post("/distributed/prepare_job")
async def prepare_job_endpoint(request):
    try:
        data = await request.json()
        multi_job_id = data.get('multi_job_id')
        if not multi_job_id:
            return await handle_api_error(request, "Missing multi_job_id", 400)

        ensure_distributed_state()
        async with prompt_server.distributed_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
        
        debug_log(f"Prepared queue for job {multi_job_id}")
        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e)

@server.PromptServer.instance.routes.post("/distributed/clear_memory")
async def clear_memory_endpoint(request):
    debug_log("Received request to clear VRAM.")
    try:
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
            debug_log(f"Warning during model unload: {e}")
        
        try:
            mm.soft_empty_cache()
        except Exception as e:
            debug_log(f"Warning during cache clear: {e}")
        
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        debug_log("VRAM cleared successfully.")
        return web.json_response({"status": "success", "message": "GPU memory cleared."})
    except Exception as e:
        # Even if there's an error, try to do basic cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        debug_log(f"Partial VRAM clear completed with warning: {e}")
        return web.json_response({"status": "success", "message": "GPU memory cleared (with warnings)"})


@server.PromptServer.instance.routes.post("/distributed/queue")
async def distributed_queue_endpoint(request):
    """Queue a distributed workflow, mirroring the UI orchestration pipeline."""
    try:
        raw_payload = await request.json()
    except Exception as exc:
        return await handle_api_error(request, f"Invalid JSON payload: {exc}", 400)

    try:
        payload = parse_queue_request_payload(raw_payload)
    except ValueError as exc:
        return await handle_api_error(request, exc, 400)

    try:
        prompt_id, prompt_number, worker_count, node_errors = await orchestrate_distributed_execution(
            payload.prompt,
            payload.workflow_meta,
            payload.client_id,
            enabled_worker_ids=payload.enabled_worker_ids,
            delegate_master=payload.delegate_master,
            trace_execution_id=payload.trace_execution_id,
        )
        return web.json_response({
            "prompt_id": prompt_id,
            "number": prompt_number,
            "node_errors": node_errors,
            "worker_count": worker_count,
            "auto_prepare_supported": True,
        })
    except Exception as exc:
        return await handle_api_error(request, exc, 500)

@server.PromptServer.instance.routes.post("/distributed/load_image")
async def load_image_endpoint(request):
    """Load an image or video file and return it as base64 data with hash."""
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
async def check_file_endpoint(request):
    """Check if a file exists and matches the given hash."""
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


async def _enqueue_worker_items(multi_job_id, items):
    """Put decoded items on the job queue, waiting briefly for the master to create it.

    Returns the queue size after enqueueing, or None if the job never appeared.
    """
    deadline = time.monotonic() + float(JOB_INIT_GRACE_PERIOD)
    while True:
        async with prompt_server.distributed_jobs_lock:
            pending = prompt_server.distributed_pending_jobs.get(multi_job_id)
            if pending is not None:
                for item in items:
                    await pending.put(item)
                return pending.qsize()
        if time.monotonic() > deadline:
            return None
        await asyncio.sleep(0.05)


def _decode_compressed_chunk_sync(envelope, parts):
    """Decode a compressed chunk into per-frame IMAGE tensors (threadpool)."""
    frames = transfer_codec.decode_chunk(envelope, parts)
    return [
        ensure_contiguous(torch.from_numpy(transfer_codec.uint8_frame_to_tensor_array(frame)))
        for frame in frames
    ]


def _validate_compressed_envelope(envelope):
    errors = []
    if not isinstance(envelope, dict):
        return ["envelope: expected JSON object"]
    for key in ("job_id", "worker_id"):
        value = envelope.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key}: expected non-empty string")
    start_index = envelope.get("start_index")
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 0:
        errors.append("start_index: expected non-negative integer")
    if not isinstance(envelope.get("is_last"), bool):
        errors.append("is_last: expected boolean")
    try:
        if transfer_codec.normalize_format(envelope.get("format")) == transfer_codec.LEGACY_FORMAT:
            errors.append("format: legacy_png must be sent as JSON")
    except transfer_codec.TransferCodecError as exc:
        errors.append(f"format: {exc}")
    audio_payload = envelope.get("audio")
    if audio_payload is not None and not isinstance(audio_payload, dict):
        errors.append("audio: expected object when provided")
    return errors


async def _job_complete_compressed(request):
    try:
        form = await request.post()
    except Exception as exc:
        return await handle_api_error(request, f"Invalid multipart payload: {exc}", 400)

    raw_envelope = form.get("envelope")
    if isinstance(raw_envelope, (bytes, bytearray)):
        raw_envelope = raw_envelope.decode("utf-8")
    elif hasattr(raw_envelope, "file"):
        raw_envelope = raw_envelope.file.read().decode("utf-8")
    try:
        envelope = json.loads(raw_envelope) if isinstance(raw_envelope, str) else None
    except ValueError as exc:
        return await handle_api_error(request, f"envelope: invalid JSON ({exc})", 400)
    if envelope is None:
        return await handle_api_error(request, "envelope: missing field", 400)

    errors = _validate_compressed_envelope(envelope)
    if errors:
        return await handle_api_error(request, errors, 400)

    parts = []
    for field in form.getall("media", []):
        if hasattr(field, "file"):
            parts.append(field.file.read())
        elif isinstance(field, (bytes, bytearray)):
            parts.append(bytes(field))
        else:
            return await handle_api_error(request, "media: expected binary file parts", 400)

    loop = asyncio.get_running_loop()
    try:
        tensors = await loop.run_in_executor(None, _decode_compressed_chunk_sync, envelope, parts)
        decoded_audio = _decode_audio_payload(envelope.get("audio")) if envelope.get("audio") is not None else None
    except ValueError as exc:
        return await handle_api_error(request, exc, 400)

    multi_job_id = envelope["job_id"].strip()
    worker_id = envelope["worker_id"].strip()
    start_index = envelope["start_index"]
    is_last = envelope["is_last"]
    last_pos = len(tensors) - 1
    items = [
        {
            "tensor": tensor,
            "worker_id": worker_id,
            "image_index": start_index + pos,
            "is_last": bool(is_last and pos == last_pos),
            "audio": decoded_audio if pos == last_pos else None,
        }
        for pos, tensor in enumerate(tensors)
    ]

    queue_size = await _enqueue_worker_items(multi_job_id, items)
    if queue_size is None:
        return await handle_api_error(request, "job not initialized", 404)

    debug_log(
        f"job_complete received {envelope['format']} chunk - job_id: {multi_job_id}, worker: {worker_id}, "
        f"frames: {start_index}..{start_index + len(items) - 1}, "
        f"bytes: {sum(len(p) for p in parts)}, is_last: {is_last}, queue_size: {queue_size}"
    )
    return web.json_response({"status": "success", "frames": len(items)})


@server.PromptServer.instance.routes.post("/distributed/job_complete")
async def job_complete_endpoint(request):
    content_type = str(getattr(request, "content_type", "") or "").lower()
    if content_type.startswith("multipart/"):
        try:
            return await _job_complete_compressed(request)
        except Exception as e:
            return await handle_api_error(request, e)

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
        is_last = data.get("is_last")

        errors = []
        if not isinstance(job_id, str) or not job_id.strip():
            errors.append("job_id: expected non-empty string")
        if not isinstance(worker_id, str) or not worker_id.strip():
            errors.append("worker_id: expected non-empty string")
        if not isinstance(batch_idx, int) or batch_idx < 0:
            errors.append("batch_idx: expected non-negative integer")
        image_provided = image_payload is not None
        audio_provided = audio_payload is not None
        if not image_provided and not audio_provided:
            errors.append("expected at least one of image or audio")
        if image_provided and (not isinstance(image_payload, str) or not image_payload.strip()):
            errors.append("image: expected non-empty base64 PNG string")
        if audio_provided and not isinstance(audio_payload, dict):
            errors.append("audio: expected object when provided")
        if not isinstance(is_last, bool):
            errors.append("is_last: expected boolean")
        if errors:
            return await handle_api_error(request, errors, 400)

        try:
            tensor = _decode_canonical_png_tensor(image_payload) if image_provided else None
            decoded_audio = _decode_audio_payload(audio_payload) if audio_provided else None
        except ValueError as exc:
            return await handle_api_error(request, exc, 400)
        multi_job_id = job_id.strip()
        worker_id = worker_id.strip()

        queue_size = await _enqueue_worker_items(
            multi_job_id,
            [
                {
                    "tensor": tensor,
                    "worker_id": worker_id,
                    "image_index": int(batch_idx),
                    "is_last": is_last,
                    "audio": decoded_audio,
                }
            ],
        )
        if queue_size is None:
            return await handle_api_error(request, "job not initialized", 404)

        debug_log(
            f"job_complete received canonical envelope - job_id: {multi_job_id}, "
            f"worker: {worker_id}, batch_idx: {batch_idx}, is_last: {is_last}, "
            f"queue_size: {queue_size}"
        )

        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e)
