import asyncio
import hashlib
import mimetypes
import os
import re
from typing import Any

import aiohttp

from ...utils.auth import distributed_auth_headers
from ...utils.config import load_config
from ...utils.logging import debug_log, log
from ...utils.network import build_worker_url, get_client_session
from ...utils.trace_logger import trace_debug, trace_info


LIKELY_FILENAME_RE = re.compile(
    r"\.(ckpt|safetensors|pt|pth|bin|yaml|json|png|jpg|jpeg|webp|gif|bmp|mp4|avi|mov|mkv|webm|"
    r"wav|mp3|flac|m4a|aac|ogg|opus|aiff|aif|wma|latent|txt|vae|lora|embedding)"
    r"(\s*\[\w+\])?$",
    re.IGNORECASE,
)
MEDIA_FILE_RE = re.compile(
    r"\.(png|jpg|jpeg|webp|gif|bmp|mp4|avi|mov|mkv|webm|wav|mp3|flac|m4a|aac|ogg|opus|aiff|aif|wma)(\s*\[\w+\])?$",
    re.IGNORECASE,
)


def _normalize_media_reference(value):
    """Normalize one media string value to a path-like reference or None."""
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s*\[\w+\]$", "", value).strip().replace("\\", "/")
    if MEDIA_FILE_RE.search(cleaned):
        return cleaned
    return None


def convert_paths_for_platform(obj: Any, target_separator: str) -> Any:
    """Recursively normalize likely file paths for the worker platform separator."""
    if target_separator not in ("/", "\\"):
        return obj

    def _convert(value):
        if isinstance(value, str):
            if ("/" in value or "\\" in value) and LIKELY_FILENAME_RE.search(value):
                trimmed = value.strip()
                has_drive = bool(re.match(r"^[A-Za-z]:(\\\\|/)", trimmed))
                is_absolute = trimmed.startswith("/") or trimmed.startswith("\\\\")
                has_protocol = bool(re.match(r"^\w+://", trimmed))

                # URLs are not local paths and should never be separator-normalized.
                if has_protocol:
                    return trimmed

                # Keep relative media-style paths in forward-slash form (Comfy-style annotated paths).
                if not has_drive and not is_absolute and not has_protocol and MEDIA_FILE_RE.search(trimmed):
                    return re.sub(r"[\\]+", "/", trimmed)

                if target_separator == "\\":
                    return re.sub(r"[\\/]+", r"\\", trimmed)
                return re.sub(r"[\\/]+", "/", trimmed)
            return value
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, dict):
            return {key: _convert(item) for key, item in value.items()}
        return value

    return _convert(obj)


def _find_media_references(prompt_obj):
    """Find media file references in image/video/audio/file inputs used by worker prompts."""
    media_refs = set()
    for node in prompt_obj.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for key in ("image", "video", "audio", "file"):
            cleaned = _normalize_media_reference(inputs.get(key))
            if cleaned:
                media_refs.add(cleaned)
    return sorted(media_refs)


def _rewrite_prompt_media_inputs(prompt_obj, worker_media_paths):
    """Rewrite media string inputs to worker-local uploaded paths."""
    if not isinstance(worker_media_paths, dict) or not worker_media_paths:
        return

    for node in prompt_obj.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key in ("image", "video", "audio", "file"):
            value = inputs.get(key)
            cleaned = _normalize_media_reference(value)
            if not cleaned:
                continue
            worker_path = worker_media_paths.get(cleaned)
            if worker_path:
                inputs[key] = worker_path


def _load_media_file_sync(filename):
    """Load local media bytes and hash for worker upload sync."""
    import folder_paths

    full_path = folder_paths.get_annotated_filepath(filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(filename)

    with open(full_path, "rb") as f:
        file_bytes = f.read()

    file_hash = hashlib.md5(file_bytes).hexdigest()
    mime_type = mimetypes.guess_type(full_path)[0]
    if not mime_type:
        ext = os.path.splitext(full_path)[1].lower()
        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            mime_type = "video/mp4"
        else:
            mime_type = "image/png"
    return file_bytes, file_hash, mime_type


async def fetch_worker_path_separator(
    worker: dict[str, Any],
    trace_execution_id: str | None = None,
) -> str | None:
    """Best-effort fetch of a worker's path separator from /distributed/system_info."""
    url = build_worker_url(worker, "/distributed/system_info")
    session = await get_client_session()
    headers = distributed_auth_headers(load_config())
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            payload = await resp.json()
            separator = ((payload or {}).get("platform") or {}).get("path_separator")
            return separator if separator in ("/", "\\") else None
    except Exception as exc:
        if trace_execution_id:
            trace_debug(trace_execution_id, f"Failed to fetch worker system info ({worker.get('id')}): {exc}")
        else:
            debug_log(f"[Distributed] Failed to fetch worker system info ({worker.get('id')}): {exc}")
        return None


async def _upload_media_to_worker(worker, filename, file_bytes, file_hash, mime_type, trace_execution_id=None):
    """Upload one media file to worker iff missing or hash-mismatched."""
    session = await get_client_session()
    headers = distributed_auth_headers(load_config())
    normalized = filename.replace("\\", "/")

    check_url = build_worker_url(worker, "/distributed/check_file")
    try:
        async with session.post(
            check_url,
            json={"filename": normalized, "hash": file_hash},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=6),
        ) as resp:
            if resp.status == 200:
                payload = await resp.json()
                if payload.get("exists") and payload.get("hash_matches"):
                    return False, normalized
    except Exception as exc:
        if trace_execution_id:
            trace_debug(trace_execution_id, f"Media check failed for '{normalized}' on worker {worker.get('id')}: {exc}")
        else:
            debug_log(f"[Distributed] Media check failed for '{normalized}' on worker {worker.get('id')}: {exc}")

    parts = normalized.split("/")
    clean_name = parts[-1]
    subfolder = "/".join(parts[:-1])

    form = aiohttp.FormData()
    form.add_field("image", file_bytes, filename=clean_name, content_type=mime_type)
    form.add_field("type", "input")
    form.add_field("subfolder", subfolder)
    form.add_field("overwrite", "true")

    upload_url = build_worker_url(worker, "/upload/image")
    async with session.post(
        upload_url,
        data=form,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        resp.raise_for_status()
        try:
            payload = await resp.json()
        except Exception:
            payload = {}

    name = str((payload or {}).get("name") or clean_name).strip()
    subfolder = str((payload or {}).get("subfolder") or "").strip().replace("\\", "/").strip("/")
    worker_path = f"{subfolder}/{name}" if subfolder else name
    return True, worker_path


async def sync_worker_media(
    worker: dict[str, Any],
    prompt_obj: dict[str, Any],
    trace_execution_id: str | None = None,
) -> None:
    """Sync referenced media files from master to a remote worker before dispatch."""
    media_refs = _find_media_references(prompt_obj)
    if not media_refs:
        return

    loop = asyncio.get_running_loop()
    uploaded = 0
    skipped = 0
    missing = 0
    worker_media_paths = {}
    for filename in media_refs:
        try:
            file_bytes, file_hash, mime_type = await loop.run_in_executor(
                None, _load_media_file_sync, filename
            )
        except FileNotFoundError:
            missing += 1
            if trace_execution_id:
                trace_info(trace_execution_id, f"Media file '{filename}' not found on master; worker may fail to load it.")
            else:
                log(f"[Distributed] Media file '{filename}' not found on master; worker may fail to load it.")
            continue
        except Exception as exc:
            if trace_execution_id:
                trace_info(trace_execution_id, f"Failed to load media '{filename}' for worker sync: {exc}")
            else:
                log(f"[Distributed] Failed to load media '{filename}' for worker sync: {exc}")
            continue

        try:
            changed, worker_path = await _upload_media_to_worker(
                worker,
                filename,
                file_bytes,
                file_hash,
                mime_type,
                trace_execution_id=trace_execution_id,
            )
            if worker_path:
                worker_media_paths[filename] = worker_path
            if changed:
                uploaded += 1
            else:
                skipped += 1
        except Exception as exc:
            if trace_execution_id:
                trace_info(trace_execution_id, f"Failed to upload media '{filename}' to worker {worker.get('id')}: {exc}")
            else:
                log(f"[Distributed] Failed to upload media '{filename}' to worker {worker.get('id')}: {exc}")

    _rewrite_prompt_media_inputs(prompt_obj, worker_media_paths)

    summary = (
        f"Media sync for worker {worker.get('id')}: "
        f"uploaded={uploaded}, skipped={skipped}, missing={missing}, referenced={len(media_refs)}"
    )
    if trace_execution_id:
        trace_debug(trace_execution_id, summary)
    else:
        debug_log(f"[Distributed] {summary}")
