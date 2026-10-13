"""Server-owned background cleanup for image-list runtime state."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Iterable
from typing import Any, cast

LOGGER = logging.getLogger(__name__)
DEFAULT_SWEEP_INTERVAL_SECONDS = 30.0
DEFAULT_ORPHAN_GRACE_SECONDS = 300.0
_LIFECYCLE_MARKER = object()


def _queue_prompt_ids(prompt_server: Any) -> set[str]:
    """Read running/pending prompt IDs, preferring ComfyUI's volatile view."""
    queue = getattr(prompt_server, "prompt_queue", None)
    readers = (
        getattr(queue, "get_current_queue_volatile", None),
        getattr(queue, "get_current_queue", None),
    )
    errors: list[Exception] = []
    for reader in readers:
        if not callable(reader):
            continue
        try:
            snapshot = reader()
            if not isinstance(snapshot, (tuple, list)) or len(snapshot) < 2:
                raise TypeError("prompt queue snapshot must contain running and pending entries")
            running, pending = snapshot[0], snapshot[1]
            prompt_ids: set[str] = set()
            for entries in (running, pending):
                if not isinstance(entries, Iterable):
                    raise TypeError("prompt queue entries must be iterable")
                for entry in entries:
                    if isinstance(entry, (tuple, list)) and len(entry) > 1:
                        prompt_id = entry[1]
                        if isinstance(prompt_id, str):
                            prompt_ids.add(prompt_id)
            return prompt_ids
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise RuntimeError("unable to read the prompt queue") from errors[-1]
    raise RuntimeError("prompt queue has no supported queue reader")


def _sweep_orphan_contexts(prompt_server: Any, runtime: Any, orphan_grace_seconds: float) -> None:
    queue_read_ok = True
    try:
        queued_prompt_ids = _queue_prompt_ids(prompt_server)
    except Exception:
        # A failed queue read must not make a submitted context look orphaned.
        LOGGER.warning("Could not read prompt queue during image-list lifecycle sweep", exc_info=True)
        queued_prompt_ids = set()
        queue_read_ok = False

    now = time.monotonic()
    with runtime._LOCK:
        for context_id, state in list(runtime._MASTER_JOBS.items()):
            if getattr(state, "active", False):
                continue
            prompt_id = getattr(state, "prompt_id", None)
            if prompt_id is not None:
                if not queue_read_ok or prompt_id in queued_prompt_ids:
                    continue
            created_at = getattr(state, "created_at", None)
            expired = (
                isinstance(created_at, (int, float))
                and not isinstance(created_at, bool)
                and math.isfinite(created_at)
                and now - created_at >= orphan_grace_seconds
            )
            if expired and runtime._MASTER_JOBS.get(context_id) is state:
                runtime.remove_master_job(context_id)


def _sweep_once(prompt_server: Any, runtime: Any, orphan_grace_seconds: float) -> None:
    try:
        runtime._sweep_worker_items()
    except Exception:
        LOGGER.exception("Could not sweep expired image-list worker staging")
    _sweep_orphan_contexts(prompt_server, runtime, orphan_grace_seconds)


async def _sweep_loop(
    prompt_server: Any,
    runtime: Any,
    interval_seconds: float,
    orphan_grace_seconds: float,
) -> None:
    while True:
        try:
            _sweep_once(prompt_server, runtime, orphan_grace_seconds)
        except Exception:
            # Keep the server-owned task alive if one pass encounters unexpected state.
            LOGGER.exception("Image-list lifecycle sweep failed")
        await asyncio.sleep(interval_seconds)


def install_lifecycle(
    prompt_server: Any,
    runtime: Any,
    *,
    interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
    orphan_grace_seconds: float = DEFAULT_ORPHAN_GRACE_SECONDS,
) -> None:
    """Register one cancellable periodic sweep in the prompt server's app context."""
    interval_seconds = float(interval_seconds)
    orphan_grace_seconds = float(orphan_grace_seconds)
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be a positive finite value")
    if not math.isfinite(orphan_grace_seconds) or orphan_grace_seconds < 0:
        raise ValueError("orphan_grace_seconds must be a non-negative finite value")

    app = getattr(prompt_server, "app", None)
    cleanup_ctx = getattr(app, "cleanup_ctx", None)
    if cleanup_ctx is None:
        raise TypeError("prompt_server must expose an app with cleanup_ctx")
    if any(getattr(factory, "_image_list_lifecycle_marker", None) is _LIFECYCLE_MARKER
           for factory in cleanup_ctx):
        return

    async def lifecycle_context(_app):
        task = asyncio.create_task(
            _sweep_loop(prompt_server, runtime, interval_seconds, orphan_grace_seconds),
            name="image-list-lifecycle-sweeper",
        )
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    cast(Any, lifecycle_context)._image_list_lifecycle_marker = _LIFECYCLE_MARKER
    cleanup_ctx.append(lifecycle_context)
