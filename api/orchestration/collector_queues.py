"""Release Collector result queues whose master prompt is gone.

A distributed run creates one result queue per Collector job before its
workers are dispatched. Normally the master's Collector removes its queue when
it finishes. If the master prompt never reaches the Collector (the user
interrupts or deletes it, or an upstream node fails), nothing else did, and the
worker results posted into the queue (decoded image tensors) stayed in memory
for the life of the process.

Queues are tied to the master prompt that will consume them, and a periodic
sweep drops the queues of prompts that are neither queued nor running.
Load-balanced runs are not registered here: they leave the master queue on
purpose while a worker runs them, and LoadBalanceQueue releases their queues.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)
SWEEP_INTERVAL_SECONDS = 30.0
_SWEEPER_MARKER = object()


def _owners(prompt_server: Any) -> dict:
    owners = getattr(prompt_server, "distributed_pending_job_owners", None)
    if owners is None:
        owners = {}
        prompt_server.distributed_pending_job_owners = owners
    return owners


def register_queue_owner(prompt_server: Any, job_ids: Iterable[str], prompt_id: str) -> None:
    """Record that ``prompt_id``'s Collectors consume these result queues."""
    if not prompt_id:
        return
    owners = _owners(prompt_server)
    for job_id in job_ids:
        owners[job_id] = prompt_id


async def release_queues(prompt_server: Any, job_ids: Iterable[str]) -> None:
    """Drop result queues no Collector will consume (the run failed to start)."""
    owners = _owners(prompt_server)
    async with prompt_server.distributed_jobs_lock:
        for job_id in job_ids:
            prompt_server.distributed_pending_jobs.pop(job_id, None)
            owners.pop(job_id, None)


def _live_prompt_ids(prompt_server: Any) -> set[str]:
    # Same reader as the image-list sweeper: ComfyUI's running and pending
    # prompt IDs, raising if the queue cannot be read.
    from ...image_list.lifecycle import _queue_prompt_ids

    return _queue_prompt_ids(prompt_server)


async def sweep_orphan_queues(prompt_server: Any) -> int:
    """Drop queues owned by prompts that left the master queue. Return the count."""
    owners = _owners(prompt_server)
    if not owners:
        return 0
    # Judge only owners registered BEFORE the queue snapshot. An owner is
    # registered after its prompt was queued, so each candidate's prompt is
    # either in the snapshot or already finished. A run that queues and
    # registers while this sweep waits for the jobs lock is not a candidate,
    # so it can never be judged against the older snapshot.
    candidates = dict(owners)
    try:
        live = _live_prompt_ids(prompt_server)
    except Exception:
        # An unreadable queue must never make a waiting run look finished.
        LOGGER.warning("Could not read the prompt queue to sweep Collector result queues", exc_info=True)
        return 0
    removed = 0
    async with prompt_server.distributed_jobs_lock:
        for job_id, prompt_id in candidates.items():
            if prompt_id in live or owners.get(job_id) != prompt_id:
                continue
            owners.pop(job_id, None)
            # The Collector usually removed its queue already; count real leaks.
            if prompt_server.distributed_pending_jobs.pop(job_id, None) is not None:
                removed += 1
    if removed:
        LOGGER.info("Released %d Collector result queue(s) whose master prompt ended before collecting", removed)
    return removed


async def _sweep_loop(prompt_server: Any, interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await sweep_orphan_queues(prompt_server)
        except Exception:
            LOGGER.exception("Collector result queue sweep failed")


def install_collector_queue_sweeper(prompt_server: Any, *, interval_seconds: float = SWEEP_INTERVAL_SECONDS) -> None:
    """Run the sweep as a server-owned task, started and stopped with the app."""
    interval_seconds = float(interval_seconds)
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be a positive finite value")
    cleanup_ctx = getattr(getattr(prompt_server, "app", None), "cleanup_ctx", None)
    if cleanup_ctx is None:
        raise TypeError("prompt_server must expose an app with cleanup_ctx")
    if any(getattr(factory, "_collector_queue_sweeper_marker", None) is _SWEEPER_MARKER
           for factory in cleanup_ctx):
        return

    async def sweeper_context(_app):
        task = asyncio.create_task(_sweep_loop(prompt_server, interval_seconds),
                                   name="distributed-collector-queue-sweeper")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    sweeper_context._collector_queue_sweeper_marker = _SWEEPER_MARKER
    cleanup_ctx.append(sweeper_context)
