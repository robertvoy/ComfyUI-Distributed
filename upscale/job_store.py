import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from ..utils.logging import debug_log
from ..utils.runtime_state import ensure_distributed_runtime_state, get_prompt_server_instance
from .job_models import BaseJobState, ImageJobState, TileJobState

# Configure maximum payload size (50MB default, configurable via environment variable)
MAX_PAYLOAD_SIZE = int(os.environ.get('COMFYUI_MAX_PAYLOAD_SIZE', str(50 * 1024 * 1024)))


@dataclass(frozen=True)
class JobQueueInitConfig:
    mode: str
    batch_size: int = 0
    num_tiles_per_image: int = 0
    all_indices: list[int] | None = None
    enabled_workers: list[str] | None = None
    batched_static: bool = False


def ensure_tile_jobs_initialized() -> Any:
    """Ensure tile job storage is initialized on the server instance."""
    prompt_server = get_prompt_server_instance()
    state = ensure_distributed_runtime_state(prompt_server)
    if not isinstance(state.distributed_pending_tile_jobs, dict):
        debug_log("Resetting invalid distributed_pending_tile_jobs state to empty dict.")
        state.distributed_pending_tile_jobs = {}
        ensure_distributed_runtime_state(prompt_server)

    invalid_job_ids = [
        job_id
        for job_id, job_data in state.distributed_pending_tile_jobs.items()
        if not isinstance(job_data, BaseJobState)
    ]
    for job_id in invalid_job_ids:
        debug_log(f"Removing invalid job state for {job_id}")
        del state.distributed_pending_tile_jobs[job_id]
    return prompt_server


async def _init_job_queue(
    multi_job_id: str,
    config: JobQueueInitConfig,
) -> None:
    """Unified initialization for job queues in static and dynamic modes."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
            debug_log(f"Queue already exists for {multi_job_id}")
            return

        if config.mode == 'dynamic':
            job_data = ImageJobState(multi_job_id=multi_job_id)
        elif config.mode == 'static':
            job_data = TileJobState(multi_job_id=multi_job_id)
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        job_data.worker_status = {w: time.time() for w in config.enabled_workers or []}
        job_data.assigned_to_workers = {w: [] for w in config.enabled_workers or []}

        if config.mode == 'dynamic':
            job_data.batch_size = int(config.batch_size or 0)
            pending_queue = job_data.pending_images
            indices = config.all_indices or list(range(int(config.batch_size or 0)))
            for i in indices:
                await pending_queue.put(i)
            debug_log(f"Initialized image queue with {config.batch_size} pending items")
        elif config.mode == 'static':
            job_data.num_tiles_per_image = int(config.num_tiles_per_image or 0)
            job_data.batch_size = int(config.batch_size or 0)
            job_data.batched_static = bool(config.batched_static)
            # For batched static distribution, populate only tile ids [0..num_tiles_per_image-1]
            pending_queue = job_data.pending_tasks
            if config.batched_static and config.num_tiles_per_image > 0:
                for i in range(config.num_tiles_per_image):
                    await pending_queue.put(i)
            else:
                total_tiles = int(config.batch_size or 0) * int(config.num_tiles_per_image or 0)
                for i in range(total_tiles):
                    await pending_queue.put(i)

        prompt_server.distributed_pending_tile_jobs[multi_job_id] = job_data


async def init_dynamic_job(
    multi_job_id: str,
    batch_size: int,
    enabled_workers: list[str],
    all_indices: list[int] | None = None,
) -> None:
    """Initialize queue for dynamic mode (per-image), with collector fields."""
    await _init_job_queue(
        multi_job_id,
        JobQueueInitConfig(
            mode='dynamic',
            batch_size=batch_size,
            all_indices=all_indices or list(range(batch_size)),
            enabled_workers=enabled_workers,
        ),
    )
    debug_log(f"Job {multi_job_id} initialized with {batch_size} images")


async def init_static_job_batched(
    multi_job_id: str,
    batch_size: int,
    num_tiles_per_image: int,
    enabled_workers: list[str],
) -> None:
    """Initialize queue for static mode (batched-per-tile)."""
    await _init_job_queue(
        multi_job_id,
        JobQueueInitConfig(
            mode='static',
            batch_size=batch_size,
            num_tiles_per_image=num_tiles_per_image,
            enabled_workers=enabled_workers,
            batched_static=True,
        ),
    )


async def drain_results_queue(multi_job_id: str) -> int:
    """Drain pending results from queue and update completed_tasks. Returns count drained."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if not isinstance(job_data, BaseJobState):
            return 0
        q = job_data.queue
        completed_tasks = job_data.completed_tasks

        collected = 0
        while True:
            try:
                result = q.get_nowait()
            except asyncio.QueueEmpty:
                break

            worker_id = result['worker_id']
            is_last = result.get('is_last', False)

            if 'image_idx' in result and 'image' in result:
                task_id = result['image_idx']
                if task_id not in completed_tasks:
                    completed_tasks[task_id] = result['image']
                    collected += 1
            elif 'tiles' in result:
                for tile_data in result['tiles']:
                    task_id = tile_data.get('global_idx', tile_data['tile_idx'])
                    if task_id not in completed_tasks:
                        completed_tasks[task_id] = tile_data
                        collected += 1
            if is_last:
                if worker_id in job_data.worker_status:
                    del job_data.worker_status[worker_id]

        return collected


async def get_completed_count(multi_job_id: str) -> int:
    """Get count of completed tasks."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if isinstance(job_data, BaseJobState):
            return len(job_data.completed_tasks)
        return 0


async def mark_task_completed(multi_job_id: str, task_id: int, result: Any) -> None:
    """Mark a task as completed."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if isinstance(job_data, BaseJobState):
            job_data.completed_tasks[task_id] = result


async def cleanup_job(multi_job_id: str) -> None:
    """Cleanup the job data."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
            del prompt_server.distributed_pending_tile_jobs[multi_job_id]
            debug_log(f"Cleaned up job {multi_job_id}")


async def init_job_queue(
    multi_job_id: str,
    config: JobQueueInitConfig,
) -> None:
    """Public entry point for unified job queue initialization."""
    await _init_job_queue(multi_job_id, config)
