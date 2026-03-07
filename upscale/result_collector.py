import asyncio, time
import comfy.model_management
import server
from ..utils.constants import DYNAMIC_MODE_MAX_POLL_TIMEOUT, HEARTBEAT_INTERVAL
from ..utils.logging import debug_log, log
from ..utils.config import get_worker_timeout_seconds
from .job_store import ensure_tile_jobs_initialized, mark_task_completed
from .job_timeout import check_and_requeue_timed_out_workers
from .job_models import BaseJobState, ImageJobState, TileJobState


class ResultCollectorMixin:
    """
    Mixin for master-side result collection in USDU distributed jobs.

    Expected co-mixins/attributes:
    - JobStateMixin methods for queue/task access.
    - `self._check_and_requeue_timed_out_workers(...)` coroutine.
    - `self.async_yield(...)` optional helper from WorkerCommsMixin.
    """

    def _log_worker_timeout_status(self, job_data, current_time: float, multi_job_id: str) -> list[str]:
        """Log timeout elapsed seconds for each tracked worker and return worker ids."""
        if not isinstance(job_data, BaseJobState):
            return []

        worker_status = dict(job_data.worker_status)
        for worker_id, last_seen in worker_status.items():
            elapsed = max(0.0, current_time - float(last_seen))
            log(
                "UltimateSDUpscale Master - Heartbeat timeout: "
                f"job={multi_job_id}, worker={worker_id}, elapsed={elapsed:.1f}s"
            )
        return list(worker_status.keys())

    async def _check_and_requeue_timed_out_workers(self, multi_job_id, batch_size):
        """Default timeout requeue hook; override in host mixins when needed."""
        return await check_and_requeue_timed_out_workers(multi_job_id, batch_size)

    async def _async_collect_results(self, multi_job_id, num_workers, mode='static', 
                                   remaining_to_collect=None, batch_size=None):
        """Unified async helper to collect results from workers (tiles or images)."""
        # Get the already initialized queue
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                raise RuntimeError(f"Job queue not initialized for {multi_job_id}")
            job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
            if mode == 'dynamic':
                if not isinstance(job_data, ImageJobState):
                    raise RuntimeError(
                        f"Mode mismatch: expected dynamic, got {getattr(job_data, 'mode', 'unknown')}"
                    )
                q = job_data.queue
                completed_images = job_data.completed_images
                expected_count = remaining_to_collect or batch_size
            elif mode == 'static':
                if not isinstance(job_data, TileJobState):
                    raise RuntimeError(
                        f"Mode mismatch: expected static, got {getattr(job_data, 'mode', 'unknown')}"
                    )
                q = job_data.queue
                expected_count = len(job_data.completed_tasks) + job_data.pending_tasks.qsize()
            else:
                raise RuntimeError(f"Unsupported mode: {mode}")
        
        item_type = "images" if mode == 'dynamic' else "tiles"
        debug_log(f"UltimateSDUpscale Master - Starting collection, expecting {expected_count} {item_type} from {num_workers} workers")
        
        collected_results = {}
        workers_done = set()
        # Unify collector/upscaler wait behavior with the UI worker timeout
        timeout = float(get_worker_timeout_seconds())
        last_heartbeat_check = time.time()
        wait_started_at = time.time()
        collected_count = 0
        
        while len(workers_done) < num_workers:
            # Check for user interruption
            if comfy.model_management.processing_interrupted():
                log("Processing interrupted by user")
                raise comfy.model_management.InterruptProcessingException()
                
            # For dynamic mode with remaining_to_collect, check if we've collected enough
            if mode == 'dynamic' and remaining_to_collect and collected_count >= remaining_to_collect:
                break

            job_data_snapshot = None
            async with prompt_server.distributed_tile_jobs_lock:
                current_job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
                if isinstance(current_job_data, BaseJobState):
                    job_data_snapshot = current_job_data

            try:
                # Shorter poll for dynamic mode, but never exceed the configured timeout
                wait_timeout = (min(DYNAMIC_MODE_MAX_POLL_TIMEOUT, timeout) if mode == 'dynamic' else timeout)
                result = await asyncio.wait_for(q.get(), timeout=wait_timeout)
                worker_id = result['worker_id']
                is_last = result.get('is_last', False)
                
                if mode == 'static':
                    # Handle tiles
                    tiles = result.get('tiles', [])
                    debug_log(
                        f"UltimateSDUpscale Master - Received batch of {len(tiles)} tiles from worker "
                        f"'{worker_id}' (is_last={is_last})"
                    )

                    for tile_data in tiles:
                        if 'batch_idx' not in tile_data:
                            log("UltimateSDUpscale Master - Missing batch_idx in tile data, skipping")
                            continue

                        tile_idx = tile_data['tile_idx']
                        key = tile_data.get('global_idx', tile_idx)
                        entry = {
                            **tile_data,
                            'tile_idx': tile_idx,
                            'worker_id': worker_id,
                            'global_idx': key,
                        }
                        _ = entry['tile_idx'], entry['worker_id'], entry['global_idx']
                        collected_results[entry['global_idx']] = entry
                
                elif mode == 'dynamic':
                    # Handle full images
                    if 'image_idx' in result and 'image' in result:
                        image_idx = result['image_idx']
                        image_pil = result['image']
                        completed_images[image_idx] = image_pil
                        collected_results[image_idx] = image_pil
                        collected_count += 1
                        debug_log(f"UltimateSDUpscale Master - Received image {image_idx} from worker {worker_id}")
                
                if is_last:
                    workers_done.add(worker_id)
                    debug_log(f"UltimateSDUpscale Master - Worker {worker_id} completed")
                    
            except asyncio.TimeoutError:
                current_time = time.time()
                waiting_workers = type(self)._log_worker_timeout_status(
                    self,
                    job_data_snapshot,
                    current_time,
                    multi_job_id,
                )
                if mode == 'dynamic':
                    # Check for worker timeouts periodically
                    if current_time - last_heartbeat_check >= HEARTBEAT_INTERVAL:
                        # Use the class hook (can be overridden by host mixins/tests).
                        requeued = await type(self)._check_and_requeue_timed_out_workers(
                            self,
                            multi_job_id,
                            batch_size,
                        )
                        if requeued > 0:
                            log(f"UltimateSDUpscale Master - Requeued {requeued} images from timed out workers")
                        last_heartbeat_check = current_time
                    
                    # Check if we've been waiting too long overall
                    if current_time - wait_started_at > timeout:
                        elapsed = current_time - wait_started_at
                        log(
                            "UltimateSDUpscale Master - Heartbeat timeout while waiting for images; "
                            f"workers={waiting_workers}, elapsed={elapsed:.1f}s"
                        )
                        break
                else:
                    elapsed = current_time - wait_started_at
                    log(
                        f"UltimateSDUpscale Master - Heartbeat timeout waiting for {item_type}; "
                        f"workers={waiting_workers}, elapsed={elapsed:.1f}s"
                    )
                    break
        
        debug_log(f"UltimateSDUpscale Master - Collection complete. Got {len(collected_results)} {item_type} from {len(workers_done)} workers")
        
        # Clean up job queue
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                del prompt_server.distributed_pending_tile_jobs[multi_job_id]
        
        return collected_results if mode == 'static' else completed_images

    async def _async_collect_worker_tiles(self, multi_job_id, num_workers):
        """Async helper to collect tiles from workers."""
        return await type(self)._async_collect_results(self, multi_job_id, num_workers, mode='static')

    async def _mark_image_completed(self, multi_job_id, image_idx, image_pil):
        """Mark an image as completed in the job data."""
        # Mark the image as completed with the image data
        await mark_task_completed(multi_job_id, image_idx, {'image': image_pil})
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if isinstance(job_data, ImageJobState):
                job_data.completed_images[image_idx] = image_pil

    async def mark_image_completed(self, multi_job_id, image_idx, image_pil):
        """Public image-completion API."""
        await self._mark_image_completed(multi_job_id, image_idx, image_pil)

    async def _async_collect_dynamic_images(self, multi_job_id, remaining_to_collect, num_workers, batch_size, master_processed_count):
        """Collect remaining processed images from workers."""
        _ = master_processed_count
        return await type(self)._async_collect_results(
            self,
            multi_job_id,
            num_workers,
            mode='dynamic',
            remaining_to_collect=remaining_to_collect,
            batch_size=batch_size,
        )

    async def collect_dynamic_images(
        self,
        multi_job_id,
        remaining_to_collect,
        num_workers,
        batch_size,
        master_processed_count,
    ):
        """Public dynamic image-collection API."""
        return await self._async_collect_dynamic_images(
            multi_job_id,
            remaining_to_collect,
            num_workers,
            batch_size,
            master_processed_count,
        )
