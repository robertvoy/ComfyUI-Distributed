import asyncio, time, torch
from PIL import Image
import comfy.model_management
from ...utils.logging import debug_log, log
from ...utils.image import tensor_to_pil, pil_to_tensor
from ...utils.async_helpers import run_async_in_server_loop
from ...utils.config import get_worker_timeout_seconds
from ...utils.constants import (
    HEARTBEAT_INTERVAL,
    JOB_POLL_INTERVAL,
    JOB_POLL_MAX_ATTEMPTS,
    MAX_BATCH,
    TILE_SEND_TIMEOUT,
    TILE_WAIT_TIMEOUT,
)
from ..job_store import (
    ensure_tile_jobs_initialized, init_static_job_batched,
    _mark_task_completed, _cleanup_job, _drain_results_queue, _get_completed_count,
)
from ..job_models import TileJobState


class StaticModeMixin:
    """
    Static (tile-queue) USDU mode behaviors for master and worker roles.

    Expected co-mixins on `self`:
    - TileOpsMixin (`calculate_tiles`, tile extract/blend helpers).
    - JobStateMixin (`_get_next_tile_index`, `_get_all_completed_tasks`, requeue checks).
    - WorkerCommsMixin (`send_tiles_batch_to_master`, `_request_tile_from_master`, `_send_heartbeat_to_master`).
    """

    def _poll_job_ready(self, multi_job_id, master_url, worker_id=None, max_attempts=JOB_POLL_MAX_ATTEMPTS):
        """Poll master for job readiness to avoid worker/master initialization race."""
        for attempt in range(max_attempts):
            ready = run_async_in_server_loop(
                self._check_job_status(multi_job_id, master_url),
                timeout=5.0
            )
            if ready:
                if worker_id:
                    debug_log(f"Worker[{worker_id[:8]}] job {multi_job_id} ready after {attempt} attempts")
                else:
                    debug_log(f"Job {multi_job_id} ready after {attempt} attempts")
                return True
            time.sleep(JOB_POLL_INTERVAL)
        return False

    def _extract_and_process_tile(
        self,
        upscaled_image,
        tile_id,
        all_tiles,
        tile_width,
        tile_height,
        padding,
        force_uniform_tiles,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        tiled_decode,
        width,
        height,
    ):
        """Extract one tile position for the whole batch and process it."""
        tx, ty = all_tiles[tile_id]
        tile_batch, x1, y1, ew, eh = self.extract_batch_tile_with_padding(
            upscaled_image, tx, ty, tile_width, tile_height, padding, force_uniform_tiles
        )
        region = (x1, y1, x1 + ew, y1 + eh)
        processed_batch = self.process_tiles_batch(
            tile_batch, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, tiled_decode,
            region, (width, height)
        )
        return processed_batch, x1, y1, ew, eh

    def _flush_tiles_to_master(
        self,
        processed_tiles,
        multi_job_id,
        master_url,
        padding,
        worker_id,
        is_final_flush=False,
    ):
        """Send accumulated tile payloads to master and return a fresh accumulator."""
        if not processed_tiles:
            if is_final_flush:
                run_async_in_server_loop(
                    self.send_tiles_batch_to_master(
                        [],
                        multi_job_id,
                        master_url,
                        padding,
                        worker_id,
                        is_final_flush=True,
                    ),
                    timeout=TILE_SEND_TIMEOUT,
                )
            return processed_tiles
        run_async_in_server_loop(
            self.send_tiles_batch_to_master(
                processed_tiles,
                multi_job_id,
                master_url,
                padding,
                worker_id,
                is_final_flush=is_final_flush,
            ),
            timeout=TILE_SEND_TIMEOUT
        )
        return []

    def _master_process_one_tile(
        self,
        tile_id,
        all_tiles,
        upscaled_image,
        result_images,
        tile_masks,
        multi_job_id,
        batch_size,
        num_tiles_per_image,
        tile_width,
        tile_height,
        padding,
        force_uniform_tiles,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        tiled_decode,
        width,
        height,
    ):
        """Process one tile_id across the batch and blend into result_images."""
        source_batch = torch.cat([pil_to_tensor(img) for img in result_images], dim=0)
        if upscaled_image.is_cuda:
            source_batch = source_batch.cuda()
        processed_batch, x1, y1, ew, eh = self._extract_and_process_tile(
            source_batch,
            tile_id,
            all_tiles,
            tile_width,
            tile_height,
            padding,
            force_uniform_tiles,
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            tiled_decode,
            width,
            height,
        )
        tile_mask = tile_masks[tile_id]
        out_bs = processed_batch.shape[0] if hasattr(processed_batch, "shape") else batch_size
        processed_items = min(batch_size, out_bs)
        for b in range(processed_items):
            tile_pil = tensor_to_pil(processed_batch, b)
            if tile_pil.size != (ew, eh):
                tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
            result_images[b] = self.blend_tile(result_images[b], tile_pil, x1, y1, (ew, eh), tile_mask, padding)
            global_idx = b * num_tiles_per_image + tile_id
            run_async_in_server_loop(
                _mark_task_completed(multi_job_id, global_idx, {'batch_idx': b, 'tile_idx': tile_id}),
                timeout=5.0
            )
        return processed_items

    def _process_worker_static_sync(self, upscaled_image, model, positive, negative, vae,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    tile_width, tile_height, padding, mask_blur,
                                    force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                    worker_id, enabled_workers):
        """Worker static mode processing with optional dynamic queue pulling."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Get dimensions and calculate tiles
        _, height, width, _ = upscaled_image.shape
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        num_tiles_per_image = len(all_tiles)
        batch_size = upscaled_image.shape[0]
        total_tiles = batch_size * num_tiles_per_image
        
        processed_tiles = []
        working_images = []
        for b in range(batch_size):
            image_pil = tensor_to_pil(upscaled_image[b:b+1], 0)
            working_images.append(image_pil.copy())
        tile_masks = []
        for tx, ty in all_tiles:
            tile_masks.append(self.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur))
        
        # Dynamic queue mode (static processing): process batched-per-tile
        log(f"USDU Dist Worker[{worker_id[:8]}]: Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Tiles/image {num_tiles_per_image} | Batch {batch_size}")
        processed_count = 0

        max_poll_attempts = JOB_POLL_MAX_ATTEMPTS
        if not self._poll_job_ready(multi_job_id, master_url, worker_id=worker_id, max_attempts=max_poll_attempts):
            log(f"Job {multi_job_id} not ready after {max_poll_attempts} attempts, aborting")
            return (upscaled_image,)

        # Main processing loop - pull tile ids from queue
        while True:
            # Request a tile to process
            tile_idx, estimated_remaining, batched_static = run_async_in_server_loop(
                self._request_tile_from_master(multi_job_id, master_url, worker_id),
                timeout=TILE_WAIT_TIMEOUT,
                cancel_on_timeout=True,
            )

            if tile_idx is None:
                debug_log(f"Worker[{worker_id[:8]}] - No more tiles to process")
                break

            # Always batched-per-tile in static mode
            debug_log(f"Worker[{worker_id[:8]}] - Assigned tile_id {tile_idx}")
            processed_count += batch_size
            tile_id = tile_idx
            source_batch = torch.cat([pil_to_tensor(img) for img in working_images], dim=0)
            if upscaled_image.is_cuda:
                source_batch = source_batch.cuda()
            processed_batch, x1, y1, ew, eh = self._extract_and_process_tile(
                source_batch,
                tile_id,
                all_tiles,
                tile_width,
                tile_height,
                padding,
                force_uniform_tiles,
                model,
                positive,
                negative,
                vae,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                tiled_decode,
                width,
                height,
            )
            # Queue results
            for b in range(batch_size):
                tile_pil = tensor_to_pil(processed_batch, b)
                if tile_pil.size != (ew, eh):
                    tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
                working_images[b] = self.blend_tile(
                    working_images[b],
                    tile_pil,
                    x1,
                    y1,
                    (ew, eh),
                    tile_masks[tile_id],
                    padding,
                )
                processed_tiles.append({
                    'tile': processed_batch[b:b+1],
                    'tile_idx': tile_id,
                    'x': x1,
                    'y': y1,
                    'extracted_width': ew,
                    'extracted_height': eh,
                    'padding': padding,
                    'batch_idx': b,
                    'global_idx': b * num_tiles_per_image + tile_id
                })

            # Send heartbeat
            try:
                run_async_in_server_loop(
                    self._send_heartbeat_to_master(multi_job_id, master_url, worker_id),
                    timeout=5.0
                )
            except Exception as e:
                debug_log(f"Worker[{worker_id[:8]}] heartbeat failed: {e}")

            # Send tiles in batches within loop
            if len(processed_tiles) >= MAX_BATCH:
                processed_tiles = self._flush_tiles_to_master(
                    processed_tiles, multi_job_id, master_url, padding, worker_id, is_final_flush=False
                )

        # Send any remaining tiles
        processed_tiles = self._flush_tiles_to_master(
            processed_tiles, multi_job_id, master_url, padding, worker_id, is_final_flush=True
        )
        
        debug_log(f"Worker {worker_id} completed all assigned and requeued tiles")
        return (upscaled_image,)

    async def _async_collect_and_monitor_static(self, multi_job_id, total_tiles, expected_total):
        """Async helper for collection and monitoring in static mode.
        Returns collected tasks dict. Caller should check if all tasks are complete."""
        last_progress_log = time.time()
        progress_interval = 5.0
        last_heartbeat_check = time.time()
        last_completed_count = 0
        
        while True:
            # Check for user interruption
            if comfy.model_management.processing_interrupted():
                log("Processing interrupted by user")
                raise comfy.model_management.InterruptProcessingException()
            
            # Drain any pending results
            collected_count = await _drain_results_queue(multi_job_id)
            
            # Check and requeue timed-out workers periodically
            current_time = time.time()
            if current_time - last_heartbeat_check >= HEARTBEAT_INTERVAL:
                requeued_count = await self._check_and_requeue_timed_out_workers(multi_job_id, expected_total)
                if requeued_count > 0:
                    log(f"Requeued {requeued_count} tasks from timed-out workers")
                last_heartbeat_check = current_time
            
            # Get current completion count
            completed_count = await _get_completed_count(multi_job_id)
            
            # Progress logging
            if current_time - last_progress_log >= progress_interval:
                log(f"Progress: {completed_count}/{expected_total} tasks completed")
                last_progress_log = current_time
            
            # Check if all tasks are completed
            if completed_count >= expected_total:
                debug_log(f"All {expected_total} tasks completed")
                break
            
            # If no active workers remain and there are pending tasks, return for local processing
            prompt_server = ensure_tile_jobs_initialized()
            async with prompt_server.distributed_tile_jobs_lock:
                job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
                if isinstance(job_data, TileJobState):
                    pending_queue = job_data.pending_tasks
                    active_workers = list(job_data.worker_status.keys())
                    if pending_queue and not pending_queue.empty() and len(active_workers) == 0:
                        log(f"No active workers remaining with {expected_total - completed_count} tasks pending. Returning for local processing.")
                        break
            
            # Wait a bit before next check
            await asyncio.sleep(0.1)
        
        # Get all completed tasks for return
        return await self._get_all_completed_tasks(multi_job_id)

    def _cleanup_static_job(self, multi_job_id):
        """Remove the master's static job state; never masks the in-flight outcome.

        Called from a ``finally`` block, so a failing cleanup is logged instead of
        replacing the result being returned or the exception being raised (for
        example, turning a user interrupt into a TimeoutError).
        """
        try:
            # Never abandon the cleanup: on a slow loop it still finishes late.
            run_async_in_server_loop(_cleanup_job(multi_job_id), timeout=5.0,
                                     cancel_on_timeout=False)
        except Exception as exc:
            log(f"UltimateSDUpscale Master - Failed to clean up job {multi_job_id}: {exc}")

    def _process_master_static_sync(self, upscaled_image, model, positive, negative, vae,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    tile_width, tile_height, padding, mask_blur,
                                    force_uniform_tiles, tiled_decode, multi_job_id, enabled_workers,
                                    all_tiles, num_tiles_per_image):
        """Static mode master processing with optional dynamic queue pulling."""
        batch_size = upscaled_image.shape[0]
        _, height, width, _ = upscaled_image.shape
        total_tiles = batch_size * num_tiles_per_image
        
        # Convert batch to PIL list for processing
        result_images = []
        for b in range(batch_size):
            image_pil = tensor_to_pil(upscaled_image[b:b+1], 0)
            result_images.append(image_pil.copy())
        
        # Initialize queue: pending queue holds tile ids (batched per tile)
        log("USDU Dist: Using tile queue distribution")
        # From here on the job is registered in distributed_pending_tile_jobs, so every
        # exit (success, user interrupt, or any exception in the master tile loop,
        # worker collection, or local fallback) must remove it exactly once. Otherwise
        # workers keep pulling the cancelled job's tiles and decoded results pile up
        # in its queue.
        try:
            run_async_in_server_loop(
                init_static_job_batched(multi_job_id, batch_size, num_tiles_per_image, enabled_workers),
                timeout=10.0
            )
            debug_log(
                f"Initialized tile-id queue with {num_tiles_per_image} ids for batch {batch_size}"
            )

            # Precompute masks for all tile positions to avoid repeated Gaussian blur work during blending
            tile_masks = []
            for idx, (tx, ty) in enumerate(all_tiles):
                tile_masks.append(self.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur))

            processed_count = 0
            consecutive_no_tile = 0
            max_consecutive_no_tile = 2

            while processed_count < total_tiles:
                comfy.model_management.throw_exception_if_processing_interrupted()
                tile_idx = run_async_in_server_loop(
                    self._get_next_tile_index(multi_job_id),
                    timeout=5.0,
                    cancel_on_timeout=True,
                )
                if tile_idx is not None:
                    consecutive_no_tile = 0
                    tile_id = tile_idx
                    processed_count += self._master_process_one_tile(
                        tile_id,
                        all_tiles,
                        upscaled_image,
                        result_images,
                        tile_masks,
                        multi_job_id,
                        batch_size,
                        num_tiles_per_image,
                        tile_width,
                        tile_height,
                        padding,
                        force_uniform_tiles,
                        model,
                        positive,
                        negative,
                        vae,
                        seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        denoise,
                        tiled_decode,
                        width,
                        height,
                    )
                    log(f"USDU Dist: Tiles progress {processed_count}/{total_tiles} (tile {tile_id})")
                else:
                    consecutive_no_tile += 1
                    if consecutive_no_tile >= max_consecutive_no_tile:
                        debug_log(f"Master processed {processed_count} tiles, moving to collection phase")
                        break
                    time.sleep(0.1)
            master_processed_count = processed_count
        
            # Continue processing any remaining tiles while collecting worker results
            remaining_tiles = total_tiles - master_processed_count
            if remaining_tiles > 0:
                debug_log(f"Master waiting for {remaining_tiles} tiles from workers")
            
                # Collect worker results using async operations
                # Wait until either all tasks are collected or there are no active workers left
                collected_tasks = run_async_in_server_loop(
                    self._async_collect_and_monitor_static(multi_job_id, total_tiles, expected_total=total_tiles),
                    timeout=None
                )
            
                # Check if we need to process any remaining tasks locally after collection
                completed_count = len(collected_tasks)
                if completed_count < total_tiles:
                    log(f"Processing remaining {total_tiles - completed_count} tasks locally after worker failures")
                
                    # Process any remaining pending tasks (batched-per-tile)
                    while True:
                        # Check for user interruption
                        comfy.model_management.throw_exception_if_processing_interrupted()

                        # Get next tile_id from pending queue
                        tile_id = run_async_in_server_loop(
                            self._get_next_tile_index(multi_job_id),
                            timeout=5.0,
                            cancel_on_timeout=True,
                        )

                        if tile_id is None:
                            break

                        self._master_process_one_tile(
                            tile_id,
                            all_tiles,
                            upscaled_image,
                            result_images,
                            tile_masks,
                            multi_job_id,
                            batch_size,
                            num_tiles_per_image,
                            tile_width,
                            tile_height,
                            padding,
                            force_uniform_tiles,
                            model,
                            positive,
                            negative,
                            vae,
                            seed,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            denoise,
                            tiled_decode,
                            width,
                            height,
                        )
            else:
                # Master processed all tiles
                collected_tasks = run_async_in_server_loop(
                    self._get_all_completed_tasks(multi_job_id),
                    timeout=5.0
                )
        
            # Blend worker tiles synchronously in deterministic tile order.
            def _sort_key(item):
                global_idx, tile_data = item
                batch_idx = tile_data.get('batch_idx', global_idx // num_tiles_per_image)
                tile_idx = tile_data.get('tile_idx', global_idx % num_tiles_per_image)
                return (tile_idx, batch_idx, global_idx)

            for global_idx, tile_data in sorted(collected_tasks.items(), key=_sort_key):
                # Skip tiles that don't have tensor data (already processed)
                if 'tensor' not in tile_data and 'image' not in tile_data:
                    continue
            
                batch_idx = tile_data.get('batch_idx', global_idx // num_tiles_per_image)
                tile_idx = tile_data.get('tile_idx', global_idx % num_tiles_per_image)
            
                if batch_idx >= batch_size:
                    continue
            
                # Blend tile synchronously
                x = tile_data.get('x', 0)
                y = tile_data.get('y', 0)
                # Prefer PIL image if present to avoid reconversion
                if 'image' in tile_data:
                    tile_pil = tile_data['image']
                else:
                    tile_tensor = tile_data['tensor']
                    tile_pil = tensor_to_pil(tile_tensor, 0)
                orig_x, orig_y = all_tiles[tile_idx]
                tile_mask = tile_masks[tile_idx]
                extracted_width = tile_data.get('extracted_width', tile_width + 2 * padding)
                extracted_height = tile_data.get('extracted_height', tile_height + 2 * padding)
                result_images[batch_idx] = self.blend_tile(result_images[batch_idx], tile_pil,
                                                          x, y, (extracted_width, extracted_height), tile_mask, padding)
        
            # Convert back to tensor
            if batch_size == 1:
                result_tensor = pil_to_tensor(result_images[0])
            else:
                result_tensors = [pil_to_tensor(img) for img in result_images]
                result_tensor = torch.cat(result_tensors, dim=0)
            
            if upscaled_image.is_cuda:
                result_tensor = result_tensor.cuda()
            
            log(f"UltimateSDUpscale Master - Job {multi_job_id} complete")
            return (result_tensor,)
        finally:
            self._cleanup_static_job(multi_job_id)
