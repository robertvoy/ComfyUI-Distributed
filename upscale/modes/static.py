import asyncio, time, torch
from PIL import Image
import comfy.model_management
from ...utils.logging import debug_log, log
from ...utils.image import blend_processed_batch_item, pil_to_tensor, tensor_to_pil
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
    cleanup_job, drain_results_queue, get_completed_count, mark_task_completed,
)
from ..job_models import TileJobState
from ..mode_contexts import (
    JobStateCollaborator,
    StaticModeContext,
    TileOpsCollaborator,
    WorkerCommsCollaborator,
)
from ..processing_args import UpscaleCoreArgs
from ..tile_processing import TileBatchArgs, extract_and_process_tile_batch


class StaticModeMixin:
    """
    Static (tile-queue) USDU mode behaviors for master and worker roles.

    Expected co-mixins on `self`:
    - TileOpsMixin (`calculate_tiles`, tile extract/blend helpers).
    - JobStateMixin (`_get_next_tile_index`, `_get_all_completed_tasks`, requeue checks).
    - WorkerCommsMixin (`send_tiles_batch`, `request_assignment`, `send_heartbeat`).
    """

    def _build_static_mode_context(self) -> StaticModeContext:
        """Build explicit collaborators for static mode execution."""
        return StaticModeContext(
            tile_ops=TileOpsCollaborator(self),
            job_state=JobStateCollaborator(self),
            worker_comms=WorkerCommsCollaborator(self),
        )

    def _poll_job_ready(
        self,
        multi_job_id,
        master_url,
        worker_id=None,
        max_attempts=JOB_POLL_MAX_ATTEMPTS,
        mode_context: StaticModeContext | None = None,
    ):
        """Poll master for job readiness to avoid worker/master initialization race."""
        context = mode_context or self._build_static_mode_context()
        for attempt in range(max_attempts):
            ready = run_async_in_server_loop(
                context.worker_comms.check_job_status(multi_job_id, master_url),
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
        tile_batch_args = TileBatchArgs(
            core=UpscaleCoreArgs(
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                tiled_decode=tiled_decode,
            ),
            tile_width=tile_width,
            tile_height=tile_height,
            padding=padding,
            force_uniform_tiles=force_uniform_tiles,
            width=width,
            height=height,
        )
        processed_batch, x1, y1, ew, eh = extract_and_process_tile_batch(
            node=self,
            upscaled_image=upscaled_image,
            tx=tx,
            ty=ty,
            args=tile_batch_args,
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
        mode_context: StaticModeContext | None = None,
    ):
        """Send accumulated tile payloads to master and return a fresh accumulator."""
        context = mode_context or self._build_static_mode_context()
        if not processed_tiles:
            if is_final_flush:
                run_async_in_server_loop(
                    context.worker_comms.send_tiles_batch(
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
            context.worker_comms.send_tiles_batch(
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
        mode_context: StaticModeContext | None = None,
    ):
        """Process one tile_id across the batch and blend into result_images."""
        context = mode_context or self._build_static_mode_context()
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
            blend_processed_batch_item(
                result_images,
                processed_batch,
                b,
                context.tile_ops.blend_tile,
                x1,
                y1,
                ew,
                eh,
                tile_mask,
                padding,
            )
            global_idx = b * num_tiles_per_image + tile_id
            run_async_in_server_loop(
                mark_task_completed(multi_job_id, global_idx, {'batch_idx': b, 'tile_idx': tile_id}),
                timeout=5.0
            )
        return processed_items

    def _process_worker_static_sync(self, upscaled_image, model, positive, negative, vae,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    tile_width, tile_height, padding, mask_blur,
                                    force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                    worker_id, enabled_workers,
                                    mode_context: StaticModeContext | None = None):
        """Worker static mode processing with optional dynamic queue pulling."""
        context = mode_context or self._build_static_mode_context()
        # Round tile dimensions
        tile_width = context.tile_ops.round_to_multiple(tile_width)
        tile_height = context.tile_ops.round_to_multiple(tile_height)
        
        # Get dimensions and calculate tiles
        _, height, width, _ = upscaled_image.shape
        all_tiles = context.tile_ops.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
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
            tile_masks.append(context.tile_ops.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur))
        
        # Dynamic queue mode (static processing): process batched-per-tile
        log(f"USDU Dist Worker[{worker_id[:8]}]: Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Tiles/image {num_tiles_per_image} | Batch {batch_size}")
        processed_count = 0

        max_poll_attempts = JOB_POLL_MAX_ATTEMPTS
        if not self._poll_job_ready(
            multi_job_id,
            master_url,
            worker_id=worker_id,
            max_attempts=max_poll_attempts,
            mode_context=context,
        ):
            log(f"Job {multi_job_id} not ready after {max_poll_attempts} attempts, aborting")
            return (upscaled_image,)

        # Main processing loop - pull tile ids from queue
        while True:
            # Request a tile to process
            assignment = run_async_in_server_loop(
                context.worker_comms.request_assignment(multi_job_id, master_url, worker_id),
                timeout=TILE_WAIT_TIMEOUT,
            )
            tile_idx = assignment.task_idx if assignment.kind == "tile" else None

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
                working_images[b] = context.tile_ops.blend_tile(
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
                    context.worker_comms.send_heartbeat(multi_job_id, master_url, worker_id),
                    timeout=5.0
                )
            except Exception as e:
                debug_log(f"Worker[{worker_id[:8]}] heartbeat failed: {e}")

            # Send tiles in batches within loop
            if len(processed_tiles) >= MAX_BATCH:
                processed_tiles = self._flush_tiles_to_master(
                    processed_tiles,
                    multi_job_id,
                    master_url,
                    padding,
                    worker_id,
                    is_final_flush=False,
                    mode_context=context,
                )

        # Send any remaining tiles
        processed_tiles = self._flush_tiles_to_master(
            processed_tiles,
            multi_job_id,
            master_url,
            padding,
            worker_id,
            is_final_flush=True,
            mode_context=context,
        )
        
        debug_log(f"Worker {worker_id} completed all assigned and requeued tiles")
        return (upscaled_image,)

    async def _async_collect_and_monitor_static(
        self,
        multi_job_id,
        total_tiles,
        expected_total,
        mode_context: StaticModeContext | None = None,
    ):
        """Async helper for collection and monitoring in static mode.
        Returns collected tasks dict. Caller should check if all tasks are complete."""
        context = mode_context or self._build_static_mode_context()
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
            collected_count = await drain_results_queue(multi_job_id)
            
            # Check and requeue timed-out workers periodically
            current_time = time.time()
            if current_time - last_heartbeat_check >= HEARTBEAT_INTERVAL:
                requeued_count = await context.job_state.check_and_requeue_timed_out_workers(multi_job_id, expected_total)
                if requeued_count > 0:
                    log(f"Requeued {requeued_count} tasks from timed-out workers")
                last_heartbeat_check = current_time
            
            # Get current completion count
            completed_count = await get_completed_count(multi_job_id)
            
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
        return await context.job_state.all_completed_tasks(multi_job_id)

    def _init_master_static_state(
        self,
        upscaled_image,
        multi_job_id,
        batch_size,
        num_tiles_per_image,
        enabled_workers,
        all_tiles,
        tile_width,
        tile_height,
        mask_blur,
        mode_context: StaticModeContext | None = None,
    ):
        context = mode_context or self._build_static_mode_context()
        _, height, width, _ = upscaled_image.shape
        result_images = [tensor_to_pil(upscaled_image[b:b + 1], 0).copy() for b in range(batch_size)]

        log("USDU Dist: Using tile queue distribution")
        run_async_in_server_loop(
            init_static_job_batched(multi_job_id, batch_size, num_tiles_per_image, enabled_workers),
            timeout=10.0,
        )
        debug_log(f"Initialized tile-id queue with {num_tiles_per_image} ids for batch {batch_size}")

        tile_masks = [
            context.tile_ops.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur)
            for tx, ty in all_tiles
        ]
        return result_images, tile_masks, width, height

    def _process_master_static_initial_tiles(
        self,
        multi_job_id,
        total_tiles,
        all_tiles,
        upscaled_image,
        result_images,
        tile_masks,
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
        mode_context: StaticModeContext | None = None,
    ):
        context = mode_context or self._build_static_mode_context()
        processed_count = 0
        consecutive_no_tile = 0
        max_consecutive_no_tile = 2

        while processed_count < total_tiles:
            comfy.model_management.throw_exception_if_processing_interrupted()
            tile_id = run_async_in_server_loop(context.job_state.next_tile_index(multi_job_id), timeout=5.0)
            if tile_id is None:
                consecutive_no_tile += 1
                if consecutive_no_tile >= max_consecutive_no_tile:
                    debug_log(f"Master processed {processed_count} tiles, moving to collection phase")
                    break
                time.sleep(0.1)
                continue

            consecutive_no_tile = 0
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
                mode_context=context,
            )
            log(f"USDU Dist: Tiles progress {processed_count}/{total_tiles} (tile {tile_id})")
        return processed_count

    def _collect_remaining_static_tiles(
        self,
        multi_job_id,
        total_tiles,
        master_processed_count,
        all_tiles,
        upscaled_image,
        result_images,
        tile_masks,
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
        mode_context: StaticModeContext | None = None,
    ):
        context = mode_context or self._build_static_mode_context()
        remaining_tiles = total_tiles - master_processed_count
        if remaining_tiles <= 0:
            return run_async_in_server_loop(context.job_state.all_completed_tasks(multi_job_id), timeout=5.0)

        debug_log(f"Master waiting for {remaining_tiles} tiles from workers")
        collected_tasks = run_async_in_server_loop(
            self._async_collect_and_monitor_static(
                multi_job_id,
                total_tiles,
                expected_total=total_tiles,
                mode_context=context,
            ),
            timeout=None,
        )

        completed_count = len(collected_tasks)
        if completed_count >= total_tiles:
            return collected_tasks

        log(f"Processing remaining {total_tiles - completed_count} tasks locally after worker failures")
        while True:
            comfy.model_management.throw_exception_if_processing_interrupted()
            tile_id = run_async_in_server_loop(context.job_state.next_tile_index(multi_job_id), timeout=5.0)
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
                mode_context=context,
            )
        return collected_tasks

    def _blend_static_collected_tiles(
        self,
        collected_tasks,
        result_images,
        all_tiles,
        tile_masks,
        num_tiles_per_image,
        batch_size,
        tile_width,
        tile_height,
        padding,
        mode_context: StaticModeContext | None = None,
    ):
        context = mode_context or self._build_static_mode_context()
        def _sort_key(item):
            global_idx, tile_data = item
            batch_idx = tile_data.get('batch_idx', global_idx // num_tiles_per_image)
            tile_idx = tile_data.get('tile_idx', global_idx % num_tiles_per_image)
            return (tile_idx, batch_idx, global_idx)

        for global_idx, tile_data in sorted(collected_tasks.items(), key=_sort_key):
            if 'tensor' not in tile_data and 'image' not in tile_data:
                continue
            batch_idx = tile_data.get('batch_idx', global_idx // num_tiles_per_image)
            tile_idx = tile_data.get('tile_idx', global_idx % num_tiles_per_image)
            if batch_idx >= batch_size:
                continue

            x = tile_data.get('x', 0)
            y = tile_data.get('y', 0)
            tile_pil = tile_data['image'] if 'image' in tile_data else tensor_to_pil(tile_data['tensor'], 0)
            tile_mask = tile_masks[tile_idx]
            extracted_width = tile_data.get('extracted_width', tile_width + 2 * padding)
            extracted_height = tile_data.get('extracted_height', tile_height + 2 * padding)
            result_images[batch_idx] = context.tile_ops.blend_tile(
                result_images[batch_idx],
                tile_pil,
                x,
                y,
                (extracted_width, extracted_height),
                tile_mask,
                padding,
            )

    def _result_images_to_tensor(self, result_images, batch_size, upscaled_image):
        if batch_size == 1:
            result_tensor = pil_to_tensor(result_images[0])
        else:
            result_tensor = torch.cat([pil_to_tensor(img) for img in result_images], dim=0)
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        return result_tensor

    def _process_master_static_sync(self, upscaled_image, model, positive, negative, vae,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    tile_width, tile_height, padding, mask_blur,
                                    force_uniform_tiles, tiled_decode, multi_job_id, enabled_workers,
                                    all_tiles, num_tiles_per_image,
                                    mode_context: StaticModeContext | None = None):
        """Static mode master processing with optional dynamic queue pulling."""
        context = mode_context or self._build_static_mode_context()
        batch_size = upscaled_image.shape[0]
        total_tiles = batch_size * num_tiles_per_image
        result_images, tile_masks, width, height = self._init_master_static_state(
            upscaled_image=upscaled_image,
            multi_job_id=multi_job_id,
            batch_size=batch_size,
            num_tiles_per_image=num_tiles_per_image,
            enabled_workers=enabled_workers,
            all_tiles=all_tiles,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            mode_context=context,
        )

        try:
            master_processed_count = self._process_master_static_initial_tiles(
                multi_job_id=multi_job_id,
                total_tiles=total_tiles,
                all_tiles=all_tiles,
                upscaled_image=upscaled_image,
                result_images=result_images,
                tile_masks=tile_masks,
                batch_size=batch_size,
                num_tiles_per_image=num_tiles_per_image,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                force_uniform_tiles=force_uniform_tiles,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                tiled_decode=tiled_decode,
                width=width,
                height=height,
                mode_context=context,
            )

            collected_tasks = self._collect_remaining_static_tiles(
                multi_job_id=multi_job_id,
                total_tiles=total_tiles,
                master_processed_count=master_processed_count,
                all_tiles=all_tiles,
                upscaled_image=upscaled_image,
                result_images=result_images,
                tile_masks=tile_masks,
                batch_size=batch_size,
                num_tiles_per_image=num_tiles_per_image,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                force_uniform_tiles=force_uniform_tiles,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                tiled_decode=tiled_decode,
                width=width,
                height=height,
                mode_context=context,
            )

            self._blend_static_collected_tiles(
                collected_tasks=collected_tasks,
                result_images=result_images,
                all_tiles=all_tiles,
                tile_masks=tile_masks,
                num_tiles_per_image=num_tiles_per_image,
                batch_size=batch_size,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                mode_context=context,
            )

            result_tensor = self._result_images_to_tensor(result_images, batch_size, upscaled_image)
            log(f"UltimateSDUpscale Master - Job {multi_job_id} complete")
            return (result_tensor,)
        finally:
            run_async_in_server_loop(cleanup_job(multi_job_id), timeout=5.0)
