from __future__ import annotations

import asyncio, torch
from typing import Any
from PIL import Image
from ...utils.logging import debug_log, log
from ...utils.image import tensor_to_pil, pil_to_tensor
from ...utils.async_helpers import run_async_in_server_loop
from ...utils.config import get_worker_timeout_seconds
from ...utils.constants import TILE_WAIT_TIMEOUT, TILE_SEND_TIMEOUT
from ..job_store import ensure_tile_jobs_initialized, init_dynamic_job
from ..processing_args import UpscaleCoreArgs


class DynamicModeMixin:
    """
    Dynamic (per-image queue) USDU mode behaviors for master and worker roles.

    Expected co-mixins on `self`:
    - TileOpsMixin (`calculate_tiles`, `_slice_conditioning`, `_process_and_blend_tile`).
    - JobStateMixin (image queue/task completion helpers).
    - WorkerCommsMixin (`_request_image_from_master`, `_send_full_image_to_master`, `_send_heartbeat_to_master`).
    """

    def _handle_master_dynamic_idle_state(
        self,
        multi_job_id,
        batch_size,
        consecutive_retries,
        max_consecutive_retries,
    ):
        """Handle queue-empty state while master waits for worker progress."""
        drained_count = run_async_in_server_loop(
            self._drain_worker_results_queue(multi_job_id),
            timeout=5.0,
        )
        run_async_in_server_loop(self._async_yield(), timeout=0.1)

        requeued_count = run_async_in_server_loop(
            self._check_and_requeue_timed_out_workers(multi_job_id, batch_size),
            timeout=5.0,
        )
        run_async_in_server_loop(self._async_yield(), timeout=0.1)

        if requeued_count > 0:
            log(f"Requeued {requeued_count} images from timed out workers")
            return False, True, 0

        completed_now = run_async_in_server_loop(
            self._get_total_completed_count(multi_job_id),
            timeout=1.0,
        )
        log(f"USDU Dist: Images progress {completed_now}/{batch_size}")
        if completed_now >= batch_size:
            return True, False, consecutive_retries

        run_async_in_server_loop(self._async_yield(), timeout=0.1)

        pending_count = run_async_in_server_loop(
            self._get_pending_count(multi_job_id),
            timeout=1.0,
        )
        if pending_count > 0:
            return False, True, 0

        consecutive_retries += 1
        if consecutive_retries >= max_consecutive_retries:
            log(f"Max retries ({max_consecutive_retries}) reached. Forcing collection of remaining results.")
            return True, False, consecutive_retries

        if drained_count > 0:
            debug_log(f"Master idle drain picked up {drained_count} images while waiting for workers")
        debug_log("Waiting for workers")
        run_async_in_server_loop(asyncio.sleep(2), timeout=3.0)
        return False, True, consecutive_retries

    def _finalize_master_dynamic_results(
        self,
        multi_job_id,
        batch_size,
        num_workers,
        processed_count,
        result_images,
        upscaled_image,
    ):
        """Collect remaining worker outputs and convert final images back to tensor."""
        all_completed = run_async_in_server_loop(
            self._get_all_completed_images(multi_job_id),
            timeout=5.0,
        )
        remaining_to_collect = batch_size - len(all_completed)
        if remaining_to_collect > 0:
            debug_log(f"Waiting for {remaining_to_collect} more images from workers")
            collection_timeout = float(get_worker_timeout_seconds())
            collected_images = run_async_in_server_loop(
                self._async_collect_dynamic_images(
                    multi_job_id,
                    remaining_to_collect,
                    num_workers,
                    batch_size,
                    processed_count,
                ),
                timeout=collection_timeout,
            )
            all_completed.update(collected_images)

        for idx, processed_img in all_completed.items():
            if idx < batch_size:
                result_images[idx] = processed_img

        result_tensor = (
            torch.cat([pil_to_tensor(img) for img in result_images], dim=0)
            if batch_size > 1
            else pil_to_tensor(result_images[0])
        )
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        return result_tensor

    def process_master_dynamic(
        self,
        upscaled_image: torch.Tensor,
        core_args: UpscaleCoreArgs,
        tile_width: int,
        tile_height: int,
        padding: int,
        mask_blur: int,
        force_uniform_tiles: bool,
        multi_job_id: str,
        enabled_workers: list[str],
    ) -> tuple[torch.Tensor]:
        """Dynamic mode for large batches - assigns whole images to workers dynamically, including master."""
        # Get batch size and dimensions
        batch_size, height, width, _ = upscaled_image.shape
        num_workers = len(enabled_workers)
        
        log(f"USDU Dist: Image queue distribution | Batch {batch_size} | Workers {num_workers} | Canvas {width}x{height} | Tile {tile_width}x{tile_height}")

        # No fixed share - all images are dynamic
        all_indices = list(range(batch_size))
        
        debug_log(f"Processing {batch_size} images dynamically across master + {num_workers} workers.")
        
        # Calculate tiles for processing
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        
        # Initialize job queue for communication
        try:
            run_async_in_server_loop(
                init_dynamic_job(multi_job_id, batch_size, enabled_workers, all_indices),
                timeout=2.0
            )
        except Exception as e:
            debug_log(f"UltimateSDUpscale Master - Queue initialization error: {e}")
            raise RuntimeError(f"Failed to initialize dynamic mode queue: {e}")
        
        # Convert batch to PIL list
        result_images = [tensor_to_pil(upscaled_image[b:b+1], 0).convert('RGB').copy() for b in range(batch_size)]
        
        # Process images dynamically with master participating
        prompt_server = ensure_tile_jobs_initialized()
        processed_count = 0
        consecutive_retries = 0
        max_consecutive_retries = 10
        
        # Process loop - master pulls from queue and processes synchronously
        while processed_count < batch_size:
            # Try to get an image to process
            image_idx = run_async_in_server_loop(
                self._get_next_image_index(multi_job_id),
                timeout=5.0  # Short timeout to allow frequent checks
            )

            if image_idx is not None:
                # Reset retry counter and process locally
                consecutive_retries = 0
                debug_log(f"Master processing image {image_idx} dynamically")
                processed_count += 1

                # Process locally
                single_tensor = upscaled_image[image_idx:image_idx+1]
                local_image = result_images[image_idx]
                image_seed = core_args.seed
                
                # Pre-slice conditioning once per image (not per tile)
                positive_sliced, negative_sliced = self._slice_conditioning(
                    core_args.positive,
                    core_args.negative,
                    image_idx,
                )
                
                for tile_idx, pos in enumerate(all_tiles):
                    source_tensor = pil_to_tensor(local_image)
                    if single_tensor.is_cuda:
                        source_tensor = source_tensor.cuda()
                    local_image = self._process_and_blend_tile(
                        tile_idx, pos, source_tensor, local_image,
                        core_args.model,
                        positive_sliced,
                        negative_sliced,
                        core_args.vae,
                        image_seed,
                        core_args.steps,
                        core_args.cfg,
                        core_args.sampler_name,
                        core_args.scheduler,
                        core_args.denoise,
                        tile_width,
                        tile_height,
                        padding, mask_blur, width, height, force_uniform_tiles,
                        core_args.tiled_decode,
                        batch_idx=image_idx,
                    )
                    
                    # Yield after each tile to minimize worker downtime
                    run_async_in_server_loop(self._async_yield(), timeout=0.1)
                    # Note: No per-tile drain here – that's what makes this "per-image"
                
                result_images[image_idx] = local_image
                
                # Mark as completed
                run_async_in_server_loop(
                    self._mark_image_completed(multi_job_id, image_idx, local_image),
                    timeout=5.0
                )
                
                # NEW: Drain after the full image is marked complete (catches workers who finished during master's processing)
                drained_count = run_async_in_server_loop(
                    self._drain_worker_results_queue(multi_job_id),
                    timeout=5.0
                )
                
                if drained_count > 0:
                    debug_log(f"Drained {drained_count} worker images after master's image {image_idx}")
                
                # NEW: Log overall progress (includes master's image + any drained workers)
                completed_now = run_async_in_server_loop(
                    self._get_total_completed_count(multi_job_id),
                    timeout=1.0
                )
                log(f"USDU Dist: Images progress {completed_now}/{batch_size}")
                
                # Yield to allow workers to get new images after completing one
                run_async_in_server_loop(self._async_yield(), timeout=0.1)
            else:
                should_break, should_continue, consecutive_retries = self._handle_master_dynamic_idle_state(
                    multi_job_id=multi_job_id,
                    batch_size=batch_size,
                    consecutive_retries=consecutive_retries,
                    max_consecutive_retries=max_consecutive_retries,
                )
                if should_break:
                    break
                if should_continue:
                    continue
        
        debug_log(f"Master processed {processed_count} images locally")
        result_tensor = self._finalize_master_dynamic_results(
            multi_job_id=multi_job_id,
            batch_size=batch_size,
            num_workers=num_workers,
            processed_count=processed_count,
            result_images=result_images,
            upscaled_image=upscaled_image,
        )
        
        debug_log(f"UltimateSDUpscale Master - Job {multi_job_id} complete")
        log(f"Completed processing all {batch_size} images")
        return (result_tensor,)

    def process_worker_dynamic(
        self,
        upscaled_image: torch.Tensor,
        core_args: UpscaleCoreArgs,
        tile_width: int,
        tile_height: int,
        padding: int,
        mask_blur: int,
        force_uniform_tiles: bool,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
        enabled_worker_ids: list[str] | str,
        dynamic_threshold: int,
    ) -> tuple[torch.Tensor]:
        """Worker processing in dynamic mode - processes whole images."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)

        # Get dimensions and tile grid
        batch_size, height, width, _ = upscaled_image.shape
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        log(f"USDU Dist Worker[{worker_id[:8]}]: Processing image queue | Batch {batch_size}")

        # Keep track of processed images for is_last detection
        processed_count = 0

        # Poll for job readiness to avoid races during master init
        max_poll_attempts = 20  # ~20s at 1s sleep
        if not self._poll_job_ready(multi_job_id, master_url, worker_id=worker_id, max_attempts=max_poll_attempts):
            log(f"Job {multi_job_id} not ready after {max_poll_attempts} attempts, aborting")
            return (upscaled_image,)

        # Loop to request and process images
        while True:
            # Request an image to process
            image_idx, estimated_remaining = run_async_in_server_loop(
                self._request_image_from_master(multi_job_id, master_url, worker_id),
                timeout=TILE_WAIT_TIMEOUT
            )

            if image_idx is None:
                debug_log(f"USDU Dist Worker - No more images to process")
                break

            debug_log(f"Worker[{worker_id[:8]}] - Assigned image {image_idx}")
            processed_count += 1

            # Determine if this should be marked as last for this worker
            is_last_for_worker = (estimated_remaining == 0)

            # Extract single image tensor
            single_tensor = upscaled_image[image_idx:image_idx+1]

            # Convert to PIL for processing
            local_image = tensor_to_pil(single_tensor, 0).copy()

            # Process all tiles for this image
            image_seed = core_args.seed

            # Pre-slice conditioning once per image (not per tile)
            positive_sliced, negative_sliced = self._slice_conditioning(
                core_args.positive,
                core_args.negative,
                image_idx,
            )

            for tile_idx, pos in enumerate(all_tiles):
                source_tensor = pil_to_tensor(local_image)
                if single_tensor.is_cuda:
                    source_tensor = source_tensor.cuda()
                local_image = self._process_and_blend_tile(
                    tile_idx, pos, source_tensor, local_image,
                    core_args.model,
                    positive_sliced,
                    negative_sliced,
                    core_args.vae,
                    image_seed,
                    core_args.steps,
                    core_args.cfg,
                    core_args.sampler_name,
                    core_args.scheduler,
                    core_args.denoise,
                    tile_width,
                    tile_height,
                    padding, mask_blur, width, height, force_uniform_tiles,
                    core_args.tiled_decode,
                    batch_idx=image_idx,
                )
                run_async_in_server_loop(
                    self._send_heartbeat_to_master(multi_job_id, master_url, worker_id),
                    timeout=5.0
                )

            # Send processed image back to master
            try:
                # Use the estimated remaining to determine if this is the last image
                is_last = is_last_for_worker
                run_async_in_server_loop(
                    self._send_full_image_to_master(local_image, image_idx, multi_job_id,
                                                    master_url, worker_id, is_last),
                    timeout=TILE_SEND_TIMEOUT
                )
                # Send heartbeat after processing
                run_async_in_server_loop(
                    self._send_heartbeat_to_master(multi_job_id, master_url, worker_id),
                    timeout=5.0
                )
                if is_last:
                    break
            except Exception as e:
                log(f"USDU Dist Worker[{worker_id[:8]}] - Error sending image {image_idx}: {e}")
                # Continue processing other images

        # Send final is_last signal
        debug_log(f"Worker[{worker_id[:8]}] processed {processed_count} images, sending completion signal")
        try:
            run_async_in_server_loop(
                self._send_worker_complete_signal(multi_job_id, master_url, worker_id),
                timeout=TILE_SEND_TIMEOUT
            )
        except Exception as e:
            log(f"USDU Dist Worker[{worker_id[:8]}] - Error sending completion signal: {e}")

        return (upscaled_image,)
