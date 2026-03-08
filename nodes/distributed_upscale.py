import math
from functools import wraps
from typing import Any, Callable

import comfy.samplers

from ..utils.logging import debug_log, log
from ..utils.async_helpers import run_async_in_server_loop
from ..upscale.job_store import ensure_tile_jobs_initialized
from .hidden_inputs import build_distributed_hidden_inputs

from ..upscale.tile_ops import TileOpsMixin
from ..upscale.result_collector import ResultCollectorMixin
from ..upscale.worker_comms import WorkerCommsMixin
from ..upscale.job_state import JobStateMixin
from ..upscale.processing_args import UpscaleCoreArgs
from ..upscale.modes.single_gpu import SingleGpuModeMixin
from ..upscale.modes.static import StaticModeMixin
from ..upscale.modes.dynamic import DynamicModeMixin
from ..utils.worker_ids import coerce_enabled_worker_ids

def sync_wrapper(async_func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to wrap async methods for synchronous execution."""
    @wraps(async_func)
    def sync_func(self, *args: Any, **kwargs: Any) -> Any:
        # Use run_async_in_server_loop for ComfyUI compatibility
        return run_async_in_server_loop(
            async_func(self, *args, **kwargs),
            timeout=600.0  # 10 minute timeout for long operations
        )
    return sync_func


def _parse_enabled_worker_ids(enabled_worker_ids: str | list[str] | None) -> list[str]:
    """Backward-compatible alias for enabled-worker normalization."""
    return coerce_enabled_worker_ids(enabled_worker_ids)

class UltimateSDUpscaleDistributed(
    DynamicModeMixin,
    StaticModeMixin,
    SingleGpuModeMixin,
    ResultCollectorMixin,
    WorkerCommsMixin,
    JobStateMixin,
    TileOpsMixin,
):

    """
    Distributed version of Ultimate SD Upscale (No Upscale).
    
    Supports three processing modes:
    1. Single GPU: No workers available, process everything locally
    2. Static Mode: Small batches, distributes tiles across workers (flattened)
    3. Dynamic Mode: Large batches, assigns whole images to workers dynamically
    
    Features:
    - Multi-mode batch handling for efficient video/image upscaling
    - Tiled VAE support for memory efficiency
    - Dynamic load balancing for large batches
    - Backward compatible with single-image workflows
    
    Environment Variables:
    - COMFYUI_MAX_BATCH: Chunk size for tile sending (default 20)
    - COMFYUI_MAX_PAYLOAD_SIZE: Max API payload bytes (default 50MB)
    
    Threshold: dynamic_threshold input controls mode switch (default 8)
    """

    def __init__(self):
        """Initialize the node and ensure persistent storage exists."""
        # Pre-initialize the persistent storage on node creation
        ensure_tile_jobs_initialized()
        debug_log("UltimateSDUpscaleDistributed - Node initialized")

    @classmethod
    def INPUT_TYPES(cls: type["UltimateSDUpscaleDistributed"]) -> dict[str, Any]:
        hidden_inputs = build_distributed_hidden_inputs()
        hidden_inputs.update(
            {
                "tile_indices": ("STRING", {"default": ""}),  # Unused - kept for compatibility
                "dynamic_threshold": ("INT", {"default": 8, "min": 1, "max": 64}),
            }
        )
        return {
            "required": {
                "upscaled_image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 256}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
            },
            "hidden": hidden_inputs,
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/upscaling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution."""
        return float("nan")  # Always re-execute

    def run(
        self,
        upscaled_image: Any,
        model: Any,
        positive: Any,
        negative: Any,
        vae: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        tile_width: int,
        tile_height: int,
        padding: int,
        mask_blur: int,
        force_uniform_tiles: bool,
        tiled_decode: bool,
        multi_job_id: str = "",
        is_worker: bool = False,
        master_url: str = "",
        enabled_worker_ids: str = "[]",
        worker_id: str = "",
        tile_indices: str = "",
        dynamic_threshold: int = 8,
    ) -> tuple[Any, ...]:
        """Entry point - runs SYNCHRONOUSLY like Ultimate SD Upscaler."""
        core_args = UpscaleCoreArgs(
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
        )
        # Strict WAN/FLOW batching: error if batch is not 4n+1 (except allow 1)
        try:
            batch_size = int(getattr(upscaled_image, 'shape', [1])[0])
        except Exception:
            batch_size = 1
        # Enforce 4n+1 batches globally for any model when batch > 1 (master only)
        if not is_worker and batch_size != 1 and (batch_size % 4 != 1):
            raise ValueError(
                f"Batch size {batch_size} is not of the form 4n+1. "
                "This node requires batch sizes of 1 or 4n+1 (1, 5, 9, 13, ...). "
                "Please adjust the batch size."
            )
        if not multi_job_id:
            # No distributed processing, run single GPU version
            return self.process_single_gpu(
                upscaled_image=upscaled_image,
                core_args=core_args,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                mask_blur=mask_blur,
                force_uniform_tiles=force_uniform_tiles,
            )
        
        if is_worker:
            # Worker mode: process tiles synchronously
            return self.process_worker(upscaled_image, model, positive, negative, vae,
                                      seed, steps, cfg, sampler_name, scheduler, denoise,
                                      tile_width, tile_height, padding, mask_blur,
                                      force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                      worker_id, enabled_worker_ids, dynamic_threshold)
        else:
            # Master mode: distribute and collect synchronously
            return self.process_master(
                upscaled_image=upscaled_image,
                core_args=core_args,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                mask_blur=mask_blur,
                force_uniform_tiles=force_uniform_tiles,
                multi_job_id=multi_job_id,
                enabled_worker_ids=enabled_worker_ids,
                dynamic_threshold=dynamic_threshold,
            )

    def process_worker(
        self,
        upscaled_image: Any,
        model: Any,
        positive: Any,
        negative: Any,
        vae: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        tile_width: int,
        tile_height: int,
        padding: int,
        mask_blur: int,
        force_uniform_tiles: bool,
        tiled_decode: bool,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
        enabled_worker_ids: str,
        dynamic_threshold: int,
    ) -> tuple[Any, ...]:
        """Unified worker processing - handles both static and dynamic modes."""
        core_args = UpscaleCoreArgs(
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
        )
        # Get batch size to determine mode
        batch_size = upscaled_image.shape[0]
        
        # Ensure mode consistency across master/workers via shared threshold
        # Determine mode (must match master's logic)
        enabled_workers = coerce_enabled_worker_ids(enabled_worker_ids)
        num_workers = len(enabled_workers)
        # Compute number of tiles for this image to decide if tile distribution makes sense
        _, height, width, _ = upscaled_image.shape
        all_tiles = self.calculate_tiles(width, height, self.round_to_multiple(tile_width), self.round_to_multiple(tile_height), force_uniform_tiles)
        num_tiles_per_image = len(all_tiles)

        mode = self._determine_processing_mode(batch_size, num_workers, dynamic_threshold)
        # For USDU-style processing, we want tile distribution whenever workers are available
        # and there is more than one tile to process for single-image runs.
        if num_workers > 0 and batch_size <= 1 and num_tiles_per_image > 1:
            mode = "static"
            
        debug_log(f"USDU Dist Worker - Batch size {batch_size}")
        
        if mode == "dynamic":
            return self.process_worker_dynamic(
                upscaled_image=upscaled_image,
                core_args=core_args,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                mask_blur=mask_blur,
                force_uniform_tiles=force_uniform_tiles,
                multi_job_id=multi_job_id,
                master_url=master_url,
                worker_id=worker_id,
                enabled_worker_ids=enabled_worker_ids,
                dynamic_threshold=dynamic_threshold,
            )
        
        # Static mode - enhanced with health monitoring and retry logic
        return self._process_worker_static_sync(upscaled_image, core_args,
                                               tile_width, tile_height, padding, mask_blur,
                                               force_uniform_tiles, multi_job_id, master_url,
                                               worker_id, enabled_workers)

    def process_master(
        self,
        upscaled_image: Any,
        core_args: UpscaleCoreArgs,
        tile_width: int,
        tile_height: int,
        padding: int,
        mask_blur: int,
        force_uniform_tiles: bool,
        multi_job_id: str,
        enabled_worker_ids: str,
        dynamic_threshold: int,
    ) -> tuple[Any, ...]:
        """Unified master processing with enhanced monitoring and failure handling."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Get image dimensions and batch size
        batch_size, height, width, _ = upscaled_image.shape
        
        # Calculate all tiles and grid
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        num_tiles_per_image = len(all_tiles)
        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        log(
            f"USDU Dist: Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Grid {rows}x{cols} ({num_tiles_per_image} tiles/image) | Batch {batch_size}"
        )
        
        # Parse enabled workers
        enabled_workers = coerce_enabled_worker_ids(enabled_worker_ids)
        num_workers = len(enabled_workers)
        
        # Determine processing mode
        mode = self._determine_processing_mode(batch_size, num_workers, dynamic_threshold)
        # Prefer tile-based static distribution when workers are available and there are multiple tiles,
        # for single-image jobs to spread tiles across GPUs like the legacy tile queue.
        if num_workers > 0 and batch_size <= 1 and num_tiles_per_image > 1:
            mode = "static"
        
        log(f"USDU Dist: Workers {num_workers} | Mode {mode} | Threshold {dynamic_threshold}")

        if mode == "single_gpu":
            # No workers, process all tiles locally
            return self.process_single_gpu(
                upscaled_image=upscaled_image,
                core_args=core_args,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                mask_blur=mask_blur,
                force_uniform_tiles=force_uniform_tiles,
            )
        
        elif mode == "dynamic":
            # Dynamic mode for large batches
            return self.process_master_dynamic(
                upscaled_image=upscaled_image,
                core_args=core_args,
                tile_width=tile_width,
                tile_height=tile_height,
                padding=padding,
                mask_blur=mask_blur,
                force_uniform_tiles=force_uniform_tiles,
                multi_job_id=multi_job_id,
                enabled_workers=enabled_workers,
            )
        
        # Static mode - enhanced with unified job management
        return self._process_master_static_sync(upscaled_image, core_args,
                                               tile_width, tile_height, padding, mask_blur,
                                               force_uniform_tiles, multi_job_id, enabled_workers,
                                               all_tiles, num_tiles_per_image)

    def _determine_processing_mode(self, batch_size: int, num_workers: int, dynamic_threshold: int) -> str:
        """Determine mode from worker availability and configured threshold."""
        if num_workers == 0:
            return "single_gpu"
        threshold = max(1, int(dynamic_threshold))
        if int(batch_size) >= threshold:
            return "dynamic"
        return "static"

class USDUDelegateCollector:
    """Lightweight stand-in used automatically in delegate-only master prompts.

    When the master is in delegate-only mode, the orchestration code swaps the
    full ``UltimateSDUpscaleDistributed`` class for this one so that no upstream
    model/image nodes execute on the master.  Workers initialise the dynamic job
    queue via ``/distributed/init_dynamic_job`` and this node simply waits for
    all images to arrive, then assembles the output tensor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "delegate_only": ("BOOLEAN", {"default": True}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
                "master_url": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/upscaling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, multi_job_id="", enabled_worker_ids="[]", **kwargs):
        import time
        import torch
        from ..utils.image import pil_to_tensor
        from ..upscale.job_models import ImageJobState

        if not multi_job_id:
            log("USDUDelegateCollector: no multi_job_id, returning empty image")
            return (torch.zeros(1, 64, 64, 3),)

        enabled_workers = coerce_enabled_worker_ids(enabled_worker_ids)
        num_workers = len(enabled_workers)
        log(f"USDU delegate-only: waiting for workers to init job {multi_job_id}")

        prompt_server = ensure_tile_jobs_initialized()

        # Wait for workers to create the job via /distributed/init_dynamic_job
        max_wait = 120.0
        poll_interval = 1.0
        start = time.time()
        job_data = None
        while time.time() - start < max_wait:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if isinstance(job_data, ImageJobState) and job_data.batch_size > 0:
                break
            job_data = None
            time.sleep(poll_interval)

        if job_data is None:
            log(f"USDUDelegateCollector: job {multi_job_id} was not initialized by workers within {max_wait}s")
            return (torch.zeros(1, 64, 64, 3),)

        batch_size = job_data.batch_size
        log(f"USDU delegate-only: job ready, collecting {batch_size} images from {num_workers} workers")

        # Collect all images using the existing result collector
        collected = run_async_in_server_loop(
            self._collect_all_images(multi_job_id, batch_size, num_workers, prompt_server),
            timeout=600.0,
        )

        # Assemble tensor from collected PIL images
        result_images = []
        for idx in range(batch_size):
            img = collected.get(idx)
            if img is not None:
                result_images.append(pil_to_tensor(img))
            else:
                log(f"USDUDelegateCollector: missing image {idx}, using blank")
                result_images.append(torch.zeros(1, 64, 64, 3))

        result_tensor = torch.cat(result_images, dim=0)
        log(f"USDU delegate-only: collected all {batch_size} images")
        return (result_tensor,)

    @staticmethod
    async def _collect_all_images(multi_job_id, batch_size, num_workers, prompt_server):
        """Wait for all images to arrive from workers."""
        import asyncio
        from ..upscale.job_models import ImageJobState
        from ..upscale.job_timeout import check_and_requeue_timed_out_workers

        timeout_seconds = 300.0
        poll_interval = 2.0
        start = asyncio.get_event_loop().time()

        while True:
            # Drain results from the queue
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if isinstance(job_data, ImageJobState):
                # Drain queue items into completed_images
                while True:
                    try:
                        result = job_data.queue.get_nowait()
                        worker_id = result.get("worker_id")
                        if "image_idx" in result and "image" in result:
                            idx = result["image_idx"]
                            if idx not in job_data.completed_images:
                                job_data.completed_images[idx] = result["image"]
                                debug_log(f"Delegate collected image {idx} from worker {worker_id}")
                    except asyncio.QueueEmpty:
                        break

                completed_count = len(job_data.completed_images)
                if completed_count >= batch_size:
                    debug_log(f"Delegate: all {batch_size} images collected")
                    completed = dict(job_data.completed_images)
                    # Cleanup
                    async with prompt_server.distributed_tile_jobs_lock:
                        prompt_server.distributed_pending_tile_jobs.pop(multi_job_id, None)
                    return completed

                # Check for timeouts periodically
                await check_and_requeue_timed_out_workers(multi_job_id, batch_size)

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout_seconds:
                log(f"USDUDelegateCollector: timed out after {elapsed:.0f}s with {len(getattr(job_data, 'completed_images', {}))} of {batch_size} images")
                if isinstance(job_data, ImageJobState):
                    completed = dict(job_data.completed_images)
                    async with prompt_server.distributed_tile_jobs_lock:
                        prompt_server.distributed_pending_tile_jobs.pop(multi_job_id, None)
                    return completed
                return {}

            await asyncio.sleep(poll_interval)


# Node registration
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleDistributed": UltimateSDUpscaleDistributed,
    "USDUDelegateCollector": USDUDelegateCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleDistributed": "Ultimate SD Upscale Distributed (No Upscale)",
    # No display name for USDUDelegateCollector — it's an internal node
}
