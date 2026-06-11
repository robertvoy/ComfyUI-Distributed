"""Distributed LTX tiled sampler node.

This node adapts the upstream LTX tiled sampler interface to the existing
ComfyUI-Distributed job queue. It distributes spatial latent tiles across the
same worker/master infrastructure used by the distributed upscaler instead of
creating a parallel queue implementation.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import aiohttp
import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from ..upscale.job_models import TileJobState
from ..upscale.job_store import (
    _cleanup_job,
    _drain_results_queue,
    _get_completed_count,
    _mark_task_completed,
    ensure_tile_jobs_initialized,
    init_static_job_batched,
)
from ..upscale.job_timeout import _check_and_requeue_timed_out_workers
from ..utils.async_helpers import run_async_in_server_loop
from ..utils.constants import HEARTBEAT_INTERVAL, TILE_WAIT_TIMEOUT
from ..utils.logging import debug_log, log
from ..utils.network import get_client_session
from ..utils.usdu_managment import _send_heartbeat_to_master

try:  # Optional external LTX tiled sampler dependency.
    from latent_tiled_sampler import LTXTiledSampler  # type: ignore
except Exception:  # pragma: no cover - depends on optional custom node install.
    LTXTiledSampler = None  # type: ignore


def _default_ltx_inputs() -> dict:
    """Fallback schema used when the source LTX sampler is not importable."""
    return {
        "required": {
            "noise": ("NOISE",),
            "guider": ("GUIDER",),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
            "h_tiles": ("INT", {"default": 1, "min": 1, "max": 64}),
            "w_tiles": ("INT", {"default": 1, "min": 1, "max": 64}),
            "overlap": ("INT", {"default": 0, "min": 0, "max": 4096}),
        },
        "optional": {
            "denoise_mask": ("MASK",),
        },
    }


def _copy_base_inputs() -> dict:
    if LTXTiledSampler is None:
        return _default_ltx_inputs()
    try:
        return copy.deepcopy(LTXTiledSampler.INPUT_TYPES())
    except Exception:
        return _default_ltx_inputs()


def _compute_tile_starts(total_size: int, n_tiles: int, overlap: int) -> Tuple[List[int], int]:
    """Compute evenly spaced tile starts and tile size for one dimension."""
    total_size = int(total_size)
    n_tiles = max(1, int(n_tiles))
    overlap = max(0, int(overlap))
    if n_tiles <= 1 or total_size <= 1:
        return [0], total_size
    tile_size = math.ceil((total_size + (n_tiles - 1) * overlap) / n_tiles)
    tile_size = min(tile_size, total_size)
    if n_tiles == 2:
        return [0, max(0, total_size - tile_size)], tile_size
    stride = (total_size - tile_size) / (n_tiles - 1)
    return [int(round(i * stride)) for i in range(n_tiles)], tile_size


def _make_window_1d(size: int, fade_left: int, fade_right: int, dtype, device):
    """Create a one-dimensional cosine edge window for tile blending."""
    win = torch.ones(int(size), dtype=dtype, device=device)
    if fade_left > 0:
        fl = min(int(fade_left), int(size))
        i = torch.arange(fl, dtype=dtype, device=device)
        # Use fl rather than fl - 1 to preserve a nonzero center plateau for small fades.
        win[:fl] = 0.5 * (1.0 - torch.cos(math.pi * i / max(1, fl)))
    if fade_right > 0:
        fr = min(int(fade_right), int(size))
        i = torch.arange(fr, dtype=dtype, device=device)
        win[int(size) - fr:] = 0.5 * (1.0 + torch.cos(math.pi * i / max(1, fr)))
    return win


def _pack_tensors(tensors: Dict[str, torch.Tensor]) -> str:
    """Serialize tensors to a base64 safetensors payload for JSON transport."""
    cpu_tensors = {
        name: tensor.detach().contiguous().to("cpu")
        for name, tensor in tensors.items()
        if tensor is not None
    }
    payload = safetensors_save(cpu_tensors)
    return base64.b64encode(payload).decode("ascii")


def _unpack_tensors(payload_b64: str, device) -> Dict[str, torch.Tensor]:
    payload = base64.b64decode(payload_b64.encode("ascii"))
    return {name: tensor.to(device) for name, tensor in safetensors_load(payload).items()}


def _extract_tensor_and_rebuilder(value) -> Tuple[torch.Tensor, Callable[[torch.Tensor], object]]:
    """Extract the tensor from common Comfy latent containers and preserve shape."""
    if isinstance(value, dict) and "samples" in value:
        original = value

        def rebuild(tensor):
            updated = dict(original)
            updated["samples"] = tensor
            return updated

        return value["samples"], rebuild

    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        original = value

        def rebuild(tensor):
            return (tensor, *original[1:])

        return value[0], rebuild

    if isinstance(value, list) and value and torch.is_tensor(value[0]):
        original = value

        def rebuild(tensor):
            updated = list(original)
            updated[0] = tensor
            return updated

        return value[0], rebuild

    if torch.is_tensor(value):
        return value, lambda tensor: tensor

    raise TypeError("Expected latent_image to be a tensor or latent container with a tensor samples field")


def _extract_optional_mask(mask):
    if mask is None:
        return None
    if isinstance(mask, dict) and "samples" in mask:
        return mask["samples"]
    return mask


def _tile_latent(value: torch.Tensor, tile: dict) -> torch.Tensor:
    return value[:, :, :, tile["h_start"]:tile["h_end"], tile["w_start"]:tile["w_end"]].contiguous()


def _tile_mask(mask, tile: dict):
    if mask is None:
        return None
    if torch.is_tensor(mask) and mask.ndim >= 5:
        return mask[:, :, :, tile["h_start"]:tile["h_end"], tile["w_start"]:tile["w_end"]].contiguous()
    if torch.is_tensor(mask) and mask.ndim >= 4:
        return mask[:, :, tile["h_start"]:tile["h_end"], tile["w_start"]:tile["w_end"]].contiguous()
    return mask


def _sample_tile(guider, tile_noise, tile_latent, sampler, sigmas, tile_mask):
    result = guider.sample(
        tile_noise,
        tile_latent,
        sampler,
        sigmas,
        denoise_mask=tile_mask,
        disable_pbar=True,
    )
    if isinstance(result, tuple):
        tile_samples = result[0]
        tile_denoised = result[1] if len(result) > 1 else None
    else:
        tile_samples = result
        tile_denoised = None
    return tile_samples, tile_denoised


def _build_tile_queue(latent_shape: Iterable[int], n_h_tiles: int, n_w_tiles: int, overlap: int) -> List[dict]:
    _batch, _channels, _frames, height, width = [int(v) for v in latent_shape]
    starts_y, tile_h = _compute_tile_starts(height, n_h_tiles, overlap)
    starts_x, tile_w = _compute_tile_starts(width, n_w_tiles, overlap)
    queue = []
    tile_idx = 0
    for grid_y, y_start in enumerate(starts_y):
        y_end = min(y_start + tile_h, height)
        for grid_x, x_start in enumerate(starts_x):
            x_end = min(x_start + tile_w, width)
            queue.append(
                {
                    "tile_index": tile_idx,
                    "h_start": y_start,
                    "h_end": y_end,
                    "w_start": x_start,
                    "w_end": x_end,
                    "grid_y": grid_y,
                    "grid_x": grid_x,
                }
            )
            tile_idx += 1
    return queue


def _blend_tile_into_sums(
    tile: dict,
    samples: torch.Tensor,
    denoised: Optional[torch.Tensor],
    output_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    denoised_sum: Optional[torch.Tensor],
    n_h_tiles: int,
    n_w_tiles: int,
    overlap: int,
):
    y_start = tile["h_start"]
    y_end = tile["h_end"]
    x_start = tile["w_start"]
    x_end = tile["w_end"]
    grid_y = tile["grid_y"]
    grid_x = tile["grid_x"]
    fade_top = overlap if grid_y > 0 else 0
    fade_bottom = overlap if grid_y < n_h_tiles - 1 else 0
    fade_left = overlap if grid_x > 0 else 0
    fade_right = overlap if grid_x < n_w_tiles - 1 else 0
    dtype = output_sum.dtype
    device = output_sum.device
    wy = _make_window_1d(y_end - y_start, fade_top, fade_bottom, dtype, device)
    wx = _make_window_1d(x_end - x_start, fade_left, fade_right, dtype, device)
    weights = wy.view(1, 1, 1, y_end - y_start, 1) * wx.view(1, 1, 1, 1, x_end - x_start)
    output_sum[:, :, :, y_start:y_end, x_start:x_end] += samples.to(device=device, dtype=dtype) * weights
    weight_sum[:, :, :, y_start:y_end, x_start:x_end] += weights
    if denoised_sum is not None and denoised is not None:
        denoised_sum[:, :, :, y_start:y_end, x_start:x_end] += denoised.to(device=device, dtype=dtype) * weights


async def _get_next_ltx_tile_index(multi_job_id: str) -> Optional[int]:
    prompt_server = ensure_tile_jobs_initialized()
    pending_queue = None
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if isinstance(job_data, TileJobState):
            pending_queue = job_data.pending_tasks
    if pending_queue is None:
        return None
    try:
        return await asyncio.wait_for(pending_queue.get(), timeout=0.1)
    except asyncio.TimeoutError:
        return None


async def _get_ltx_completed_tasks(multi_job_id: str) -> dict:
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if isinstance(job_data, TileJobState):
            return dict(job_data.completed_tasks)
    return {}


async def _check_ltx_job_status(multi_job_id: str, master_url: str) -> bool:
    session = await get_client_session()
    async with session.get(f"{master_url}/distributed/job_status?multi_job_id={multi_job_id}") as response:
        if response.status != 200:
            return False
        data = await response.json()
        return bool(data.get("ready", False))


async def _request_tile_from_master(multi_job_id: str, master_url: str, worker_id: str):
    session = await get_client_session()
    async with session.post(
        f"{master_url}/distributed/request_image",
        json={"multi_job_id": multi_job_id, "worker_id": str(worker_id)},
        timeout=aiohttp.ClientTimeout(total=TILE_WAIT_TIMEOUT),
    ) as response:
        if response.status == 200:
            return await response.json()
        if response.status == 404:
            return None
        response.raise_for_status()


async def _submit_ltx_tile_to_master(
    multi_job_id: str,
    master_url: str,
    worker_id: str,
    tile_index: int,
    payload_b64: Optional[str] = None,
    is_last: bool = False,
):
    session = await get_client_session()
    body = {
        "multi_job_id": multi_job_id,
        "worker_id": str(worker_id),
        "tile_index": int(tile_index) if tile_index is not None else None,
        "payload": payload_b64,
        "is_last": bool(is_last),
    }
    async with session.post(f"{master_url}/distributed/ltx/submit_tile", json=body) as response:
        response.raise_for_status()
        return await response.json()


async def _ltx_worker_process(
    node,
    noise,
    guider,
    sampler,
    sigmas,
    latent_tensor,
    denoise_mask,
    n_h_tiles,
    n_w_tiles,
    tile_overlap,
):
    job_id = node.multi_job_id
    master_url = node.master_url
    worker_id = node.worker_id or "ltx-worker"
    tiles_by_index = {
        tile["tile_index"]: tile
        for tile in _build_tile_queue(latent_tensor.shape, n_h_tiles, n_w_tiles, tile_overlap)
    }
    for attempt in range(120):
        if await _check_ltx_job_status(job_id, master_url):
            break
        await asyncio.sleep(1.0)
    else:
        log(f"LTX Dist Worker[{worker_id[:8]}]: job {job_id} was not ready; aborting")
        return

    full_noise = noise.generate_noise({"samples": latent_tensor})

    last_heartbeat = 0.0
    while True:
        data = await _request_tile_from_master(job_id, master_url, worker_id)
        if not data:
            log(f"LTX Dist Worker[{worker_id[:8]}]: master job unavailable; stopping")
            break
        tile_index = data.get("tile_idx")
        if tile_index is None:
            await _submit_ltx_tile_to_master(job_id, master_url, worker_id, -1, is_last=True)
            break
        tile = tiles_by_index.get(int(tile_index))
        if tile is None:
            raise RuntimeError(f"Master assigned unknown LTX tile index {tile_index}")

        tile_latent = _tile_latent(latent_tensor, tile)
        tile_noise = _tile_latent(full_noise, tile)
        tile_mask = _tile_mask(denoise_mask, tile)
        tile_samples, tile_denoised = _sample_tile(guider, tile_noise, tile_latent, sampler, sigmas, tile_mask)
        tensors = {"samples": tile_samples}
        if tile_denoised is not None:
            tensors["denoised"] = tile_denoised
        await _submit_ltx_tile_to_master(
            job_id,
            master_url,
            worker_id,
            int(tile_index),
            _pack_tensors(tensors),
        )
        now = asyncio.get_running_loop().time()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            await _send_heartbeat_to_master(job_id, master_url, worker_id)
            last_heartbeat = now


async def _ltx_master_process(
    node,
    noise,
    guider,
    sampler,
    sigmas,
    latent_tensor,
    denoise_mask,
    n_h_tiles,
    n_w_tiles,
    tile_overlap,
):
    job_id = node.multi_job_id
    enabled_workers = node.enabled_worker_ids
    tiles = _build_tile_queue(latent_tensor.shape, n_h_tiles, n_w_tiles, tile_overlap)
    tiles_by_index = {tile["tile_index"]: tile for tile in tiles}
    total_tiles = len(tiles)
    await init_static_job_batched(job_id, batch_size=1, num_tiles_per_image=total_tiles, enabled_workers=enabled_workers)

    full_noise = noise.generate_noise({"samples": latent_tensor})
    output_sum = torch.zeros_like(latent_tensor)
    weight_sum = torch.zeros_like(latent_tensor)
    denoised_sum = torch.zeros_like(latent_tensor) if denoise_mask is not None else None
    integrated = set()
    last_timeout_check = 0.0

    try:
        while len(integrated) < total_tiles:
            await _drain_results_queue(job_id)
            completed = await _get_ltx_completed_tasks(job_id)
            for tile_index, result in list(completed.items()):
                if tile_index in integrated:
                    continue
                if not isinstance(result, dict) or "payload" not in result:
                    # Locally completed tiles are blended immediately and then marked complete.
                    continue
                tile = tiles_by_index.get(int(tile_index))
                if tile is None:
                    continue
                tensors = _unpack_tensors(result["payload"], latent_tensor.device)
                _blend_tile_into_sums(
                    tile,
                    tensors["samples"],
                    tensors.get("denoised"),
                    output_sum,
                    weight_sum,
                    denoised_sum,
                    n_h_tiles,
                    n_w_tiles,
                    tile_overlap,
                )
                integrated.add(int(tile_index))

            if len(integrated) >= total_tiles:
                break

            tile_index = await _get_next_ltx_tile_index(job_id)
            if tile_index is not None and int(tile_index) not in integrated:
                tile = tiles_by_index[int(tile_index)]
                tile_latent = _tile_latent(latent_tensor, tile)
                tile_noise = _tile_latent(full_noise, tile)
                tile_mask = _tile_mask(denoise_mask, tile)
                tile_samples, tile_denoised = _sample_tile(guider, tile_noise, tile_latent, sampler, sigmas, tile_mask)
                _blend_tile_into_sums(
                    tile,
                    tile_samples,
                    tile_denoised,
                    output_sum,
                    weight_sum,
                    denoised_sum,
                    n_h_tiles,
                    n_w_tiles,
                    tile_overlap,
                )
                integrated.add(int(tile_index))
                await _mark_task_completed(job_id, int(tile_index), {"tile_idx": int(tile_index), "source": "master"})
                continue

            now = asyncio.get_running_loop().time()
            if now - last_timeout_check >= 10.0:
                requeued = await _check_and_requeue_timed_out_workers(job_id, total_tiles)
                if requeued:
                    log(f"LTX Dist: Requeued {requeued} timed-out tile(s)")
                last_timeout_check = now

            completed_count = await _get_completed_count(job_id)
            debug_log(f"LTX Dist: Progress {completed_count}/{total_tiles} completed, {len(integrated)} integrated")
            await asyncio.sleep(0.1)
    finally:
        await _cleanup_job(job_id)

    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    samples = output_sum / weight_sum
    denoised = denoised_sum / weight_sum if denoised_sum is not None else None
    return samples, denoised


def _ltx_local_tiled_process(
    noise,
    guider,
    sampler,
    sigmas,
    latent_tensor,
    denoise_mask,
    n_h_tiles,
    n_w_tiles,
    tile_overlap,
):
    """Process all LTX latent tiles locally in the current ComfyUI execution context.

    This path is used when no workers are enabled. Keeping it synchronous avoids
    crossing ComfyUI's execution/inference context boundary while still using the
    same tile extraction and blend logic as the distributed master/worker path.
    """
    tiles = _build_tile_queue(latent_tensor.shape, n_h_tiles, n_w_tiles, tile_overlap)
    full_noise = noise.generate_noise({"samples": latent_tensor})
    output_sum = torch.zeros_like(latent_tensor)
    weight_sum = torch.zeros_like(latent_tensor)
    denoised_sum = None

    for tile in tiles:
        tile_latent = _tile_latent(latent_tensor, tile)
        tile_noise = _tile_latent(full_noise, tile)
        tile_mask = _tile_mask(denoise_mask, tile)
        tile_samples, tile_denoised = _sample_tile(guider, tile_noise, tile_latent, sampler, sigmas, tile_mask)
        if denoised_sum is None and tile_denoised is not None:
            denoised_sum = torch.zeros_like(latent_tensor)
        _blend_tile_into_sums(
            tile,
            tile_samples,
            tile_denoised,
            output_sum,
            weight_sum,
            denoised_sum,
            n_h_tiles,
            n_w_tiles,
            tile_overlap,
        )

    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    samples = output_sum / weight_sum
    denoised = denoised_sum / weight_sum if denoised_sum is not None else None
    return samples, denoised


class LTXTiledSamplerDistributed:
    """Distributed wrapper around the LTX tiled sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = _copy_base_inputs()
        base_inputs.setdefault("hidden", {})
        base_inputs["hidden"].update(
            {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "worker_id": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
            }
        )
        return base_inputs

    RETURN_TYPES = getattr(LTXTiledSampler, "RETURN_TYPES", ("LATENT", "LATENT", "LATENT"))
    RETURN_NAMES = getattr(LTXTiledSampler, "RETURN_NAMES", ("samples", "denoised", "preview"))
    FUNCTION = "sample_tiled_distributed"
    CATEGORY = "latent/diffusion"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def __init__(self):
        self.base = LTXTiledSampler() if LTXTiledSampler is not None else None

    def _run_base(self, noise, guider, sampler, sigmas, latent_image, denoise_mask, h_tiles, w_tiles, overlap):
        if self.base is None:
            raise RuntimeError(
                "LTXTiledSamplerDistributed requires the base LTXTiledSampler node for local fallback. "
                "Install the LTX tiled sampler dependency or run with distributed multi_job_id."
            )
        return self.base.sample_tiled(
            noise,
            guider,
            sampler,
            sigmas,
            latent_image,
            denoise_mask,
            h_tiles,
            w_tiles,
            overlap,
        )

    def sample_tiled_distributed(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        denoise_mask=None,
        h_tiles: int = 1,
        w_tiles: int = 1,
        overlap: int = 0,
        multi_job_id: str = "",
        is_worker: bool = False,
        master_url: str = "",
        worker_id: str = "",
        enabled_worker_ids: str = "[]",
    ):
        if not multi_job_id:
            return self._run_base(noise, guider, sampler, sigmas, latent_image, denoise_mask, h_tiles, w_tiles, overlap)

        latent_tensor, rebuild_latent = _extract_tensor_and_rebuilder(latent_image)
        if latent_tensor.ndim != 5:
            raise ValueError(
                f"LTXTiledSamplerDistributed expects a 5D latent tensor [B,C,F,H,W]; got shape {tuple(latent_tensor.shape)}"
            )
        mask_tensor = _extract_optional_mask(denoise_mask)
        self.multi_job_id = str(multi_job_id)
        self.master_url = str(master_url or "").rstrip("/")
        self.worker_id = str(worker_id or "")
        if isinstance(enabled_worker_ids, str):
            try:
                import json

                parsed_workers = json.loads(enabled_worker_ids) if enabled_worker_ids else []
            except Exception:
                parsed_workers = []
        else:
            parsed_workers = enabled_worker_ids or []
        self.enabled_worker_ids = [str(worker) for worker in parsed_workers]

        if is_worker:
            run_async_in_server_loop(
                _ltx_worker_process(
                    self,
                    noise,
                    guider,
                    sampler,
                    sigmas,
                    latent_tensor,
                    mask_tensor,
                    int(h_tiles),
                    int(w_tiles),
                    int(overlap),
                ),
                timeout=None,
            )
            placeholder = rebuild_latent(latent_tensor)
            return (placeholder, placeholder, placeholder)

        if not self.enabled_worker_ids:
            samples, denoised = _ltx_local_tiled_process(
                noise,
                guider,
                sampler,
                sigmas,
                latent_tensor,
                mask_tensor,
                int(h_tiles),
                int(w_tiles),
                int(overlap),
            )
        else:
            samples, denoised = run_async_in_server_loop(
                _ltx_master_process(
                    self,
                    noise,
                    guider,
                    sampler,
                    sigmas,
                    latent_tensor,
                    mask_tensor,
                    int(h_tiles),
                    int(w_tiles),
                    int(overlap),
                ),
                timeout=None,
            )
        samples_latent = rebuild_latent(samples)
        denoised_latent = rebuild_latent(denoised if denoised is not None else samples)
        return (samples_latent, denoised_latent, samples_latent)


NODE_CLASS_MAPPINGS = {"LTXTiledSamplerDistributed": LTXTiledSamplerDistributed}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXTiledSamplerDistributed": "LTX Tiled Sampler (Distributed)"}
