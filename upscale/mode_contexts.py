from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TileOpsCollaborator:
    """Typed facade over tile-related operations used by USDU modes."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def round_to_multiple(self, value: int) -> int:
        return self._delegate.round_to_multiple(value)

    def calculate_tiles(
        self,
        width: int,
        height: int,
        tile_width: int,
        tile_height: int,
        force_uniform_tiles: bool,
    ) -> list[tuple[int, int]]:
        return self._delegate.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)

    def create_tile_mask(
        self,
        width: int,
        height: int,
        tx: int,
        ty: int,
        tile_width: int,
        tile_height: int,
        mask_blur: int,
    ) -> Any:
        return self._delegate.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur)

    def blend_tile(
        self,
        base_image: Any,
        tile_pil: Any,
        x1: int,
        y1: int,
        size: tuple[int, int],
        tile_mask: Any,
        padding: int,
    ) -> Any:
        return self._delegate.blend_tile(base_image, tile_pil, x1, y1, size, tile_mask, padding)

    def slice_conditioning(self, positive: Any, negative: Any, image_idx: int) -> tuple[Any, Any]:
        return self._delegate.slice_conditioning(positive, negative, image_idx)

    def process_and_blend_tile(
        self,
        tile_idx: int,
        tile_pos: tuple[int, int],
        upscaled_image: Any,
        result_image: Any,
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
        image_width: int,
        image_height: int,
        force_uniform_tiles: bool,
        tiled_decode: bool,
        batch_idx: int,
    ) -> Any:
        return self._delegate.process_and_blend_tile(
            tile_idx,
            tile_pos,
            upscaled_image,
            result_image,
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
            tile_width,
            tile_height,
            padding,
            mask_blur,
            image_width,
            image_height,
            force_uniform_tiles,
            tiled_decode,
            batch_idx=batch_idx,
        )


class JobStateCollaborator:
    """Typed facade over job-state operations used by USDU modes."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def next_tile_index(self, multi_job_id: str) -> int | None:
        return await self._delegate.next_tile_index(multi_job_id)

    async def all_completed_tasks(self, multi_job_id: str) -> dict[int, dict[str, Any]]:
        return await self._delegate.all_completed_tasks(multi_job_id)

    async def check_and_requeue_timed_out_workers(self, multi_job_id: str, expected_total: int) -> int:
        return await self._delegate.check_and_requeue_timed_out_workers(multi_job_id, expected_total)

    async def next_image_index(self, multi_job_id: str) -> int | None:
        return await self._delegate.next_image_index(multi_job_id)

    async def drain_worker_results_queue(self, multi_job_id: str) -> int:
        return await self._delegate.drain_worker_results_queue(multi_job_id)

    async def total_completed_count(self, multi_job_id: str) -> int:
        return await self._delegate.total_completed_count(multi_job_id)

    async def pending_count(self, multi_job_id: str) -> int:
        return await self._delegate.pending_count(multi_job_id)

    async def all_completed_images(self, multi_job_id: str) -> dict[int, Any]:
        return await self._delegate.all_completed_images(multi_job_id)


class WorkerCommsCollaborator:
    """Typed facade over worker/master communication operations used by USDU modes."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def request_assignment(self, multi_job_id: str, master_url: str, worker_id: str) -> Any:
        return await self._delegate.request_assignment(multi_job_id, master_url, worker_id)

    async def send_tiles_batch(
        self,
        processed_tiles: list[dict[str, Any]],
        multi_job_id: str,
        master_url: str,
        padding: int,
        worker_id: str,
        is_final_flush: bool = False,
    ) -> None:
        await self._delegate.send_tiles_batch(
            processed_tiles,
            multi_job_id,
            master_url,
            padding,
            worker_id,
            is_final_flush=is_final_flush,
        )

    async def send_heartbeat(self, multi_job_id: str, master_url: str, worker_id: str) -> None:
        await self._delegate.send_heartbeat(multi_job_id, master_url, worker_id)

    async def check_job_status(self, multi_job_id: str, master_url: str) -> bool:
        return await self._delegate.check_job_status(multi_job_id, master_url)

    async def async_yield(self) -> None:
        await self._delegate.async_yield()

    async def send_full_image(
        self,
        image_pil: Any,
        image_idx: int,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
        is_last: bool,
    ) -> None:
        await self._delegate.send_full_image(
            image_pil,
            image_idx,
            multi_job_id,
            master_url,
            worker_id,
            is_last,
        )

    async def send_worker_complete_signal(self, multi_job_id: str, master_url: str, worker_id: str) -> None:
        await self._delegate.send_worker_complete_signal(multi_job_id, master_url, worker_id)


class ResultCollectorCollaborator:
    """Typed facade over result-collection operations used by dynamic mode."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def mark_image_completed(self, multi_job_id: str, image_idx: int, image_pil: Any) -> None:
        await self._delegate.mark_image_completed(multi_job_id, image_idx, image_pil)

    async def collect_dynamic_images(
        self,
        multi_job_id: str,
        remaining_to_collect: int,
        num_workers: int,
        batch_size: int,
        master_processed_count: int,
    ) -> dict[int, Any]:
        return await self._delegate.collect_dynamic_images(
            multi_job_id,
            remaining_to_collect,
            num_workers,
            batch_size,
            master_processed_count,
        )


@dataclass(frozen=True)
class SingleGpuModeContext:
    tile_ops: TileOpsCollaborator


@dataclass(frozen=True)
class StaticModeContext:
    tile_ops: TileOpsCollaborator
    job_state: JobStateCollaborator
    worker_comms: WorkerCommsCollaborator


@dataclass(frozen=True)
class DynamicModeContext:
    tile_ops: TileOpsCollaborator
    job_state: JobStateCollaborator
    worker_comms: WorkerCommsCollaborator
    result_collector: ResultCollectorCollaborator
