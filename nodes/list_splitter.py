import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp

from ..upscale.job_models import BaseJobState, ImageJobState
from ..upscale.job_store import ensure_tile_jobs_initialized, init_dynamic_job
from ..utils.async_helpers import run_async_in_server_loop
from ..utils.logging import debug_log
from ..utils.network import get_client_session
from .utilities import _chunk_bounds


@dataclass(frozen=True)
class ListSplitterRunContext:
    participant_index: int = 0
    total_participants: int = 1
    multi_job_id: str = ""
    is_worker: bool = False
    master_url: str = ""
    worker_id: str = ""


class DistributedListSplitter:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls: type["DistributedListSplitter"]) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["static", "dynamic"], {"default": "static"}),
            },
            "hidden": {
                "participant_index": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "total_participants": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "split"
    CATEGORY = "image"

    def _build_run_context(self, **kwargs: Any) -> ListSplitterRunContext:
        try:
            participant_index = int(kwargs.get("participant_index", 0))
        except (TypeError, ValueError):
            participant_index = 0
        try:
            total_participants = int(kwargs.get("total_participants", 1))
        except (TypeError, ValueError):
            total_participants = 1

        return ListSplitterRunContext(
            participant_index=participant_index,
            total_participants=max(total_participants, 1),
            multi_job_id=str(kwargs.get("multi_job_id", "")),
            is_worker=bool(kwargs.get("is_worker", False)),
            master_url=str(kwargs.get("master_url", "")),
            worker_id=str(kwargs.get("worker_id", "")),
        )

    def split(
        self,
        images: list[Any] | Any | None,
        mode: str = "static",
        **kwargs: Any,
    ) -> tuple[list[Any]]:
        context = self._build_run_context(**kwargs)
        if images is None:
            return ([],)

        image_list = images if isinstance(images, list) else [images]
        if mode != "dynamic" or not context.multi_job_id:
            return self._split_static(
                image_list,
                context.participant_index,
                context.total_participants,
            )

        return run_async_in_server_loop(
            self._split_dynamic(
                image_list,
                context.multi_job_id,
                context.is_worker,
                context.master_url,
                context.worker_id,
            )
        )

    def _split_static(self, image_list, participant_index, total_participants):
        split_count = max(1, int(total_participants or 1))
        if split_count <= 1:
            return (list(image_list),)

        bounds = _chunk_bounds(len(image_list), split_count)
        index = int(participant_index or 0)
        if index < 0:
            index = 0
        if index >= split_count:
            index = split_count - 1

        start, end = bounds[index]
        return (list(image_list[start:end]),)

    async def _ensure_local_dynamic_queue(self, multi_job_id, total_items):
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            existing = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if isinstance(existing, ImageJobState):
                return
        await init_dynamic_job(
            multi_job_id,
            batch_size=total_items,
            enabled_workers=[],
            all_indices=list(range(total_items)),
        )

    async def _pull_items_from_local_queue(self, multi_job_id, worker_id):
        prompt_server = ensure_tile_jobs_initialized()
        pulled = []
        worker_key = str(worker_id or "master")

        while True:
            async with prompt_server.distributed_tile_jobs_lock:
                job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
                if not isinstance(job_data, BaseJobState):
                    break
                pending_queue = getattr(job_data, "pending_images", None) or getattr(job_data, "pending_tasks", None)
                if pending_queue is None:
                    break

                try:
                    item_idx = pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                job_data.assigned_to_workers.setdefault(worker_key, []).append(item_idx)
                pulled.append(int(item_idx))

        return pulled

    async def _request_item_from_master(self, multi_job_id, master_url, worker_id):
        session = await get_client_session()
        url = f"{master_url}/distributed/request_list_item"
        try:
            async with session.post(
                url,
                json={"multi_job_id": str(multi_job_id), "worker_id": str(worker_id or "")},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 404:
                    return "not_ready", None
                if response.status != 200:
                    return "error", None
                payload = await response.json()
                return "ok", payload.get("item_idx")
        except Exception:
            return "error", None

    async def _pull_items_from_master(self, multi_job_id, master_url, worker_id):
        pulled = []
        consecutive_not_ready = 0
        max_not_ready_polls = 40

        while True:
            status, item_idx = await self._request_item_from_master(multi_job_id, master_url, worker_id)

            if status == "not_ready":
                if consecutive_not_ready >= max_not_ready_polls:
                    break
                consecutive_not_ready += 1
                await asyncio.sleep(0.25)
                continue

            if status == "error":
                if pulled:
                    break
                await asyncio.sleep(0.25)
                continue

            consecutive_not_ready = 0
            if item_idx is None:
                break

            try:
                pulled.append(int(item_idx))
            except (TypeError, ValueError) as exc:
                debug_log(f"DistributedListSplitter dynamic mode: invalid item index '{item_idx}' ignored: {exc}")
                continue

        return pulled

    async def _split_dynamic(self, image_list, multi_job_id, is_worker, master_url, worker_id):
        if not image_list:
            return ([],)

        if is_worker:
            pulled_indices = await self._pull_items_from_master(multi_job_id, master_url, worker_id)
        else:
            await self._ensure_local_dynamic_queue(multi_job_id, len(image_list))
            pulled_indices = await self._pull_items_from_local_queue(multi_job_id, worker_id)

        selected_images = [
            image_list[index]
            for index in pulled_indices
            if isinstance(index, int) and 0 <= index < len(image_list)
        ]
        debug_log(
            f"DistributedListSplitter dynamic mode: role={'worker' if is_worker else 'master'} "
            f"pulled={len(selected_images)} items for job {multi_job_id}"
        )
        return (selected_images,)
