import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch

from ..utils.async_helpers import run_async_in_server_loop
from ..utils.config import is_master_delegate_only
from ..utils.image import encode_tensor_png_data_url, ensure_contiguous
from ..utils.logging import debug_log, log
from ..utils.network import get_client_session
from ..utils.worker_ids import parse_enabled_worker_ids
from .context_kwargs import parse_distributed_hidden_context
from .hidden_inputs import build_distributed_hidden_inputs
from .queue_wait import collect_worker_queue_results
from .runtime_helpers import (
    get_prompt_server_instance as _get_prompt_server_instance,
    throw_if_processing_interrupted as _throw_if_processing_interrupted,
)


@dataclass(frozen=True)
class ListCollectorRunContext:
    multi_job_id: str = ""
    is_worker: bool = False
    master_url: str = ""
    enabled_worker_ids: str = "[]"
    worker_id: str = ""
    delegate_only: bool = False


class DistributedListCollector:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls: type["DistributedListCollector"]) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": build_distributed_hidden_inputs(),
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "image"

    def _build_run_context(self, **kwargs: Any) -> ListCollectorRunContext:
        return ListCollectorRunContext(**parse_distributed_hidden_context(kwargs))

    def run(
        self,
        images: list[Any] | torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[list[Any]]:
        context = self._build_run_context(**kwargs)
        image_list = self._normalize_image_list(images)
        if not context.multi_job_id:
            return (image_list,)

        return run_async_in_server_loop(
            self.execute(
                image_list,
                context=context,
            )
        )

    def _normalize_image_list(self, images: list[Any] | torch.Tensor | None) -> list[Any]:
        if images is None:
            return []
        if isinstance(images, list):
            return list(images)
        return [images]

    async def send_list_to_master(
        self,
        image_list: list[Any],
        multi_job_id: str,
        master_url: str,
        worker_id: str,
    ) -> None:
        if not image_list:
            return

        sendable = []
        for idx, image in enumerate(image_list):
            if not isinstance(image, torch.Tensor):
                continue
            batch = image
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
            if batch.ndim != 4 or batch.shape[0] <= 0:
                continue
            sendable.append((idx, batch))

        if not sendable:
            return

        session = await get_client_session()
        url = f"{master_url}/distributed/job_complete"

        for send_pos, (image_index, image_batch) in enumerate(sendable):
            payload = {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "batch_idx": int(image_index),
                "image": encode_tensor_png_data_url(image_batch, 0),
                "is_last": bool(send_pos == len(sendable) - 1),
            }
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response.raise_for_status()
            except Exception as exc:
                log(f"Worker - Failed to send list item to master: {exc}")
                debug_log(f"Worker - Full error details: URL={url}")
                raise

    def _store_worker_result(self, worker_images, item):
        worker_id = str(item.get("worker_id", ""))
        tensor = item.get("tensor")
        image_index = item.get("image_index")
        if not worker_id or tensor is None or image_index is None:
            return 0

        worker_images.setdefault(worker_id, {})
        worker_images[worker_id][int(image_index)] = tensor
        return 1

    def _reorder_and_combine_list(self, worker_images, worker_order, master_images, delegate_mode):
        ordered = []
        if not delegate_mode:
            ordered.extend(master_images)

        ordered_worker_ids = [str(worker_id) for worker_id in (worker_order or [])]
        seen = set()

        for worker_id in ordered_worker_ids:
            seen.add(worker_id)
            worker_entries = worker_images.get(worker_id, {})
            for image_index in sorted(worker_entries.keys()):
                ordered.append(worker_entries[image_index])

        for worker_id in sorted(worker_images.keys()):
            if worker_id in seen:
                continue
            worker_entries = worker_images.get(worker_id, {})
            for image_index in sorted(worker_entries.keys()):
                ordered.append(worker_entries[image_index])

        combined = []
        for image in ordered:
            if isinstance(image, torch.Tensor):
                if image.is_cuda:
                    image = image.cpu()
                image = ensure_contiguous(image)
            combined.append(image)

        return combined

    async def execute(self, *args: Any, **kwargs: Any) -> tuple[list[Any]]:
        """Compatibility wrapper with a uniform execute() signature across collectors."""
        return await self._execute_list(*args, **kwargs)

    async def _execute_list(
        self,
        images: list[Any] | torch.Tensor | None,
        context: ListCollectorRunContext | None = None,
    ) -> tuple[list[Any]]:
        run_context = context or ListCollectorRunContext()
        image_list = self._normalize_image_list(images)

        if run_context.is_worker:
            debug_log(
                "Worker - Job "
                f"{run_context.multi_job_id} complete. Sending {len(image_list)} image list item(s) to master"
            )
            await self.send_list_to_master(
                image_list,
                run_context.multi_job_id,
                run_context.master_url,
                run_context.worker_id,
            )
            return (image_list,)

        delegate_mode = bool(run_context.delegate_only or is_master_delegate_only())
        enabled_workers = parse_enabled_worker_ids(run_context.enabled_worker_ids)
        expected_workers = set(enabled_workers)
        if not expected_workers:
            return (image_list,)

        prompt_server = _get_prompt_server_instance()

        async with prompt_server.distributed_jobs_lock:
            if run_context.multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[run_context.multi_job_id] = asyncio.Queue()
                debug_log(
                    "Master - Initialized queue early for list collector job "
                    f"{run_context.multi_job_id}"
                )

        master_images = [] if delegate_mode else image_list

        worker_images = {}
        workers_done: set[str] = set()

        def _handle_queue_result(result: dict[str, Any]) -> None:
            self._store_worker_result(worker_images, result)

        try:
            workers_done = await collect_worker_queue_results(
                prompt_server=prompt_server,
                multi_job_id=run_context.multi_job_id,
                expected_workers=expected_workers,
                on_result=_handle_queue_result,
                timeout_log_prefix=(
                    "Master - List collector heartbeat timeout. "
                    "Still waiting for workers: "
                ),
                throw_if_interrupted=_throw_if_processing_interrupted,
            )

        finally:
            async with prompt_server.distributed_jobs_lock:
                prompt_server.distributed_pending_jobs.pop(run_context.multi_job_id, None)

        combined = self._reorder_and_combine_list(worker_images, enabled_workers, master_images, delegate_mode)
        debug_log(
            "Master - List collector job "
            f"{run_context.multi_job_id} complete. Combined {len(combined)} image list item(s)."
        )
        return (combined,)
