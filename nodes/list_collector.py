import asyncio
import base64
import io
import json
import time

import aiohttp
import torch

from ..utils.async_helpers import run_async_in_server_loop
from ..utils.config import get_worker_timeout_seconds, is_master_delegate_only
from ..utils.constants import HEARTBEAT_INTERVAL
from ..utils.image import ensure_contiguous, tensor_to_pil
from ..utils.logging import debug_log, log
from ..utils.network import get_client_session


def _get_prompt_server_instance():
    import server as _server

    return _server.PromptServer.instance


def _throw_if_processing_interrupted():
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except Exception:
        # Tests and non-Comfy contexts may not have interruption support.
        return


class DistributedListCollector:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
                "delegate_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(
        self,
        images,
        multi_job_id="",
        is_worker=False,
        master_url="",
        enabled_worker_ids="[]",
        worker_id="",
        delegate_only=False,
    ):
        image_list = self._normalize_image_list(images)
        if not multi_job_id:
            return (image_list,)

        return run_async_in_server_loop(
            self.execute(
                image_list,
                multi_job_id=multi_job_id,
                is_worker=is_worker,
                master_url=master_url,
                enabled_worker_ids=enabled_worker_ids,
                worker_id=worker_id,
                delegate_only=delegate_only,
            )
        )

    def _normalize_image_list(self, images):
        if images is None:
            return []
        if isinstance(images, list):
            return list(images)
        return [images]

    async def send_list_to_master(self, image_list, multi_job_id, master_url, worker_id):
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
            img = tensor_to_pil(image_batch, 0)
            byte_io = io.BytesIO()
            img.save(byte_io, format="PNG", compress_level=0)
            encoded_image = base64.b64encode(byte_io.getvalue()).decode("utf-8")
            payload = {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "batch_idx": int(image_index),
                "image": f"data:image/png;base64,{encoded_image}",
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

    def _parse_enabled_workers(self, enabled_worker_ids):
        try:
            raw = json.loads(enabled_worker_ids)
        except Exception:
            raw = []

        workers = []
        seen = set()
        for worker_id in raw if isinstance(raw, list) else []:
            worker_id_str = str(worker_id)
            if worker_id_str in seen:
                continue
            seen.add(worker_id_str)
            workers.append(worker_id_str)
        return workers

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

    async def execute(
        self,
        images,
        multi_job_id="",
        is_worker=False,
        master_url="",
        enabled_worker_ids="[]",
        worker_id="",
        delegate_only=False,
    ):
        image_list = self._normalize_image_list(images)

        if is_worker:
            debug_log(f"Worker - Job {multi_job_id} complete. Sending {len(image_list)} image list item(s) to master")
            await self.send_list_to_master(image_list, multi_job_id, master_url, worker_id)
            return (image_list,)

        delegate_mode = bool(delegate_only or is_master_delegate_only())
        enabled_workers = self._parse_enabled_workers(enabled_worker_ids)
        expected_workers = set(enabled_workers)
        if not expected_workers:
            return (image_list,)

        prompt_server = _get_prompt_server_instance()

        async with prompt_server.distributed_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
                debug_log(f"Master - Initialized queue early for list collector job {multi_job_id}")

        master_images = [] if delegate_mode else image_list

        worker_images = {}
        workers_done = set()
        base_timeout = float(get_worker_timeout_seconds())
        slice_timeout = min(max(0.1, HEARTBEAT_INTERVAL / 20.0), base_timeout)
        last_activity = time.time()

        try:
            while len(workers_done) < len(expected_workers):
                _throw_if_processing_interrupted()
                try:
                    async with prompt_server.distributed_jobs_lock:
                        queue = prompt_server.distributed_pending_jobs[multi_job_id]
                    result = await asyncio.wait_for(queue.get(), timeout=slice_timeout)
                except asyncio.TimeoutError:
                    if (time.time() - last_activity) < base_timeout:
                        continue
                    missing_workers = sorted(expected_workers - workers_done)
                    log(
                        "Master - List collector heartbeat timeout. "
                        f"Still waiting for workers: {missing_workers}"
                    )
                    break

                worker_id_value = str(result.get("worker_id", ""))
                is_last = bool(result.get("is_last", False))
                self._store_worker_result(worker_images, result)
                last_activity = time.time()
                base_timeout = float(get_worker_timeout_seconds())
                if is_last and worker_id_value in expected_workers:
                    workers_done.add(worker_id_value)

        finally:
            async with prompt_server.distributed_jobs_lock:
                prompt_server.distributed_pending_jobs.pop(multi_job_id, None)

        combined = self._reorder_and_combine_list(worker_images, enabled_workers, master_images, delegate_mode)
        debug_log(
            f"Master - List collector job {multi_job_id} complete. Combined {len(combined)} image list item(s)."
        )
        return (combined,)
