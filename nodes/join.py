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
from .utilities import any_type


MAX_BRANCH_OUTPUTS = 10


def _get_prompt_server_instance():
    import server as _server

    return _server.PromptServer.instance


def _throw_if_processing_interrupted():
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except Exception:
        return


class DistributedJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (any_type,),
                "num_branches": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1}),
            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
                "assigned_branch": ("INT", {"default": -1, "min": -1, "max": 9}),
                "delegate_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = tuple([any_type] * MAX_BRANCH_OUTPUTS)
    RETURN_NAMES = tuple([f"branch_{idx + 1}" for idx in range(MAX_BRANCH_OUTPUTS)])
    FUNCTION = "run"
    CATEGORY = "utils"

    def run(
        self,
        input,
        num_branches=2,
        multi_job_id="",
        is_worker=False,
        master_url="",
        enabled_worker_ids="[]",
        worker_id="",
        assigned_branch=-1,
        delegate_only=False,
    ):
        if not multi_job_id:
            return self._build_outputs(input, assigned_branch)

        return run_async_in_server_loop(
            self.execute(
                input,
                num_branches=num_branches,
                multi_job_id=multi_job_id,
                is_worker=is_worker,
                master_url=master_url,
                enabled_worker_ids=enabled_worker_ids,
                worker_id=worker_id,
                assigned_branch=assigned_branch,
                delegate_only=delegate_only,
            )
        )

    def _build_outputs(self, value, assigned_branch):
        outputs = [None] * MAX_BRANCH_OUTPUTS
        try:
            branch_idx = int(assigned_branch)
        except (TypeError, ValueError):
            branch_idx = -1

        if 0 <= branch_idx < MAX_BRANCH_OUTPUTS:
            outputs[branch_idx] = value
        return tuple(outputs)

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

    def _expected_worker_ids_for_branches(self, enabled_workers, delegate_mode, num_branches):
        slots_by_participant = self._participant_branch_slots(enabled_workers, delegate_mode, num_branches)
        expected = []
        for participant_id, slots in slots_by_participant.items():
            if participant_id == "master" or not slots:
                continue
            expected.append(str(participant_id))
        return expected

    def _participant_branch_slots(self, enabled_workers, delegate_mode, num_branches):
        participants = list(enabled_workers) if delegate_mode else ["master"] + list(enabled_workers)
        participant_count = len(participants)
        if participant_count <= 0:
            return {}

        slots_by_participant = {}
        for branch_slot in range(int(num_branches)):
            participant_id = str(participants[branch_slot % participant_count])
            slots_by_participant.setdefault(participant_id, []).append(branch_slot)
        return slots_by_participant

    def _fallback_for_missing_branch(self, input_value):
        if isinstance(input_value, torch.Tensor):
            tensor = input_value
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 4:
                fallback = torch.zeros_like(tensor)
                if fallback.is_cuda:
                    fallback = fallback.cpu()
                return ensure_contiguous(fallback)
        return input_value

    async def _send_branch_result_to_master(self, value, branch_idx, multi_job_id, master_url, worker_id):
        if not isinstance(value, torch.Tensor):
            raise ValueError("DistributedJoin currently supports torch.Tensor results for worker transfer.")

        image_batch = value
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)
        if image_batch.ndim != 4 or image_batch.shape[0] <= 0:
            raise ValueError("DistributedJoin tensor result must be IMAGE-like with shape [B,H,W,C] or [H,W,C].")

        img = tensor_to_pil(image_batch, 0)
        byte_io = io.BytesIO()
        img.save(byte_io, format="PNG", compress_level=0)
        encoded_image = base64.b64encode(byte_io.getvalue()).decode("utf-8")

        payload = {
            "job_id": str(multi_job_id),
            "worker_id": str(worker_id),
            "batch_idx": int(branch_idx),
            "image": f"data:image/png;base64,{encoded_image}",
            "is_last": True,
        }

        session = await get_client_session()
        url = f"{master_url}/distributed/job_complete"
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            response.raise_for_status()

    async def execute(
        self,
        input,
        num_branches=2,
        multi_job_id="",
        is_worker=False,
        master_url="",
        enabled_worker_ids="[]",
        worker_id="",
        assigned_branch=-1,
        delegate_only=False,
    ):
        try:
            branch_idx = int(assigned_branch)
        except (TypeError, ValueError):
            branch_idx = -1

        outputs = [None] * MAX_BRANCH_OUTPUTS

        if is_worker:
            if 0 <= branch_idx < MAX_BRANCH_OUTPUTS:
                try:
                    await self._send_branch_result_to_master(
                        input,
                        branch_idx,
                        multi_job_id,
                        master_url,
                        worker_id,
                    )
                    outputs[branch_idx] = input
                except Exception as exc:
                    log(f"Worker - DistributedJoin failed to send branch result: {exc}")
            return tuple(outputs)

        delegate_mode = bool(delegate_only or is_master_delegate_only())
        try:
            branch_count = int(num_branches)
        except (TypeError, ValueError):
            branch_count = 2
        branch_count = max(2, min(branch_count, MAX_BRANCH_OUTPUTS))

        enabled_workers = self._parse_enabled_workers(enabled_worker_ids)
        slots_by_participant = self._participant_branch_slots(enabled_workers, delegate_mode, branch_count)
        expected_worker_ids = self._expected_worker_ids_for_branches(enabled_workers, delegate_mode, branch_count)
        expected_workers = set(expected_worker_ids)

        if not delegate_mode and 0 <= branch_idx < MAX_BRANCH_OUTPUTS:
            outputs[branch_idx] = input

        if not expected_workers:
            return tuple(outputs)

        prompt_server = _get_prompt_server_instance()
        async with prompt_server.distributed_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()

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
                        "Master - DistributedJoin heartbeat timeout. "
                        f"Still waiting for workers: {missing_workers}"
                    )
                    break

                worker_id_value = str(result.get("worker_id", ""))
                image_index = result.get("image_index")
                tensor = result.get("tensor")
                is_last = bool(result.get("is_last", False))

                try:
                    slot_idx = int(image_index)
                except (TypeError, ValueError):
                    slot_idx = -1

                if 0 <= slot_idx < MAX_BRANCH_OUTPUTS and tensor is not None:
                    if isinstance(tensor, torch.Tensor):
                        if tensor.is_cuda:
                            tensor = tensor.cpu()
                        tensor = ensure_contiguous(tensor)
                    outputs[slot_idx] = tensor

                last_activity = time.time()
                base_timeout = float(get_worker_timeout_seconds())
                if is_last and worker_id_value in expected_workers:
                    workers_done.add(worker_id_value)

        finally:
            async with prompt_server.distributed_jobs_lock:
                prompt_server.distributed_pending_jobs.pop(multi_job_id, None)

        missing_workers = sorted(expected_workers - workers_done)
        if missing_workers:
            fallback = self._fallback_for_missing_branch(input)
            for missing_worker_id in missing_workers:
                for slot_idx in slots_by_participant.get(str(missing_worker_id), []):
                    if 0 <= slot_idx < MAX_BRANCH_OUTPUTS and outputs[slot_idx] is None:
                        outputs[slot_idx] = fallback

        return tuple(outputs)
