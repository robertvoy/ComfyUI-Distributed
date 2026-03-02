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
from ..utils.logging import log
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


class DistributedBranchCollector:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"branch_{idx + 1}": (any_type,)
            for idx in range(MAX_BRANCH_OUTPUTS)
        }
        return {
            "required": {
                "num_branches": ("INT", {"default": 2, "min": 2, "max": MAX_BRANCH_OUTPUTS, "step": 1}),
            },
            "optional": optional_inputs,
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
                "assigned_branch": ("INT", {"default": -1, "min": -1, "max": MAX_BRANCH_OUTPUTS - 1}),
                "delegate_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = tuple([any_type] * MAX_BRANCH_OUTPUTS)
    RETURN_NAMES = tuple([f"branch_{idx + 1}" for idx in range(MAX_BRANCH_OUTPUTS)])
    FUNCTION = "run"
    CATEGORY = "utils"

    def _branch_values(self, **kwargs):
        values = []
        for idx in range(MAX_BRANCH_OUTPUTS):
            values.append(kwargs.get(f"branch_{idx + 1}", None))
        return values

    def _resolve_local_branch_value(self, branch_values, assigned_branch):
        try:
            assigned_idx = int(assigned_branch)
        except (TypeError, ValueError):
            assigned_idx = -1

        if 0 <= assigned_idx < MAX_BRANCH_OUTPUTS:
            assigned_value = branch_values[assigned_idx]
            if assigned_value is not None:
                return assigned_value, assigned_idx

            # In participant-pruned prompts, each participant should typically have
            # exactly one local branch input. If socket names drifted (e.g. branch_2
            # connected for a participant assigned to slot 0), remap that sole value
            # to the assigned slot so the participant still contributes correctly.
            non_none_indices = [idx for idx, value in enumerate(branch_values) if value is not None]
            if len(non_none_indices) == 1:
                return branch_values[non_none_indices[0]], assigned_idx

        for idx, value in enumerate(branch_values):
            if value is not None:
                return value, idx
        return None, assigned_idx

    def _build_outputs(self, branch_values, num_branches):
        outputs = [None] * MAX_BRANCH_OUTPUTS
        try:
            branch_count = int(num_branches)
        except (TypeError, ValueError):
            branch_count = 2
        branch_count = max(2, min(branch_count, MAX_BRANCH_OUTPUTS))

        for idx in range(branch_count):
            value = branch_values[idx]
            if value is not None:
                outputs[idx] = value
        return tuple(outputs)

    def run(
        self,
        num_branches=2,
        multi_job_id="",
        is_worker=False,
        master_url="",
        enabled_worker_ids="[]",
        worker_id="",
        assigned_branch=-1,
        delegate_only=False,
        branch_1=None,
        branch_2=None,
        branch_3=None,
        branch_4=None,
        branch_5=None,
        branch_6=None,
        branch_7=None,
        branch_8=None,
        branch_9=None,
        branch_10=None,
    ):
        branch_values = [
            branch_1,
            branch_2,
            branch_3,
            branch_4,
            branch_5,
            branch_6,
            branch_7,
            branch_8,
            branch_9,
            branch_10,
        ]

        if not multi_job_id:
            return self._build_outputs(branch_values, num_branches)

        return run_async_in_server_loop(
            self.execute(
                branch_values,
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

    def _expected_worker_ids_for_branches(self, enabled_workers, delegate_mode, num_branches):
        slots_by_participant = self._participant_branch_slots(enabled_workers, delegate_mode, num_branches)
        expected = []
        for participant_id, slots in slots_by_participant.items():
            if participant_id == "master" or not slots:
                continue
            expected.append(str(participant_id))
        return expected

    def _fallback_for_missing_branch(self, source_value):
        if isinstance(source_value, torch.Tensor):
            tensor = source_value
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 4:
                fallback = torch.zeros_like(tensor)
                if fallback.is_cuda:
                    fallback = fallback.cpu()
                return ensure_contiguous(fallback)
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    async def _send_branch_result_to_master(self, value, branch_idx, multi_job_id, master_url, worker_id):
        if not isinstance(value, torch.Tensor):
            raise ValueError("DistributedBranchCollector currently supports torch.Tensor results for worker transfer.")

        image_batch = value
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)
        if image_batch.ndim != 4 or image_batch.shape[0] <= 0:
            raise ValueError(
                "DistributedBranchCollector tensor result must be IMAGE-like with shape [B,H,W,C] or [H,W,C]."
            )

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
        branch_values,
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
            branch_count = int(num_branches)
        except (TypeError, ValueError):
            branch_count = 2
        branch_count = max(2, min(branch_count, MAX_BRANCH_OUTPUTS))

        outputs = [None] * MAX_BRANCH_OUTPUTS
        for idx in range(branch_count):
            if branch_values[idx] is not None:
                outputs[idx] = branch_values[idx]

        local_value, local_branch_idx = self._resolve_local_branch_value(branch_values, assigned_branch)
        if 0 <= local_branch_idx < MAX_BRANCH_OUTPUTS and local_value is not None:
            outputs[local_branch_idx] = local_value

        if is_worker:
            if 0 <= local_branch_idx < MAX_BRANCH_OUTPUTS and local_value is not None:
                try:
                    await self._send_branch_result_to_master(
                        local_value,
                        local_branch_idx,
                        multi_job_id,
                        master_url,
                        worker_id,
                    )
                    outputs[local_branch_idx] = local_value
                except Exception as exc:
                    log(f"Worker - DistributedBranchCollector failed to send branch result: {exc}")
            return tuple(outputs)

        delegate_mode = bool(delegate_only or is_master_delegate_only())
        enabled_workers = self._parse_enabled_workers(enabled_worker_ids)
        slots_by_participant = self._participant_branch_slots(enabled_workers, delegate_mode, branch_count)
        expected_worker_ids = self._expected_worker_ids_for_branches(enabled_workers, delegate_mode, branch_count)
        expected_workers = set(expected_worker_ids)

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
                        "Master - DistributedBranchCollector heartbeat timeout. "
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
            fallback = self._fallback_for_missing_branch(local_value)
            for missing_worker_id in missing_workers:
                for slot_idx in slots_by_participant.get(str(missing_worker_id), []):
                    if 0 <= slot_idx < MAX_BRANCH_OUTPUTS and outputs[slot_idx] is None:
                        outputs[slot_idx] = fallback

        return tuple(outputs)
