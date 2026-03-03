import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch

from ..utils.async_helpers import run_async_in_server_loop
from ..utils.config import is_master_delegate_only
from ..utils.image import encode_tensor_png_data_url, ensure_contiguous
from ..utils.logging import log
from ..utils.network import get_client_session
from ..utils.worker_ids import parse_enabled_worker_ids
from .context_kwargs import parse_distributed_hidden_context
from .hidden_inputs import build_distributed_hidden_inputs
from .queue_wait import collect_worker_queue_results
from .runtime_helpers import (
    get_prompt_server_instance as _get_prompt_server_instance,
    throw_if_processing_interrupted as _throw_if_processing_interrupted,
)
from .utilities import any_type


MAX_BRANCH_OUTPUTS = 10


@dataclass(frozen=True)
class BranchRunContext:
    multi_job_id: str = ""
    is_worker: bool = False
    master_url: str = ""
    enabled_worker_ids: str = "[]"
    worker_id: str = ""
    assigned_branch: int = -1
    delegate_only: bool = False

class DistributedBranchCollector:
    @classmethod
    def INPUT_TYPES(cls: type["DistributedBranchCollector"]) -> dict[str, Any]:
        optional_inputs = {
            f"branch_{idx + 1}": (any_type,)
            for idx in range(MAX_BRANCH_OUTPUTS)
        }
        return {
            "required": {
                "num_branches": ("INT", {"default": 2, "min": 2, "max": MAX_BRANCH_OUTPUTS, "step": 1}),
            },
            "optional": optional_inputs,
            "hidden": build_distributed_hidden_inputs(
                include_assigned_branch=True,
                assigned_branch_max=MAX_BRANCH_OUTPUTS - 1,
            ),
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

    def _build_run_context(self, **kwargs: Any) -> BranchRunContext:
        try:
            assigned_branch = int(kwargs.get("assigned_branch", -1))
        except (TypeError, ValueError):
            assigned_branch = -1

        common_context = parse_distributed_hidden_context(kwargs)
        return BranchRunContext(
            assigned_branch=assigned_branch,
            **common_context,
        )

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
        num_branches: int = 2,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        branch_values = self._branch_values(**kwargs)
        context = self._build_run_context(**kwargs)

        if not context.multi_job_id:
            return self._build_outputs(branch_values, num_branches)

        return run_async_in_server_loop(
            self.execute(
                branch_values,
                num_branches=num_branches,
                context=context,
            )
        )

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

        payload = {
            "job_id": str(multi_job_id),
            "worker_id": str(worker_id),
            "batch_idx": int(branch_idx),
            "image": encode_tensor_png_data_url(image_batch, 0),
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

    async def execute(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Compatibility wrapper with a uniform execute() signature across collectors."""
        return await self._execute_branch(*args, **kwargs)

    async def _execute_branch(
        self,
        branch_values: list[Any],
        num_branches: int = 2,
        context: BranchRunContext | None = None,
    ) -> tuple[Any, ...]:
        run_context = context or BranchRunContext()
        try:
            branch_count = int(num_branches)
        except (TypeError, ValueError):
            branch_count = 2
        branch_count = max(2, min(branch_count, MAX_BRANCH_OUTPUTS))

        outputs = [None] * MAX_BRANCH_OUTPUTS
        for idx in range(branch_count):
            if branch_values[idx] is not None:
                outputs[idx] = branch_values[idx]

        local_value, local_branch_idx = self._resolve_local_branch_value(branch_values, run_context.assigned_branch)
        if 0 <= local_branch_idx < MAX_BRANCH_OUTPUTS and local_value is not None:
            outputs[local_branch_idx] = local_value

        if run_context.is_worker:
            if 0 <= local_branch_idx < MAX_BRANCH_OUTPUTS and local_value is not None:
                try:
                    await self._send_branch_result_to_master(
                        local_value,
                        local_branch_idx,
                        run_context.multi_job_id,
                        run_context.master_url,
                        run_context.worker_id,
                    )
                    outputs[local_branch_idx] = local_value
                except Exception as exc:
                    log(f"Worker - DistributedBranchCollector failed to send branch result: {exc}")
            return tuple(outputs)

        delegate_mode = bool(run_context.delegate_only or is_master_delegate_only())
        enabled_workers = parse_enabled_worker_ids(run_context.enabled_worker_ids)
        slots_by_participant = self._participant_branch_slots(enabled_workers, delegate_mode, branch_count)
        expected_worker_ids = self._expected_worker_ids_for_branches(enabled_workers, delegate_mode, branch_count)
        expected_workers = set(expected_worker_ids)

        if not expected_workers:
            return tuple(outputs)

        prompt_server = _get_prompt_server_instance()
        async with prompt_server.distributed_jobs_lock:
            if run_context.multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[run_context.multi_job_id] = asyncio.Queue()

        workers_done: set[str] = set()

        def _handle_queue_result(result: dict[str, Any]) -> None:
            image_index = result.get("image_index")
            tensor = result.get("tensor")
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

        try:
            workers_done = await collect_worker_queue_results(
                prompt_server=prompt_server,
                multi_job_id=run_context.multi_job_id,
                expected_workers=expected_workers,
                on_result=_handle_queue_result,
                timeout_log_prefix=(
                    "Master - DistributedBranchCollector heartbeat timeout. "
                    "Still waiting for workers: "
                ),
                throw_if_interrupted=_throw_if_processing_interrupted,
            )

        finally:
            async with prompt_server.distributed_jobs_lock:
                prompt_server.distributed_pending_jobs.pop(run_context.multi_job_id, None)

        missing_workers = sorted(expected_workers - workers_done)
        if missing_workers:
            fallback = self._fallback_for_missing_branch(local_value)
            for missing_worker_id in missing_workers:
                for slot_idx in slots_by_participant.get(str(missing_worker_id), []):
                    if 0 <= slot_idx < MAX_BRANCH_OUTPUTS and outputs[slot_idx] is None:
                        outputs[slot_idx] = fallback

        return tuple(outputs)
