import asyncio
import base64
import io
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import server as _server
import comfy.model_management
import torch
from comfy.utils import ProgressBar

from ..utils.auth import distributed_auth_headers
from ..utils.logging import debug_log, log
from ..utils.config import get_worker_timeout_seconds, load_config, is_master_delegate_only
from ..utils.constants import HEARTBEAT_INTERVAL
from ..utils.image import tensor_to_pil, pil_to_tensor, ensure_contiguous
from ..utils.network import build_worker_url, get_client_session, probe_worker
from ..utils.worker_ids import coerce_enabled_worker_ids
from ..utils.audio_payload import encode_audio_payload
from ..utils.async_helpers import run_async_in_server_loop
from .context_kwargs import parse_distributed_hidden_context
from .hidden_inputs import build_distributed_hidden_inputs

prompt_server = _server.PromptServer.instance


@dataclass(frozen=True)
class CollectorRunContext:
    multi_job_id: str = ""
    is_worker: bool = False
    master_url: str = ""
    enabled_worker_ids: list[str] = field(default_factory=list)
    worker_batch_size: int = 1
    worker_id: str = ""
    pass_through: bool = False
    delegate_only: bool = False


class DistributedCollectorNode:
    EMPTY_AUDIO = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

    @classmethod
    def INPUT_TYPES(cls: type["DistributedCollectorNode"]) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "load_balance": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run this workflow on one least-busy participant (master included when participating).",
                    },
                ),
            },
            "optional": { "audio": ("AUDIO",) },
            "hidden": build_distributed_hidden_inputs(
                include_worker_batch_size=True,
                include_pass_through=True,
            ),
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "image"

    def _build_run_context(self, **kwargs: Any) -> CollectorRunContext:
        try:
            worker_batch_size = int(kwargs.get("worker_batch_size", 1))
        except (TypeError, ValueError):
            worker_batch_size = 1

        common_context = parse_distributed_hidden_context(kwargs)
        return CollectorRunContext(
            worker_batch_size=max(worker_batch_size, 1),
            pass_through=bool(kwargs.get("pass_through", False)),
            **common_context,
        )

    def run(
        self,
        images: torch.Tensor,
        load_balance: bool = False,
        audio: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        context = self._build_run_context(**kwargs)
        # Create empty audio if not provided
        empty_audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

        if not context.multi_job_id or context.pass_through:
            if context.pass_through:
                debug_log("Collector: pass-through mode enabled, returning images unchanged")
            return (images, audio if audio is not None else empty_audio)

        # Use async helper to run in server loop
        result = run_async_in_server_loop(
            self.execute(
                images=images,
                audio=audio,
                load_balance=load_balance,
                context=context,
            )
        )
        return result

    async def send_batch_to_master(
        self,
        image_batch: torch.Tensor,
        audio: dict[str, Any] | None,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
    ) -> None:
        """Send image batch to master via canonical JSON envelopes."""
        batch_size = image_batch.shape[0]
        if batch_size == 0:
            return

        encoded_audio = encode_audio_payload(audio)

        session = await get_client_session()
        url = f"{master_url}/distributed/job_complete"
        for batch_idx in range(batch_size):
            img = tensor_to_pil(image_batch[batch_idx:batch_idx+1], 0)
            byte_io = io.BytesIO()
            img.save(byte_io, format='PNG', compress_level=0)
            encoded_image = base64.b64encode(byte_io.getvalue()).decode('utf-8')
            payload = {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "batch_idx": int(batch_idx),
                "image": f"data:image/png;base64,{encoded_image}",
                "is_last": bool(batch_idx == batch_size - 1),
            }
            if payload["is_last"] and encoded_audio is not None:
                payload["audio"] = encoded_audio

            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=distributed_auth_headers(load_config()),
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response.raise_for_status()
            except Exception as e:
                log(f"Worker - Failed to send canonical image envelope to master: {e}")
                debug_log(f"Worker - Full error details: URL={url}")
                raise  # Re-raise to handle at caller level

    def _combine_audio(self, master_audio, worker_audio, empty_audio, worker_order=None):
        """Combine audio from master and workers into a single audio output.

        Ordering: master first, then workers in `worker_order` (if provided),
        then any unexpected worker ids in sorted order.
        """
        audio_pieces = []
        sample_rate = 44100

        # Add master audio first if present
        if master_audio is not None:
            waveform = master_audio.get("waveform")
            if waveform is not None and waveform.numel() > 0:
                audio_pieces.append(waveform)
                sample_rate = master_audio.get("sample_rate", 44100)

        # Add worker audio in configured enabled-worker order first.
        ordered_worker_ids = [str(worker_id) for worker_id in (worker_order or [])]
        seen = set()
        for worker_id_str in ordered_worker_ids:
            seen.add(worker_id_str)
            w_audio = worker_audio.get(worker_id_str)
            if w_audio is not None:
                waveform = w_audio.get("waveform")
                if waveform is not None and waveform.numel() > 0:
                    audio_pieces.append(waveform)
                    # Use first available sample rate
                    if sample_rate == 44100:
                        sample_rate = w_audio.get("sample_rate", 44100)

        # Append any audio from unexpected worker ids deterministically.
        for worker_id_str in sorted(worker_audio.keys()):
            if worker_id_str in seen:
                continue
            w_audio = worker_audio[worker_id_str]
            if w_audio is not None:
                waveform = w_audio.get("waveform")
                if waveform is not None and waveform.numel() > 0:
                    audio_pieces.append(waveform)
                    if sample_rate == 44100:
                        sample_rate = w_audio.get("sample_rate", 44100)

        if not audio_pieces:
            return empty_audio

        try:
            # Concatenate along the samples dimension (dim=-1)
            # Ensure all pieces have same batch and channel dimensions
            combined_waveform = torch.cat(audio_pieces, dim=-1)
            debug_log(f"Master - Combined audio: {len(audio_pieces)} pieces, final shape={combined_waveform.shape}")
            return {"waveform": combined_waveform, "sample_rate": sample_rate}
        except Exception as e:
            log(f"[Distributed] Master - Audio combination failed, returning silence: {e}")
            return empty_audio

    def _store_worker_result(self, worker_images: dict, item: dict) -> int:
        """Store one canonical queue item in worker_images in-place.

        Canonical format:
        - item has 'worker_id', 'image_index', and 'tensor'
        Returns 1 when stored, otherwise 0.
        """
        worker_id = item['worker_id']
        tensor = item.get('tensor')
        image_index = item.get('image_index')
        if tensor is None or image_index is None:
            return 0

        worker_images.setdefault(worker_id, {})
        worker_images[worker_id][image_index] = tensor
        return 1

    def _reorder_and_combine_tensors(
        self,
        worker_images: dict,
        worker_order: list,
        master_batch_size: int,
        images_on_cpu,
        delegate_mode: bool,
        fallback_images,
    ) -> torch.Tensor:
        """Assemble final tensor: master first, then workers in enabled order."""
        ordered_tensors = []
        if not delegate_mode and images_on_cpu is not None:
            for i in range(master_batch_size):
                ordered_tensors.append(images_on_cpu[i:i+1])

        ordered_worker_ids = [str(worker_id) for worker_id in (worker_order or [])]
        seen = set()
        for worker_id_str in ordered_worker_ids:
            seen.add(worker_id_str)
            if worker_id_str not in worker_images:
                continue
            for idx in sorted(worker_images[worker_id_str].keys()):
                ordered_tensors.append(worker_images[worker_id_str][idx])

        # Append any unexpected worker ids deterministically.
        for worker_id_str in sorted(worker_images.keys()):
            if worker_id_str in seen:
                continue
            for idx in sorted(worker_images[worker_id_str].keys()):
                ordered_tensors.append(worker_images[worker_id_str][idx])

        cpu_tensors = []
        for t in ordered_tensors:
            if t.is_cuda:
                t = t.cpu()
            t = ensure_contiguous(t)
            cpu_tensors.append(t)

        if cpu_tensors:
            return ensure_contiguous(torch.cat(cpu_tensors, dim=0))
        elif fallback_images is not None:
            return ensure_contiguous(fallback_images)
        else:
            raise ValueError("No image data collected from master or workers")

    async def _ensure_pending_queue(self, multi_job_id):
        async with prompt_server.distributed_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_jobs:
                prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
                debug_log(f"Master - Initialized queue early for job {multi_job_id}")
            else:
                existing_size = prompt_server.distributed_pending_jobs[multi_job_id].qsize()
                debug_log(f"Master - Using existing queue for job {multi_job_id} (current size: {existing_size})")

    async def _cleanup_pending_queue(self, multi_job_id):
        async with prompt_server.distributed_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_jobs:
                del prompt_server.distributed_pending_jobs[multi_job_id]
            if hasattr(prompt_server, "distributed_job_allowed_workers"):
                prompt_server.distributed_job_allowed_workers.pop(multi_job_id, None)

    def _build_master_inputs(self, images, audio, delegate_mode, multi_job_id, num_workers):
        if delegate_mode:
            debug_log(
                f"Master - Job {multi_job_id}: Delegate-only mode enabled, collecting exclusively from {num_workers} workers"
            )
            return 0, None, None

        images_on_cpu = ensure_contiguous(images.cpu())
        master_batch_size = images.shape[0]
        debug_log(
            f"Master - Job {multi_job_id}: Master has {master_batch_size} images, collecting from {num_workers} workers..."
        )
        return master_batch_size, images_on_cpu, audio

    async def _probe_missing_workers_busy(self, missing_workers):
        any_busy = False
        try:
            cfg = load_config()
            cfg_workers = cfg.get('workers', [])
            for wid in list(missing_workers):
                wrec = next((w for w in cfg_workers if str(w.get('id')) == str(wid)), None)
                if not wrec:
                    debug_log(f"Collector probe: worker {wid} not found in config")
                    continue
                worker_url = build_worker_url(wrec)
                try:
                    payload = await probe_worker(worker_url, timeout=2.0)
                    queue_remaining = None
                    if payload is not None:
                        queue_remaining = int(payload.get('exec_info', {}).get('queue_remaining', 0))
                    debug_log(
                        "Collector probe: worker "
                        f"{wid} online={payload is not None} queue_remaining={queue_remaining}"
                    )
                    if payload is not None and queue_remaining and queue_remaining > 0:
                        any_busy = True
                        log(
                            f"Master - Probe grace: worker {wid} appears busy "
                            f"(queue_remaining={queue_remaining}). Continuing to wait."
                        )
                        break
                except Exception as e:
                    debug_log(f"Collector probe failed for worker {wid}: {e}")
        except Exception as e:
            debug_log(f"Collector probe setup error: {e}")
        return any_busy

    async def _drain_remaining_queue_items(
        self,
        multi_job_id: str,
        worker_images: dict[str, dict[int, torch.Tensor]],
        mark_worker_done: Callable[[str], None],
    ) -> int:
        collected_count = 0
        async with prompt_server.distributed_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_jobs:
                log(f"Master - Queue {multi_job_id} no longer exists!")
                return collected_count

            final_q = prompt_server.distributed_pending_jobs[multi_job_id]
            remaining_items = []
            while not final_q.empty():
                try:
                    remaining_items.append(final_q.get_nowait())
                except asyncio.QueueEmpty:
                    break

        for item in remaining_items:
            worker_id = item['worker_id']
            is_last = item.get('is_last', False)
            collected_count += self._store_worker_result(worker_images, item)
            if is_last:
                mark_worker_done(worker_id)
        return collected_count

    async def _collect_worker_results(
        self,
        multi_job_id: str,
        enabled_workers: list[str],
        expected_workers: set[str],
        worker_images: dict[str, dict[int, torch.Tensor]],
        worker_audio: dict[str, dict[str, Any]],
    ) -> int:
        num_workers = len(expected_workers)
        workers_done = set()
        collected_count = 0
        base_timeout = float(get_worker_timeout_seconds())
        slice_timeout = min(max(0.1, HEARTBEAT_INTERVAL / 20.0), base_timeout)
        last_activity = time.time()
        progress = ProgressBar(num_workers)

        def mark_worker_done(done_worker_id: str) -> None:
            done_worker_id = str(done_worker_id)
            if done_worker_id not in expected_workers:
                debug_log(
                    f"Master - Ignoring completion from unexpected worker {done_worker_id} for job {multi_job_id}"
                )
                return
            if done_worker_id in workers_done:
                debug_log(
                    f"Master - Ignoring duplicate completion from worker {done_worker_id} for job {multi_job_id}"
                )
                return
            workers_done.add(done_worker_id)
            progress.update(1)

        while len(workers_done) < num_workers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            try:
                async with prompt_server.distributed_jobs_lock:
                    q = prompt_server.distributed_pending_jobs[multi_job_id]

                result = await asyncio.wait_for(q.get(), timeout=slice_timeout)
                worker_id = result['worker_id']
                is_last = result.get('is_last', False)
                collected_count += self._store_worker_result(worker_images, result)
                debug_log(
                    f"Master - Got canonical result from worker {worker_id}, "
                    f"image {result.get('image_index', 0)}, is_last={is_last}"
                )

                result_audio = result.get('audio')
                if result_audio is not None:
                    worker_audio[worker_id] = result_audio
                    debug_log(f"Master - Got audio from worker {worker_id}")

                last_activity = time.time()
                base_timeout = float(get_worker_timeout_seconds())
                if is_last:
                    mark_worker_done(worker_id)
            except asyncio.TimeoutError:
                if (time.time() - last_activity) < base_timeout:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    continue

                comfy.model_management.throw_exception_if_processing_interrupted()
                missing_workers = set(str(w) for w in enabled_workers) - workers_done
                elapsed = time.time() - last_activity
                for missing_worker_id in sorted(missing_workers):
                    log(
                        "Master - Heartbeat timeout: "
                        f"worker={missing_worker_id}, elapsed={elapsed:.1f}s"
                    )
                log(
                    f"Master - Heartbeat timeout. Still waiting for workers: {list(missing_workers)} "
                    f"(elapsed={elapsed:.1f}s)"
                )

                if await self._probe_missing_workers_busy(missing_workers):
                    last_activity = time.time()
                    base_timeout = float(get_worker_timeout_seconds())
                    continue

                collected_count += await self._drain_remaining_queue_items(
                    multi_job_id,
                    worker_images,
                    mark_worker_done,
                )
                break

        return collected_count

    async def _execute_master(
        self,
        images,
        audio,
        multi_job_id,
        enabled_worker_ids,
        delegate_only,
    ):
        delegate_mode = delegate_only or is_master_delegate_only()
        enabled_workers = coerce_enabled_worker_ids(enabled_worker_ids)
        expected_workers = set(enabled_workers)
        num_workers = len(expected_workers)
        if num_workers == 0:
            return (images, audio if audio is not None else self.EMPTY_AUDIO)

        await self._ensure_pending_queue(multi_job_id)
        master_batch_size, images_on_cpu, master_audio = self._build_master_inputs(
            images, audio, delegate_mode, multi_job_id, num_workers
        )

        worker_images = {}
        worker_audio = {}
        try:
            await self._collect_worker_results(
                multi_job_id=multi_job_id,
                enabled_workers=enabled_workers,
                expected_workers=expected_workers,
                worker_images=worker_images,
                worker_audio=worker_audio,
            )
        except comfy.model_management.InterruptProcessingException:
            await self._cleanup_pending_queue(multi_job_id)
            raise

        await self._cleanup_pending_queue(multi_job_id)
        try:
            combined = self._reorder_and_combine_tensors(
                worker_images, enabled_workers, master_batch_size, images_on_cpu, delegate_mode, images
            )
            debug_log(
                f"Master - Job {multi_job_id} complete. Combined {combined.shape[0]} images total "
                f"(master: {master_batch_size}, workers: {combined.shape[0] - master_batch_size})"
            )
            combined_audio = self._combine_audio(master_audio, worker_audio, self.EMPTY_AUDIO, enabled_workers)
            return (combined, combined_audio)
        except Exception as e:
            log(f"Master - Error combining images: {e}")
            return (images, audio if audio is not None else self.EMPTY_AUDIO)

    async def execute(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compatibility wrapper with a uniform execute() signature across collectors."""
        return await self._execute_collector(*args, **kwargs)

    async def _execute_collector(
        self,
        images: torch.Tensor,
        audio: dict[str, Any] | None,
        load_balance: bool = False,
        context: CollectorRunContext | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        run_context = context or CollectorRunContext()
        _ = load_balance
        _ = run_context.worker_batch_size
        if run_context.is_worker:
            debug_log(
                "Worker - Job "
                f"{run_context.multi_job_id} complete. Sending {images.shape[0]} image(s) to master"
            )
            await self.send_batch_to_master(
                images,
                audio,
                run_context.multi_job_id,
                run_context.master_url,
                run_context.worker_id,
            )
            return (images, audio if audio is not None else self.EMPTY_AUDIO)

        return await self._execute_master(
            images=images,
            audio=audio,
            multi_job_id=run_context.multi_job_id,
            enabled_worker_ids=run_context.enabled_worker_ids,
            delegate_only=run_context.delegate_only,
        )
