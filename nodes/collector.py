import torch
import io
import json
import asyncio
import time
import base64

import aiohttp
import server as _server
import comfy.model_management
from comfy.utils import ProgressBar

from ..utils.logging import debug_log, log
from ..utils.config import get_worker_timeout_seconds, load_config, is_master_delegate_only
from ..utils.constants import HEARTBEAT_INTERVAL
from ..utils.image import tensor_to_pil, pil_to_tensor, ensure_contiguous
from ..utils.network import build_worker_url, get_client_session, probe_worker
from ..utils.audio_payload import encode_audio_payload
from ..utils.async_helpers import run_async_in_server_loop

prompt_server = _server.PromptServer.instance


class DistributedCollectorNode:
    INPUT_IS_LIST = True
    EMPTY_AUDIO = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "load_balance": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run this workflow on one least-busy participant (master included when participating).",
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "worker_id": ("STRING", {"default": ""}),
                "pass_through": ("BOOLEAN", {"default": False}),
                "delegate_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"
    CATEGORY = "image"
    
    @staticmethod
    def _unwrap_list_input(value):
        """Unwrap scalar inputs when ComfyUI passes them via INPUT_IS_LIST."""
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    def _normalize_images_input(self, images):
        """Collapse ComfyUI list IMAGE inputs into a normal batched IMAGE tensor."""
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("Collector received an empty image list")
            if not all(isinstance(image, torch.Tensor) for image in images):
                raise TypeError("Collector expected IMAGE list items to be torch.Tensor instances")
            if len(images) == 1:
                return ensure_contiguous(images[0])
            return ensure_contiguous(torch.cat([ensure_contiguous(image) for image in images], dim=0))
        return ensure_contiguous(images)

    def _normalize_audio_input(self, audio):
        """Collapse ComfyUI list AUDIO inputs into a single AUDIO payload when present."""
        if not isinstance(audio, (list, tuple)):
            return audio

        audio_items = [item for item in audio if item is not None]
        if not audio_items:
            return None
        if len(audio_items) == 1:
            return audio_items[0]

        waveforms = []
        sample_rate = 44100
        for item in audio_items:
            if not isinstance(item, dict):
                raise TypeError("Collector expected AUDIO list items to be dictionaries")
            waveform = item.get("waveform")
            if waveform is None or waveform.numel() == 0:
                continue
            waveforms.append(waveform)
            if sample_rate == 44100:
                sample_rate = item.get("sample_rate", 44100)

        if not waveforms:
            return None
        return {"waveform": torch.cat(waveforms, dim=-1), "sample_rate": sample_rate}

    def run(self, images=None, load_balance=False, audio=None, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id="", pass_through=False, delegate_only=False):
        if images is not None:
            images = self._normalize_images_input(images)
        audio = self._normalize_audio_input(audio)
        load_balance = self._unwrap_list_input(load_balance)
        multi_job_id = self._unwrap_list_input(multi_job_id)
        is_worker = self._unwrap_list_input(is_worker)
        master_url = self._unwrap_list_input(master_url)
        enabled_worker_ids = self._unwrap_list_input(enabled_worker_ids)
        worker_batch_size = self._unwrap_list_input(worker_batch_size)
        worker_id = self._unwrap_list_input(worker_id)
        pass_through = self._unwrap_list_input(pass_through)
        delegate_only = self._unwrap_list_input(delegate_only)

        remote_only_master = (
            bool(multi_job_id)
            and not is_worker
            and (delegate_only or is_master_delegate_only())
        )
        if images is None and audio is None and not remote_only_master:
            raise ValueError("DistributedCollector requires at least one image or audio input")

        # Create empty audio if not provided
        empty_audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

        if not multi_job_id or pass_through:
            if pass_through:
                debug_log("Collector: pass-through mode enabled, returning images unchanged")
            return (images, audio if audio is not None else empty_audio)

        # Use async helper to run in server loop
        result = run_async_in_server_loop(
            self.execute(
                images,
                audio,
                load_balance,
                multi_job_id,
                is_worker,
                master_url,
                enabled_worker_ids,
                worker_batch_size,
                worker_id,
                delegate_only,
            )
        )
        return result

    async def send_batch_to_master(self, image_batch, audio, multi_job_id, master_url, worker_id):
        """Send an image batch, optionally with audio, or an audio-only completion."""
        encoded_audio = encode_audio_payload(audio)
        session = await get_client_session()
        url = f"{master_url}/distributed/job_complete"

        payloads = []
        batch_size = 0 if image_batch is None else image_batch.shape[0]
        if batch_size == 0:
            if encoded_audio is None:
                raise ValueError("Worker completion requires image or audio data")
            payloads.append(
                {
                    "job_id": str(multi_job_id),
                    "worker_id": str(worker_id),
                    "batch_idx": 0,
                    "audio": encoded_audio,
                    "is_last": True,
                }
            )
        else:
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
                payloads.append(payload)

        for payload in payloads:
            timeout_seconds = 60 if "image" in payload else 600
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as response:
                    response.raise_for_status()
            except Exception as e:
                media_type = "image/audio" if "image" in payload else "audio-only"
                log(f"Worker - Failed to send canonical {media_type} envelope to master: {e}")
                debug_log(f"Worker - Full error details: URL={url}")
                raise

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
    ):
        """Assemble final tensor, or return None when the job contains only audio."""
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
        if fallback_images is not None:
            return ensure_contiguous(fallback_images)
        return None

    async def execute(self, images, audio, load_balance=False, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id="", delegate_only=False):
        if is_worker:
            # Worker mode: send images and audio to master in a single batch
            image_count = 0 if images is None else images.shape[0]
            debug_log(f"Worker - Job {multi_job_id} complete. Sending {image_count} image(s) to master")
            await self.send_batch_to_master(images, audio, multi_job_id, master_url, worker_id)
            return (images, audio if audio is not None else self.EMPTY_AUDIO)
        else:
            delegate_mode = delegate_only or is_master_delegate_only()
            # Master mode: collect images and audio from workers
            enabled_workers_raw = json.loads(enabled_worker_ids)
            enabled_workers = []
            seen_enabled = set()
            for worker_id in enabled_workers_raw:
                worker_id_str = str(worker_id)
                if worker_id_str in seen_enabled:
                    continue
                seen_enabled.add(worker_id_str)
                enabled_workers.append(worker_id_str)
            expected_workers = set(enabled_workers)
            num_workers = len(expected_workers)
            if num_workers == 0:
                return (images, audio if audio is not None else self.EMPTY_AUDIO)

            # Create the queue before any expensive local work to avoid job_complete race.
            async with prompt_server.distributed_jobs_lock:
                if multi_job_id not in prompt_server.distributed_pending_jobs:
                    prompt_server.distributed_pending_jobs[multi_job_id] = asyncio.Queue()
                    debug_log(f"Master - Initialized queue early for job {multi_job_id}")
                else:
                    existing_size = prompt_server.distributed_pending_jobs[multi_job_id].qsize()
                    debug_log(f"Master - Using existing queue for job {multi_job_id} (current size: {existing_size})")

            if delegate_mode:
                master_batch_size = 0
                images_on_cpu = None
                master_audio = None
                debug_log(f"Master - Job {multi_job_id}: Delegate-only mode enabled, collecting exclusively from {num_workers} workers")
            else:
                if images is None:
                    images_on_cpu = None
                    master_batch_size = 0
                else:
                    images_on_cpu = ensure_contiguous(images.cpu())
                    master_batch_size = images.shape[0]
                master_audio = audio  # Keep master's audio for later
                debug_log(f"Master - Job {multi_job_id}: Master has {master_batch_size} images, collecting from {num_workers} workers...")


            # Initialize storage for collected images and audio
            worker_images = {}  # Dict to store images by worker_id and index
            worker_audio = {}   # Dict to store audio by worker_id
            
            # Collect images until all workers report they're done
            collected_count = 0
            workers_done = set()
            
            # Use unified worker timeout from config/UI with simple sliced waits
            base_timeout = float(get_worker_timeout_seconds())
            slice_timeout = min(max(0.1, HEARTBEAT_INTERVAL / 20.0), base_timeout)
            last_activity = time.time()
            
            
            # Get queue size before starting
            async with prompt_server.distributed_jobs_lock:
                q = prompt_server.distributed_pending_jobs[multi_job_id]
                initial_size = q.qsize()

            # NEW: Initialize progress bar for workers (total = num_workers)
            p = ProgressBar(num_workers)

            def mark_worker_done(done_worker_id):
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
                p.update(1)  # +1 per completed expected worker

            try:
                while len(workers_done) < num_workers:
                    # Check for user interruption to abort collection promptly
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    try:
                        # Get the queue again each time to ensure we have the right reference
                        async with prompt_server.distributed_jobs_lock:
                            q = prompt_server.distributed_pending_jobs[multi_job_id]
                            current_size = q.qsize()
                        
                        result = await asyncio.wait_for(q.get(), timeout=slice_timeout)
                        worker_id = result['worker_id']
                        is_last = result.get('is_last', False)
                        count = self._store_worker_result(worker_images, result)
                        collected_count += count
                        debug_log(
                            f"Master - Got canonical result from worker {worker_id}, "
                            f"image {result.get('image_index', 0)}, is_last={is_last}"
                        )

                        # Collect audio data if present
                        result_audio = result.get('audio')
                        if result_audio is not None:
                            worker_audio[worker_id] = result_audio
                            debug_log(f"Master - Got audio from worker {worker_id}")

                        # Record activity and refresh timeout baseline
                        last_activity = time.time()
                        base_timeout = float(get_worker_timeout_seconds())

                        if is_last:
                            mark_worker_done(worker_id)
                        
                    except asyncio.TimeoutError:
                        # If we still have time, continue polling; otherwise handle timeout
                        if (time.time() - last_activity) < base_timeout:
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            continue
                        # Re-check for user interruption after timeout expiry
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

                        # Probe missing workers' /prompt endpoints to check if they are actively processing
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

                        if any_busy:
                            # Refresh last_activity and continue waiting
                            last_activity = time.time()
                            # Refresh base timeout in case the user changed it in UI
                            base_timeout = float(get_worker_timeout_seconds())
                            continue
                        
                        # Check queue size again with lock
                        async with prompt_server.distributed_jobs_lock:
                            if multi_job_id in prompt_server.distributed_pending_jobs:
                                final_q = prompt_server.distributed_pending_jobs[multi_job_id]
                                final_size = final_q.qsize()
                                
                                # Try to drain any remaining items
                                remaining_items = []
                                while not final_q.empty():
                                    try:
                                        item = final_q.get_nowait()
                                        remaining_items.append(item)
                                    except asyncio.QueueEmpty:
                                        break
                                
                                if remaining_items:
                                    # Process them
                                    for item in remaining_items:
                                        worker_id = item['worker_id']
                                        is_last = item.get('is_last', False)

                                        collected_count += self._store_worker_result(worker_images, item)
                                        
                                        if is_last:
                                            mark_worker_done(worker_id)
                            else:
                                log(f"Master - Queue {multi_job_id} no longer exists!")
                        break
            except comfy.model_management.InterruptProcessingException:
                # Cleanup queue on interruption and re-raise to abort prompt cleanly
                async with prompt_server.distributed_jobs_lock:
                    if multi_job_id in prompt_server.distributed_pending_jobs:
                        del prompt_server.distributed_pending_jobs[multi_job_id]
                raise
            
            total_collected = sum(len(imgs) for imgs in worker_images.values())
            
            # Clean up job queue
            async with prompt_server.distributed_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_jobs:
                    del prompt_server.distributed_pending_jobs[multi_job_id]

            combined_audio = self._combine_audio(master_audio, worker_audio, self.EMPTY_AUDIO, enabled_workers)
            try:
                combined = self._reorder_and_combine_tensors(
                    worker_images, enabled_workers, master_batch_size, images_on_cpu, delegate_mode, images
                )
                if combined is None:
                    debug_log(f"Master - Job {multi_job_id} complete with audio only")
                else:
                    debug_log(f"Master - Job {multi_job_id} complete. Combined {combined.shape[0]} images total "
                              f"(master: {master_batch_size}, workers: {combined.shape[0] - master_batch_size})")

                return (combined, combined_audio)
            except Exception as e:
                log(f"Master - Error combining images: {e}")
                # Preserve collected audio even when image assembly fails.
                return (images, combined_audio)
