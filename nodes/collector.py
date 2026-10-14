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
from ..utils.constants import HEARTBEAT_INTERVAL, TRANSFER_MAX_CHUNK_BYTES, TRANSFER_MAX_REQUEST_BYTES
from ..utils.image import tensor_to_pil, pil_to_tensor, ensure_contiguous
from ..utils.network import build_worker_url, get_client_session, probe_worker
from ..utils.audio_payload import encode_audio_payload
from ..utils.async_helpers import run_async_in_server_loop
from ..utils import transfer_codec

prompt_server = _server.PromptServer.instance


def _encode_all_chunks(frames, fmt, quality, max_chunk_bytes, max_frame_bytes):
    """Materialize every transfer chunk (runs in an executor)."""
    return list(transfer_codec.encode_chunks(frames, fmt, quality, max_chunk_bytes, max_frame_bytes))


def _exact_frame_bytes(image_batch):
    """Wire size of one exact frame, computed from shape/dtype before encoding."""
    from ..serverless_bridge import codec as bridge_codec
    if not isinstance(image_batch, torch.Tensor) or image_batch.layout != torch.strided or image_batch.is_nested:
        raise transfer_codec.TransferCodecError("exact transfer requires a dense IMAGE tensor")
    if image_batch.dtype not in bridge_codec._DTYPE_NAMES:
        raise transfer_codec.TransferCodecError(f"exact transfer does not support dtype {image_batch.dtype}")
    if image_batch.ndim != 4:
        raise transfer_codec.TransferCodecError(
            f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(image_batch.shape)}")
    batch, height, width, channels = (int(value) for value in image_batch.shape)
    if min(batch, height, width) <= 0 or channels not in (1, 3, 4):
        raise transfer_codec.TransferCodecError("exact IMAGE dimensions must be positive with 1, 3 or 4 channels")
    size = height * width * channels * image_batch.element_size()
    if size > bridge_codec.MAX_PAYLOAD_BYTES:
        raise transfer_codec.FrameTooLargeError("exact frame exceeds the tensor codec byte limit")
    return size


def _encode_exact_frames(frames):
    """Encode [1,H,W,C] CPU slices losslessly (runs in an executor)."""
    from ..serverless_bridge import codec as bridge_codec
    descriptors, parts = [], []
    for frame in frames:
        try:
            descriptor, payload = bridge_codec.encode_payload("IMAGE", frame)
        except ValueError as exc:
            raise transfer_codec.TransferCodecError(f"exact transfer cannot encode this IMAGE: {exc}") from None
        descriptors.append(descriptor)
        parts.append(payload)
    return descriptors, parts


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
                "transfer_format": (
                    list(transfer_codec.TRANSFER_FORMATS),
                    {
                        "default": transfer_codec.DEFAULT_TRANSFER_FORMAT,
                        "tooltip": transfer_codec.TRANSFER_FORMAT_TOOLTIP,
                    },
                ),
                "transfer_quality": (
                    "INT",
                    {
                        "default": transfer_codec.DEFAULT_TRANSFER_QUALITY,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": transfer_codec.TRANSFER_QUALITY_TOOLTIP,
                    },
                ),
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

    @staticmethod
    def _combine_image_tensors(tensors, sources):
        """Validate BHWC batches, retaining alpha and deterministic input order."""
        nonempty = []
        reference_shape = None
        reference_source = None
        for tensor, source in zip(tensors, sources):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Collector expected an IMAGE tensor from {source}")
            shape = tuple(tensor.shape)
            if tensor.ndim != 4 or shape[-1] not in (1, 3, 4) or min(shape[1:3]) <= 0:
                raise ValueError(
                    f"Collector received invalid IMAGE shape {shape} from {source}; "
                    "expected [B, H, W, C] with 1, 3, or 4 channels"
                )
            # Empty master placeholders must not determine the worker image grid.
            if shape[0] == 0:
                continue
            if reference_shape is None:
                reference_shape, reference_source = shape, source
            elif shape[1:3] != reference_shape[1:3]:
                raise ValueError(
                    "Collector cannot combine images with different spatial dimensions: "
                    f"{reference_source} has shape {reference_shape}; {source} has shape {shape}"
                )
            nonempty.append(tensor)
        if not nonempty:
            raise ValueError("Collector cannot return an empty image batch")

        channels = max(tensor.shape[-1] for tensor in nonempty)
        normalized = []
        for tensor in nonempty:
            if tensor.shape[-1] == 1 and channels > 1:
                tensor = tensor.expand(-1, -1, -1, 3)
            if tensor.shape[-1] == 3 and channels == 4:
                # RGB is opaque; never discard alpha from another participant.
                tensor = torch.cat((tensor, torch.ones_like(tensor[..., :1])), dim=-1)
            normalized.append(ensure_contiguous(tensor))
        return ensure_contiguous(torch.cat(normalized, dim=0))

    def _normalize_images_input(self, images):
        """Collapse ComfyUI list IMAGE inputs into a normal batched IMAGE tensor."""
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("Collector received an empty image list")
            if not all(isinstance(image, torch.Tensor) for image in images):
                raise TypeError("Collector expected IMAGE list items to be torch.Tensor instances")
            if len(images) == 1:
                return ensure_contiguous(images[0])
            return self._combine_image_tensors(
                images, [f"IMAGE list item {index}" for index in range(len(images))]
            )
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

    def run(self, images=None, load_balance=False, audio=None, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id="", pass_through=False, delegate_only=False, transfer_format=transfer_codec.DEFAULT_TRANSFER_FORMAT, transfer_quality=transfer_codec.DEFAULT_TRANSFER_QUALITY):
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
        transfer_format = self._unwrap_list_input(transfer_format)
        transfer_quality = self._unwrap_list_input(transfer_quality)

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
                transfer_format,
                transfer_quality,
            )
        )
        return result

    @staticmethod
    def _get_serverless_manager():
        """Return the serverless manager only for collector jobs with serverless slots."""
        manager = getattr(prompt_server, "distributed_serverless", None)
        if manager is not None:
            return manager
        from ..api.orchestration.serverless_runtime import get_manager
        return get_manager()

    async def send_batch_to_master(
        self,
        image_batch,
        audio,
        multi_job_id,
        master_url,
        worker_id,
        transfer_format=transfer_codec.DEFAULT_TRANSFER_FORMAT,
        transfer_quality=transfer_codec.DEFAULT_TRANSFER_QUALITY,
    ):
        """Send an image batch, optionally with audio, or an audio-only completion."""
        batch_size = 0 if image_batch is None else image_batch.shape[0]
        fmt = self._resolve_transfer_format(transfer_format) if batch_size else transfer_codec.LEGACY_FORMAT
        if fmt == transfer_codec.EXACT_FORMAT:
            # Never downgrade exact to 8-bit: an older master fails the job clearly.
            await self._send_exact_batch(image_batch, audio, multi_job_id, master_url, worker_id)
            return
        if fmt != transfer_codec.LEGACY_FORMAT:
            sent = await self._send_compressed_batch(
                image_batch, audio, multi_job_id, master_url, worker_id, fmt, transfer_quality
            )
            if sent:
                return
            log(
                "Worker - Master did not accept compressed transfer "
                "(older ComfyUI-Distributed on the master?). Falling back to legacy_png."
            )
        await self._send_legacy_batch(image_batch, audio, multi_job_id, master_url, worker_id)

    @staticmethod
    def _resolve_transfer_format(transfer_format):
        try:
            fmt = transfer_codec.normalize_format(transfer_format)
        except transfer_codec.TransferCodecError as exc:
            log(f"Worker - {exc}; using legacy_png.")
            return transfer_codec.LEGACY_FORMAT
        if fmt not in transfer_codec.VIDEO_FORMATS or transfer_codec.video_codec_available(fmt):
            return fmt
        fallback = "h264" if fmt != "ffv1" and transfer_codec.video_codec_available("h264") else "png"
        log(f"Worker - Encoder for '{fmt}' is not available in this PyAV build; using '{fallback}'.")
        return fallback

    @staticmethod
    def _build_multipart(envelope, parts):
        form = aiohttp.FormData()
        form.add_field("envelope", json.dumps(envelope), content_type="application/json")
        for i, data in enumerate(parts):
            form.add_field("media", data, filename=f"{i}.bin", content_type="application/octet-stream")
        return form

    # Room left in each request for the multipart envelope (ids, frame
    # metadata, part headers). Audio is budgeted separately.
    TRANSFER_ENVELOPE_RESERVE_BYTES = 256 * 1024
    # Body of a 400 from a master without compressed-transfer support: it
    # tries request.json() on the multipart body and fails.
    _OLD_MASTER_REJECTION = "Invalid JSON payload"

    async def _send_compressed_batch(self, image_batch, audio, multi_job_id, master_url, worker_id, fmt, quality):
        """Send the batch as compressed multipart chunks.

        All chunks are encoded before anything is sent, so a frame that cannot
        fit in one request fails the transfer without delivering a partial
        batch. Every request stays within TRANSFER_MAX_REQUEST_BYTES.

        Returns False (without having delivered anything) when the master
        rejects the first chunk because it predates compressed transfer, so
        the caller can retry with legacy_png. Any other rejection raises.
        """
        encoded_audio = encode_audio_payload(audio)
        session = await get_client_session()
        url = f"{master_url}/distributed/job_complete"
        loop = asyncio.get_running_loop()

        frames = await loop.run_in_executor(None, transfer_codec.tensor_batch_to_uint8, image_batch)
        raw_bytes = int(frames.nbytes)
        total = int(frames.shape[0])
        media_limit = max(1, TRANSFER_MAX_REQUEST_BYTES - self.TRANSFER_ENVELOPE_RESERVE_BYTES)
        started = time.time()
        # Encoding is CPU-bound (seconds for video); keep the server loop responsive.
        chunks = await loop.run_in_executor(
            None, _encode_all_chunks, frames, fmt, quality, min(TRANSFER_MAX_CHUNK_BYTES, media_limit), media_limit,
        )
        del frames

        # Attach audio to the final chunk only if the request stays in bounds;
        # otherwise finish with a separate audio-only completion.
        audio_bytes = len(json.dumps(encoded_audio)) if encoded_audio is not None else 0
        last_media = sum(len(p) for p in chunks[-1]["parts"])
        audio_inline = encoded_audio is not None and last_media + audio_bytes <= media_limit
        audio_separate = encoded_audio is not None and not audio_inline

        sent_bytes = 0
        for index, chunk in enumerate(chunks):
            parts = chunk.pop("parts")
            is_final_chunk = index == len(chunks) - 1
            envelope = {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "is_last": bool(is_final_chunk and not audio_separate),
                **chunk,
            }
            if is_final_chunk and audio_inline:
                envelope["audio"] = encoded_audio
            chunk_bytes = sum(len(p) for p in parts)
            timeout_seconds = min(600, 60 + chunk_bytes // (256 * 1024))
            try:
                async with session.post(
                    url,
                    data=self._build_multipart(envelope, parts),
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as response:
                    if response.status >= 400:
                        body = await response.text()
                        if index == 0 and response.status == 400 and self._OLD_MASTER_REJECTION in body:
                            debug_log(f"Worker - Master rejected compressed chunk: {body[:300]}")
                            return False
                        raise RuntimeError(
                            f"Master rejected {fmt} chunk {index + 1}/{len(chunks)} "
                            f"({chunk_bytes} bytes) with HTTP {response.status}: {body[:300]}. "
                            "If this is a size limit, make COMFYUI_DISTRIBUTED_TRANSFER_MAX_REQUEST_BYTES "
                            "on the worker smaller than the master's --max-upload-size."
                        )
            except Exception as e:
                log(f"Worker - Failed to send {fmt} chunk to master: {e}")
                debug_log(f"Worker - Full error details: URL={url}")
                raise
            sent_bytes += chunk_bytes

        requests = len(chunks)
        if audio_separate:
            await self._post_json_completion(session, url, {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "batch_idx": 0,
                "audio": encoded_audio,
                "is_last": True,
            })
            requests += 1

        log(
            f"Worker - Sent {total} image(s) as {fmt} in {requests} request(s): "
            f"{sent_bytes / 1e6:.2f} MB (raw {raw_bytes / 1e6:.2f} MB, "
            f"{transfer_codec.describe_ratio(sent_bytes, raw_bytes)} smaller) in {time.time() - started:.1f}s"
        )
        return True

    async def _send_exact_batch(self, image_batch, audio, multi_job_id, master_url, worker_id):
        """Send original floating-point frames as binary multipart chunks.

        Frame sizes are known from shape and dtype, so every request is planned
        (and an oversized frame rejected) before anything is sent. Frames are
        encoded one chunk at a time to avoid holding a second copy of the batch.
        Any master rejection raises: there is no lossy or 8-bit fallback.
        """
        encoded_audio = encode_audio_payload(audio)
        session = await get_client_session()
        url = f"{master_url}/distributed/job_complete"
        loop = asyncio.get_running_loop()
        media_limit = max(1, TRANSFER_MAX_REQUEST_BYTES - self.TRANSFER_ENVELOPE_RESERVE_BYTES)
        chunk_limit = min(TRANSFER_MAX_CHUNK_BYTES, media_limit)

        frame_bytes = _exact_frame_bytes(image_batch)
        total = int(image_batch.shape[0])
        if frame_bytes > media_limit:
            raise transfer_codec.FrameTooLargeError(
                f"Frame 0 is {frame_bytes} bytes as exact, above the {media_limit}-byte request limit. "
                "Use png or a lossy format, or raise the master's --max-upload-size together with "
                "COMFYUI_DISTRIBUTED_TRANSFER_MAX_REQUEST_BYTES."
            )
        # Bound descriptor/part-header overhead even for tiny frames.
        per_chunk = min(128, max(1, chunk_limit // frame_bytes))
        ranges = [(start, min(per_chunk, total - start)) for start in range(0, total, per_chunk)]
        audio_bytes = len(json.dumps(encoded_audio)) if encoded_audio is not None else 0
        last_media = ranges[-1][1] * frame_bytes
        audio_inline = encoded_audio is not None and last_media + audio_bytes <= media_limit
        audio_separate = encoded_audio is not None and not audio_inline
        height, width = int(image_batch.shape[1]), int(image_batch.shape[2])
        if audio_separate and audio_bytes > media_limit:
            raise transfer_codec.FrameTooLargeError("exact completion audio exceeds the request limit")
        cpu_batch = ensure_contiguous(image_batch.detach().cpu())

        started = time.time()
        sent_bytes = 0
        for index, (start, count) in enumerate(ranges):
            frames = [cpu_batch[i:i + 1] for i in range(start, start + count)]
            descriptors, parts = await loop.run_in_executor(None, _encode_exact_frames, frames)
            is_final_chunk = index == len(ranges) - 1
            envelope = {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "format": transfer_codec.EXACT_FORMAT,
                "start_index": start,
                "count": count,
                "width": width,
                "height": height,
                "descriptors": descriptors,
                "is_last": bool(is_final_chunk and not audio_separate),
            }
            if is_final_chunk and audio_inline:
                envelope["audio"] = encoded_audio
            chunk_bytes = sum(len(p) for p in parts)
            body = self._build_multipart(envelope, parts)()
            if body.size is None or body.size > TRANSFER_MAX_REQUEST_BYTES:
                raise transfer_codec.FrameTooLargeError("exact multipart body exceeds the request limit")
            timeout_seconds = min(600, 60 + chunk_bytes // (256 * 1024))
            try:
                async with session.post(
                    url,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"Master rejected exact chunk {index + 1}/{len(ranges)} ({chunk_bytes} bytes) "
                            f"with HTTP {response.status}: {body[:300]}. exact needs a master with exact "
                            "transfer support and requests within its --max-upload-size; it is never "
                            "resent as an 8-bit format. Update the master or choose png."
                        )
            except Exception as e:
                log(f"Worker - Failed to send exact chunk to master: {e}")
                debug_log(f"Worker - Full error details: URL={url}")
                raise
            sent_bytes += chunk_bytes

        requests = len(ranges)
        if audio_separate:
            await self._post_json_completion(session, url, {
                "job_id": str(multi_job_id),
                "worker_id": str(worker_id),
                "batch_idx": 0,
                "audio": encoded_audio,
                "is_last": True,
            })
            requests += 1
        log(
            f"Worker - Sent {total} image(s) as exact in {requests} request(s): "
            f"{sent_bytes / 1e6:.2f} MB in {time.time() - started:.1f}s"
        )

    @staticmethod
    async def _post_json_completion(session, url, payload):
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

    async def _send_legacy_batch(self, image_batch, audio, multi_job_id, master_url, worker_id):
        """Original transfer: one JSON request per frame with a base64 PNG."""
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
            await self._post_json_completion(session, url, payload)

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

    # Collector sockets a comfy-media-v1 serverless pool may return.
    TYPED_MEDIA_SOCKETS = ("images", "audio")

    def _store_typed_media(self, worker_images: dict, worker_audio: dict, worker_id: str, media) -> int:
        """Store one atomic typed serverless delivery: ordered IMAGE/AUDIO item lists.

        Each IMAGE item is a BHWC batch kept at its list position; AUDIO items are
        joined in order, exactly as list inputs are on the master.
        """
        if (not isinstance(media, dict) or not media
                or any(socket not in self.TYPED_MEDIA_SOCKETS for socket in media)):
            raise RuntimeError(f"Invalid serverless media delivery from {worker_id}")
        for socket, items in media.items():
            if not isinstance(items, list) or not items or any(item is None for item in items):
                raise RuntimeError(f"Serverless participant {worker_id} returned no {socket} items")
        stored = 0
        for index, tensor in enumerate(media.get("images", [])):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise RuntimeError(f"Serverless participant {worker_id} returned an invalid IMAGE item")
            worker_images.setdefault(worker_id, {})[index] = tensor
            stored += 1
        if "audio" in media:
            worker_audio[worker_id] = self._normalize_audio_input(media["audio"])
        return stored

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
        sources = []
        if not delegate_mode and images_on_cpu is not None:
            for i in range(master_batch_size):
                ordered_tensors.append(images_on_cpu[i:i+1])
                sources.append(f"master image {i}")

        ordered_worker_ids = [str(worker_id) for worker_id in (worker_order or [])]
        seen = set()
        for worker_id_str in ordered_worker_ids:
            seen.add(worker_id_str)
            if worker_id_str not in worker_images:
                continue
            for idx in sorted(worker_images[worker_id_str].keys()):
                ordered_tensors.append(worker_images[worker_id_str][idx])
                sources.append(f"worker {worker_id_str!r} image {idx}")

        # Append any unexpected worker ids deterministically.
        for worker_id_str in sorted(worker_images.keys()):
            if worker_id_str in seen:
                continue
            for idx in sorted(worker_images[worker_id_str].keys()):
                ordered_tensors.append(worker_images[worker_id_str][idx])
                sources.append(f"worker {worker_id_str!r} image {idx}")

        cpu_tensors = []
        for t in ordered_tensors:
            if t.is_cuda:
                t = t.cpu()
            t = ensure_contiguous(t)
            cpu_tensors.append(t)

        if cpu_tensors:
            return self._combine_image_tensors(cpu_tensors, sources)
        if fallback_images is not None:
            return self._combine_image_tensors([fallback_images], ["master fallback"])
        return None

    async def execute(self, images, audio, load_balance=False, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id="", delegate_only=False, transfer_format=transfer_codec.DEFAULT_TRANSFER_FORMAT, transfer_quality=transfer_codec.DEFAULT_TRANSFER_QUALITY):
        if is_worker:
            # Worker mode: send images and audio to master in a single batch
            image_count = 0 if images is None else images.shape[0]
            debug_log(f"Worker - Job {multi_job_id} complete. Sending {image_count} image(s) to master")
            await self.send_batch_to_master(
                images, audio, multi_job_id, master_url, worker_id, transfer_format, transfer_quality
            )
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
            serverless_workers = {
                worker_id for worker_id in expected_workers if worker_id.startswith("serverless:")
            }
            serverless_manager = None

            def get_serverless_manager():
                nonlocal serverless_manager
                if serverless_workers and serverless_manager is None:
                    serverless_manager = self._get_serverless_manager()
                return serverless_manager

            try:
                exact_requested = transfer_codec.normalize_format(transfer_format) == transfer_codec.EXACT_FORMAT
            except transfer_codec.TransferCodecError:
                exact_requested = False
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

            def consume_result(result):
                nonlocal collected_count
                if "serverless_error" in result:
                    error = result.get("serverless_error")
                    message = str(error).strip() if error is not None else "unspecified serverless failure"
                    participant = result.get("worker_id", "unknown")
                    raise RuntimeError(f"Serverless participant {participant} failed: {message}")

                result_worker_id = str(result["worker_id"])
                if "media" in result:
                    if result_worker_id not in serverless_workers:
                        raise RuntimeError(f"Unexpected serverless media participant {result_worker_id}")
                    if result_worker_id in workers_done:
                        raise RuntimeError(f"Duplicate serverless media delivery from {result_worker_id}")
                    if result.get("is_last") is not True:
                        raise RuntimeError("Serverless media delivery must be atomic and final")
                    collected_count += self._store_typed_media(
                        worker_images, worker_audio, result_worker_id, result["media"]
                    )
                    debug_log(f"Master - Got typed media from serverless participant {result_worker_id}")
                    mark_worker_done(result_worker_id)
                    return

                if (exact_requested and result.get("tensor") is not None
                        and result.get("format") != transfer_codec.EXACT_FORMAT):
                    raise RuntimeError(
                        f"Worker {result_worker_id} sent {result.get('format')} images, but this Collector "
                        "requested exact. Update ComfyUI-Distributed on that worker; exact is never "
                        "replaced by an 8-bit format."
                    )
                is_last = result.get("is_last", False)
                collected_count += self._store_worker_result(worker_images, result)
                debug_log(
                    f"Master - Got canonical result from worker {result_worker_id}, "
                    f"image {result.get('image_index', 0)}, is_last={is_last}"
                )
                result_audio = result.get("audio")
                if result_audio is not None:
                    worker_audio[result_worker_id] = result_audio
                    debug_log(f"Master - Got audio from worker {result_worker_id}")
                if is_last:
                    mark_worker_done(result_worker_id)

            try:
                while len(workers_done) < num_workers:
                    # Check for user interruption to abort collection promptly.
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    try:
                        async with prompt_server.distributed_jobs_lock:
                            q = prompt_server.distributed_pending_jobs[multi_job_id]
                        result = await asyncio.wait_for(q.get(), timeout=slice_timeout)
                        consume_result(result)

                        # Real results refresh the regular-worker timeout baseline.
                        last_activity = time.time()
                        base_timeout = float(get_worker_timeout_seconds())

                    except asyncio.TimeoutError:
                        if (time.time() - last_activity) < base_timeout:
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            continue
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        missing_workers = expected_workers - workers_done

                        # A queue delivery can race the timed wait. Process it before
                        # declaring an owned provider slot stopped without a result.
                        async with prompt_server.distributed_jobs_lock:
                            current_q = prompt_server.distributed_pending_jobs.get(multi_job_id)
                            queue_has_results = current_q is not None and not current_q.empty()
                        if queue_has_results:
                            continue

                        missing_serverless = missing_workers & serverless_workers
                        if missing_serverless:
                            manager = get_serverless_manager()
                            if manager is None:
                                raise RuntimeError("Serverless manager is unavailable for pending collector slots")
                            pending_serverless = []
                            for participant_id in sorted(missing_serverless):
                                if manager.is_pending(multi_job_id, participant_id):
                                    pending_serverless.append(participant_id)
                                elif manager.owns(multi_job_id, participant_id):
                                    raise RuntimeError(
                                        f"Serverless participant {participant_id} stopped without delivering a result"
                                    )
                            # Provider deadlines are managed by the serverless runtime;
                            # do not apply the regular worker heartbeat timeout to it.
                            if pending_serverless:
                                continue

                        missing_regular = missing_workers - serverless_workers
                        elapsed = time.time() - last_activity
                        for missing_worker_id in sorted(missing_regular):
                            log(
                                "Master - Heartbeat timeout: "
                                f"worker={missing_worker_id}, elapsed={elapsed:.1f}s"
                            )
                        if missing_regular:
                            log(
                                f"Master - Heartbeat timeout. Still waiting for workers: {list(sorted(missing_regular))} "
                                f"(elapsed={elapsed:.1f}s)"
                            )

                        # Probe only regular worker /prompt endpoints.
                        any_busy = False
                        if missing_regular:
                            try:
                                cfg = load_config()
                                cfg_workers = cfg.get('workers', [])
                                for wid in sorted(missing_regular):
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
                            last_activity = time.time()
                            base_timeout = float(get_worker_timeout_seconds())
                            continue

                        async with prompt_server.distributed_jobs_lock:
                            final_q = prompt_server.distributed_pending_jobs.get(multi_job_id)
                            remaining_items = []
                            if final_q is not None:
                                while not final_q.empty():
                                    try:
                                        remaining_items.append(final_q.get_nowait())
                                    except asyncio.QueueEmpty:
                                        break
                        for item in remaining_items:
                            consume_result(item)
                        break

                if exact_requested and workers_done != expected_workers:
                    raise RuntimeError(
                        f"Exact transfer incomplete: missing completion from {sorted(expected_workers - workers_done)}"
                    )
                total_collected = sum(len(imgs) for imgs in worker_images.values())

                # Remove the queue before combining so a late provider callback cannot
                # resurrect results for a collector that has completed.
                async with prompt_server.distributed_jobs_lock:
                    prompt_server.distributed_pending_jobs.pop(multi_job_id, None)

                combined_audio = self._combine_audio(master_audio, worker_audio, self.EMPTY_AUDIO, enabled_workers)
                serverless_results = set(worker_images) & serverless_workers
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
                    if serverless_results or exact_requested:
                        raise RuntimeError(
                            "Serverless or exact result batch shape mismatch while combining participants "
                            f"{sorted(worker_images)}: {e}"
                        ) from e
                    log(f"Master - Error combining images: {e}")
                    # Do not silently discard worker images or pass an empty
                    # fallback batch downstream to SaveImage.
                    raise
            except BaseException:
                # Remove the queue first, preventing cancelled work from delivering into
                # a collector that is already unwinding.
                async with prompt_server.distributed_jobs_lock:
                    prompt_server.distributed_pending_jobs.pop(multi_job_id, None)
                if serverless_workers:
                    try:
                        manager = get_serverless_manager()
                        if manager is not None:
                            await manager.cancel(collector_job_id=multi_job_id)
                    except Exception as cancel_error:
                        debug_log(f"Collector - Serverless cancellation failed: {cancel_error}")
                raise
