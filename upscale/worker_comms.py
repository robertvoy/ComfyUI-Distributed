import asyncio, io, json, time
from dataclasses import dataclass
from typing import Any, Literal
import aiohttp
from PIL import Image
from ..utils.logging import debug_log, log
from ..utils.auth import distributed_auth_headers
from ..utils.config import load_config
from ..utils.network import get_client_session
from ..utils.constants import TILE_SEND_TIMEOUT
from ..utils.usdu_management import MAX_PAYLOAD_SIZE, send_heartbeat_to_master
from ..utils.image import tensor_to_pil


WorkAssignmentKind = Literal["image", "tile", "none"]


@dataclass(frozen=True, slots=True)
class WorkAssignment:
    """Canonical typed representation of a single work-item assignment."""

    kind: WorkAssignmentKind
    task_idx: int | None
    estimated_remaining: int = 0
    batched_static: bool = False


class WorkerCommsMixin:
    @staticmethod
    def _master_auth_headers() -> dict[str, str]:
        return distributed_auth_headers(load_config())

    async def send_heartbeat(
        self,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
    ) -> None:
        """Send worker heartbeat to the master."""
        await send_heartbeat_to_master(multi_job_id, master_url, worker_id)

    async def send_tiles_batch(
        self,
        processed_tiles: list[dict[str, Any]],
        multi_job_id: str,
        master_url: str,
        padding: int,
        worker_id: str,
        is_final_flush: bool = False,
    ) -> None:
        """Send all processed tiles to master, chunked if large."""
        if not processed_tiles:
            if is_final_flush:
                await self._send_tiles_completion_signal(multi_job_id, master_url, worker_id)
            return  # Early exit if empty

        total_tiles = len(processed_tiles)
        debug_log(f"Worker[{worker_id[:8]}] - Preparing to send {total_tiles} tiles (size-aware chunks)")

        # Prepare encoded images and sizes to enable size-aware chunking
        encoded = []
        for idx, tile_data in enumerate(processed_tiles):
            img = tensor_to_pil(tile_data['tile'], 0)
            bio = io.BytesIO()
            # Keep compression low to balance speed and size; adjust if needed
            img.save(bio, format='PNG', compress_level=0)
            raw = bio.getvalue()
            encoded.append({
                'bytes': raw,
                'meta': {
                    'tile_idx': tile_data['tile_idx'],
                    'x': tile_data['x'],
                    'y': tile_data['y'],
                    'extracted_width': tile_data['extracted_width'],
                    'extracted_height': tile_data['extracted_height'],
                    **({'batch_idx': tile_data['batch_idx']} if 'batch_idx' in tile_data else {}),
                    **({'global_idx': tile_data['global_idx']} if 'global_idx' in tile_data else {}),
                }
            })

        # Size-aware chunking
        max_bytes = int(MAX_PAYLOAD_SIZE) - (1024 * 1024)  # 1MB headroom
        i = 0
        chunk_index = 0
        while i < total_tiles:
            metadata = []
            chunk_images: list[tuple[int, int, bytes]] = []
            used = 0
            j = i
            while j < total_tiles:
                img_bytes = encoded[j]['bytes']
                meta = encoded[j]['meta']
                # Rough overhead for fields + JSON
                overhead = 1024
                if used + len(img_bytes) + overhead > max_bytes and j > i:
                    break
                # Accept this tile in this chunk
                metadata.append(meta)
                chunk_images.append((j - i, j, img_bytes))
                used += len(img_bytes) + overhead
                j += 1

            # Ensure at least one tile per chunk
            if j == i:
                # Single oversized tile, send anyway
                meta = encoded[j]['meta']
                metadata.append(meta)
                chunk_images.append((0, j, encoded[j]['bytes']))
                j += 1

            chunk_size = j - i
            is_chunk_last = (j >= total_tiles)

            def _build_chunk_form() -> aiohttp.FormData:
                data = aiohttp.FormData()
                data.add_field('multi_job_id', multi_job_id)
                data.add_field('worker_id', str(worker_id))
                data.add_field('padding', str(padding))
                data.add_field('is_last', str(bool(is_final_flush and is_chunk_last)))
                data.add_field('batch_size', str(chunk_size))
                data.add_field('tiles_metadata', json.dumps(metadata), content_type='application/json')
                for relative_idx, source_idx, img_bytes in chunk_images:
                    data.add_field(
                        f'tile_{relative_idx}',
                        io.BytesIO(img_bytes),
                        filename=f'tile_{source_idx}.png',
                        content_type='image/png',
                    )
                return data

            # Retry logic with exponential backoff
            max_retries = 5
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    session = await get_client_session()
                    url = f"{master_url}/distributed/submit_tiles"
                    async with session.post(
                        url,
                        data=_build_chunk_form(),
                        headers=self._master_auth_headers(),
                    ) as response:
                        response.raise_for_status()
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 5.0)
                    else:
                        log(f"UltimateSDUpscale Worker - Failed to send chunk {chunk_index} after {max_retries} attempts: {e}")
                        raise

            debug_log(f"Worker[{worker_id[:8]}] - Sent chunk {chunk_index} ({chunk_size} tiles, ~{used/1e6:.2f} MB)")
            chunk_index += 1
            i = j

    async def _send_tiles_completion_signal(self, multi_job_id, master_url, worker_id):
        """Send completion signal to master in static mode when no tiles are left."""
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('is_last', 'true')
        data.add_field('batch_size', '0')

        session = await get_client_session()
        url = f"{master_url}/distributed/submit_tiles"
        async with session.post(url, data=data, headers=self._master_auth_headers()) as response:
            response.raise_for_status()
            debug_log(f"Worker {worker_id} sent static completion signal")

    async def _request_work_item_from_master(
        self,
        multi_job_id,
        master_url,
        worker_id,
        endpoint="/distributed/request_image",
    ):
        """Request one work item from master with retry/backoff and total timeout."""
        max_retries = 10
        retry_delay = 0.5
        start_time = time.monotonic()
        url = f"{master_url}{endpoint}"

        for attempt in range(max_retries):
            if time.monotonic() - start_time > 30:
                log(f"Total request timeout after 30s for worker {worker_id}")
                return None

            try:
                session = await get_client_session()
                async with session.post(url, json={
                    'worker_id': str(worker_id),
                    'multi_job_id': multi_job_id
                }, headers=self._master_auth_headers()) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 404:
                        text = await response.text()
                        debug_log(f"Job not found (404), will retry: {text}")
                        await asyncio.sleep(1.0)
                    else:
                        text = await response.text()
                        debug_log(
                            f"Request work item failed ({response.status}) for worker {worker_id}: {text}"
                        )

            except Exception as exc:
                if attempt < max_retries - 1:
                    debug_log(f"Retry {attempt + 1}/{max_retries} after error: {exc}")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 5.0)
                else:
                    log(f"Failed to request work item after {max_retries} attempts: {exc}")
                    raise

        return None

    @staticmethod
    def _parse_work_assignment(data: dict[str, Any] | None) -> WorkAssignment:
        """Normalize assignment payloads into a single discriminated contract."""
        if not data:
            return WorkAssignment(kind="none", task_idx=None)

        kind_value = data.get("kind")
        if kind_value is None:
            # Backward-compatible parsing for older masters.
            if "image_idx" in data:
                kind_value = "image"
                data = {
                    "kind": "image",
                    "task_idx": data.get("image_idx"),
                    "estimated_remaining": data.get("estimated_remaining", 0),
                }
            elif "tile_idx" in data:
                kind_value = "tile"
                data = {
                    "kind": "tile",
                    "task_idx": data.get("tile_idx"),
                    "estimated_remaining": data.get("estimated_remaining", 0),
                    "batched_static": data.get("batched_static", False),
                }
            else:
                kind_value = "none"

        if kind_value not in {"image", "tile", "none"}:
            return WorkAssignment(kind="none", task_idx=None)

        task_idx_raw = data.get("task_idx")
        if task_idx_raw is None:
            task_idx = None
        else:
            try:
                task_idx = int(task_idx_raw)
            except (TypeError, ValueError):
                task_idx = None

        try:
            estimated_remaining = int(data.get("estimated_remaining", 0) or 0)
        except (TypeError, ValueError):
            estimated_remaining = 0

        return WorkAssignment(
            kind=kind_value,
            task_idx=task_idx,
            estimated_remaining=estimated_remaining,
            batched_static=bool(data.get("batched_static", False)),
        )

    async def request_assignment(self, multi_job_id, master_url, worker_id) -> WorkAssignment:
        """Request one assignment and parse into the canonical discriminated contract."""
        data = await self._request_work_item_from_master(multi_job_id, master_url, worker_id)
        return self._parse_work_assignment(data)

    async def send_full_image(self, image_pil, image_idx, multi_job_id, 
                              master_url, worker_id, is_last):
        """Send a processed full image back to master in dynamic mode."""
        # Serialize image to PNG
        byte_io = io.BytesIO()
        image_pil.save(byte_io, format='PNG', compress_level=0)
        image_bytes = byte_io.getvalue()

        def _build_image_form() -> aiohttp.FormData:
            data = aiohttp.FormData()
            data.add_field('multi_job_id', multi_job_id)
            data.add_field('worker_id', str(worker_id))
            data.add_field('image_idx', str(image_idx))
            data.add_field('is_last', str(is_last))
            data.add_field(
                'full_image',
                io.BytesIO(image_bytes),
                filename=f'image_{image_idx}.png',
                content_type='image/png',
            )
            return data
        
        # Retry logic
        max_retries = 5
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/submit_image"
                
                async with session.post(
                    url,
                    data=_build_image_form(),
                    headers=self._master_auth_headers(),
                ) as response:
                    response.raise_for_status()
                    debug_log(f"Successfully sent image {image_idx} to master")
                    return
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    debug_log(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log(f"Failed to send image {image_idx} after {max_retries} attempts: {e}")
                    raise

    async def send_worker_complete_signal(self, multi_job_id, master_url, worker_id):
        """Send completion signal to master in dynamic mode."""
        # Send a dummy request with is_last=True
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('is_last', 'true')
        # No image data - just completion signal
        
        session = await get_client_session()
        url = f"{master_url}/distributed/submit_image"
        
        async with session.post(url, data=data, headers=self._master_auth_headers()) as response:
            response.raise_for_status()
            debug_log(f"Worker {worker_id} sent completion signal")

    async def check_job_status(self, multi_job_id, master_url):
        """Check if job is ready on the master."""
        try:
            session = await get_client_session()
            url = f"{master_url}/distributed/job_status?multi_job_id={multi_job_id}"
            async with session.get(url, headers=self._master_auth_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('ready', False)
                return False
        except Exception as e:
            debug_log(f"Job status check failed: {e}")
            return False

    async def async_yield(self):
        """Simple async yield to allow event loop processing."""
        await asyncio.sleep(0)
