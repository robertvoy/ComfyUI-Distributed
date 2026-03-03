import asyncio, io, json, time
from typing import Any
import aiohttp
from PIL import Image
from ..utils.logging import debug_log, log
from ..utils.network import get_client_session
from ..utils.constants import TILE_SEND_TIMEOUT
from ..utils.usdu_managment import MAX_PAYLOAD_SIZE, send_heartbeat_to_master
from ..utils.image import tensor_to_pil


class WorkerCommsMixin:
    async def _send_heartbeat_to_master(
        self,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
    ) -> None:
        """Proxy heartbeat helper used by worker processing mixins."""
        await send_heartbeat_to_master(multi_job_id, master_url, worker_id)

    async def send_tiles_batch_to_master(
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
            data = aiohttp.FormData()
            data.add_field('multi_job_id', multi_job_id)
            data.add_field('worker_id', str(worker_id))
            data.add_field('padding', str(padding))

            metadata = []
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
                data.add_field(f'tile_{j - i}', io.BytesIO(img_bytes), filename=f'tile_{j}.png', content_type='image/png')
                used += len(img_bytes) + overhead
                j += 1

            # Ensure at least one tile per chunk
            if j == i:
                # Single oversized tile, send anyway
                meta = encoded[j]['meta']
                metadata.append(meta)
                data.add_field('tile_0', io.BytesIO(encoded[j]['bytes']), filename=f'tile_{j}.png', content_type='image/png')
                j += 1

            chunk_size = j - i
            is_chunk_last = (j >= total_tiles)
            data.add_field('is_last', str(bool(is_final_flush and is_chunk_last)))
            data.add_field('batch_size', str(chunk_size))
            data.add_field('tiles_metadata', json.dumps(metadata), content_type='application/json')

            # Retry logic with exponential backoff
            max_retries = 5
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    session = await get_client_session()
                    url = f"{master_url}/distributed/submit_tiles"
                    async with session.post(url, data=data) as response:
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
        async with session.post(url, data=data) as response:
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
                }) as response:
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

    async def _request_image_from_master(self, multi_job_id, master_url, worker_id):
        """Request an image index to process from master in dynamic mode."""
        data = await self._request_work_item_from_master(multi_job_id, master_url, worker_id)
        if not data:
            return None, 0
        image_idx = data.get('image_idx')
        estimated_remaining = data.get('estimated_remaining', 0)
        return image_idx, estimated_remaining

    async def _request_tile_from_master(self, multi_job_id, master_url, worker_id):
        """Request a tile index to process from master in static mode (reusing dynamic infrastructure)."""
        data = await self._request_work_item_from_master(multi_job_id, master_url, worker_id)
        if not data:
            return None, 0, False
        tile_idx = data.get('tile_idx')
        estimated_remaining = data.get('estimated_remaining', 0)
        batched_static = data.get('batched_static', False)
        return tile_idx, estimated_remaining, batched_static

    async def _send_full_image_to_master(self, image_pil, image_idx, multi_job_id, 
                                        master_url, worker_id, is_last):
        """Send a processed full image back to master in dynamic mode."""
        # Serialize image to PNG
        byte_io = io.BytesIO()
        image_pil.save(byte_io, format='PNG', compress_level=0)
        byte_io.seek(0)
        
        # Prepare form data
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('image_idx', str(image_idx))
        data.add_field('is_last', str(is_last))
        data.add_field('full_image', byte_io, filename=f'image_{image_idx}.png', 
                      content_type='image/png')
        
        # Retry logic
        max_retries = 5
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/submit_image"
                
                async with session.post(url, data=data) as response:
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

    async def _send_worker_complete_signal(self, multi_job_id, master_url, worker_id):
        """Send completion signal to master in dynamic mode."""
        # Send a dummy request with is_last=True
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('is_last', 'true')
        # No image data - just completion signal
        
        session = await get_client_session()
        url = f"{master_url}/distributed/submit_image"
        
        async with session.post(url, data=data) as response:
            response.raise_for_status()
            debug_log(f"Worker {worker_id} sent completion signal")

    async def _check_job_status(self, multi_job_id, master_url):
        """Check if job is ready on the master."""
        try:
            session = await get_client_session()
            url = f"{master_url}/distributed/job_status?multi_job_id={multi_job_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('ready', False)
                return False
        except Exception as e:
            debug_log(f"Job status check failed: {e}")
            return False

    async def _async_yield(self):
        """Simple async yield to allow event loop processing."""
        await asyncio.sleep(0)
