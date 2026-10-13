import asyncio
import json
import uuid

import aiohttp

from ...utils.logging import debug_log, log
from ...utils.network import build_worker_url, get_client_session, probe_worker
# Must match image_list.runtime.PROTOCOL_VERSION (kept local so this module
# stays importable without the image-list runtime; a test pins them equal).
IMAGE_LIST_PROTOCOL_VERSION = 2
try:
    from ...utils.trace_logger import trace_debug, trace_info
except ImportError:
    def trace_debug(*_args, **_kwargs):
        return None

    def trace_info(*_args, **_kwargs):
        return None

try:
    from ..schemas import parse_positive_int
except ImportError:
    def parse_positive_int(value, default):
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

_least_busy_rr_index = 0


async def worker_is_active(worker):
    """Ping worker's /prompt endpoint to confirm it's reachable."""
    url = build_worker_url(worker)
    return await probe_worker(url, timeout=3.0) is not None


async def worker_ws_is_active(worker):
    """Ping worker's websocket endpoint to confirm it's reachable."""
    session = await get_client_session()
    url = build_worker_url(worker, "/distributed/worker_ws")
    try:
        ws = await session.ws_connect(url, heartbeat=20, timeout=3)
        await ws.close()
        return True
    except asyncio.TimeoutError:
        debug_log(f"[Distributed] Worker WS probe timed out: {url}")
        return False
    except aiohttp.ClientConnectorError:
        debug_log(f"[Distributed] Worker WS unreachable: {url}")
        return False
    except Exception as e:
        debug_log(f"[Distributed] Worker WS probe unexpected error: {e}")
        return False


async def _probe_worker_active(worker, use_websocket, semaphore):
    async with semaphore:
        if worker.get("type") == "serverless":
            from .serverless_runtime import get_manager
            # Authentication/configuration failures must surface, not silently skip paid work.
            status = await get_manager().health(worker)
            return worker, status["ok"]
        is_active = await (worker_ws_is_active(worker) if use_websocket else worker_is_active(worker))
        return worker, is_active


async def _dispatch_via_websocket(worker_url, payload, client_id, timeout=60.0):
    """Open a fresh worker websocket, dispatch one prompt, wait for ack, then close."""
    request_id = uuid.uuid4().hex
    ws_payload = {
        "type": "dispatch_prompt",
        "request_id": request_id,
        "prompt": payload.get("prompt"),
        "workflow": payload.get("workflow"),
        "client_id": client_id,
    }
    ws_url = worker_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/distributed/worker_ws"
    session = await get_client_session()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    ws = None

    try:
        async with asyncio.timeout_at(deadline):
            ws = await session.ws_connect(ws_url, heartbeat=20)
            await ws.send_json(ws_payload)
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data or "{}")
                    if data.get("type") == "dispatch_ack" and data.get("request_id") == request_id:
                        if data.get("ok"):
                            return data.get("prompt_id")
                        error_text = data.get("error") or "Worker rejected websocket dispatch."
                        validation_error = data.get("validation_error")
                        node_errors = data.get("node_errors")
                        if validation_error:
                            error_text = f"{error_text} | validation_error={validation_error}"
                        if node_errors:
                            error_text = f"{error_text} | node_errors={node_errors}"
                        raise RuntimeError(error_text)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    raise RuntimeError(f"Worker websocket closed before acknowledgement: {msg.type}; submission status is ambiguous")
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Worker websocket dispatch timed out before dispatch_ack; submission status is ambiguous."
        ) from exc
    finally:
        if ws is not None:
            # Closing must not discard an already acknowledged prompt ID or
            # replace a definite rejection with an ambiguous timeout.
            try:
                async with asyncio.timeout_at(min(deadline, loop.time() + 1.0)):
                    await ws.close()
            except (asyncio.TimeoutError, aiohttp.ClientError):
                pass


async def dispatch_worker_prompt(
    worker,
    prompt_obj,
    workflow_meta,
    client_id=None,
    use_websocket=False,
    trace_execution_id=None,
    return_prompt_id=False,
):
    """Send the prepared prompt to a worker ComfyUI instance."""
    worker_url = build_worker_url(worker)
    url = build_worker_url(worker, "/prompt")
    payload = {"prompt": prompt_obj}
    extra_data = {}
    if workflow_meta:
        extra_data.setdefault("extra_pnginfo", {})["workflow"] = workflow_meta
    if extra_data:
        payload["extra_data"] = extra_data

    if use_websocket:
        try:
            prompt_id = await _dispatch_via_websocket(
                worker_url,
                {
                    "prompt": prompt_obj,
                    "workflow": workflow_meta,
                },
                client_id,
            )
            if return_prompt_id and (not isinstance(prompt_id, str) or not prompt_id):
                raise RuntimeError("Worker websocket acknowledgement omitted prompt_id")
            return prompt_id if return_prompt_id else None
        except Exception as exc:
            worker_id = worker.get("id")
            if trace_execution_id:
                trace_info(trace_execution_id, f"Websocket dispatch failed for worker {worker_id}: {exc}")
            else:
                log(f"[Distributed] Websocket dispatch failed for worker {worker_id}: {exc}")
            raise

    session = await get_client_session()
    async with session.post(
        url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        resp.raise_for_status()
        if not return_prompt_id:
            return None
        try:
            response_data = await resp.json()
        except Exception as exc:
            raise RuntimeError("Worker /prompt response did not contain JSON acknowledgement") from exc
        prompt_id = response_data.get("prompt_id") if isinstance(response_data, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError("Worker /prompt response omitted prompt_id")
        return prompt_id if return_prompt_id else None


async def preflight_image_list_workers(workers, require_idle=True, timeout=3.0):
    """Require protocol/node support and report which direct workers are idle.

    This performs read-only GETs only; it never submits a prompt or contacts a
    serverless provider. Unsupported worker releases fail before master inputs run.
    """
    if not workers:
        raise RuntimeError("Distributed Image List requires at least one selected direct worker")
    session = await get_client_session()
    required_nodes = {"DistributedImageListStart", "DistributedImageListEnd"}
    semaphore = asyncio.Semaphore(8)

    async def inspect(worker):
        worker_id = str(worker.get("id") or "unknown")
        check = "capability"
        try:
            async with semaphore:
                cap_url = build_worker_url(worker, "/distributed/image-list/capabilities")
                async with session.get(cap_url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Image-list worker {worker_id} does not expose the capability endpoint "
                            f"(HTTP {response.status}); upgrade that worker before running this feature"
                        )
                    capability = await response.json()
                if not isinstance(capability, dict):
                    raise RuntimeError(f"Image-list worker {worker_id} returned malformed capability data")
                supported = capability.get("supported")
                protocol_version = capability.get("protocol_version")
                nodes = capability.get("node_ids")
                if (
                    not isinstance(supported, bool)
                    or type(protocol_version) is not int
                    or not isinstance(nodes, list)
                    or any(not isinstance(node_id, str) for node_id in nodes)
                ):
                    raise RuntimeError(f"Image-list worker {worker_id} returned malformed capability data")
                if (
                    supported is not True
                    or protocol_version != IMAGE_LIST_PROTOCOL_VERSION
                    or not required_nodes.issubset(nodes)
                ):
                    raise RuntimeError(
                        f"Image-list worker {worker_id} is incompatible: requires protocol "
                        f"v{IMAGE_LIST_PROTOCOL_VERSION} and nodes "
                        f"{sorted(required_nodes)}; no work was submitted"
                    )

                check = "queue"
                queue_url = build_worker_url(worker, "/queue")
                async with session.get(queue_url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Image-list worker {worker_id} queue preflight returned HTTP {response.status}")
                    queue = await response.json()
                if (
                    not isinstance(queue, dict)
                    or not isinstance(queue.get("queue_running"), list)
                    or not isinstance(queue.get("queue_pending"), list)
                ):
                    raise RuntimeError(f"Image-list worker {worker_id} returned malformed /queue state")
                idle = not queue["queue_running"] and not queue["queue_pending"]
                return worker_id, idle
        except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as exc:
            log(
                f"[Distributed] Image-list worker {worker_id} unavailable during {check} check; "
                f"skipping: {exc}"
            )
            return worker_id, None

    statuses = await asyncio.gather(*(inspect(worker) for worker in workers))
    idle_workers = {
        worker_id: True
        for worker_id, idle in statuses
        if idle is True
    }
    if require_idle and not idle_workers:
        raise RuntimeError("No selected feature-compatible direct worker is idle; unrelated worker prompts were left untouched")
    return idle_workers


async def select_active_workers(
    workers,
    use_websocket,
    delegate_master,
    trace_execution_id=None,
    probe_concurrency=8,
):
    """Probe workers and return (active_workers, updated_delegate_master)."""
    probe_limit = parse_positive_int(probe_concurrency, 8)
    probe_semaphore = asyncio.Semaphore(probe_limit)

    if trace_execution_id and workers:
        trace_debug(
            trace_execution_id,
            f"Probing {len(workers)} workers with probe_concurrency={probe_limit}",
        )

    probe_results = await asyncio.gather(
        *[
            _probe_worker_active(worker, use_websocket, probe_semaphore)
            for worker in workers
        ]
    )

    active_workers = []
    for worker, is_active in probe_results:
        if is_active:
            active_workers.append(worker)
        else:
            if trace_execution_id:
                trace_info(trace_execution_id, f"Worker {worker['name']} ({worker['id']}) is offline, skipping.")
            else:
                log(f"[Distributed] Worker {worker['name']} ({worker['id']}) is offline, skipping.")

    if trace_execution_id and workers:
        trace_debug(
            trace_execution_id,
            f"Worker probe complete: active={len(active_workers)}/{len(workers)}",
        )

    if not active_workers and delegate_master:
        if trace_execution_id:
            trace_debug(trace_execution_id, "All workers offline while delegate-only requested; enabling master participation.")
        else:
            debug_log("All workers offline while delegate-only requested; enabling master participation.")
        delegate_master = False

    return active_workers, delegate_master


def _extract_queue_remaining(payload):
    if not isinstance(payload, dict):
        return 0
    try:
        queue_remaining = int(payload.get("exec_info", {}).get("queue_remaining", 0))
    except (TypeError, ValueError):
        queue_remaining = 0
    return max(queue_remaining, 0)


async def _probe_worker_queue(worker, semaphore, probe_timeout):
    async with semaphore:
        if worker.get("type") == "serverless":
            from .serverless_runtime import get_manager
            status = await get_manager().health(worker)
            return {"worker": worker, "queue_remaining": status["queueRemaining"]}
        worker_url = build_worker_url(worker)
        payload = await probe_worker(worker_url, timeout=probe_timeout)
        if payload is None:
            return None
        return {
            "worker": worker,
            "queue_remaining": _extract_queue_remaining(payload),
        }


def _select_idle_round_robin(statuses):
    global _least_busy_rr_index
    if not statuses:
        return None
    index = _least_busy_rr_index % len(statuses)
    _least_busy_rr_index += 1
    return statuses[index]


async def select_least_busy_worker(
    workers,
    trace_execution_id=None,
    probe_concurrency=8,
    probe_timeout=3.0,
):
    """Select one worker by queue depth, round-robin among idle workers."""
    if not workers:
        return None

    probe_limit = parse_positive_int(probe_concurrency, 8)
    probe_semaphore = asyncio.Semaphore(probe_limit)
    statuses = await asyncio.gather(
        *[
            _probe_worker_queue(worker, probe_semaphore, probe_timeout)
            for worker in workers
        ]
    )
    statuses = [status for status in statuses if status is not None]
    if not statuses:
        if trace_execution_id:
            trace_info(trace_execution_id, "Least-busy selection failed: no worker queue probes succeeded.")
        else:
            log("[Distributed] Least-busy selection failed: no worker queue probes succeeded.")
        return None

    idle_statuses = [status for status in statuses if status["queue_remaining"] == 0]
    if idle_statuses:
        selected = _select_idle_round_robin(idle_statuses)
    else:
        selected = min(statuses, key=lambda status: status["queue_remaining"])

    worker = selected["worker"]
    queue_remaining = selected["queue_remaining"]
    if trace_execution_id:
        trace_debug(
            trace_execution_id,
            f"Least-busy worker selected: {worker.get('name')} ({worker.get('id')}), queue_remaining={queue_remaining}",
        )
    else:
        debug_log(
            f"Least-busy worker selected: {worker.get('name')} ({worker.get('id')}), queue_remaining={queue_remaining}"
        )
    return worker
