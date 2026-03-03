import asyncio
import time
from collections.abc import Callable
from typing import Any

from ..utils.config import get_worker_timeout_seconds
from ..utils.constants import HEARTBEAT_INTERVAL
from ..utils.logging import log


async def collect_worker_queue_results(
    *,
    prompt_server: Any,
    multi_job_id: str,
    expected_workers: set[str],
    on_result: Callable[[dict[str, Any]], None],
    timeout_log_prefix: str,
    throw_if_interrupted: Callable[[], None],
) -> set[str]:
    workers_done: set[str] = set()
    base_timeout = float(get_worker_timeout_seconds())
    slice_timeout = min(max(0.1, HEARTBEAT_INTERVAL / 20.0), base_timeout)
    last_activity = time.time()

    while len(workers_done) < len(expected_workers):
        throw_if_interrupted()
        try:
            async with prompt_server.distributed_jobs_lock:
                queue = prompt_server.distributed_pending_jobs[multi_job_id]
            result = await asyncio.wait_for(queue.get(), timeout=slice_timeout)
        except asyncio.TimeoutError:
            if (time.time() - last_activity) < base_timeout:
                continue
            missing_workers = sorted(expected_workers - workers_done)
            log(f"{timeout_log_prefix}{missing_workers}")
            break

        on_result(result)
        worker_id_value = str(result.get("worker_id", ""))
        is_last = bool(result.get("is_last", False))
        last_activity = time.time()
        base_timeout = float(get_worker_timeout_seconds())
        if is_last and worker_id_value in expected_workers:
            workers_done.add(worker_id_value)

    return workers_done
