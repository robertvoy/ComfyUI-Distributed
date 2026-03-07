"""Canonical USDU helper exports.

This module is the stable import target for USDU queue/heartbeat helpers.
"""

from ..upscale.conditioning import clone_conditioning, clone_control_chain
from ..upscale.job_store import (
    MAX_PAYLOAD_SIZE,
    cleanup_job,
    drain_results_queue,
    ensure_tile_jobs_initialized,
    get_completed_count,
    init_dynamic_job,
    init_job_queue,
    init_static_job_batched,
    mark_task_completed,
)
from ..upscale.job_timeout import check_and_requeue_timed_out_workers
from ..upscale.payload_parsers import parse_tiles_from_form
from .auth import distributed_auth_headers
from .config import load_config
from .logging import debug_log
from .network import get_client_session


async def send_heartbeat_to_master(multi_job_id: str, master_url: str, worker_id: str) -> None:
    """Send heartbeat to master from worker-side processing loops."""
    try:
        data = {"multi_job_id": multi_job_id, "worker_id": str(worker_id)}
        session = await get_client_session()
        url = f"{master_url}/distributed/heartbeat"
        async with session.post(
            url,
            json=data,
            headers=distributed_auth_headers(load_config()),
        ) as response:
            response.raise_for_status()
    except Exception as e:
        debug_log(f"Heartbeat failed: {e}")

__all__ = [
    "MAX_PAYLOAD_SIZE",
    "check_and_requeue_timed_out_workers",
    "cleanup_job",
    "drain_results_queue",
    "get_completed_count",
    "init_job_queue",
    "mark_task_completed",
    "parse_tiles_from_form",
    "send_heartbeat_to_master",
    "clone_conditioning",
    "clone_control_chain",
    "ensure_tile_jobs_initialized",
    "init_dynamic_job",
    "init_static_job_batched",
]
