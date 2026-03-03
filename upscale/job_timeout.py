import time

from ..utils.config import load_config
from ..utils.constants import HEARTBEAT_TIMEOUT
from ..utils.logging import debug_log, log
from ..utils.network import build_worker_url, probe_worker
from .job_models import BaseJobState
from .job_store import ensure_tile_jobs_initialized


def _find_worker_record(worker_id):
    """Return worker config entry by id, or None when missing."""
    workers = load_config().get("workers", [])
    return next((w for w in workers if str(w.get("id")) == str(worker_id)), None)


async def _check_and_requeue_timed_out_workers(multi_job_id, total_tasks):
    """Check timed out workers and requeue their tasks. Returns requeued count."""
    prompt_server = ensure_tile_jobs_initialized()
    current_time = time.time()

    # Allow override via config setting 'worker_timeout_seconds'
    cfg = load_config()
    hb_timeout = int(cfg.get("settings", {}).get("worker_timeout_seconds", HEARTBEAT_TIMEOUT))

    # Snapshot timed-out workers and job details under lock.
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if not isinstance(job_data, BaseJobState):
            return 0

        completed_tasks_snapshot = set(job_data.completed_tasks.keys())
        batched_static_snapshot = bool(job_data.batched_static)
        num_tiles_per_image_snapshot = int(job_data.num_tiles_per_image or 1)
        batch_size_snapshot = int(job_data.batch_size or 1)

        timed_out_workers = []
        for worker, last_heartbeat in list(job_data.worker_status.items()):
            age = current_time - float(last_heartbeat)
            debug_log(f"Timeout check: worker={worker} age={age:.1f}s threshold={hb_timeout}s")
            if age > hb_timeout:
                timed_out_workers.append(
                    {
                        "worker_id": worker,
                        "last_heartbeat": float(last_heartbeat),
                        "assigned_tasks": list(job_data.assigned_to_workers.get(worker, [])),
                    }
                )

    if not timed_out_workers:
        return 0

    # Probe outside lock to avoid lock contention on network latency.
    workers_to_requeue = []
    workers_graced = []
    for worker_info in timed_out_workers:
        worker = worker_info["worker_id"]
        assigned = worker_info["assigned_tasks"]
        age = current_time - worker_info["last_heartbeat"]

        incomplete_assigned = 0
        try:
            if assigned:
                if batched_static_snapshot:
                    for task_id in assigned:
                        for b in range(batch_size_snapshot):
                            gidx = b * num_tiles_per_image_snapshot + task_id
                            if gidx not in completed_tasks_snapshot:
                                incomplete_assigned += 1
                                break
                else:
                    for task_id in assigned:
                        if task_id not in completed_tasks_snapshot:
                            incomplete_assigned += 1
            debug_log(
                f"Assigned diagnostics: total_assigned={len(assigned)} "
                f"incomplete_assigned={incomplete_assigned}"
            )
        except Exception as e:
            debug_log(f"Assigned diagnostics failed for worker {worker}: {e}")

        busy = False
        probe_queue = None
        try:
            worker_record = _find_worker_record(worker)
            if worker_record:
                worker_url = build_worker_url(worker_record)
                debug_log(f"Probing worker {worker} at {worker_url}/prompt")
                payload = await probe_worker(worker_url, timeout=2.0)
                if payload is not None:
                    probe_queue = int(payload.get("exec_info", {}).get("queue_remaining", 0))
                    busy = probe_queue is not None and probe_queue > 0
            else:
                debug_log(f"Probe skipped; worker {worker} not found in config")
        except Exception as e:
            debug_log(f"Probe failed for worker {worker}: {e}")
        finally:
            debug_log(
                f"Probe diagnostics: online={probe_queue is not None} queue_remaining={probe_queue}"
            )

        if busy:
            workers_graced.append(worker)
            debug_log(f"Heartbeat grace: worker {worker} busy via probe; skipping requeue")
            continue

        log(f"Worker {worker} heartbeat timed out after {age:.1f}s")
        workers_to_requeue.append((worker, assigned))

    # Re-acquire lock and apply requeue/cleanup decisions.
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if not isinstance(job_data, BaseJobState):
            return 0

        # Refresh heartbeat for workers that we proved are still busy.
        for worker in workers_graced:
            if worker in job_data.worker_status:
                job_data.worker_status[worker] = current_time

        requeued_count = 0
        completed_tasks = job_data.completed_tasks
        batched_static = bool(job_data.batched_static)
        num_tiles_per_image = int(job_data.num_tiles_per_image or 1)
        batch_size = int(job_data.batch_size or 1)
        for worker, assigned_snapshot in workers_to_requeue:
            # Use current assignments if present, falling back to the snapshot.
            assigned_tasks = list(job_data.assigned_to_workers.get(worker, assigned_snapshot))
            for task_id in assigned_tasks:
                # If batched_static, task_id is a tile_idx; consider it complete only if
                # all corresponding global_idx entries are present in completed_tasks.
                if batched_static:
                    all_done = True
                    for b in range(batch_size):
                        gidx = b * num_tiles_per_image + task_id
                        if gidx not in completed_tasks:
                            all_done = False
                            break
                    if not all_done:
                        await job_data.pending_tasks.put(task_id)
                        requeued_count += 1
                else:
                    if task_id not in completed_tasks:
                        await job_data.pending_tasks.put(task_id)
                        requeued_count += 1
            job_data.worker_status.pop(worker, None)
            if worker in job_data.assigned_to_workers:
                job_data.assigned_to_workers[worker] = []

        return requeued_count


async def check_and_requeue_timed_out_workers(multi_job_id, total_tasks):
    """Public wrapper for worker-timeout requeue checks."""
    return await _check_and_requeue_timed_out_workers(multi_job_id, total_tasks)
