import asyncio
import time
import uuid
from typing import Any

from ..utils.async_helpers import queue_prompt_payload
from ..utils.config import load_config
from ..utils.constants import (
    ORCHESTRATION_MEDIA_SYNC_CONCURRENCY,
    ORCHESTRATION_MEDIA_SYNC_TIMEOUT,
    ORCHESTRATION_WORKER_PROBE_CONCURRENCY,
    ORCHESTRATION_WORKER_PREP_CONCURRENCY,
)
from ..utils.logging import debug_log, log
from ..utils.network import build_master_url
from ..utils.runtime_state import ensure_distributed_runtime_state, get_prompt_server_instance
from ..utils.trace_logger import trace_debug
from .schemas import coerce_positive_float, coerce_positive_int
from .orchestration.dispatch import (
    dispatch_worker_prompt,
    rank_workers_by_load,
    select_active_workers,
    select_least_busy_worker,
)
from .orchestration.media_sync import convert_paths_for_platform, fetch_worker_path_separator, sync_worker_media
from .orchestration.prompt_transform import (
    PromptIndex,
    apply_participant_overrides,
    find_nodes_by_class,
    generate_job_id_map,
    prepare_delegate_master_prompt,
    prune_prompt_for_worker,
)

def _generate_execution_trace_id():
    return f"exec_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def ensure_distributed_state(server_instance=None):
    """Ensure prompt_server has the state used by distributed queue orchestration."""
    ensure_distributed_runtime_state(server_instance)


async def _ensure_distributed_queue(job_id):
    """Ensure a queue exists for the given distributed job ID."""
    runtime_state = ensure_distributed_runtime_state()
    async with runtime_state.distributed_jobs_lock:
        if job_id not in runtime_state.distributed_pending_jobs:
            runtime_state.distributed_pending_jobs[job_id] = asyncio.Queue()


def _resolve_enabled_workers(config, requested_ids=None):
    """Return a list of worker configs that should participate."""
    workers = []
    for worker in config.get("workers", []):
        worker_id = str(worker.get("id") or "").strip()
        if not worker_id:
            continue

        if requested_ids is not None:
            if worker_id not in requested_ids:
                continue
        elif not worker.get("enabled", False):
            continue

        raw_port = worker.get("port", worker.get("listen_port", 8188))
        try:
            port = int(raw_port or 8188)
        except (TypeError, ValueError):
            log(f"[Distributed] Invalid port '{raw_port}' for worker {worker_id}; defaulting to 8188.")
            port = 8188

        workers.append(
            {
                "id": worker_id,
                "name": worker.get("name", worker_id),
                "host": worker.get("host"),
                "port": port,
                "type": worker.get("type", "local"),
            }
        )
    return workers


def _resolve_orchestration_limits(config):
    """Resolve bounded concurrency/timeouts for worker preparation pipeline."""
    settings = (config or {}).get("settings", {}) or {}
    worker_probe_concurrency = coerce_positive_int(
        settings.get("worker_probe_concurrency"),
        ORCHESTRATION_WORKER_PROBE_CONCURRENCY,
    )
    worker_prep_concurrency = coerce_positive_int(
        settings.get("worker_prep_concurrency"),
        ORCHESTRATION_WORKER_PREP_CONCURRENCY,
    )
    media_sync_concurrency = coerce_positive_int(
        settings.get("media_sync_concurrency"),
        ORCHESTRATION_MEDIA_SYNC_CONCURRENCY,
    )
    media_sync_timeout_seconds = coerce_positive_float(
        settings.get("media_sync_timeout_seconds"),
        ORCHESTRATION_MEDIA_SYNC_TIMEOUT,
    )
    return (
        worker_probe_concurrency,
        worker_prep_concurrency,
        media_sync_concurrency,
        media_sync_timeout_seconds,
    )


def _is_load_balance_enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _prompt_requests_load_balance(prompt_index):
    for node_id in prompt_index.nodes_for_class("DistributedCollector"):
        inputs = prompt_index.inputs_by_node.get(node_id, {})
        if _is_load_balance_enabled(inputs.get("load_balance", False)):
            return True
    return False


async def _prepare_worker_payload(
    worker,
    prompt_index,
    enabled_ids,
    job_id_map,
    master_url,
    delegate_master,
    trace_execution_id,
    worker_prep_semaphore,
    media_sync_semaphore,
    media_sync_timeout_seconds,
):
    """Prepare one worker prompt payload with bounded concurrency and media-sync timeout."""
    async with worker_prep_semaphore:
        worker_prompt = prompt_index.copy_prompt()

        worker_type = str(worker.get("type") or "local").strip().lower()
        is_remote_like = bool(worker.get("host")) and worker_type != "local"
        if is_remote_like:
            path_separator = await fetch_worker_path_separator(worker, trace_execution_id=trace_execution_id)
            if path_separator:
                worker_prompt = convert_paths_for_platform(worker_prompt, path_separator)

        worker_prompt = prune_prompt_for_worker(worker_prompt)
        worker_prompt = apply_participant_overrides(
            worker_prompt,
            worker["id"],
            enabled_ids,
            job_id_map,
            master_url,
            delegate_master,
            prompt_index,
        )

        if is_remote_like:
            async with media_sync_semaphore:
                try:
                    await asyncio.wait_for(
                        sync_worker_media(worker, worker_prompt, trace_execution_id=trace_execution_id),
                        timeout=media_sync_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    trace_debug(
                        trace_execution_id,
                        (
                            f"Media sync timed out after {media_sync_timeout_seconds:.1f}s "
                            f"for worker {worker.get('name')} ({worker.get('id')}); continuing dispatch."
                        ),
                    )

        return worker, worker_prompt


async def _select_execution_workers(
    workers,
    use_websocket,
    delegate_master,
    load_balance_requested,
    has_branch_nodes,
    master_url,
    execution_trace_id,
    worker_probe_concurrency,
):
    active_workers, delegate_master = await select_active_workers(
        workers,
        use_websocket,
        delegate_master,
        trace_execution_id=execution_trace_id,
        probe_concurrency=worker_probe_concurrency,
    )

    if load_balance_requested:
        candidate_workers = list(active_workers)
        if not delegate_master:
            candidate_workers.append(
                {
                    "id": "master",
                    "name": "Master",
                    "host": master_url,
                    "type": "local",
                }
            )

        selected_worker = None
        if candidate_workers:
            selected_worker = await select_least_busy_worker(
                candidate_workers,
                trace_execution_id=execution_trace_id,
                probe_concurrency=worker_probe_concurrency,
            )
        if selected_worker is None and candidate_workers:
            trace_debug(
                execution_trace_id,
                "Load-balance selection probe failed; using first available candidate.",
            )
            selected_worker = candidate_workers[0]

        if selected_worker is not None and str(selected_worker.get("id")) == "master":
            active_workers = []
            delegate_master = False
            trace_debug(
                execution_trace_id,
                "Load-balance selected master for execution (workers skipped).",
            )
        elif selected_worker is not None:
            active_workers = [selected_worker]
            delegate_master = True
            trace_debug(
                execution_trace_id,
                f"Load-balance selected worker {selected_worker.get('id')} (master set to delegate-only).",
            )
        else:
            trace_debug(
                execution_trace_id,
                "Load-balance requested but no execution candidates were available.",
            )
            active_workers = []
            delegate_master = False

    if has_branch_nodes and len(active_workers) > 1:
        active_workers = await rank_workers_by_load(
            active_workers,
            trace_execution_id=execution_trace_id,
            probe_concurrency=worker_probe_concurrency,
        )

    return active_workers, delegate_master


def _distributed_queue_nodes(prompt_index):
    return (
        prompt_index.nodes_for_class("DistributedCollector")
        + prompt_index.nodes_for_class("DistributedBranchCollector")
        + prompt_index.nodes_for_class("DistributedBranch")
        + prompt_index.nodes_for_class("UltimateSDUpscaleDistributed")
    )


def _register_job_allowed_workers(job_id_map, enabled_ids):
    runtime_state = ensure_distributed_runtime_state()
    allowed = {str(worker_id) for worker_id in (enabled_ids or []) if str(worker_id).strip()}
    for job_id in job_id_map.values():
        if job_id:
            runtime_state.distributed_job_allowed_workers[str(job_id)] = set(allowed)


async def _ensure_job_queues(prompt_index, job_id_map):
    for node_id in _distributed_queue_nodes(prompt_index):
        job_id = job_id_map.get(node_id)
        if job_id:
            await _ensure_distributed_queue(job_id)


def _build_master_prompt(
    prompt_index,
    enabled_ids,
    job_id_map,
    master_url,
    delegate_master,
):
    master_prompt = prompt_index.copy_prompt()
    master_prompt = apply_participant_overrides(
        master_prompt,
        "master",
        enabled_ids,
        job_id_map,
        master_url,
        delegate_master,
        prompt_index,
    )

    if not delegate_master:
        return master_prompt

    collector_ids = (
        find_nodes_by_class(master_prompt, "DistributedCollector")
        + find_nodes_by_class(master_prompt, "DistributedBranchCollector")
    )
    upscale_nodes = find_nodes_by_class(master_prompt, "UltimateSDUpscaleDistributed")
    # Include USDU nodes as collector-like for delegate pruning
    collector_ids.extend(upscale_nodes)
    if not collector_ids:
        debug_log(
            "Delegate-only master mode requested but no collector/branch-collector nodes found in master prompt. Running full prompt on master."
        )
        return master_prompt
    return prepare_delegate_master_prompt(master_prompt, collector_ids)


async def _prepare_worker_payloads_for_dispatch(
    active_workers,
    prompt_index,
    enabled_ids,
    job_id_map,
    master_url,
    delegate_master,
    execution_trace_id,
    worker_prep_concurrency,
    media_sync_concurrency,
    media_sync_timeout_seconds,
):
    if not active_workers:
        return []

    worker_prep_semaphore = asyncio.Semaphore(worker_prep_concurrency)
    media_sync_semaphore = asyncio.Semaphore(media_sync_concurrency)
    return await asyncio.gather(
        *[
            _prepare_worker_payload(
                worker,
                prompt_index,
                enabled_ids,
                job_id_map,
                master_url,
                delegate_master,
                execution_trace_id,
                worker_prep_semaphore,
                media_sync_semaphore,
                media_sync_timeout_seconds,
            )
            for worker in active_workers
        ]
    )


async def _dispatch_worker_payloads(
    worker_payloads,
    workflow_meta,
    client_id,
    use_websocket,
    execution_trace_id,
):
    if not worker_payloads:
        return

    await asyncio.gather(
        *[
            dispatch_worker_prompt(
                worker,
                wprompt,
                workflow_meta,
                client_id,
                use_websocket=use_websocket,
                trace_execution_id=execution_trace_id,
            )
            for worker, wprompt in worker_payloads
        ]
    )


async def orchestrate_distributed_execution(
    prompt_obj: dict[str, Any],
    workflow_meta: dict[str, Any] | None,
    client_id: str | None,
    enabled_worker_ids: list[str] | set[str] | None = None,
    delegate_master: bool | None = None,
    trace_execution_id: str | None = None,
) -> tuple[str, int]:
    """Core orchestration logic for the /distributed/queue endpoint.

    Returns:
        tuple[str, int]: (prompt_id, worker_count)
    """
    ensure_distributed_state()
    execution_trace_id = trace_execution_id or _generate_execution_trace_id()

    config = load_config()
    use_websocket = bool(config.get("settings", {}).get("websocket_orchestration", False))
    master_url = build_master_url(config=config, prompt_server_instance=get_prompt_server_instance())
    (
        worker_probe_concurrency,
        worker_prep_concurrency,
        media_sync_concurrency,
        media_sync_timeout_seconds,
    ) = _resolve_orchestration_limits(config)
    requested_ids = enabled_worker_ids if enabled_worker_ids is not None else None
    workers = _resolve_enabled_workers(config, requested_ids)
    prompt_index = PromptIndex(prompt_obj)
    load_balance_requested = _prompt_requests_load_balance(prompt_index)
    has_branch_nodes = bool(prompt_index.nodes_for_class("DistributedBranch"))
    trace_debug(
        execution_trace_id,
        (
            f"Orchestration start: requested_workers={len(workers)}, "
            f"requested_ids={requested_ids if requested_ids is not None else 'enabled_only'}, "
            f"websocket={use_websocket}, "
            f"probe_concurrency={worker_probe_concurrency}, "
            f"prep_concurrency={worker_prep_concurrency}, "
            f"media_sync_concurrency={media_sync_concurrency}, "
            f"media_sync_timeout={media_sync_timeout_seconds:.1f}s, "
            f"load_balance={load_balance_requested}, has_branch_nodes={has_branch_nodes}"
        ),
    )

    # Respect master delegate-only configuration
    if delegate_master is None:
        delegate_master = bool(config.get("settings", {}).get("master_delegate_only", False))

    if not workers and delegate_master:
        trace_debug(
            execution_trace_id,
            "Delegate-only requested but no workers are enabled. Falling back to master execution.",
        )
        delegate_master = False

    active_workers, delegate_master = await _select_execution_workers(
        workers=workers,
        use_websocket=use_websocket,
        delegate_master=delegate_master,
        load_balance_requested=load_balance_requested,
        has_branch_nodes=has_branch_nodes,
        master_url=master_url,
        execution_trace_id=execution_trace_id,
        worker_probe_concurrency=worker_probe_concurrency,
    )

    enabled_ids = [worker["id"] for worker in active_workers]

    discovery_prefix = f"exec_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    job_id_map = generate_job_id_map(prompt_index, discovery_prefix)

    if not job_id_map:
        trace_debug(execution_trace_id, "No distributed nodes detected; queueing prompt on master only.")
        prompt_id = await queue_prompt_payload(prompt_obj, workflow_meta, client_id)
        return prompt_id, 0

    _register_job_allowed_workers(job_id_map, enabled_ids)
    await _ensure_job_queues(prompt_index, job_id_map)

    master_prompt = _build_master_prompt(
        prompt_index=prompt_index,
        enabled_ids=enabled_ids,
        job_id_map=job_id_map,
        master_url=master_url,
        delegate_master=delegate_master,
    )

    if active_workers:
        trace_debug(
            execution_trace_id,
            "Active distributed workers: "
            + ", ".join(f"{worker['name']} ({worker['id']})" for worker in active_workers),
        )
    worker_payloads = await _prepare_worker_payloads_for_dispatch(
        active_workers=active_workers,
        prompt_index=prompt_index,
        enabled_ids=enabled_ids,
        job_id_map=job_id_map,
        master_url=master_url,
        delegate_master=delegate_master,
        execution_trace_id=execution_trace_id,
        worker_prep_concurrency=worker_prep_concurrency,
        media_sync_concurrency=media_sync_concurrency,
        media_sync_timeout_seconds=media_sync_timeout_seconds,
    )
    await _dispatch_worker_payloads(
        worker_payloads=worker_payloads,
        workflow_meta=workflow_meta,
        client_id=client_id,
        use_websocket=use_websocket,
        execution_trace_id=execution_trace_id,
    )

    prompt_id = await queue_prompt_payload(master_prompt, workflow_meta, client_id)
    trace_debug(
        execution_trace_id,
        f"Orchestration complete: prompt_id={prompt_id}, dispatched_workers={len(worker_payloads)}, delegate_master={delegate_master}",
    )
    return prompt_id, len(worker_payloads)
