import asyncio
import copy
import time
import uuid

import server

from ..utils.async_helpers import queue_prompt_payload
from ..utils.config import load_config
from ..utils.constants import (
    ORCHESTRATION_MEDIA_SYNC_CONCURRENCY,
    ORCHESTRATION_MEDIA_SYNC_TIMEOUT,
    ORCHESTRATION_WORKER_PROBE_CONCURRENCY,
    ORCHESTRATION_WORKER_PREP_CONCURRENCY,
)
from ..utils.logging import debug_log, log
from ..utils.network import build_master_url, build_master_callback_url
from ..utils.trace_logger import trace_debug
from .schemas import parse_positive_float, parse_positive_int
from .orchestration.dispatch import (
    dispatch_worker_prompt,
    select_active_workers,
    select_least_busy_worker,
)
from .orchestration.media_sync import (
    convert_paths_for_platform,
    fetch_worker_has_node,
    fetch_worker_path_separator,
    sync_worker_media,
)
from .orchestration.prompt_transform import (
    PromptIndex,
    apply_participant_overrides,
    find_nodes_by_class,
    generate_job_id_map,
    prepare_delegate_master_prompt,
    prune_prompt_for_worker,
    WORKER_SINK_CLASS,
    _resolve_direct_workers_for_image_list,
)
from ..image_list.runtime import validate_image_list_regions, _has_selected_serverless_workers
from ..image_list.execution_plan import prepare_image_list_execution


prompt_server = server.PromptServer.instance


def _generate_execution_trace_id():
    return f"exec_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def ensure_distributed_state(server_instance=None):
    """Ensure prompt_server has the state used by distributed queue orchestration."""
    ps = server_instance or prompt_server
    if not hasattr(ps, "distributed_pending_jobs"):
        ps.distributed_pending_jobs = {}
    if not hasattr(ps, "distributed_jobs_lock"):
        ps.distributed_jobs_lock = asyncio.Lock()


# Initialize top-level distributed queue state at module import time.
ensure_distributed_state()


async def _ensure_distributed_queue(job_id):
    """Ensure a queue exists for the given distributed job ID."""
    ensure_distributed_state()
    async with prompt_server.distributed_jobs_lock:
        if job_id not in prompt_server.distributed_pending_jobs:
            prompt_server.distributed_pending_jobs[job_id] = asyncio.Queue()


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

        if worker.get("type") == "serverless":
            from ..utils.serverless_config import validate_serverless_worker
            pool = validate_serverless_worker(worker)
            if pool.get("media_profile") == "comfy-media-v1":
                from ..serverless_bridge.contract import identifier
                # Do not repair malformed media identities by stripping them.
                identifier(pool["id"])
            for slot in range(pool["jobs_per_run"]):
                workers.append(dict(pool, pool_id=worker_id, id=f"serverless:{worker_id}:{slot + 1}",
                                    name=f"{pool.get('name', worker_id)} / job {slot + 1}"))
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
    worker_probe_concurrency = parse_positive_int(
        settings.get("worker_probe_concurrency"),
        ORCHESTRATION_WORKER_PROBE_CONCURRENCY,
    )
    worker_prep_concurrency = parse_positive_int(
        settings.get("worker_prep_concurrency"),
        ORCHESTRATION_WORKER_PREP_CONCURRENCY,
    )
    media_sync_concurrency = parse_positive_int(
        settings.get("media_sync_concurrency"),
        ORCHESTRATION_MEDIA_SYNC_CONCURRENCY,
    )
    media_sync_timeout_seconds = parse_positive_float(
        settings.get("media_sync_timeout_seconds"),
        ORCHESTRATION_MEDIA_SYNC_TIMEOUT,
    )
    return (
        worker_probe_concurrency,
        worker_prep_concurrency,
        media_sync_concurrency,
        media_sync_timeout_seconds,
    )


def _build_master_prompt(prompt_index, enabled_ids, job_id_map, master_url, delegate_master,
                         workflow_meta, image_list_mode=False):
    """Master copy of the prompt; pruned to post-Collector nodes in delegate mode."""
    master_prompt = apply_participant_overrides(
        prompt_index.copy_prompt(),
        "master",
        enabled_ids,
        job_id_map,
        master_url,
        delegate_master,
        prompt_index,
        workflow_meta=workflow_meta,
    )
    if delegate_master and not image_list_mode:
        collector_ids = find_nodes_by_class(master_prompt, "DistributedCollector")
        upscale_nodes = find_nodes_by_class(master_prompt, "UltimateSDUpscaleDistributed")
        if upscale_nodes:
            debug_log(
                "Delegate-only master mode currently does not support UltimateSDUpscaleDistributed nodes; running full prompt on master."
            )
        elif not collector_ids:
            debug_log(
                "Delegate-only master mode requested but no collectors found in master prompt. Running full prompt on master."
            )
        else:
            master_prompt = prepare_delegate_master_prompt(master_prompt, collector_ids)
    return master_prompt


def _load_balance_can_wait(prompt_index, workers):
    """Only plain Collector runs on direct workers can wait for the next free GPU.

    Upscale nodes cannot run in delegate mode, and serverless slots are
    started through their own manager rather than a worker prompt queue.
    """
    if prompt_index.nodes_for_class("UltimateSDUpscaleDistributed"):
        return False
    return not any(worker.get("type") == "serverless" for worker in workers)


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


def _validate_media_workers(workers, prompt_index=None):
    """Reject unknown serverless media profiles before probing or dispatching.

    A Distributed Collector accepts direct workers, legacy-image-v1 pools and
    comfy-media-v1 pools in the same run; each pool's profile selects its
    endpoint contract. Unknown profiles fail rather than guessing a contract.
    legacy-image-v1 endpoints return 8-bit files, so a Collector set to exact
    cannot use them.
    """
    exact = prompt_index is not None and any(
        str(prompt_index.inputs_by_node.get(node_id, {}).get("transfer_format", "")).strip().lower() == "exact"
        for node_id in prompt_index.nodes_for_class("DistributedCollector")
    )
    for worker in workers:
        if worker.get("type") != "serverless":
            continue
        profile = worker.get("media_profile", "legacy-image-v1")
        if profile not in ("legacy-image-v1", "comfy-media-v1"):
            raise ValueError("Unsupported serverless media_profile")
        if exact and profile == "legacy-image-v1":
            raise ValueError(
                "transfer_format exact requires comfy-media-v1 serverless pools; "
                "legacy-image-v1 endpoints return 8-bit images"
            )


async def _prepare_worker_payload(
    worker,
    prompt_index,
    enabled_ids,
    job_id_map,
    master_url,
    config,
    delegate_master,
    trace_execution_id,
    worker_prep_semaphore,
    media_sync_semaphore,
    media_sync_timeout_seconds,
):
    """Prepare one worker prompt payload with bounded concurrency and media-sync timeout."""
    async with worker_prep_semaphore:
        _validate_media_workers([worker], prompt_index)
        worker_prompt = prompt_index.copy_prompt()
        if worker.get("type") == "serverless":
            from .orchestration.runpod_client import serialize_request
            if worker.get("media_profile", "legacy-image-v1") == "comfy-media-v1":
                from ..serverless_bridge.contract import PROFILE, VERSION
                from .orchestration.serverless_media_prompt import compile_media_prompt
                from .orchestration.serverless_typed_inputs import prepare_media_inputs
                # The typed compiler validates the complete private graph and
                # prunes it itself, retaining every connected Collector branch.
                worker_prompt = apply_participant_overrides(
                    worker_prompt, worker["id"], enabled_ids, job_id_map,
                    "", delegate_master, prompt_index,
                )
                request_id = str(uuid.uuid4())
                workflow, manifest = compile_media_prompt(worker_prompt, request_id)
                images, artifacts, refs = await prepare_media_inputs(workflow)
                media = {
                    "version": VERSION, "profile": PROFILE, "request_id": request_id,
                    "artifacts": artifacts, "inputs": refs, "outputs": copy.deepcopy(manifest),
                }
                serialize_request(workflow, images, worker["timeout_seconds"], media=media)
                return worker, {"worker": worker, "workflow": workflow, "manifest": manifest,
                                "images": images, "media": media}

            from .orchestration.serverless_prompt import compile_serverless_prompt
            from .orchestration.serverless_media import prepare_images
            worker_prompt = apply_participant_overrides(
                prune_prompt_for_worker(worker_prompt), worker["id"], enabled_ids,
                job_id_map, "", delegate_master, prompt_index,
            )
            workflow, manifest = compile_serverless_prompt(worker_prompt)
            images = await prepare_images(workflow)
            # Fail during preparation, before regular workers or the master are
            # dispatched. submit() uses this same serializer for the sent bytes.
            serialize_request(workflow, images, worker["timeout_seconds"])
            return worker, {"worker": worker, "workflow": workflow, "manifest": manifest, "images": images}

        worker_master_url = build_master_callback_url(
            worker,
            config=config,
            prompt_server_instance=prompt_server,
        )

        worker_type = str(worker.get("type") or "local").strip().lower()
        is_remote_like = bool(worker.get("host")) and worker_type != "local"
        # Older workers lack the sink node; they keep the PreviewImage terminator.
        has_sink_task = asyncio.ensure_future(
            fetch_worker_has_node(worker, WORKER_SINK_CLASS, trace_execution_id=trace_execution_id)
        )
        try:
            if is_remote_like:
                path_separator = await fetch_worker_path_separator(worker, trace_execution_id=trace_execution_id)
                if path_separator:
                    worker_prompt = convert_paths_for_platform(worker_prompt, path_separator)
            use_worker_sink = await has_sink_task
        finally:
            if not has_sink_task.done():
                has_sink_task.cancel()

        worker_prompt = prune_prompt_for_worker(worker_prompt, use_worker_sink=use_worker_sink)
        worker_prompt = apply_participant_overrides(
            worker_prompt,
            worker["id"],
            enabled_ids,
            job_id_map,
            worker_master_url,
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


def _register_waiting_run(prompt_index, job_id_map, candidates, *, master_may_run, master_url, config,
                          workflow_meta, client_id, use_websocket, media_sync_timeout_seconds,
                          execution_trace_id, dispatched_worker=None):
    """Register a load-balanced run with the next-free-GPU dispatcher.

    Returns the prompt id the run is tracked under. Without ``dispatched_worker``
    the run waits for a free participant: its master prompt must be queued
    under that id, the worker prompt is only prepared (media sync included)
    once a worker frees up, and the master prompt is then re-queued in its
    delegate form under the same id so clients keep tracking one prompt.

    With ``dispatched_worker`` the run is already being sent to that worker
    while the master participates: nothing is queued on the master now, and
    the dispatcher queues the delegate prompt once the worker delivers.
    """
    from .orchestration.load_balance_queue import PendingRun, get_load_balance_queue

    prompt_id = str(uuid.uuid4())

    async def prepare_for_worker(worker):
        _, worker_prompt = await _prepare_worker_payload(
            worker,
            prompt_index,
            [worker["id"]],
            job_id_map,
            master_url,
            config,
            True,
            execution_trace_id,
            asyncio.Semaphore(1),
            asyncio.Semaphore(1),
            media_sync_timeout_seconds,
        )
        return worker_prompt

    async def send_to_worker(worker, worker_prompt):
        await dispatch_worker_prompt(
            worker,
            worker_prompt,
            workflow_meta,
            client_id,
            use_websocket=use_websocket,
            trace_execution_id=execution_trace_id,
        )

    async def requeue_as_delegate(worker):
        delegate_prompt = _build_master_prompt(
            prompt_index, [worker["id"]], job_id_map, master_url, True, workflow_meta
        )
        # Validation yields to the loop; a clear or delete that lands meanwhile
        # must stop the insertion itself, not just the attempt.
        await queue_prompt_payload(
            delegate_prompt, workflow_meta, client_id, prompt_id=prompt_id,
            insert_if=lambda: not record.cancelled,
        )

    record = PendingRun(
        prompt_id,
        job_id_map.values(),
        candidates,
        master_may_run,
        prepare_for_worker,
        send_to_worker,
        requeue_as_delegate,
        trace_execution_id=execution_trace_id,
    )
    if dispatched_worker is not None:
        get_load_balance_queue().register_dispatched(record, dispatched_worker)
    else:
        get_load_balance_queue().register(record)
    return prompt_id


def _waiting_run_queued(prompt_id):
    """The master prompt is in the queue: the dispatcher may now hand it out."""
    from .orchestration.load_balance_queue import get_load_balance_queue
    queue = get_load_balance_queue()
    record = queue._records.get(str(prompt_id))
    if record is not None:
        queue.mark_queued(record)


def _discard_waiting_run(prompt_id):
    """The run never made it: forget it and its result queues."""
    from .orchestration.load_balance_queue import get_load_balance_queue
    get_load_balance_queue().discard_run(prompt_id)


def _hold_unresolved_run(prompt_id, worker):
    """The eager submission to ``worker`` may have been accepted: never resend it.

    The dispatcher confirms with the worker; if it took the run its result
    is still collected, otherwise the run is dropped (there is nothing on the
    master to put back, and the caller already reported the failure).
    """
    from .orchestration.load_balance_queue import get_load_balance_queue
    queue = get_load_balance_queue()
    record = queue._records.get(str(prompt_id))
    if record is not None:
        queue.mark_unresolved(record, worker)


async def orchestrate_distributed_execution(
    prompt_obj,
    workflow_meta,
    client_id,
    enabled_worker_ids=None,
    delegate_master=None,
    trace_execution_id=None,
):
    """Core orchestration logic for the /distributed/queue endpoint.

    Returns:
        tuple[str, int, int, dict]: (prompt_id, number, worker_count, node_errors)
    """
    ensure_distributed_state()
    execution_trace_id = trace_execution_id or _generate_execution_trace_id()

    config = load_config()
    if delegate_master is None:
        delegate_master = bool(config.get("settings", {}).get("master_delegate_only", False))
    use_websocket = bool(config.get("settings", {}).get("websocket_orchestration", False))
    master_url = build_master_url(config=config, prompt_server_instance=prompt_server)
    (
        worker_probe_concurrency,
        worker_prep_concurrency,
        media_sync_concurrency,
        media_sync_timeout_seconds,
    ) = _resolve_orchestration_limits(config)
    requested_ids = enabled_worker_ids if enabled_worker_ids is not None else None
    image_list_mode = any(isinstance(node, dict) and node.get("class_type") == "DistributedImageListStart"
                          for node in prompt_obj.values())
    target_kwargs = {}
    if image_list_mode:
        # Check the source graph before execution-root pruning can remove an
        # incompatible mode (or mask it with an unrelated boundary error).
        if any(isinstance(node, dict) and node.get("class_type") in (
                "DistributedCollector", "UltimateSDUpscaleDistributed") for node in prompt_obj.values()):
            raise ValueError(
                "Distributed Image List v1 cannot be mixed with Distributed Collector or "
                "Ultimate SD Upscale nodes in one prompt"
            )
        prompt_obj, outer_targets = prepare_image_list_execution(prompt_obj)
        target_kwargs["partial_execution_targets"] = outer_targets
    prompt_index = PromptIndex(prompt_obj)
    if image_list_mode:
        # Validate before probing or queueing anything so malformed boundaries and
        # old worker releases fail before any upstream image is materialized.
        validate_image_list_regions(prompt_obj, explicit_targets=True)
        from ..serverless_bridge.image_transfer import DEFAULT_FORMAT, DEFAULT_QUALITY, validate_options
        required_transfer_formats = set()
        for end_id in prompt_index.nodes_for_class("DistributedImageListEnd"):
            end_inputs = prompt_obj[end_id].get("inputs", {})
            fmt, _ = validate_options(end_inputs.get("transfer_format", DEFAULT_FORMAT),
                                      end_inputs.get("transfer_quality", DEFAULT_QUALITY))
            if fmt != DEFAULT_FORMAT:
                required_transfer_formats.add(fmt)
        workers = _resolve_direct_workers_for_image_list(config, requested_ids)
        if _has_selected_serverless_workers(config, requested_ids):
            from ..image_list.serverless import resolve_serverless_workers
            # Pool identity is retained once, never expanded by Images per run.
            # Validate profiles before reachability can hide an invalid selection.
            workers += resolve_serverless_workers(config, requested_ids)
        if not workers and delegate_master:
            raise RuntimeError(
                "Distributed Image List in orchestrator-only mode requires a selected direct worker "
                "or comfy-media-v1 serverless pool"
            )
    else:
        workers = _resolve_enabled_workers(config, requested_ids)
        _validate_media_workers(workers, prompt_index)
        if any(worker.get("media_profile") == "comfy-media-v1" for worker in workers):
            from ..serverless_bridge.contract import identifier
            # Validate the source, not PromptIndex's JSON copy (which coerces
            # integer keys to strings), before probing or dispatching workers.
            for node_id in prompt_obj:
                identifier(node_id)
    load_balance_requested = _prompt_requests_load_balance(prompt_index)
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
            f"load_balance={load_balance_requested}"
        ),
    )

    # Keep the legacy Collector fallback, but never override image-list
    # orchestrator-only mode merely because its direct workers are unavailable.
    if not image_list_mode and not workers and delegate_master:
        trace_debug(
            execution_trace_id,
            "Delegate-only requested but no workers are enabled. Falling back to master execution.",
        )
        delegate_master = False

    active_workers, selected_delegate_master = await select_active_workers(
        workers,
        use_websocket,
        False if image_list_mode else delegate_master,
        trace_execution_id=execution_trace_id,
        probe_concurrency=worker_probe_concurrency,
    )
    if image_list_mode:
        from .orchestration.dispatch import preflight_image_list_workers
        if not active_workers and delegate_master:
            raise RuntimeError(
                "No selected worker is reachable for Distributed Image List in orchestrator-only mode"
            )
        from ..image_list.runtime import END_CLASS, end_override_names
        requires_values = any(end_override_names(node.get("inputs", {}))
                              for node in prompt_obj.values() if node.get("class_type") == END_CLASS)
        direct_workers = [worker for worker in active_workers if worker.get("type") != "serverless"]
        preflight_kwargs = ({"required_transfer_formats": sorted(required_transfer_formats)}
                            if required_transfer_formats else {})
        if requires_values:
            preflight_kwargs["require_value_results"] = True
        # Preserve busy direct candidates for queued runs; runtime rechecks idle
        # availability before assignment. Never direct-probe serverless pools.
        if direct_workers:
            await preflight_image_list_workers(direct_workers, require_idle=False, **preflight_kwargs)
    else:
        delegate_master = selected_delegate_master

    wait_for_free_participant = False
    deferred_candidates = []
    # Participating master handed the run to an idle worker: queue the master's
    # collect-only prompt once the worker delivers, not now, so the master
    # executor stays free for the next full workflow.
    hold_delegate_for_worker = None
    if load_balance_requested and not image_list_mode:
        candidate_workers = list(active_workers)
        if not delegate_master:
            # Include master in load balancing only when master participation is enabled.
            candidate_workers.append(
                {
                    "id": "master",
                    "name": "Master",
                    "host": master_url,
                    "type": "local",
                }
            )

        selected_worker = None
        can_wait = bool(candidate_workers) and _load_balance_can_wait(prompt_index, active_workers)
        if can_wait:
            from .orchestration.load_balance_queue import select_idle_worker
            selected_worker, probed = await select_idle_worker(
                candidate_workers,
                trace_execution_id=execution_trace_id,
                probe_concurrency=worker_probe_concurrency,
            )
            if selected_worker is None and probed:
                # Everyone is busy: keep the run in the master queue and hand it
                # to whichever participant frees up first.
                wait_for_free_participant = True
        elif candidate_workers:
            selected_worker = await select_least_busy_worker(
                candidate_workers,
                trace_execution_id=execution_trace_id,
                probe_concurrency=worker_probe_concurrency,
            )
        if selected_worker is None and candidate_workers and not wait_for_free_participant:
            trace_debug(
                execution_trace_id,
                "Load-balance selection probe failed; using first available candidate.",
            )
            selected_worker = candidate_workers[0]

        if wait_for_free_participant:
            deferred_candidates = list(active_workers)
            active_workers = []
            trace_debug(
                execution_trace_id,
                "Load-balance: all participants busy; run waits in the master queue for the next free one "
                f"(master_may_run={not delegate_master}, workers={[w.get('id') for w in deferred_candidates]}).",
            )
        elif selected_worker is not None and str(selected_worker.get("id")) == "master":
            # Master selected as least busy; run master workload only.
            active_workers = []
            delegate_master = False
            trace_debug(
                execution_trace_id,
                "Load-balance selected master for execution (workers skipped).",
            )
        elif selected_worker is not None:
            active_workers = [selected_worker]
            if can_wait and not delegate_master:
                hold_delegate_for_worker = selected_worker
            # Worker selected as least busy; keep master orchestrator-only for this run.
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

    enabled_ids = [worker["id"] for worker in active_workers]

    discovery_prefix = f"exec_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    job_id_map = generate_job_id_map(prompt_index, discovery_prefix)

    if not job_id_map:
        trace_debug(execution_trace_id, "No distributed nodes detected; queueing prompt on master only.")
        queue_result = await queue_prompt_payload(
            prompt_obj,
            workflow_meta,
            client_id,
            include_queue_metadata=True,
            **target_kwargs,
        )
        return (
            queue_result["prompt_id"],
            queue_result["number"],
            0,
            queue_result.get("node_errors", {}),
        )

    master_prompt = _build_master_prompt(
        prompt_index, enabled_ids, job_id_map, master_url, delegate_master, workflow_meta, image_list_mode
    )

    if active_workers:
        trace_debug(
            execution_trace_id,
            "Active distributed workers: "
            + ", ".join(f"{worker['name']} ({worker['id']})" for worker in active_workers),
        )
    worker_payloads = []
    if active_workers and not image_list_mode:
        worker_prep_semaphore = asyncio.Semaphore(worker_prep_concurrency)
        media_sync_semaphore = asyncio.Semaphore(media_sync_concurrency)
        worker_payloads = await asyncio.gather(
            *[
                _prepare_worker_payload(
                    worker,
                    prompt_index,
                    enabled_ids,
                    job_id_map,
                    master_url,
                    config,
                    delegate_master,
                    execution_trace_id,
                    worker_prep_semaphore,
                    media_sync_semaphore,
                    media_sync_timeout_seconds,
                )
                for worker in active_workers
            ]
        )

    serverless_jobs = [payload for worker, payload in worker_payloads if worker.get("type") == "serverless"]
    regular_payloads = [(worker, payload) for worker, payload in worker_payloads if worker.get("type") != "serverless"]
    manager = None
    if serverless_jobs:
        from .orchestration.serverless_runtime import get_manager
        # Repeat the manager's whole-batch identity/profile preflight *before*
        # any direct dispatch, master queueing or collector queue allocation.
        # Pure legacy jobs are a no-op in that validator; keep their path free
        # of typed imports (and compatible with existing legacy media adapters).
        for prepared in serverless_jobs:
            if (prepared["worker"].get("media_profile", "legacy-image-v1") != "legacy-image-v1"
                    or prepared.get("media") is not None):
                from .orchestration.serverless_typed_results import validate_prepared_job
                validate_prepared_job(prepared)
        manager = get_manager()
        manager.ensure_capacity(len(serverless_jobs))

    # Do not leave collector queues behind when offline compilation/media validation fails.
    if not image_list_mode:
        for job_id in job_id_map.values():
            await _ensure_distributed_queue(job_id)

    held_prompt_id = None
    if hold_delegate_for_worker is not None:
        # Register before dispatching so the worker's result is never missed.
        held_prompt_id = _register_waiting_run(
            prompt_index,
            job_id_map,
            [hold_delegate_for_worker],
            master_may_run=False,
            master_url=master_url,
            config=config,
            workflow_meta=workflow_meta,
            client_id=client_id,
            use_websocket=use_websocket,
            media_sync_timeout_seconds=media_sync_timeout_seconds,
            execution_trace_id=execution_trace_id,
            dispatched_worker=hold_delegate_for_worker,
        )

    if regular_payloads:
        try:
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
                    for worker, wprompt in regular_payloads
                ]
            )
        except Exception as exc:
            if held_prompt_id is not None:
                from .orchestration import dispatch as _dispatch
                ambiguous_error = getattr(_dispatch, "AmbiguousDispatchError", ())
                if isinstance(exc, ambiguous_error):
                    _hold_unresolved_run(held_prompt_id, hold_delegate_for_worker)
                else:
                    _discard_waiting_run(held_prompt_id)
            raise

    if held_prompt_id is not None:
        trace_debug(
            execution_trace_id,
            f"Orchestration complete: prompt_id={held_prompt_id} runs on worker "
            f"{hold_delegate_for_worker.get('id')}; master collector queued when it delivers.",
        )
        return held_prompt_id, getattr(prompt_server, "number", 0), len(worker_payloads), {}

    if wait_for_free_participant:
        target_kwargs["prompt_id"] = _register_waiting_run(
            prompt_index,
            job_id_map,
            deferred_candidates,
            master_may_run=not delegate_master,
            master_url=master_url,
            config=config,
            workflow_meta=workflow_meta,
            client_id=client_id,
            use_websocket=use_websocket,
            media_sync_timeout_seconds=media_sync_timeout_seconds,
            execution_trace_id=execution_trace_id,
        )

    try:
        queue_result = await queue_prompt_payload(
            master_prompt,
            workflow_meta,
            client_id,
            include_queue_metadata=True,
            **target_kwargs,
        )
    except Exception:
        if wait_for_free_participant:
            _discard_waiting_run(target_kwargs["prompt_id"])
        if serverless_jobs:
            async with prompt_server.distributed_jobs_lock:
                for job_id in job_id_map.values():
                    prompt_server.distributed_pending_jobs.pop(job_id, None)
        raise
    prompt_id = queue_result["prompt_id"]
    if wait_for_free_participant:
        _waiting_run_queued(prompt_id)
    if manager is not None:
        try:
            await manager.start(execution_trace_id, prompt_id, serverless_jobs)
        except Exception:
            # Wake queued collectors with an explicit error if journal/capacity setup failed.
            await manager.cancel(execution_id=execution_trace_id)
            async with prompt_server.distributed_jobs_lock:
                for prepared in serverless_jobs:
                    # Typed manifests have several sockets per collector, but
                    # each participant gets only one terminal collector event.
                    for job_id in {binding["job_id"] for binding in prepared["manifest"]}:
                        q = prompt_server.distributed_pending_jobs.get(job_id)
                        if q is not None:
                            q.put_nowait({"worker_id": prepared["worker"]["id"], "is_last": True,
                                          "serverless_error": "Serverless job setup failed; no automatic retry was attempted"})
            raise
    prompt_number = queue_result["number"]
    node_errors = queue_result.get("node_errors", {})
    trace_debug(
        execution_trace_id,
        f"Orchestration complete: prompt_id={prompt_id}, dispatched_workers={len(worker_payloads)}, delegate_master={delegate_master}",
    )
    return prompt_id, prompt_number, (len(active_workers) if image_list_mode else len(worker_payloads)), node_errors
