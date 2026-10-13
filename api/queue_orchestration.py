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
from ..image_list.runtime import validate_image_list_regions


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


def _is_load_balance_enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _prompt_requests_load_balance(prompt_index):
    collector_ids = (prompt_index.nodes_for_class("DistributedCollector")
                     + prompt_index.nodes_for_class("DistributedMediaCollector"))
    for node_id in collector_ids:
        inputs = prompt_index.inputs_by_node.get(node_id, {})
        if _is_load_balance_enabled(inputs.get("load_balance", False)):
            return True
    return False


def _validate_media_workers(prompt_index, workers):
    """Reject incompatible selections before probing, pruning or dispatching.

    Typed collectors currently have only a master/serverless transport. Check
    all selected workers, not just reachable or load-balanced participants, so
    a bad selection cannot silently turn into a different execution mode.
    """
    typed = bool(prompt_index.nodes_for_class("DistributedMediaCollector"))
    if typed and (prompt_index.nodes_for_class("DistributedCollector")
                  or prompt_index.nodes_for_class("UltimateSDUpscaleDistributed")):
        raise ValueError(
            "DistributedMediaCollector cannot be mixed with DistributedCollector or "
            "UltimateSDUpscaleDistributed in one prompt"
        )
    for worker in workers:
        if worker.get("type") != "serverless":
            if typed:
                raise ValueError(
                    "DistributedMediaCollector does not support selected direct workers; "
                    "use master execution or comfy-media-v1 serverless pools"
                )
            continue
        profile = worker.get("media_profile", "legacy-image-v1")
        if profile not in ("legacy-image-v1", "comfy-media-v1"):
            raise ValueError("Unsupported serverless media_profile")
        if typed != (profile == "comfy-media-v1"):
            raise ValueError(
                "DistributedMediaCollector requires comfy-media-v1 serverless pools; "
                "legacy collectors require legacy-image-v1"
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
        _validate_media_workers(prompt_index, [worker])
        worker_prompt = prompt_index.copy_prompt()
        if worker.get("type") == "serverless":
            from .orchestration.runpod_client import serialize_request
            if worker.get("media_profile", "legacy-image-v1") == "comfy-media-v1":
                from ..serverless_bridge.contract import PROFILE, VERSION
                from .orchestration.serverless_media_prompt import compile_media_prompt
                from .orchestration.serverless_typed_inputs import prepare_media_inputs
                # The legacy pruner only recognizes Collector/Upscale roots.
                # The typed compiler validates the complete private graph and
                # prunes it itself, retaining every connected typed branch.
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
    prompt_index = PromptIndex(prompt_obj)
    typed_media_mode = bool(prompt_index.nodes_for_class("DistributedMediaCollector"))
    if typed_media_mode:
        from ..serverless_bridge.contract import identifier
        # PromptIndex's JSON copy would otherwise turn integer keys into strings.
        for node_id in prompt_obj:
            identifier(node_id)
    image_list_start_ids = prompt_index.nodes_for_class("DistributedImageListStart")
    image_list_mode = bool(image_list_start_ids)
    if image_list_mode:
        # Validate before probing or queueing anything so malformed boundaries and
        # old worker releases fail before any upstream image is materialized.
        other_distributed = (
            prompt_index.nodes_for_class("DistributedCollector")
            + prompt_index.nodes_for_class("DistributedMediaCollector")
            + prompt_index.nodes_for_class("UltimateSDUpscaleDistributed")
        )
        if other_distributed:
            raise ValueError(
                "Distributed Image List v1 cannot be mixed with Distributed Collector, "
                "DistributedMediaCollector or Ultimate SD Upscale nodes in one prompt"
            )
        validate_image_list_regions(prompt_obj)
        configured_serverless = [
            str(worker.get("id")) for worker in config.get("workers", [])
            if worker.get("enabled", False)
            and str(worker.get("type") or "local").lower() == "serverless"
            and (requested_ids is None or str(worker.get("id")) in {str(x) for x in requested_ids})
        ]
        if configured_serverless:
            trace_debug(
                execution_trace_id,
                "Image List v1 explicitly excludes unsupported serverless pools: "
                + ", ".join(configured_serverless),
            )
        workers = _resolve_direct_workers_for_image_list(config, requested_ids)
        if not workers and delegate_master:
            raise RuntimeError(
                "Distributed Image List in orchestrator-only mode requires a selected direct worker; "
                "serverless pools are unsupported"
            )
    else:
        if typed_media_mode and requested_ids is not None:
            configured_ids = {worker.get("id") for worker in config.get("workers", [])}
            if any(worker_id not in configured_ids for worker_id in requested_ids):
                raise ValueError("Unknown explicitly selected media worker ID")
        workers = _resolve_enabled_workers(config, requested_ids)
        _validate_media_workers(prompt_index, workers)
        if typed_media_mode and delegate_master and not workers:
            raise RuntimeError("Delegate-only DistributedMediaCollector requires a selected comfy-media-v1 pool")
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

    # Keep the legacy Collector fallback. Image-list and typed-media delegate
    # modes must not silently enable master participation.
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
    if typed_media_mode and delegate_master and not active_workers:
        raise RuntimeError("No selected comfy-media-v1 worker is reachable for delegate-only DistributedMediaCollector")
    if image_list_mode:
        from .orchestration.dispatch import preflight_image_list_workers
        if not active_workers and delegate_master:
            raise RuntimeError(
                "No selected direct worker is reachable for Distributed Image List in orchestrator-only mode"
            )
        idle_ids = (await preflight_image_list_workers(active_workers, require_idle=delegate_master)
                    if active_workers else {})
        active_workers = [worker for worker in active_workers if worker["id"] in idle_ids]
        if not active_workers and delegate_master:
            raise RuntimeError("No feature-compatible direct worker is idle in orchestrator-only mode")
    else:
        delegate_master = selected_delegate_master

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
            # Master selected as least busy; run master workload only.
            active_workers = []
            delegate_master = False
            trace_debug(
                execution_trace_id,
                "Load-balance selected master for execution (workers skipped).",
            )
        elif selected_worker is not None:
            active_workers = [selected_worker]
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
        )
        return (
            queue_result["prompt_id"],
            queue_result["number"],
            0,
            queue_result.get("node_errors", {}),
        )

    master_prompt = prompt_index.copy_prompt()
    master_prompt = apply_participant_overrides(
        master_prompt,
        "master",
        enabled_ids,
        job_id_map,
        master_url,
        delegate_master,
        prompt_index,
        workflow_meta=workflow_meta,
    )

    if delegate_master and not image_list_mode:
        collector_ids = (find_nodes_by_class(master_prompt, "DistributedCollector")
                         + find_nodes_by_class(master_prompt, "DistributedMediaCollector"))
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

    if regular_payloads:
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

    try:
        queue_result = await queue_prompt_payload(
            master_prompt,
            workflow_meta,
            client_id,
            include_queue_metadata=True,
        )
    except Exception:
        if serverless_jobs:
            async with prompt_server.distributed_jobs_lock:
                for job_id in job_id_map.values():
                    prompt_server.distributed_pending_jobs.pop(job_id, None)
        raise
    prompt_id = queue_result["prompt_id"]
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
