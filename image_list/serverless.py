"""Image-list items on explicitly selected comfy-media-v1 pools.

Only graph/media adaptation lives here. The shared manager owns admission,
provider submission, durable reservations, cancellation and atomic decoding.
A provider terminal state is not proof that an arbitrary endpoint stopped its
local Comfy prompt: the selected profile must honour its supervision contract.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import uuid

from . import runtime
from ..api.orchestration.runpod_client import serialize_request
from ..api.orchestration.serverless_media_prompt import compile_media_prompt
from ..api.orchestration.serverless_runtime import PER_WORKER_JOB_LIMIT, get_manager
from ..api.orchestration.serverless_typed_inputs import prepare_media_inputs, _artifact_id
from ..serverless_bridge import codec, contract
from ..utils.serverless_config import validate_serverless_worker


def _identity(value):
    contract.identifier(value)
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Unsafe image-list media identifier")
    return value


def resolve_serverless_workers(config, requested_ids):
    """Return pool configurations, never jobs_per_run-expanded participants.

    None means enabled defaults; an explicit selection includes disabled pools.
    Direct worker IDs may coexist in that selection and are left to their own
    resolver. Unknown, duplicate configured and malformed identities fail closed.
    """
    workers = (config or {}).get("workers", [])
    indexed = {}
    for worker in workers:
        wid = worker.get("id")
        if not isinstance(wid, str) or not wid or wid != wid.strip():
            raise ValueError("Invalid configured image-list worker ID")
        if wid in indexed:
            raise ValueError("Duplicate configured image-list worker ID")
        indexed[wid] = worker
    requested = None
    if requested_ids is not None:
        if not isinstance(requested_ids, (list, tuple, set)):
            raise ValueError("Image-list worker selection must be a list of exact IDs")
        if any(not isinstance(wid, str) or not wid or wid != wid.strip() for wid in requested_ids):
            raise ValueError("Invalid selected image-list worker ID")
        requested = set(requested_ids)
        if requested - indexed.keys():
            raise ValueError("Unknown explicitly selected image-list worker ID(s)")
    result = []
    for wid, worker in indexed.items():
        if requested is None:
            if worker.get("enabled", False) is not True:
                continue
        elif wid not in requested:
            continue
        if str(worker.get("type") or "").strip().lower() != "serverless":
            continue
        clean = validate_serverless_worker(worker)
        _identity(wid)
        if clean.get("media_profile") != contract.PROFILE:
            raise ValueError(f"Image-list serverless pool {wid!r} requires comfy-media-v1")
        # Pool IDs, not virtual legacy participant IDs, own capacity and secrets.
        if "pool_id" in clean and clean["pool_id"] != wid:
            raise ValueError("Image-list selection requires a pool, not an expanded participant")
        clean["pool_id"] = wid
        result.append(clean)
    return result


def _link(value, prompt):
    if not isinstance(value, (list, tuple)):
        return None
    if (len(value) != 2 or not isinstance(value[0], str)
            or type(value[1]) is not int or value[1] < 0):
        raise ValueError("Invalid image-list graph link")
    _identity(value[0])
    if value[0] not in prompt:
        raise ValueError("Image-list graph link source is missing")
    return value[0], value[1]


def _new_node_id(graph, label):
    base = "image-list:input:" + label
    candidate, suffix = base, 0
    while candidate in graph:
        suffix += 1
        candidate = f"{base}:{suffix}"
    return candidate


def _merge_artifacts(*groups):
    indexed = {}
    for group in groups:
        for artifact in group:
            contract.validate_artifact(artifact)
            aid = artifact["id"]
            if aid in indexed and indexed[aid] != artifact:
                raise ValueError("Conflicting image-list input artifact identity")
            indexed[aid] = artifact
    if (len(indexed) > contract.MAX_ARTIFACTS or
            sum(a["byte_length"] for a in indexed.values()) > contract.MAX_TOTAL_BYTES):
        raise ValueError("Image-list media inputs exceed artifact limits")
    return [indexed[aid] for aid in sorted(indexed)]


async def prepare_serverless_item(prompt, region, item, worker, *, request_id, job_id):
    """Compile one immutable Start/End region to an endpoint-native media job.

    Tensor Start outputs use exact codec artifacts and references to existing
    consumer inputs. PrimitiveString placeholders keep exporter sockets linked
    until the endpoint handler injects MediaImportV1 and prunes unused ancestors.
    Scalar values remain typed literals, except exported text stays linked.
    """
    _identity(request_id)
    _identity(job_id)
    worker = validate_serverless_worker(worker)
    _identity(worker["id"])
    if worker.get("media_profile") != contract.PROFILE:
        raise ValueError("Image-list serverless execution requires comfy-media-v1")
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("Image-list prompt must be an API graph")
    # Check original tokens before looking them up or pruning any nodes. Do not
    # let int/bool IDs or slots become valid through string/int normalization.
    for node_id, node in prompt.items():
        _identity(node_id)
        if not isinstance(node, dict) or not isinstance(node.get("inputs", {}), dict):
            raise ValueError("Invalid image-list prompt node")
        for name, value in node.get("inputs", {}).items():
            _identity(name)
            _link(value, prompt)
    start, end = region["start_id"], region["end_id"]
    _identity(start)
    _identity(end)
    if (prompt[start].get("class_type") != runtime.START_CLASS or
            prompt[end].get("class_type") != runtime.END_CLASS):
        raise ValueError("Image-list region has mismatched Start/End identities")
    if _link(prompt[end].get("inputs", {}).get("context"), prompt) != (start, runtime.CONTEXT_OUT):
        raise ValueError("Image-list End context must reference its exact Start")
    ids = region["node_ids"]
    for node_id in ids:
        _identity(node_id)
    if not {start, end} <= set(ids):
        raise ValueError("Image-list region is missing its boundary nodes")
    graph = {nid: copy.deepcopy(prompt[nid]) for nid in sorted(ids)}
    end_inputs = graph[end].get("inputs", {})
    from ..serverless_bridge.image_transfer import DEFAULT_FORMAT, DEFAULT_QUALITY, validate_options
    transfer_format, transfer_quality = validate_options(
        end_inputs.get("transfer_format", DEFAULT_FORMAT),
        end_inputs.get("transfer_quality", DEFAULT_QUALITY))
    if "images" not in end_inputs:
        raise ValueError("Image-list End requires images")
    graph[end] = {"class_type": "DistributedMediaCollector", "inputs": {
        **{key: end_inputs[key] for key in ("images", "masks", "texts") if key in end_inputs},
        "multi_job_id": job_id, "worker_id": worker["id"]}}
    if transfer_format != DEFAULT_FORMAT:
        graph[end]["inputs"].update(transfer_format=transfer_format, transfer_quality=transfer_quality)
    del graph[start]
    if type(item.get("index")) is not int or item["index"] < 0:
        raise ValueError("Image-list item index must be a nonnegative integer")
    image = item.get("image")
    if (not isinstance(image, runtime._torch().Tensor) or image.ndim != 4
            or image.shape[0] != 1 or image.shape[-1] not in (3, 4)):
        raise ValueError("Image-list item must contain exactly one RGB/RGBA image frame")
    values = item.get("values", {})
    if not isinstance(values, dict) or any(name not in runtime.VALUE_NAMES for name in values):
        raise ValueError("Invalid image-list value_N slots")
    materialized, artifacts = {}, {}
    placeholders = {}

    def replacement(slot):
        if slot in materialized:
            return copy.deepcopy(materialized[slot])
        if slot == runtime.IMAGE_OUT:
            value, kind = image, "IMAGE"
        elif slot == runtime.MASK_OUT:
            value, kind = item.get("mask"), "MASK"
        elif slot == runtime.INDEX_OUT:
            value, kind = item["index"], "INT"
        elif runtime.VALUE_OUT_BASE <= slot < runtime.VALUE_OUT_BASE + runtime.MAX_VALUE_SLOTS:
            name = runtime.VALUE_NAMES[slot - runtime.VALUE_OUT_BASE]
            value = values.get(name)
            kind = runtime.value_kind(value, name)
            if kind == "TENSOR":
                if value.ndim == 4:
                    kind = "IMAGE"
                elif value.ndim == 3:
                    kind = "MASK"
                else:
                    raise ValueError(f"{name} serverless tensor requires IMAGE [B,H,W,C] or MASK [B,H,W]")
        else:
            raise ValueError("Unsupported Start context/control output in endpoint computation")
        if value is None:
            raise ValueError("Connected Start output is empty; None cannot be submitted")
        if kind in ("IMAGE", "MASK"):
            if kind == "MASK" and slot == runtime.MASK_OUT:
                if value.ndim != 3 or value.shape[0] != 1 or value.shape[1:3] != image.shape[1:3]:
                    raise ValueError("Image-list item mask must be paired with its image grid")
            if value.numel() * value.element_size() > contract.MAX_ARTIFACT_BYTES:
                raise ValueError("Image-list tensor exceeds media artifact byte limit")
            descriptor, payload = codec.encode_payload(kind, value.detach().to("cpu"),
                                                        max_bytes=contract.MAX_ARTIFACT_BYTES)
            aid = _artifact_id(kind, descriptor, payload)
            if aid not in artifacts and (len(artifacts) >= contract.MAX_ARTIFACTS or
                    len(payload) + sum(a["byte_length"] for a in artifacts.values()) > contract.MAX_TOTAL_BYTES):
                raise ValueError("Image-list media inputs exceed artifact limits")
            artifacts[aid] = contract.artifact_from_payload(aid, kind, descriptor, payload)
            placeholder = _new_node_id(graph, str(slot))
            graph[placeholder] = {"class_type": "PrimitiveString", "inputs": {"value": ""}}
            placeholders[placeholder] = aid
            output = [placeholder, 0]
        else:
            # value_kind rejects objects/None/nonfinite values and bool remains bool.
            runtime.value_kind(value, "Start output")
            output = value
        materialized[slot] = output
        return copy.deepcopy(output)

    for node_id in list(graph):
        inputs = graph[node_id].setdefault("inputs", {})
        for name, value in list(inputs.items()):
            link = _link(value, prompt)
            if link and link[0] == start:
                inputs[name] = replacement(link[1])
            elif link:
                inputs[name] = list(link)
    # Start text/index/generic scalar substitutions may have removed End's link.
    text = graph[end]["inputs"].get("texts")
    if "texts" in graph[end]["inputs"] and not isinstance(text, list):
        if not isinstance(text, str):
            raise ValueError("Image-list End texts must be STRING")
        nid = _new_node_id(graph, "text")
        graph[nid] = {"class_type": "PrimitiveString", "inputs": {"value": text}}
        graph[end]["inputs"]["texts"] = [nid, 0]
    workflow, manifest = compile_media_prompt(graph, request_id)
    refs = []
    for nid, node in workflow.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and value[0] in placeholders:
                refs.append({"node_id": nid, "input_name": name, "artifact_id": placeholders[value[0]]})
    # Only actual surviving consumer slots count as uses of an input artifact.
    used = {ref["artifact_id"] for ref in refs}
    images, fixed_artifacts, fixed_refs = await prepare_media_inputs(workflow)
    merged = _merge_artifacts([artifacts[aid] for aid in used], fixed_artifacts)
    refs.extend(fixed_refs)
    media = {"version": contract.VERSION, "profile": contract.PROFILE,
             "request_id": request_id, "artifacts": merged,
             "inputs": sorted(refs, key=lambda ref: (ref["node_id"], ref["input_name"])),
             "outputs": manifest}
    # Includes handler identity/link/ref validation and aggregate + exact JSON caps.
    serialize_request(workflow, images, worker["timeout_seconds"], media=media)
    return {"worker": worker, "workflow": workflow, "images": images,
            "manifest": manifest, "media": media}


class ServerlessSlots:
    """Nonblocking slot adapter; completion means the manager task has drained."""
    def __init__(self, state, region, workers, manager, event):
        self.state, self.region, self.manager, self.event = state, region, manager, event
        self.workers = {worker["id"]: worker for worker in workers}
        # The manager remains the atomic authority at start(). Bound this local
        # scheduler too, including other callers' currently pending tasks.
        available = max(0, 64 - sum(not task.done() for task in manager.tasks.values())) if manager else 0
        self.slots = [("serverless", worker["id"], index)
                      for index in range(PER_WORKER_JOB_LIMIT) for worker in workers][:available]
        if workers and not self.slots:
            raise ValueError("Distributed already has too many pending serverless jobs (limit 64)")
        digest = hashlib.sha256(state.context_id.encode()).hexdigest()
        self._scope = f"image-list:{digest}:{uuid.uuid4().hex}"
        self._active = {}
        self._queues = {}
        self._executions = set()
        self._assigned = set()
        self._submissions = set()
        self._closed = False
        self._close_task = None

    async def dispatch(self, slot, index):
        if self._closed:
            raise RuntimeError("Image-list serverless controller is closed")
        if slot not in self.slots or slot in self._active:
            raise RuntimeError("Unknown or occupied serverless image-list slot")
        if type(index) is not int or not 0 <= index < len(self.state.items) or index in self._assigned:
            raise ValueError("Invalid or duplicate serverless image-list index")
        if self.state.items[index].get("index") != index:
            raise ValueError("Serverless image-list item index mismatch")
        self._assigned.add(index)
        identity = f"{self._scope}:{index}"
        handle = {"index": index, "execution_id": identity, "job_id": identity,
                  "worker_id": slot[1], "task": None}
        self._active[slot] = handle
        # Own execution identity BEFORE start, even if its caller loses the handle.
        self._executions.add(identity)
        task = asyncio.create_task(self._submit(handle))
        self._submissions.add(task)
        try:
            await asyncio.shield(task)
            return handle
        except BaseException:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._remove_queue(handle["job_id"])
            await self.manager.cancel(execution_id=identity)
            raise
        finally:
            self._submissions.discard(task)

    async def _submit(self, handle):
        job = await prepare_serverless_item(
            self.state.original_prompt, self.region, self.state.items[handle["index"]],
            self.workers[handle["worker_id"]], request_id=handle["execution_id"], job_id=handle["job_id"])
        if self._closed:
            raise RuntimeError("Image-list serverless controller closed during preparation")
        self.manager.ensure_capacity(1)
        queue = asyncio.Queue()
        async with self.manager.server.distributed_jobs_lock:
            if self._closed:
                raise RuntimeError("Image-list serverless controller is closed")
            if handle["job_id"] in self.manager.server.distributed_pending_jobs:
                raise RuntimeError("Duplicate serverless image-list queue identity")
            self._queues[handle["job_id"]] = queue
            self.manager.server.distributed_pending_jobs[handle["job_id"]] = queue
        handle["sockets"] = {binding["socket"] for binding in job["manifest"]}
        record_ids = await self.manager.start(handle["execution_id"], self.state.prompt_id, [job])
        if len(record_ids) != 1 or record_ids[0] not in self.manager.tasks:
            raise RuntimeError("Missing serverless image-list lifecycle task")
        handle["task"] = self.manager.tasks[record_ids[0]]
        handle["task"].add_done_callback(lambda _: self.event.set())

    def _remove_queue(self, job_id):
        queue = self._queues.pop(job_id, None)
        # Same event-loop turn, without an await: never delete a replacement queue.
        if queue is not None and self.manager.server.distributed_pending_jobs.get(job_id) is queue:
            del self.manager.server.distributed_pending_jobs[job_id]

    def poll(self, inflight):
        if self._closed:
            raise RuntimeError("Image-list serverless controller is closed")
        for slot, handle in inflight.items():
            if slot not in self.slots:
                continue  # Direct/master slots belong to the surrounding scheduler.
            if self._active.get(slot) is not handle:
                raise RuntimeError("Mismatched or duplicate serverless image-list completion handle")
            task = handle["task"]
            if task is None or not task.done():
                continue
            if task.cancelled():
                raise RuntimeError("Serverless image-list task was cancelled")
            if task.exception() is not None:
                raise RuntimeError("Serverless image-list lifecycle task failed") from task.exception()
            queue = self._queues.get(handle["job_id"])
            if queue is None or self.manager.server.distributed_pending_jobs.get(handle["job_id"]) is not queue:
                raise RuntimeError("Missing serverless image-list result queue")
            if queue.qsize() != 1:
                raise RuntimeError("Missing or duplicate serverless image-list terminal result")
            message = queue.get_nowait()
            if (not isinstance(message, dict) or message.get("worker_id") != handle["worker_id"]
                    or message.get("is_last") is not True):
                raise RuntimeError("Mismatched serverless image-list result identity")
            if "serverless_error" in message:
                raise RuntimeError(f"Serverless image-list item {handle['index']} failed: {message['serverless_error']}")
            media = message.get("media")
            if not isinstance(media, dict) or set(media) != handle["sockets"]:
                raise RuntimeError("Missing or unexpected serverless image-list result sockets")
            if any(not isinstance(values, list) or len(values) != 1 or values[0] is None
                   for values in media.values()):
                raise RuntimeError("Serverless image-list result requires exactly one value per socket")
            result = runtime.normalize_worker_result(media["images"], media.get("masks"), media.get("texts"))
            if result["mask"] is not None and result["mask"].shape[1:3] != result["image"].shape[1:3]:
                raise ValueError("Serverless image-list result mask is not paired with the image grid")
            self._remove_queue(handle["job_id"])
            del self._active[slot]
            return slot, handle["index"], result
        return None

    async def _close(self):
        # Stop preparation/start before cancellation discovery, then retire exact
        # destinations before manager callbacks can publish cancellation errors.
        submissions = tuple(self._submissions)
        for task in submissions:
            if not task.done():
                task.cancel()
        await asyncio.gather(*submissions, return_exceptions=True)
        for job_id in list(self._queues):
            self._remove_queue(job_id)
        await asyncio.gather(*(self.manager.cancel(execution_id=eid) for eid in self._executions))
        self._active.clear()

    async def close(self):
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError


async def create_serverless_slots(state, region, config, event):
    workers = resolve_serverless_workers(config, state.worker_ids)
    return ServerlessSlots(state, region, workers, get_manager() if workers else None, event)
