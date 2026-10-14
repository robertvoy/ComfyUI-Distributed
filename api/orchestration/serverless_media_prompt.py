"""Pure compilation of prepared Distributed Collectors for comfy-media-v1 pools.

The legacy image compiler remains separate. This compiler emits one MediaExportV1
execution target per connected Collector socket (``images`` and/or ``audio``),
retaining only its upstream computation. It never imports ComfyUI or
inspects/executes runtime node classes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from graphlib import CycleError, TopologicalSorter

from ...serverless_bridge.contract import SOCKET_KINDS, identifier, validate_bindings
from ...serverless_bridge.image_transfer import DEFAULT_FORMAT, validate_options
from .serverless_prompt import (
    _PRIMITIVE_NODES,
    _REMOVED_SCALAR_NODES,
    _is_link,
    _is_unsupported_distributed,
    _make_scalar_resolver,
    _prune_graph,
)


_COLLECTOR = "DistributedCollector"
# Collector inputs, in output-slot order, that a typed pool returns.
COLLECTOR_SOCKETS = ("images", "audio")
# Collector widget defaults (the saved-workflow contract is legacy_png / 90).
COLLECTOR_DEFAULT_FORMAT = "legacy_png"
COLLECTOR_DEFAULT_QUALITY = 90


def collector_image_policy(inputs):
    """Return the endpoint IMAGE (format, quality) for one Collector.

    legacy_png is the Collector's saved default and names a JSON/base64 framing
    that only applies to direct workers; serverless returns the same 8-bit
    pixels as compressed png. exact stays opt-in. No other format is altered.
    """
    fmt, quality = validate_options(inputs.get("transfer_format", COLLECTOR_DEFAULT_FORMAT),
                                    inputs.get("transfer_quality", COLLECTOR_DEFAULT_QUALITY))
    return ("png" if fmt == "legacy_png" else fmt), quality


def _validate_identifier(value, context):
    # Validate the original token before any lookup. In particular, neither
    # integer node IDs nor whitespace in prepared identities is normalized.
    try:
        identifier(value)
    except ValueError:
        raise ValueError(f"{context} must be a valid nonempty media identifier") from None


def _validate_prompt(prepared_prompt):
    """Validate the entire source graph, including branches later discarded."""
    if not isinstance(prepared_prompt, dict) or not prepared_prompt:
        raise ValueError("prepared prompt must be a nonempty API graph mapping")

    collectors = []
    for node_id, node in prepared_prompt.items():
        _validate_identifier(node_id, "prompt node ID")
        if not isinstance(node, dict):
            raise ValueError(f"prompt node {node_id!r} must be an object")
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type.strip():
            raise ValueError(f"prompt node {node_id!r} must have a nonempty class_type")
        if class_type == _COLLECTOR:
            collectors.append(node_id)
        elif _is_unsupported_distributed(class_type):
            raise ValueError(f"unsupported distributed or batch node: {class_type}")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"prompt node {node_id!r} inputs must be an object")
        if any(not isinstance(name, str) or not name for name in inputs):
            raise ValueError(f"prompt node {node_id!r} input names must be nonempty strings")

    if not collectors:
        raise ValueError("prepared prompt contains no DistributedCollector nodes")

    dependencies = {}
    for node_id, node in prepared_prompt.items():
        parents = set()
        for input_name, value in node.get("inputs", {}).items():
            # Ordinary inputs may contain literal lists. As in the legacy
            # compiler, exactly two elements denotes an API link, not media.
            if not _is_link(value):
                continue
            source_id, slot = value
            context = f"link at node {node_id!r} input {input_name!r}"
            _validate_identifier(source_id, context + " source ID")
            if source_id not in prepared_prompt:
                raise ValueError(f"invalid {context}: source {source_id!r} does not exist")
            if type(slot) is not int or slot < 0:
                raise ValueError(f"invalid {context}: output slot must be a nonnegative integer")
            source_class = prepared_prompt[source_id]["class_type"]
            if source_class in _REMOVED_SCALAR_NODES or source_class in _PRIMITIVE_NODES:
                if slot != 0:
                    raise ValueError(f"invalid output slot {slot} for scalar node {source_id!r}")
            elif source_class == _COLLECTOR and slot >= len(COLLECTOR_SOCKETS):
                raise ValueError(f"invalid output slot {slot} for collector {source_id!r}")
            parents.add(source_id)
        dependencies[node_id] = parents

    for collector_id in collectors:
        inputs = prepared_prompt[collector_id].get("inputs", {})
        collector_image_policy(inputs)
        for field in ("multi_job_id", "worker_id"):
            _validate_identifier(inputs.get(field), f"collector {collector_id!r} {field}")
        connected = [socket for socket in COLLECTOR_SOCKETS if socket in inputs]
        if not connected:
            raise ValueError(f"collector {collector_id!r} requires at least one connected media socket")
        for socket in connected:
            value = inputs[socket]
            if not _is_link(value):
                raise ValueError(f"collector {collector_id!r} {socket} must be linked; use PrimitiveString for literal text")
            if prepared_prompt[value[0]]["class_type"] in _REMOVED_SCALAR_NODES:
                # Export sockets must remain actual links after scalar folding.
                # A PrimitiveString wrapper supports distributed text values.
                raise ValueError(f"collector {collector_id!r} {socket} cannot link directly to a distributed scalar; use a typed node or PrimitiveString")

    # Iterative topological sorting checks ordinary and unused cycles too, not
    # only the primitive chains that the shared scalar resolver traverses.
    try:
        ordered = list(TopologicalSorter(dependencies).static_order())
    except CycleError:
        raise ValueError("cycle in prepared media prompt") from None
    collector_set = set(collectors)
    has_collector_ancestor = {}
    for node_id in ordered:
        has_collector_ancestor[node_id] = any(
            parent in collector_set or has_collector_ancestor[parent]
            for parent in dependencies[node_id]
        )
        if node_id in collector_set and has_collector_ancestor[node_id]:
            raise ValueError("collector-to-collector dependence is unsupported")

    return sorted(collectors)


def _build_exports(source_nodes, collector_ids, request_id):
    bindings = []
    exports = {}
    reserved = set(source_nodes)
    for collector_id in collector_ids:
        inputs = source_nodes[collector_id].get("inputs", {})
        for socket in COLLECTOR_SOCKETS:
            kind = SOCKET_KINDS[socket]
            if socket not in inputs:
                continue
            # Hashing avoids separator ambiguity and keeps long valid source
            # identities within the bridge's identifier length bound. Identity
            # is stable across request IDs and dictionary insertion orders.
            identity = json.dumps([collector_id, socket], separators=(",", ":"))
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            base_id = f"serverless:media:{digest}"
            node_id = base_id
            suffix = 0
            while node_id in reserved:
                suffix += 1
                node_id = f"{base_id}:{suffix}"
            reserved.add(node_id)
            binding = {
                "binding_id": f"b:{digest}",
                "collector_id": collector_id,
                "job_id": inputs["multi_job_id"],
                "participant_id": inputs["worker_id"],
                "node_id": node_id,
                "socket": socket,
                "kind": kind,
            }
            bindings.append(binding)
            exports[node_id] = {
                "class_type": "MediaExportV1",
                "inputs": {
                    "request_id": request_id,
                    "binding_json": json.dumps(binding),
                    socket: copy.deepcopy(inputs[socket]),
                },
            }
            if kind == "IMAGE":
                fmt, quality = collector_image_policy(inputs)
                if fmt != DEFAULT_FORMAT:
                    exports[node_id]["inputs"].update(transfer_format=fmt, transfer_quality=quality)
    validate_bindings(bindings)
    return bindings, exports


def compile_media_prompt(prepared_prompt: dict, request_id: str) -> tuple[dict, list[dict]]:
    """Return a worker graph and canonical typed-media export bindings.

    Each DistributedCollector must have prepared literal ``multi_job_id`` and
    ``worker_id`` identities and at least one linked ``images``/``audio`` input.
    Present media inputs must be API links. Collector transport controls become
    the IMAGE export policy (``legacy_png`` is sent as ``png``); other controls
    are not exported.

    Only MediaExportV1 targets and their ancestors survive. Required ordinary
    output/utility nodes are preserved regardless of their runtime OUTPUT_NODE
    flag; callers must dispatch exactly the returned binding ``node_id`` values
    as partial_execution_targets. Distributed scalar dependencies use the same
    resolver as the legacy compiler. Direct scalar-to-media links need a typed
    wrapper so that each export input remains a link.

    Validation and compilation never modify the caller's source graph. This
    structural compiler does not infer socket types or unknown nodes' output
    counts; ComfyUI performs runtime type validation at the endpoint.
    """
    _validate_identifier(request_id, "request_id")
    collector_ids = _validate_prompt(prepared_prompt)
    bindings, exports = _build_exports(prepared_prompt, collector_ids, request_id)
    compiled = _prune_graph(
        prepared_prompt, bindings, exports, _make_scalar_resolver(prepared_prompt)
    )
    return {node_id: compiled[node_id] for node_id in sorted(compiled)}, bindings
