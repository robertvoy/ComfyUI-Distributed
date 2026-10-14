"""Compile prepared distributed prompts into worker-side ComfyUI image jobs.

This module is deliberately pure: it only copies and rewrites an API prompt
mapping and never imports or executes ComfyUI nodes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re


_SUPPORTED_DISTRIBUTED = {
    "DistributedCollector",
    "DistributedSeed",
    "DistributedValue",
    "DistributedModelName",
}
_REMOVED_SCALAR_NODES = {
    "DistributedSeed",
    "DistributedValue",
    "DistributedModelName",
}
_PRIMITIVE_NODES = {
    "PrimitiveBoolean": False,
    "PrimitiveFloat": 0.0,
    "PrimitiveInt": 0,
    "PrimitiveString": "",
}
_REMOVED_NODE_OUTPUTS = {
    "DistributedCollector": {0, 1},
    "DistributedSeed": {0},
    "DistributedValue": {0},
    "DistributedModelName": {0},
}
_SEED_DEFAULT = 1125899906842


def _is_link(value):
    return isinstance(value, list) and len(value) == 2


def _is_unsupported_distributed(class_type):
    if not isinstance(class_type, str) or class_type in _SUPPORTED_DISTRIBUTED:
        return False
    lowered = class_type.lower()
    return "distributed" in lowered or "batchdivider" in lowered


def _safe_component(value):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    return cleaned[:24] or "empty"


def _filename_prefix(job_id, worker_id, collector_id, kind):
    identity = json.dumps(
        [job_id, worker_id, collector_id, kind],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    parts = (
        "serverless",
        _safe_component(job_id),
        _safe_component(worker_id),
        _safe_component(collector_id),
        kind,
        digest,
    )
    return "_".join(parts)


def _worker_index(worker_id):
    """Mirror nodes.utilities worker-index parsing; None means fallback."""
    try:
        if worker_id.startswith("worker_"):
            return int(worker_id.split("_")[1])
        return int(worker_id)
    except (ValueError, IndexError):
        return None


def _coerce(value, value_type):
    if value_type == "INT":
        return int(float(value))
    if value_type == "FLOAT":
        return float(value)
    return value


def _coerce_safe(value, value_type):
    try:
        return _coerce(value, value_type)
    except (TypeError, ValueError):
        return value


def _stringify_model_name(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, indent=4)
    except (TypeError, ValueError):
        return str(value)


def _model_name_result(text):
    if isinstance(text, list):
        values = [_stringify_model_name(value) for value in text]
        return values[0] if len(values) == 1 else values
    return _stringify_model_name(text)


def _validate_prepared_prompt(prepared_prompt):
    """Validate graph shape, links, collectors, and supported topology."""
    if not isinstance(prepared_prompt, dict):
        raise ValueError("prepared prompt must be an API graph mapping")

    source_nodes = {}
    for raw_id, node in prepared_prompt.items():
        node_id = str(raw_id)
        if node_id in source_nodes:
            raise ValueError("prompt has duplicate node IDs after string normalization")
        if not isinstance(node, dict):
            raise ValueError(f"prompt node {node_id!r} must be an object")
        class_type = node.get("class_type")
        if not isinstance(class_type, str):
            raise ValueError(f"prompt node {node_id!r} is missing a class_type")
        if _is_unsupported_distributed(class_type):
            raise ValueError(f"unsupported distributed or batch node: {class_type}")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"prompt node {node_id!r} inputs must be an object")
        source_nodes[node_id] = node

    if not source_nodes:
        raise ValueError("prepared prompt contains no nodes")

    # Validate every API link before any graph rewriting. Slots on the nodes
    # removed by this compiler have a known schema and are checked exactly.
    for node_id, node in source_nodes.items():
        for input_name, value in node.get("inputs", {}).items():
            if not _is_link(value):
                continue
            upstream_id, output_slot = value
            upstream_id = str(upstream_id)
            if upstream_id not in source_nodes:
                raise ValueError(
                    f"invalid link at node {node_id!r} input {input_name!r}: "
                    f"source {upstream_id!r} does not exist"
                )
            if isinstance(output_slot, bool) or not isinstance(output_slot, int) or output_slot < 0:
                raise ValueError(
                    f"invalid link at node {node_id!r} input {input_name!r}: "
                    "output slot must be a nonnegative integer"
                )
            upstream_class = source_nodes[upstream_id]["class_type"]
            valid_slots = _REMOVED_NODE_OUTPUTS.get(upstream_class)
            if valid_slots is not None and output_slot not in valid_slots:
                raise ValueError(
                    f"invalid output slot {output_slot} for removed node "
                    f"{upstream_class} {upstream_id!r}"
                )

    collector_ids = sorted(
        (node_id for node_id, node in source_nodes.items() if node["class_type"] == "DistributedCollector"),
        key=str,
    )
    if not collector_ids:
        raise ValueError("prepared prompt contains no DistributedCollector nodes")

    # Only image collectors are supported by the current generic handler.
    for collector_id in collector_ids:
        inputs = source_nodes[collector_id].get("inputs", {})
        if "audio" in inputs:
            raise ValueError("collector audio input is unsupported by this image-only compiler")
        if not _is_link(inputs.get("images")):
            raise ValueError(f"collector {collector_id!r} is missing a linked image input")
        source_id = str(inputs["images"][0])
        source_class = source_nodes[source_id]["class_type"]
        if source_class in _REMOVED_SCALAR_NODES:
            raise ValueError(f"collector {collector_id!r} image input must link to an image node")

    # A collector cannot be used, directly or through processing, as another
    # collector's input: this first compiler handles disconnected jobs only.
    for collector_id in collector_ids:
        pending = [str(source_nodes[collector_id]["inputs"]["images"][0])]
        seen = set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            if source_nodes[node_id]["class_type"] == "DistributedCollector":
                raise ValueError("collector-to-collector dependence is unsupported")
            for value in source_nodes[node_id].get("inputs", {}).values():
                if _is_link(value):
                    pending.append(str(value[0]))

    return source_nodes, collector_ids


def _make_scalar_resolver(source_nodes):
    """Build pure resolvers for distributed scalar nodes and stock primitives."""
    # Resolve only stock primitive nodes. A visited set makes malformed
    # primitive-to-primitive cycles fail clearly instead of recursing forever.
    def resolve_primitive(node_id, visiting=None):
        node_id = str(node_id)
        node = source_nodes[node_id]
        class_type = node["class_type"]
        if class_type not in _PRIMITIVE_NODES:
            raise ValueError("linked value must be literal or supported PrimitiveInt/Float/String/Boolean")
        if visiting is None:
            visiting = set()
        if node_id in visiting:
            raise ValueError(f"cycle while resolving primitive value at node {node_id!r}")
        visiting.add(node_id)
        value = node.get("inputs", {}).get("value", _PRIMITIVE_NODES[class_type])
        if _is_link(value):
            parent_id, output_slot = str(value[0]), value[1]
            if output_slot != 0:
                raise ValueError(f"invalid output slot {output_slot} for primitive node {parent_id!r}")
            if source_nodes[parent_id]["class_type"] not in _PRIMITIVE_NODES:
                raise ValueError("linked primitive value must come from a stock Primitive node")
            result = resolve_primitive(parent_id, visiting)
        else:
            result = copy.deepcopy(value)
        visiting.remove(node_id)
        return result

    def resolve_literal_or_primitive(value, context):
        if _is_link(value):
            source_id, output_slot = str(value[0]), value[1]
            if output_slot != 0 or source_nodes[source_id]["class_type"] not in _PRIMITIVE_NODES:
                raise ValueError(f"{context} must be a literal or supported Primitive link")
            try:
                return resolve_primitive(source_id)
            except ValueError as exc:
                if "cycle" in str(exc):
                    raise
                raise ValueError(f"{context} must be a literal or supported Primitive link") from None
        return copy.deepcopy(value)

    def resolve_hidden(node_id, name, default):
        value = source_nodes[node_id].get("inputs", {}).get(name, default)
        if _is_link(value):
            raise ValueError(f"{name} on distributed node {node_id!r} must be a literal")
        return copy.deepcopy(value)

    resolved_distributed = {}

    def resolve_distributed(node_id, output_slot):
        node_id = str(node_id)
        class_type = source_nodes[node_id]["class_type"]
        if output_slot != 0:
            raise ValueError(f"invalid output slot {output_slot} for {class_type} {node_id!r}")
        cache_key = (node_id, output_slot)
        if cache_key in resolved_distributed:
            return copy.deepcopy(resolved_distributed[cache_key])
        inputs = source_nodes[node_id].get("inputs", {})
        if class_type == "DistributedSeed":
            seed = resolve_literal_or_primitive(inputs.get("seed", _SEED_DEFAULT), f"seed on {node_id!r}")
            is_worker = resolve_hidden(node_id, "is_worker", False)
            worker_id = resolve_hidden(node_id, "worker_id", "")
            if is_worker:
                worker_index = _worker_index(worker_id)
                if worker_index is not None:
                    seed = seed + worker_index + 1
            result = seed
        elif class_type == "DistributedValue":
            default_value = resolve_literal_or_primitive(
                inputs.get("default_value", ""), f"default_value on {node_id!r}"
            )
            worker_values = resolve_literal_or_primitive(
                inputs.get("worker_values", "{}"), f"worker_values on {node_id!r}"
            )
            is_worker = resolve_hidden(node_id, "is_worker", False)
            worker_id = resolve_hidden(node_id, "worker_id", "")
            try:
                values = json.loads(worker_values) if isinstance(worker_values, str) else worker_values
            except json.JSONDecodeError:
                values = {}
            if not isinstance(values, dict):
                values = {}
            value_type = values.get("_type", "STRING")
            fallback = _coerce_safe(default_value, value_type)
            result = fallback
            if is_worker:
                worker_index = _worker_index(worker_id)
                if worker_index is not None:
                    raw_value = values.get(str(worker_index + 1), "")
                    if raw_value:
                        try:
                            result = _coerce(raw_value, value_type)
                        except (ValueError, IndexError):
                            result = fallback
        elif class_type == "DistributedModelName":
            text = resolve_literal_or_primitive(
                inputs.get("text", ""), f"DistributedModelName text on {node_id!r}"
            )
            result = _model_name_result(text)
        else:
            raise ValueError(f"unsupported distributed node: {class_type}")
        resolved_distributed[cache_key] = copy.deepcopy(result)
        return result

    return resolve_distributed


def _build_savers(source_nodes, collector_ids):
    """Construct stable SaveImage nodes and their collector metadata."""
    # The prepared collector values define the artifact identity. The fallback
    # for absent hidden values is stable and retains the collector's job scope.
    collector_specs = []
    max_numeric_id = 0
    for node_id in source_nodes:
        try:
            max_numeric_id = max(max_numeric_id, int(node_id))
        except ValueError:
            pass
    next_numeric_id = max_numeric_id + 1
    for collector_id in collector_ids:
        inputs = source_nodes[collector_id].get("inputs", {})
        raw_job_id = inputs.get("multi_job_id", collector_id)
        if _is_link(raw_job_id):
            raise ValueError(f"collector {collector_id!r} multi_job_id must be a literal")
        job_id = str(collector_id if raw_job_id in (None, "") else raw_job_id)
        raw_worker_id = inputs.get("worker_id", "")
        if _is_link(raw_worker_id):
            raise ValueError(f"collector {collector_id!r} worker_id must be a literal")
        worker_id = "" if raw_worker_id is None else str(raw_worker_id)
        saver_id = str(next_numeric_id)
        next_numeric_id += 1
        prefix = _filename_prefix(job_id, worker_id, collector_id, "image")
        collector_specs.append(
            {
                "collector_id": collector_id,
                "job_id": job_id,
                "node_id": saver_id,
                "kind": "image",
                "prefix": prefix,
                "source_link": copy.deepcopy(inputs["images"]),
            }
        )

    saver_nodes = {}
    for spec in collector_specs:
        saver_nodes[spec["node_id"]] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": spec["source_link"],
                "filename_prefix": spec["prefix"],
            },
            "_meta": {"title": f"Serverless output: collector {spec['collector_id']} (image)"},
        }

    return collector_specs, saver_nodes


def _prune_graph(source_nodes, collector_specs, saver_nodes, resolve_distributed):
    """Keep saver ancestors, replacing distributed scalar links with literals."""
    all_nodes = {node_id: copy.deepcopy(node) for node_id, node in source_nodes.items()}
    all_nodes.update(copy.deepcopy(saver_nodes))
    needed = set()
    pending = [spec["node_id"] for spec in collector_specs]
    while pending:
        node_id = pending.pop()
        if node_id in needed:
            continue
        needed.add(node_id)
        node = all_nodes[node_id]
        if node["class_type"] == "DistributedCollector":
            raise ValueError("collector-to-collector dependence is unsupported")
        for input_name, value in list(node.get("inputs", {}).items()):
            if not _is_link(value):
                continue
            upstream_id, output_slot = str(value[0]), value[1]
            upstream_class = source_nodes[upstream_id]["class_type"]
            if upstream_class in _REMOVED_SCALAR_NODES:
                node["inputs"][input_name] = resolve_distributed(upstream_id, output_slot)
            else:
                pending.append(upstream_id)

    compiled = {
        node_id: copy.deepcopy(node)
        for node_id, node in all_nodes.items()
        if node_id in needed
    }
    if any("distributed" in node["class_type"].lower() for node in compiled.values()):
        raise ValueError("compiled prompt still contains a distributed dependency")

    return compiled


def compile_serverless_prompt(prepared_prompt: dict) -> tuple[dict, list[dict]]:
    """Compile a prepared worker prompt to an image-only serverless graph.

    Every DistributedCollector's original ``images`` input is connected to a
    new SaveImage node. Only those savers and their upstream dependencies are
    retained. DistributedSeed/Value/ModelName links are resolved as scalar
    literals; unsupported node execution is never attempted.
    """
    source_nodes, collector_ids = _validate_prepared_prompt(prepared_prompt)
    resolve_distributed = _make_scalar_resolver(source_nodes)
    collector_specs, saver_nodes = _build_savers(source_nodes, collector_ids)
    compiled = _prune_graph(
        source_nodes,
        collector_specs,
        saver_nodes,
        resolve_distributed,
    )
    manifest = [
        {key: spec[key] for key in ("collector_id", "job_id", "node_id", "kind", "prefix")}
        for spec in collector_specs
    ]
    return compiled, manifest
