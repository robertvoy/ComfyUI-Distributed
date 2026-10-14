"""Runtime primitives for the V3 dynamic image-list feature.

The network protocol transports typed float tensors as bounded JSON/base64 payloads;
no Python object deserialization or arbitrary-path access is used.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# v2: Start carries generic value_N slots instead of fixed reference/text/strength/seed fields.
PROTOCOL_VERSION = 2
START_CLASS = "DistributedImageListStart"
END_CLASS = "DistributedImageListEnd"
START_NODE_ID = "DistributedImageListStart"
END_NODE_ID = "DistributedImageListEnd"
# Start output indexes are a stable internal prompt protocol. Fixed outputs come
# first; value_1..value_N follow so the frontend can hide unused trailing slots
# without renumbering any connected output.
IMAGE_OUT, MASK_OUT, INDEX_OUT, CONTEXT_OUT = range(4)
VALUE_OUT_BASE = 4
MAX_VALUE_SLOTS = 10
VALUE_NAMES = tuple(f"value_{i}" for i in range(1, MAX_VALUE_SLOTS + 1))
VALUES_INPUT = "values"
MAX_PAYLOAD_BYTES = 50 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_PAYLOAD_BYTES * 2 + 1024 * 1024
MAX_ITEMS = 4096
SUBMISSION_TIMEOUT_SECONDS = 190.0
SUBMISSION_CANCEL_GRACE_SECONDS = 1.0
CLEANUP_TIMEOUT_SECONDS = 16.0
_ALLOWED_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
_VALUE_KINDS = {"TENSOR", "STRING", "INT", "FLOAT", "BOOLEAN"}

_LOCK = threading.RLock()
_MASTER_JOBS: dict[str, "ImageListJob"] = {}
_WORKER_ITEMS: dict[str, tuple[float, dict[str, Any]]] = {}
_WORKER_ITEM_TTL_SECONDS = 3600.0


class ImageListCancelledError(RuntimeError):
    """Raised after owned worker prompts have been cleaned up on cancellation."""


@dataclass
class ImageListJob:
    context_id: str
    start_id: str
    end_id: str
    original_prompt: dict[str, Any]
    items: list[dict[str, Any]]
    worker_ids: list[str]
    master_callback_url: str
    workflow_meta: dict[str, Any] | None = None
    results: dict[int, dict[str, Any]] = field(default_factory=dict)
    received_indexes: set[int] = field(default_factory=set)
    expected_tokens: dict[int, str] = field(default_factory=dict)
    expected_workers: dict[int, str] = field(default_factory=dict)
    event: asyncio.Event | None = None
    created_at: float = field(default_factory=time.monotonic)
    failed: str | None = None
    prompt_id: str | None = None
    active: bool = False
    master_participates: bool = False
    event_loop: asyncio.AbstractEventLoop | None = None
    local_model_callback: Any = None


def _torch():
    import torch
    return torch


def unwrap_single(value: Any) -> Any:
    """Unwrap the one-element wrappers V3/INPUT_IS_LIST uses for scalar inputs."""
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _image_frames(value: Any, name: str, *, mask: bool = False) -> list[Any]:
    torch = _torch()
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        frames: list[Any] = []
        for item in value:
            frames.extend(_image_frames(item, name, mask=mask))
        return frames
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a ComfyUI tensor or a list of tensors")
    if not value.is_floating_point():
        raise TypeError(f"{name} tensor must have a floating-point dtype")
    if mask:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        elif value.ndim == 4 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim != 3:
            raise ValueError(f"{name} MASK must have shape [B,H,W] or [B,H,W,1]")
        if min(value.shape) < 1:
            raise ValueError(f"{name} MASK dimensions must be positive")
        return [value[i:i + 1] for i in range(value.shape[0])]
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError(f"{name} IMAGE must have shape [B,H,W,C] or [H,W,C]")
    if min(value.shape[:3]) < 1 or value.shape[-1] not in (3, 4):
        raise ValueError(f"{name} IMAGE must have positive dimensions and 3 or 4 channels")
    return [value[i:i + 1] for i in range(value.shape[0])]


def _scalar_items(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _align(items: list[Any], count: int, name: str) -> list[Any]:
    """One value is shared by every image; exactly one value per image pairs in order."""
    if len(items) == 1:
        return [items[0] for _ in range(count)]
    if len(items) == count:
        return items
    raise ValueError(f"{name} has {len(items)} values; connect exactly 1 (shared by every image) "
                     f"or {count} (one per image, in order)")


def value_kind(value: Any, name: str) -> str:
    """Return the transport kind for one generic value, or raise for unsupported types."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{name} tensor must have a floating-point dtype")
        if value.ndim < 1 or value.ndim > 4 or min(value.shape) < 1:
            raise ValueError(f"{name} tensor must have 1 to 4 positive dimensions")
        return "TENSOR"
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{name} numbers must be finite")
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    if value is None:
        raise ValueError(f"{name} contains an empty value")
    raise TypeError(f"{name} has unsupported type {type(value).__name__}; "
                    "use IMAGE, MASK, STRING, INT, FLOAT or BOOLEAN")


def _normalize_values(values: Any, count: int) -> dict[str, list[Any]]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError("Start values must be a mapping of value_N slots")
    aligned = {}
    for name, raw in values.items():
        if name not in VALUE_NAMES:
            raise ValueError(f"Unknown Start value slot {name!r}; expected value_1..value_{MAX_VALUE_SLOTS}")
        if raw is None or (isinstance(raw, (list, tuple)) and all(item is None for item in raw)):
            continue  # Unconnected Autogrow slot.
        # Each list element is one value. A batch tensor stays one value; only
        # `images` is split into frames.
        items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        for item in items:
            value_kind(item, name)
        aligned[name] = _align(items, count, name)
    return aligned


def normalize_image_list_inputs(images: Any, mask: Any = None, values: Any = None) -> list[dict[str, Any]]:
    """Split images into items and pair masks/generic values by the 1-or-N rule."""
    frames = _image_frames(images, "images")
    if not frames:
        raise ValueError("Distributed Image List Start requires at least one image")
    if len(frames) > MAX_ITEMS:
        raise ValueError(f"Image list contains {len(frames)} items; maximum is {MAX_ITEMS}")
    count = len(frames)
    mask_frames = _image_frames(mask, "mask", mask=True)
    masks = _align(mask_frames, count, "mask") if mask_frames else [None] * count
    aligned = _normalize_values(values, count)
    return [
        {"image": frames[i], "mask": masks[i],
         "values": {name: column[i] for name, column in aligned.items()}, "index": i}
        for i in range(count)
    ]


def _tensor_dtype_name(tensor: Any) -> str:
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _ALLOWED_FLOAT_DTYPES:
        raise TypeError(f"Tensor dtype {name!r} is unsupported; use float16, bfloat16, float32, or float64")
    return name


def encode_value(value: Any, kind: str) -> dict[str, Any] | None:
    """Encode only the explicit IMAGE/MASK/TENSOR/STRING/FLOAT/INT/BOOLEAN protocol types."""
    if value is None:
        return None
    if kind in ("IMAGE", "MASK", "TENSOR"):
        torch = _torch()
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{kind} result must be a floating-point tensor")
        if kind == "TENSOR":
            if value.ndim < 1 or value.ndim > 4 or min(value.shape) < 1:
                raise ValueError("TENSOR value must have 1 to 4 positive dimensions")
        elif kind == "IMAGE":
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] not in (3, 4):
                raise ValueError("IMAGE item must have shape [1,H,W,3|4]")
        else:
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim == 4 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError("MASK item must have shape [1,H,W] or [1,H,W,1]")
        if value.numel() * value.element_size() > MAX_PAYLOAD_BYTES:
            raise ValueError(f"{kind} item exceeds the {MAX_PAYLOAD_BYTES}-byte transport limit")
        cpu = value.detach().to("cpu").contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"{kind} item exceeds the {MAX_PAYLOAD_BYTES}-byte transport limit")
        return {"type": kind, "dtype": _tensor_dtype_name(cpu),
                "shape": list(cpu.shape), "data": base64.b64encode(raw).decode("ascii")}
    if kind == "STRING":
        if not isinstance(value, str):
            raise TypeError("STRING transport value must be a string")
        return {"type": kind, "value": value}
    if kind == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("INT transport value must be an integer")
        return {"type": kind, "value": value}
    if kind == "BOOLEAN":
        if not isinstance(value, bool):
            raise TypeError("BOOLEAN transport value must be a boolean")
        return {"type": kind, "value": value}
    if kind == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TypeError("FLOAT transport value must be finite")
        return {"type": kind, "value": float(value)}
    raise ValueError(f"Unsupported image-list transport type {kind!r}")


def decode_value(data: Any, expected_kind: str) -> Any:
    if data is None:
        return None
    if not isinstance(data, dict) or data.get("type") != expected_kind:
        raise ValueError(f"Expected {expected_kind} transport value")
    if expected_kind in ("IMAGE", "MASK", "TENSOR"):
        torch = _torch()
        dtype_name = data.get("dtype")
        if dtype_name not in _ALLOWED_FLOAT_DTYPES:
            raise ValueError(f"Unsupported tensor dtype {dtype_name!r}")
        shape = data.get("shape")
        if not isinstance(shape, list) or not shape or len(shape) > 4:
            raise ValueError("Tensor shape must be a list of 1 to 4 positive dimensions")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape):
            raise ValueError("Tensor shape dimensions must be positive integers")
        expected_count = math.prod(shape)
        element_size = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}[dtype_name]
        expected_bytes = expected_count * element_size
        if expected_bytes > MAX_PAYLOAD_BYTES:
            raise ValueError("Tensor exceeds the image-list transport limit")
        encoded = data.get("data")
        if not isinstance(encoded, str) or len(encoded) > ((MAX_PAYLOAD_BYTES + 2) // 3) * 4:
            raise ValueError("Tensor data is missing or exceeds the transport limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("Tensor data must be valid base64") from exc
        if len(raw) != expected_bytes:
            raise ValueError("Tensor byte length does not match shape and dtype")
        dtype = getattr(torch, dtype_name)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape).clone()
    value = data.get("value")
    return encode_value(value, expected_kind) and value


def encode_request_payload(payload: dict[str, Any]) -> bytes:
    """Budget the final JSON envelope, including base64, before HTTP upload."""
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError(
            f"Image-list complete JSON/base64 request is {len(body)} bytes; "
            f"limit is {MAX_REQUEST_BYTES} bytes (image/mask/values combined)"
        )
    return body


def encode_item(item: dict[str, Any]) -> dict[str, Any]:
    values = item.get("values") or {}
    encoded_values = {}
    for name, value in values.items():
        if name not in VALUE_NAMES:
            raise ValueError(f"Unknown Start value slot {name!r}")
        encoded_values[name] = encode_value(value, value_kind(value, name))
    return {"image": encode_value(item["image"], "IMAGE"),
            "mask": encode_value(item.get("mask"), "MASK"),
            "values": encoded_values,
            "index": int(item["index"])}


def decode_item(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("item must be an object")
    index = data.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("item.index must be a non-negative integer")
    image = decode_value(data.get("image"), "IMAGE")
    if image is None or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError("item.image must contain exactly one RGB/RGBA image frame")
    mask = decode_value(data.get("mask"), "MASK")
    if mask is not None and (mask.ndim != 3 or mask.shape[0] != 1):
        raise ValueError("item.mask must contain exactly one MASK frame")
    raw_values = data.get("values", {})
    if not isinstance(raw_values, dict):
        raise ValueError("item.values must be an object")
    values = {}
    for name, encoded in raw_values.items():
        if name not in VALUE_NAMES:
            raise ValueError(f"item.values has unknown slot {name!r}")
        if not isinstance(encoded, dict) or encoded.get("type") not in _VALUE_KINDS:
            raise ValueError(f"item.values.{name} has an unsupported type")
        values[name] = decode_value(encoded, encoded["type"])
        if values[name] is None:
            raise ValueError(f"item.values.{name} is empty")
    return {"image": image, "mask": mask, "values": values, "index": index}


def _sweep_worker_items() -> None:
    now = time.monotonic()
    with _LOCK:
        for token, (expires, _) in list(_WORKER_ITEMS.items()):
            if expires <= now:
                del _WORKER_ITEMS[token]


def store_worker_item(token: str, item: dict[str, Any]) -> None:
    if not isinstance(token, str) or len(token) < 32 or len(token) > 128:
        raise ValueError("token must be a 32-128 character capability")
    clean = decode_item(item)
    _sweep_worker_items()
    with _LOCK:
        if token in _WORKER_ITEMS:
            raise ValueError("duplicate item token")
        _WORKER_ITEMS[token] = (time.monotonic() + _WORKER_ITEM_TTL_SECONDS, clean)


def consume_worker_item(token: str) -> dict[str, Any]:
    if not isinstance(token, str) or not token:
        raise ValueError("worker item token is required")
    _sweep_worker_items()
    with _LOCK:
        entry = _WORKER_ITEMS.pop(token, None)
    if entry is None:
        raise RuntimeError("Image-list item is missing, expired, or already consumed")
    return entry[1]


def delete_worker_item(token: str) -> bool:
    with _LOCK:
        return _WORKER_ITEMS.pop(str(token), None) is not None


def create_master_job(start_id: str, end_id: str, prompt: dict[str, Any], items: list[dict[str, Any]],
                      worker_ids: list[str], master_callback_url: str,
                      workflow_meta: dict[str, Any] | None = None,
                      prompt_id: str | None = None, *, master_participates: bool = False,
                      explicit_targets: bool = False) -> str:
    if not isinstance(prompt, dict):
        raise ValueError("Image List Start requires the original ComfyUI prompt for region extraction")
    if not isinstance(master_participates, bool):
        raise ValueError("master_participates must be a boolean")
    if not worker_ids and not master_participates:
        raise RuntimeError("Distributed Image List requires at least one selected worker; master fallback is disabled")
    # Only the target-controlled Start caller may relax the outer prompt guard.
    regions = validate_image_list_regions(prompt, explicit_targets=explicit_targets is True)
    if not any(region["start_id"] == str(start_id) and region["end_id"] == str(end_id)
               for region in regions):
        raise ValueError("Image-list region does not match the requested Start/End pair")
    context_id = secrets.token_urlsafe(36)
    state = ImageListJob(context_id, str(start_id), str(end_id), prompt, items,
                         [str(worker_id) for worker_id in worker_ids], master_callback_url,
                         workflow_meta, prompt_id=prompt_id, master_participates=master_participates)
    with _LOCK:
        _MASTER_JOBS[context_id] = state
    return context_id


def get_master_job(context_id: str) -> ImageListJob | None:
    with _LOCK:
        return _MASTER_JOBS.get(context_id)


def remove_master_job(context_id: str) -> None:
    with _LOCK:
        _MASTER_JOBS.pop(context_id, None)


def _prompt_links(node: Any, prompt: dict[str, Any]):
    if not isinstance(node, dict):
        return
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return
    for input_name, value in inputs.items():
        if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) in prompt:
            yield input_name, str(value[0]), value[1]


def _direct_link(value: Any, prompt: dict[str, Any], node_id: str, input_name: str) -> tuple[str, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) in prompt):
        raise ValueError(f"{node_id}.{input_name} must be connected to an in-graph node")
    try:
        return str(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{node_id}.{input_name} has an invalid link") from exc


def _ancestors_for_inputs(prompt: dict[str, Any], end_id: str, start_id: str,
                          input_names: tuple[str, ...]) -> set[str]:
    result = {end_id}
    pending = []
    end_inputs = prompt[end_id].get("inputs", {})
    for name in input_names:
        if name in end_inputs:
            source, _ = _direct_link(end_inputs[name], prompt, end_id, name)
            pending.append(source)
    while pending:
        node_id = pending.pop()
        if node_id in result:
            continue
        result.add(node_id)
        if node_id == start_id:
            continue
        node = prompt[node_id]
        for _, source, _ in _prompt_links(node, prompt):
            if source not in result:
                pending.append(source)
    return result


def _downstream_until(prompt: dict[str, Any], start_id: str, end_id: str) -> set[str]:
    downstream: dict[str, set[str]] = {str(node_id): set() for node_id in prompt}
    for node_id, node in prompt.items():
        for _, source, _ in _prompt_links(node, prompt):
            downstream[source].add(str(node_id))
    visited = set()
    pending = [start_id]
    while pending:
        node_id = pending.pop()
        for target in downstream.get(node_id, ()):
            if target == end_id or target in visited:
                continue
            visited.add(target)
            pending.append(target)
    return visited


def _start_ancestors(prompt: dict[str, Any], start_id: str) -> set[str]:
    """All dependencies materialized on the master, without crossing Start again."""
    ancestors = set()
    pending = [source for _, source, _ in _prompt_links(prompt[start_id], prompt)]
    while pending:
        node_id = pending.pop()
        if node_id == start_id or node_id in ancestors:
            continue
        ancestors.add(node_id)
        pending.extend(source for _, source, _ in _prompt_links(prompt[node_id], prompt))
    return ancestors


def _safe_shared_dependency(node: dict[str, Any], prompt: dict[str, Any], registry) -> bool:
    """Only typed model dependencies or scalar constants may straddle Start.

    A leaf alone is not sufficient: LoadImage/EmptyImage are leaves too. Unknown
    classes and list-producing nodes fail closed. All shared ancestors are checked,
    so a model-producing node cannot hide a shared image/latent dependency.
    """
    node_class = registry.get(node.get("class_type"))
    if node_class is None:
        return False
    info = node_class.GET_NODE_INFO_V1() if hasattr(node_class, "GET_NODE_INFO_V1") else {}
    outputs = info.get("output", getattr(node_class, "RETURN_TYPES", ()))
    if not outputs or any(info.get("output_is_list", getattr(node_class, "OUTPUT_IS_LIST", ()))):
        return False
    types = {str(output) for output in outputs}
    model_types = {"MODEL", "CLIP", "VAE", "CLIP_VISION", "CONTROL_NET"}
    scalar_types = {"INT", "FLOAT", "STRING", "BOOLEAN"}
    if types <= model_types:
        return True
    return types <= scalar_types and not any(_prompt_links(node, prompt))


def _is_output_node(node: dict[str, Any], registry) -> bool:
    node_class = registry.get(node.get("class_type"))
    if node_class is None:
        return False
    info = node_class.GET_NODE_INFO_V1() if hasattr(node_class, "GET_NODE_INFO_V1") else {}
    return bool(info.get("output_node", getattr(node_class, "OUTPUT_NODE", False)))


def _image_list_pairs(prompt: dict[str, Any]) -> list[tuple[str, str]]:
    """Pair boundaries before execution-copy pruning or full validation."""
    if not isinstance(prompt, dict):
        raise ValueError("Prompt must be an object")
    starts = sorted(str(k) for k, v in prompt.items() if isinstance(v, dict) and v.get("class_type") == START_CLASS)
    ends = sorted(str(k) for k, v in prompt.items() if isinstance(v, dict) and v.get("class_type") == END_CLASS)
    if len(starts) != len(ends) or not starts:
        raise ValueError("Distributed Image List needs one paired Start and End per region")
    pairs = []
    for start_id in starts:
        candidates = []
        for end_id in ends:
            context = prompt[end_id].get("inputs", {}).get("context")
            if isinstance(context, (list, tuple)) and len(context) == 2 and str(context[0]) == start_id and context[1] == CONTEXT_OUT:
                candidates.append(end_id)
        if len(candidates) != 1:
            raise ValueError(f"Start node {start_id} must connect its typed context output directly to exactly one End")
        pairs.append((start_id, candidates[0]))
    return pairs


def validate_image_list_regions(prompt: dict[str, Any], *, explicit_targets: bool = False) -> list[dict[str, Any]]:
    """Validate pairing, non-nesting, and closed per-image graph boundaries.

    Required output-node utilities are allowed only when callers explicitly
    select execution roots, rather than letting every OUTPUT_NODE run independently.
    All data-boundary checks still apply in that mode.
    """
    regions = []
    for start_id, end_id in _image_list_pairs(prompt):
        if "images" not in prompt[end_id].get("inputs", {}):
            raise ValueError(f"End node {end_id}.images must be connected to the per-item result branch")
        end_inputs = prompt[end_id].get("inputs", {})
        for name in ("images", "masks", "texts"):
            if name in end_inputs:
                _direct_link(end_inputs[name], prompt, end_id, name)
        region_nodes = _ancestors_for_inputs(prompt, end_id, start_id, ("images", "masks", "texts"))
        if start_id not in region_nodes:
            raise ValueError(f"Start node {start_id} is not an ancestor of End node {end_id}")
        forbidden = {n for n in region_nodes if n != start_id and n != end_id and
                     isinstance(prompt[n], dict) and prompt[n].get("class_type") in
                     {START_CLASS, END_CLASS, "DistributedCollector", "UltimateSDUpscaleDistributed"}}
        if forbidden:
            raise ValueError("Nested/ambiguous regions and other distributed nodes are unsupported in an image-list region: " + ", ".join(sorted(forbidden)))
        import sys
        registry = getattr(sys.modules.get("nodes"), "NODE_CLASS_MAPPINGS", {})
        shared = region_nodes & _start_ancestors(prompt, start_id)
        unsafe = {node_id for node_id in shared
                  if not _safe_shared_dependency(prompt[node_id], prompt, registry)}
        if unsafe:
            links = [f"{node_id}.{name} <- {source}:{slot}"
                     for node_id in sorted(region_nodes - {start_id})
                     for name, source, slot in _prompt_links(prompt[node_id], prompt)
                     if source in unsafe]
            raise ValueError(
                f"Image-list region {start_id}->{end_id} bypasses Start through upstream nodes "
                f"{sorted(unsafe)} ({'; '.join(links)}); route per-item data through Start"
            )
        if not explicit_targets:
            for node_id in region_nodes - {start_id, end_id}:
                if _is_output_node(prompt[node_id], registry):
                    raise ValueError(f"Output node {node_id} must be placed after Distributed Image List End")
        downstream = _downstream_until(prompt, start_id, end_id)
        outside = sorted(downstream - region_nodes)
        if outside:
            raise ValueError(f"Image-list region {start_id}->{end_id} has unsafe cross-boundary nodes: {outside}")
        # Model loaders and constants can be shared; item-dependent processing cannot.
        regions.append({"start_id": start_id, "end_id": end_id,
                        "node_ids": region_nodes, "branch_ids": (region_nodes & downstream) - {start_id}})
    regions.sort(key=lambda region: (0, int(region["start_id"])) if region["start_id"].isdigit()
                 else (1, region["start_id"]))
    internal_sets = [r["branch_ids"] for r in regions]
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            overlap = internal_sets[i] & internal_sets[j]
            if overlap:
                raise ValueError(f"Image-list regions share ambiguous processing nodes: {sorted(overlap)}")
    return regions


def build_worker_prompt(prompt: dict[str, Any], region: dict[str, Any], item_token: str,
                        context_id: str, item_index: int, callback_url: str) -> dict[str, Any]:
    """Copy only the per-image branch and its worker-side dependencies."""
    start_id, end_id = region["start_id"], region["end_id"]
    _direct_link(prompt[end_id].get("inputs", {}).get("context"), prompt, end_id, "context")
    kept = {node_id: json.loads(json.dumps(prompt[node_id])) for node_id in region["node_ids"]}
    start_inputs = kept[start_id].setdefault("inputs", {})
    # Per-item data reaches workers only through the staged item token.
    for name in list(start_inputs):
        if name in ("images", "mask") or name.startswith(VALUES_INPUT + "."):
            start_inputs.pop(name)
    start_inputs.update({"worker_item_token": item_token, "worker_context_id": context_id})
    end_inputs = kept[end_id].setdefault("inputs", {})
    end_inputs.update({"worker_result_token": item_token,
                       "worker_item_index": int(item_index),
                       "master_callback_url": callback_url})
    return kept


def create_worker_task_token() -> str:
    return secrets.token_urlsafe(36)


async def dynamic_schedule(item_count: int, workers: list[Any], dispatch, wait_next,
                           cleanup, interrupted=None):
    """One in-flight job per worker; completed workers immediately take next item."""
    if item_count <= 0:
        raise ValueError("Dynamic image-list queue has no items")
    if not workers:
        raise RuntimeError("No free, reachable, feature-compatible direct workers are available")
    pending_index = 0
    inflight: dict[Any, Any] = {}
    results: dict[int, Any] = {}

    def check_interrupted():
        if interrupted is not None and interrupted():
            raise ImageListCancelledError("Image-list job interrupted by user")

    async def next_result():
        # Keep one waiter alive while checking ComfyUI's thread-safe interrupt flag.
        # Restarting wait_next on a timeout can lose a result it already consumed.
        waiter = asyncio.create_task(wait_next(inflight))
        try:
            while True:
                check_interrupted()
                done, _ = await asyncio.wait({waiter}, timeout=0.1)
                check_interrupted()
                if done:
                    return waiter.result()
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def submit(worker, index):
        # Bound the shield itself, and inspect the real Comfy interrupt flag while
        # awaiting an acknowledgement. Briefly retain a late acknowledged handle.
        task = asyncio.create_task(asyncio.wait_for(
            dispatch(worker, index), SUBMISSION_TIMEOUT_SECONDS))
        try:
            while True:
                check_interrupted()
                done, _ = await asyncio.wait({task}, timeout=0.1)
                if done:
                    handle = task.result()
                    inflight[worker] = handle
                    check_interrupted()
                    return handle
        except (asyncio.CancelledError, ImageListCancelledError):
            try:
                inflight[worker] = await asyncio.wait_for(
                    asyncio.shield(task), SUBMISSION_CANCEL_GRACE_SECONDS)
            except (Exception, asyncio.CancelledError):
                pass
            raise
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        for worker in workers:
            check_interrupted()
            if pending_index >= item_count:
                break
            inflight[worker] = await submit(worker, pending_index)
            pending_index += 1
        while inflight:
            check_interrupted()
            worker, index, result = await next_result()
            if worker not in inflight:
                raise RuntimeError(f"Unexpected completion from unassigned worker {worker}")
            if isinstance(index, bool) or not isinstance(index, int) or index != inflight[worker].get("index"):
                raise RuntimeError(f"Worker {worker} returned an unexpected image index {index!r}")
            if index in results:
                raise RuntimeError(f"Duplicate image-list result for item {index}")
            results[index] = result
            del inflight[worker]
            if pending_index < item_count:
                inflight[worker] = await submit(worker, pending_index)
                pending_index += 1
        missing = sorted(set(range(item_count)) - set(results))
        if missing:
            raise RuntimeError(f"Image-list results are incomplete; missing indexes {missing}")
        return [results[i] for i in range(item_count)]
    except BaseException:
        if inflight:
            await asyncio.gather(*(asyncio.wait_for(cleanup(worker, handle), CLEANUP_TIMEOUT_SECONDS)
                                   for worker, handle in inflight.items()), return_exceptions=True)
        raise


def set_master_result(context_id: str, item_index: int, token: str, result: dict[str, Any]) -> None:
    with _LOCK:
        state = _MASTER_JOBS.get(context_id)
        if state is None:
            raise ValueError("Unknown or expired image-list context")
        expected = state.expected_tokens.get(item_index)
        if expected is None or not secrets.compare_digest(expected, token):
            raise ValueError("Image-list result token does not match an assigned item")
        if item_index in state.received_indexes:
            raise ValueError(f"Duplicate image-list result for item {item_index}")
        state.received_indexes.add(item_index)
        state.results[item_index] = result
        event = state.event
    if event is not None:
        if state.event_loop is not None:
            state.event_loop.call_soon_threadsafe(event.set)
        else:
            event.set()


def register_assignment(context_id: str, item_index: int, worker_id: str, token: str) -> None:
    state = get_master_job(context_id)
    if state is None:
        raise RuntimeError("Image-list context expired before worker dispatch")
    with _LOCK:
        if item_index in state.expected_tokens:
            raise RuntimeError(f"Image-list item {item_index} was already assigned")
        state.expected_tokens[item_index] = token
        state.expected_workers[item_index] = worker_id


def decode_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    image = decode_value(result.get("image"), "IMAGE")
    if image is None or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError("result.image must contain exactly one RGB/RGBA image")
    mask = decode_value(result.get("mask"), "MASK")
    if mask is not None and (mask.ndim != 3 or mask.shape[0] != 1):
        raise ValueError("result.mask must contain exactly one MASK frame")
    text = decode_value(result.get("text"), "STRING")
    return {"image": image, "mask": mask, "text": text}


def normalize_worker_result(images: Any, masks: Any = None, texts: Any = None) -> dict[str, Any]:
    def one(value, name, kind):
        if value is None:
            return None
        if kind in ("IMAGE", "MASK"):
            values = _image_frames(value, name, mask=(kind == "MASK"))
            if len(values) != 1:
                raise ValueError(f"Worker End expects exactly one {kind} result, received {len(values)}")
            return values[0]
        values = _scalar_items(value, name)
        values = [v for v in values if v is not None]
        if not values:
            return None
        if len(values) != 1:
            raise ValueError(f"Worker End expects at most one {kind} result, received {len(values)}")
        return values[0]
    image = one(images, "images", "IMAGE")
    if image is None:
        raise ValueError("Worker End received no image result")
    mask = one(masks, "masks", "MASK")
    text = one(texts, "texts", "STRING")
    if text is not None and not isinstance(text, str):
        raise TypeError("Worker End text result must be a string")
    return {"image": image, "mask": mask, "text": text}


def encode_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_worker_result(result.get("image"), result.get("mask"), result.get("text"))
    return {"image": encode_value(normalized["image"], "IMAGE"),
            "mask": encode_value(normalized["mask"], "MASK"),
            "text": encode_value(normalized["text"], "STRING")}


def _collect_output_lists(results: list[dict[str, Any]]) -> tuple[list[Any], list[Any], list[Any]]:
    if not results:
        raise RuntimeError("Image-list execution returned no results")
    images, masks, texts = [], [], []
    for expected_index, result in enumerate(results):
        if not isinstance(result, dict) or "image" not in result:
            raise RuntimeError(f"Image-list result {expected_index} is missing its image")
        images.append(result["image"])
        masks.append(result.get("mask"))
        texts.append(result.get("text"))
    return images, masks, texts


async def run_image_list_job(context_id: str, *, model_callback=None) -> tuple[list[Any], list[Any], list[Any]]:
    """Schedule one queue across master, direct peers and bounded cloud slots."""
    state = get_master_job(context_id)
    if state is None:
        raise RuntimeError("Distributed Image List context is missing or expired")
    with _LOCK:
        state.active = True
        state.local_model_callback = model_callback
    try:
        from ..utils.config import get_image_list_job_timeout_seconds
        deadline = get_image_list_job_timeout_seconds()
        timer = asyncio.timeout(deadline if deadline > 0 else None)
        try:
            async with timer:
                return await _run_image_list_job(state)
        except TimeoutError:
            if timer.expired():
                raise TimeoutError(f"Image-list total deadline exceeded ({deadline:g}s); cleanup attempted") from None
            raise
    except asyncio.CancelledError as exc:
        # The synchronous End bridge catches Exception, not BaseException.
        # Cleanup (including the local thread join) has finished at this point.
        raise ImageListCancelledError("Image-list job cancelled") from exc
    finally:
        remove_master_job(context_id)


def _has_selected_serverless_workers(config, requested_ids):
    """Keep direct/master-only execution free of optional cloud adapter imports.

    This is only an import guard, not validation. The adapter validates original
    pool IDs/configuration without repairing them, before any cloud work starts.
    """
    requested = None if requested_ids is None else {str(value) for value in requested_ids}
    return any(
        str(worker.get("type") or "local").strip().lower() == "serverless"
        and (worker.get("enabled", False) if requested is None
             else str(worker.get("id") or "").strip() in requested)
        for worker in config.get("workers", [])
    )


async def _run_image_list_job(state: ImageListJob):
    state.event = asyncio.Event()
    state.event_loop = asyncio.get_running_loop()
    from ..api.orchestration.prompt_transform import _resolve_direct_workers_for_image_list
    from ..utils.config import load_config

    config = load_config()
    workers = _resolve_direct_workers_for_image_list(config, state.worker_ids)
    # Every item path selects End explicitly, never all OUTPUT_NODE roots.
    regions = validate_image_list_regions(state.original_prompt, explicit_targets=True)
    region = next((entry for entry in regions if entry["start_id"] == state.start_id and entry["end_id"] == state.end_id), None)
    if region is None:
        raise RuntimeError("Image-list region no longer matches the submitted prompt")
    controller = None
    try:
        if _has_selected_serverless_workers(config, state.worker_ids):
            from .serverless import create_serverless_slots
            controller = await create_serverless_slots(state, region, config, state.event)
        return await _run_image_list_slots(state, config, workers, region, controller)
    finally:
        # Own cloud jobs even if direct preflight/local setup fails, or dispatch
        # is cancelled before returning an acknowledged handle to the scheduler.
        if controller is not None:
            await controller.close()


async def _run_image_list_slots(state, config, workers, region, controller):
    context_id = state.context_id
    from ..api.orchestration.dispatch import (
        preflight_image_list_workers, dispatch_worker_prompt,
    )
    from ..utils.config import get_worker_timeout_seconds
    from ..utils.network import build_master_callback_url, build_worker_url, get_client_session
    from ..utils.logging import log
    import aiohttp

    serverless_slots = list(controller.slots) if controller is not None else []
    # Direct capability/idle probes must never be sent to a serverless pool.
    worker_statuses = await preflight_image_list_workers(workers, require_idle=not state.master_participates) if workers else {}
    workers = [worker for worker in workers if worker["id"] in worker_statuses]
    if not workers and not serverless_slots and not state.master_participates:
        raise RuntimeError("No reachable, feature-compatible image-list workers are available; master execution is disabled")
    session = await get_client_session() if workers else None
    timeout_seconds = max(5.0, float(get_worker_timeout_seconds()))
    use_websocket = bool((config.get("settings", {}) or {}).get("websocket_orchestration", False))
    prepared_worker_prompts: dict[str, dict[str, Any]] = {}
    owned_uploads: dict[str, dict[str, Any]] = {}
    # An opaque key cannot collide with any configured direct worker ID, even
    # a worker named "master" or a separate local worker process.
    master_slot = object()
    local_executor = None
    local_tokens: set[str] = set()
    if state.master_participates:
        from .local_executor import LocalExecutionSlot
        local_executor = LocalExecutionSlot(model_callback=state.local_model_callback)

    async def dispatch(worker_id: Any, index: int) -> dict[str, Any]:
        if worker_id in serverless_slots:
            return await controller.dispatch(worker_id, index)
        if worker_id is master_slot:
            assert local_executor is not None
            token = create_worker_task_token()
            register_assignment(context_id, index, "in-process-master", token)
            local_tokens.add(token)
            # Already normalized tensors stay in memory; no HTTP, serialization,
            # /prompt queue, or repeated execution of the original source.
            with _LOCK:
                _WORKER_ITEMS[token] = (time.monotonic() + _WORKER_ITEM_TTL_SECONDS, state.items[index])
            prompt = build_worker_prompt(state.original_prompt, region, token, context_id, index, "")
            prompt[state.start_id]["inputs"]["image_list_local_execution"] = True
            prompt[state.end_id]["inputs"]["image_list_local_execution"] = True
            future = local_executor.submit(prompt, f"{context_id}:{index}", state.end_id, state.workflow_meta)
            future.add_done_callback(lambda _: state.event_loop.call_soon_threadsafe(state.event.set))
            return {"index": index, "token": token, "future": future}
        worker = next((entry for entry in workers if entry["id"] == worker_id), None)
        if worker is None:
            raise RuntimeError(f"Worker {worker_id} was removed before dispatch")
        assert session is not None
        token = create_worker_task_token()
        register_assignment(context_id, index, worker_id, token)
        item_payload = {"context_id": context_id, "item": encode_item(state.items[index])}
        body = encode_request_payload({"token": token, **item_payload})
        handle = {"prompt_id": None, "token": token, "index": index, "worker": worker,
                  "submission_pending": False}
        # Own even a lost upload response; DELETE is idempotent.
        owned_uploads[token] = handle
        input_url = build_worker_url(worker, "/distributed/image-list/item")
        async with session.post(input_url, data=body, headers={"Content-Type": "application/json"},
                                timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status >= 400:
                raise RuntimeError(f"Worker {worker_id} rejected image-list item {index}: HTTP {response.status}")
        callback_url = build_master_callback_url(worker, config=config)
        if worker_id not in prepared_worker_prompts:
            worker_prompt = build_worker_prompt(state.original_prompt, region, token, context_id,
                                                index, callback_url)
            if worker.get("host") and str(worker.get("type") or "local").lower() != "local":
                from ..api.orchestration.media_sync import fetch_worker_path_separator, convert_paths_for_platform
                separator = await fetch_worker_path_separator(worker)
                if separator:
                    worker_prompt = convert_paths_for_platform(worker_prompt, separator)
            from ..api.orchestration.media_sync import sync_worker_media
            await asyncio.wait_for(sync_worker_media(worker, worker_prompt), timeout=60)
            prepared_worker_prompts[worker_id] = worker_prompt
        # Reuse media rewrites, never assignment identity or a mutable submitted graph.
        worker_prompt = build_worker_prompt(prepared_worker_prompts[worker_id], region, token,
                                            context_id, index, callback_url)
        handle["submission_pending"] = True
        prompt_id = await dispatch_worker_prompt(
            worker, worker_prompt, state.workflow_meta,
            client_id=f"image-list-{context_id[:20]}", use_websocket=use_websocket,
            return_prompt_id=True,
            partial_execution_targets=[state.end_id],
        )
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"Worker {worker_id} did not return a prompt_id; submission status is ambiguous")
        handle.update(prompt_id=prompt_id, last_seen=time.monotonic(), submission_pending=False)
        return handle

    async def wait_next(inflight: dict[Any, Any]):
        while True:
            state.event.clear()
            if state.failed:
                raise RuntimeError(state.failed)
            if controller is not None:
                completed = controller.poll(inflight)
                if completed is not None:
                    return completed
            local = inflight.get(master_slot)
            if local is not None and local["future"].done():
                # A posted result does not release this slot until execution has
                # actually finished, including restoration of parent state.
                local["future"].result()
                with _LOCK:
                    result = state.results.pop(local["index"], None)
                if result is None:
                    raise RuntimeError(f"Master completed item {local['index']} without posting a result")
                delete_worker_item(local["token"])
                local_tokens.discard(local["token"])
                return master_slot, local["index"], result
            # Treat worker-side execution errors as failures even when End never posts a result.
            async def history_status(worker_id, handle):
                assert session is not None
                url = build_worker_url(handle["worker"], f"/history/{handle['prompt_id']}")
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                        if response.status >= 400:
                            return None
                        data = await response.json()
                    entry = data.get(handle["prompt_id"])
                    if not isinstance(entry, dict):
                        async with session.get(build_worker_url(handle["worker"], "/queue"),
                                               timeout=aiohttp.ClientTimeout(total=3)) as response:
                            response.raise_for_status()
                            queue = await response.json()
                        if isinstance(queue, dict) and any(
                            isinstance(row, (list, tuple)) and len(row) > 1 and row[1] == handle["prompt_id"]
                            for key in ("queue_running", "queue_pending")
                            for row in (queue.get(key) if isinstance(queue.get(key), list) else [])
                        ):
                            handle["last_seen"] = time.monotonic()
                        return None
                    status = entry.get("status", {}) or {}
                    if status.get("status_str") == "error":
                        return f"Worker {worker_id} failed image-list item {handle['index']}: {status.get('messages', status)}"
                    if status.get("completed") and status.get("status_str") not in (None, "success"):
                        return f"Worker {worker_id} failed image-list item {handle['index']}: {status}"
                    if status.get("completed"):
                        with _LOCK:
                            result = state.results.get(handle["index"])
                        if result is None:
                            return f"Worker {worker_id} completed item {handle['index']} without posting a result"
                        return worker_id, handle["index"], result
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    return None
                return None
            errors = await asyncio.gather(*(history_status(wid, handle) for wid, handle in inflight.items()
                                            if wid is not master_slot and wid not in serverless_slots))
            failure = next((error for error in errors if isinstance(error, str)), None)
            if failure:
                raise RuntimeError(failure)
            # A cloud manager task may finish during direct HTTP history waits.
            if controller is not None:
                completed = controller.poll(inflight)
                if completed is not None:
                    return completed
            completed = next((entry for entry in errors if isinstance(entry, tuple)), None)
            if completed is not None:
                # The callback precedes the worker prompt finishing. Wait for its
                # successful history before releasing the slot / next region.
                with _LOCK:
                    state.results.pop(completed[1], None)
                owned_uploads.pop(inflight[completed[0]]["token"], None)
                return completed
            now = time.monotonic()
            expired = [f"{wid}/item {handle['index']}" for wid, handle in inflight.items()
                       if wid is not master_slot and wid not in serverless_slots
                       and now - handle["last_seen"] > timeout_seconds]
            if expired:
                raise TimeoutError(f"Image-list owned-prompt liveness timeout after {timeout_seconds:.1f}s: {', '.join(expired)}")
            try:
                await asyncio.wait_for(state.event.wait(), timeout=0.25)
            except TimeoutError:
                pass

    async def cleanup(worker_id: Any, handle: dict[str, Any]) -> None:
        if worker_id in serverless_slots:
            # The controller owns exact manager records, including dispatches
            # that never returned a handle. Close it once in the outer finally.
            return
        if worker_id is master_slot:
            # The outer finally owns the non-abandonable stop/join. The generic
            # scheduler's bounded remote cleanup must never detach local work.
            assert local_executor is not None
            local_executor.request_stop()
            return
        assert session is not None
        worker = handle["worker"]
        prompt_id = handle.get("prompt_id")
        if prompt_id:
            # Try both independently: failure of queue deletion must not suppress
            # the targeted interrupt. No global interrupt or queue clearing.
            for route, body in (("/queue", {"delete": [prompt_id]}),
                                ("/interrupt", {"prompt_id": prompt_id})):
                try:
                    async with session.post(build_worker_url(worker, route), json=body,
                                            timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response.raise_for_status()
                except Exception as exc:
                    log(f"[Distributed Image List] Owned prompt cleanup unconfirmed for worker {worker_id} ({route}): {exc}")
        elif handle.get("submission_pending"):
            log(f"[Distributed Image List] Worker {worker_id} submission status is ambiguous; "
                "no prompt ID to target. Check worker manually; not retrying.")
        try:
            async with session.delete(build_worker_url(worker, f"/distributed/image-list/item/{handle['token']}"),
                                      timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
        except Exception as exc:
            log(f"[Distributed Image List] Staged input cleanup unconfirmed for worker {worker_id}: {exc}")
        finally:
            owned_uploads.pop(handle["token"], None)

    def interrupted() -> bool:
        try:
            import comfy.model_management
            return bool(comfy.model_management.processing_interrupted())
        except Exception:
            return False

    try:
        slots = (([master_slot] if local_executor is not None else [])
                 + [worker["id"] for worker in workers] + serverless_slots)
        results = await dynamic_schedule(len(state.items), slots,
                                         dispatch, wait_next, cleanup, interrupted)
        return _collect_output_lists(results)
    finally:
        if local_executor is not None:
            # No timeout here: Python cannot safely kill a synchronous node.
            # Stop at a node/progress boundary and join BEFORE releasing End.
            # Shield against repeated cancellation as well as the job deadline.
            closing = asyncio.create_task(local_executor.close())
            try:
                while not closing.done():
                    try:
                        await asyncio.shield(closing)
                    except asyncio.CancelledError:
                        continue
                closing.result()
            finally:
                for token in local_tokens:
                    delete_worker_item(token)
        # Upload ownership exists before dynamic_schedule has an acknowledged
        # handle. Cover rejection/lost upload responses as well as cancellation.
        pending = list(owned_uploads.values())
        outcomes = await asyncio.gather(*(
            asyncio.wait_for(cleanup(handle["worker"]["id"], handle), CLEANUP_TIMEOUT_SECONDS)
            for handle in pending), return_exceptions=True)
        for handle, outcome in zip(pending, outcomes):
            if isinstance(outcome, BaseException):
                log(f"[Distributed Image List] Cleanup unconfirmed for worker {handle['worker']['id']}: {outcome}")
