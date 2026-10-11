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

PROTOCOL_VERSION = 1
START_CLASS = "DistributedImageListStart"
END_CLASS = "DistributedImageListEnd"
START_NODE_ID = "DistributedImageListStart"
END_NODE_ID = "DistributedImageListEnd"
# Start output indexes are a stable internal prompt protocol.
IMAGE_OUT, REFERENCE_OUT, MASK_OUT, TEXT_OUT, STRENGTH_OUT, SEED_OUT, INDEX_OUT, CONTEXT_OUT = range(8)
MAX_PAYLOAD_BYTES = 50 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_PAYLOAD_BYTES * 2 + 1024 * 1024
MAX_ITEMS = 4096
SUBMISSION_TIMEOUT_SECONDS = 190.0
SUBMISSION_CANCEL_GRACE_SECONDS = 1.0
CLEANUP_TIMEOUT_SECONDS = 16.0
_ALLOWED_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}

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


def _normalize_aligned(value: Any, mode: str, count: int, name: str, kind: str) -> list[Any]:
    if value is None:
        return [None] * count
    if mode not in ("per_item", "shared"):
        raise ValueError(f"{name}_mode must be 'per_item' or 'shared'")
    if kind == "image":
        items = _image_frames(value, name)
    elif kind == "mask":
        items = _image_frames(value, name, mask=True)
    else:
        items = _scalar_items(value, name)
    if mode == "shared":
        if len(items) != 1:
            raise ValueError(f"{name} is shared but has {len(items)} values; shared inputs require exactly one value")
        items = [items[0] for _ in range(count)]
    if len(items) != count:
        raise ValueError(f"{name} has {len(items)} per-item values; expected exactly {count}")
    if kind == "text" and any(item is not None and not isinstance(item, str) for item in items):
        raise TypeError(f"{name} per-item values must be strings")
    if kind == "int" and any(item is not None and (isinstance(item, bool) or not isinstance(item, int)) for item in items):
        raise TypeError(f"{name} per-item values must be integers")
    if kind == "float" and any(item is not None and (isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))) for item in items):
        raise TypeError(f"{name} per-item values must be finite numbers")
    return items


def normalize_image_list_inputs(images: Any, reference_image: Any = None, mask: Any = None,
                                text: Any = None, strength: Any = None, seed: Any = None,
                                reference_image_mode: Any = "per_item", mask_mode: Any = "per_item",
                                text_mode: Any = "per_item", strength_mode: Any = "per_item",
                                seed_mode: Any = "per_item") -> list[dict[str, Any]]:
    """Normalize batches and explicitly shared/per-item optional inputs."""
    frames = _image_frames(images, "images")
    if not frames:
        raise ValueError("Distributed Image List Start requires at least one image")
    if len(frames) > MAX_ITEMS:
        raise ValueError(f"Image list contains {len(frames)} items; maximum is {MAX_ITEMS}")
    count = len(frames)
    refs = _normalize_aligned(reference_image, str(unwrap_single(reference_image_mode)), count,
                              "reference_image", "image")
    masks = _normalize_aligned(mask, str(unwrap_single(mask_mode)), count, "mask", "mask")
    texts = _normalize_aligned(text, str(unwrap_single(text_mode)), count, "text", "text")
    strengths = _normalize_aligned(strength, str(unwrap_single(strength_mode)), count, "strength", "float")
    seeds = _normalize_aligned(seed, str(unwrap_single(seed_mode)), count, "seed", "int")
    return [
        {"image": frames[i], "reference_image": refs[i], "mask": masks[i],
         "text": texts[i], "strength": strengths[i], "seed": seeds[i], "index": i}
        for i in range(count)
    ]


def _tensor_dtype_name(tensor: Any) -> str:
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _ALLOWED_FLOAT_DTYPES:
        raise TypeError(f"Tensor dtype {name!r} is unsupported; use float16, bfloat16, float32, or float64")
    return name


def encode_value(value: Any, kind: str) -> dict[str, Any] | None:
    """Encode only the explicit IMAGE/MASK/STRING/FLOAT/INT protocol types."""
    if value is None:
        return None
    if kind in ("IMAGE", "MASK"):
        torch = _torch()
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{kind} result must be a floating-point tensor")
        if kind == "IMAGE":
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
    if expected_kind in ("IMAGE", "MASK"):
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
            f"limit is {MAX_REQUEST_BYTES} bytes (image/reference/mask combined)"
        )
    return body


def encode_item(item: dict[str, Any]) -> dict[str, Any]:
    return {"image": encode_value(item["image"], "IMAGE"),
            "reference_image": encode_value(item.get("reference_image"), "IMAGE"),
            "mask": encode_value(item.get("mask"), "MASK"),
            "text": encode_value(item.get("text"), "STRING"),
            "strength": encode_value(item.get("strength"), "FLOAT"),
            "seed": encode_value(item.get("seed"), "INT"),
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
    reference = decode_value(data.get("reference_image"), "IMAGE")
    if reference is not None and (reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[-1] not in (3, 4)):
        raise ValueError("item.reference_image must contain exactly one RGB/RGBA image frame")
    mask = decode_value(data.get("mask"), "MASK")
    if mask is not None and (mask.ndim != 3 or mask.shape[0] != 1):
        raise ValueError("item.mask must contain exactly one MASK frame")
    return {"image": image,
            "reference_image": reference,
            "mask": mask,
            "text": decode_value(data.get("text"), "STRING"),
            "strength": decode_value(data.get("strength"), "FLOAT"),
            "seed": decode_value(data.get("seed"), "INT"), "index": index}


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
                      prompt_id: str | None = None) -> str:
    if not isinstance(prompt, dict):
        raise ValueError("Image List Start requires the original ComfyUI prompt for region extraction")
    if not worker_ids:
        raise RuntimeError("Distributed Image List requires at least one preflighted direct worker; master fallback is disabled")
    context_id = secrets.token_urlsafe(36)
    state = ImageListJob(context_id, str(start_id), str(end_id), prompt, items,
                         [str(worker_id) for worker_id in worker_ids], master_callback_url,
                         workflow_meta, prompt_id=prompt_id)
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


def validate_image_list_regions(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate pairing, non-nesting, and closed per-image graph boundaries."""
    if not isinstance(prompt, dict):
        raise ValueError("Prompt must be an object")
    starts = sorted(str(k) for k, v in prompt.items() if isinstance(v, dict) and v.get("class_type") == START_CLASS)
    ends = sorted(str(k) for k, v in prompt.items() if isinstance(v, dict) and v.get("class_type") == END_CLASS)
    if len(starts) != len(ends) or not starts:
        raise ValueError("Distributed Image List needs one paired Start and End per region")
    regions = []
    for start_id in starts:
        candidates = []
        for end_id in ends:
            context = prompt[end_id].get("inputs", {}).get("context")
            if isinstance(context, (list, tuple)) and len(context) == 2 and str(context[0]) == start_id and context[1] == CONTEXT_OUT:
                candidates.append(end_id)
        if len(candidates) != 1:
            raise ValueError(f"Start node {start_id} must connect its typed context output directly to exactly one End")
        end_id = candidates[0]
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
        # Output nodes execute independently of the lazy End, so they would
        # evaluate the processing branch on the master as well as the worker.
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
        for node_id in region_nodes - {start_id, end_id}:
            node_class = registry.get(prompt[node_id].get("class_type"))
            if node_class is None:
                continue
            info = node_class.GET_NODE_INFO_V1() if hasattr(node_class, "GET_NODE_INFO_V1") else {}
            if info.get("output_node", getattr(node_class, "OUTPUT_NODE", False)):
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
    for name in ("images", "reference_image", "mask", "text", "strength", "seed"):
        start_inputs.pop(name, None)
    start_inputs.update({"worker_item_token": item_token, "worker_context_id": context_id})
    end_inputs = kept[end_id].setdefault("inputs", {})
    end_inputs.update({"worker_result_token": item_token,
                       "worker_item_index": int(item_index),
                       "master_callback_url": callback_url})
    return kept


def create_worker_task_token() -> str:
    return secrets.token_urlsafe(36)


async def dynamic_schedule(item_count: int, workers: list[str], dispatch, wait_next,
                           cleanup, interrupted=None):
    """One in-flight job per worker; completed workers immediately take next item."""
    if item_count <= 0:
        raise ValueError("Dynamic image-list queue has no items")
    if not workers:
        raise RuntimeError("No free, reachable, feature-compatible direct workers are available")
    pending_index = 0
    inflight: dict[str, Any] = {}
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


async def run_image_list_job(context_id: str) -> tuple[list[Any], list[Any], list[Any]]:
    """Resolve compatible idle direct workers and schedule one image at a time."""
    state = get_master_job(context_id)
    if state is None:
        raise RuntimeError("Distributed Image List context is missing or expired")
    with _LOCK:
        state.active = True
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
    finally:
        remove_master_job(context_id)


async def _run_image_list_job(state: ImageListJob):
    context_id = state.context_id
    state.event = asyncio.Event()
    from ..api.orchestration.dispatch import (
        preflight_image_list_workers, dispatch_worker_prompt,
    )
    from ..api.orchestration.prompt_transform import _resolve_direct_workers_for_image_list
    from ..utils.config import load_config, get_worker_timeout_seconds
    from ..utils.network import build_master_callback_url, build_worker_url, get_client_session
    from ..utils.logging import log
    import aiohttp

    config = load_config()
    workers = _resolve_direct_workers_for_image_list(config, state.worker_ids)
    if not workers:
        raise RuntimeError("Distributed Image List requires enabled direct workers; serverless pools are unsupported")
    # Recheck worker capability and idle state immediately before the first item.
    worker_statuses = await preflight_image_list_workers(workers, require_idle=True)
    workers = [worker for worker in workers if worker["id"] in worker_statuses]
    if not workers:
        raise RuntimeError("No free, reachable, feature-compatible direct workers are available; master execution is disabled")
    session = await get_client_session()
    regions = validate_image_list_regions(state.original_prompt)
    region = next((entry for entry in regions if entry["start_id"] == state.start_id and entry["end_id"] == state.end_id), None)
    if region is None:
        raise RuntimeError("Image-list region no longer matches the submitted prompt")
    timeout_seconds = max(5.0, float(get_worker_timeout_seconds()))
    use_websocket = bool((config.get("settings", {}) or {}).get("websocket_orchestration", False))
    prepared_worker_prompts: dict[str, dict[str, Any]] = {}
    owned_uploads: dict[str, dict[str, Any]] = {}

    async def dispatch(worker_id: str, index: int) -> dict[str, Any]:
        worker = next((entry for entry in workers if entry["id"] == worker_id), None)
        if worker is None:
            raise RuntimeError(f"Worker {worker_id} was removed before dispatch")
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
        )
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"Worker {worker_id} did not return a prompt_id; submission status is ambiguous")
        handle.update(prompt_id=prompt_id, last_seen=time.monotonic(), submission_pending=False)
        return handle

    async def wait_next(inflight: dict[str, Any]):
        while True:
            if state.failed:
                raise RuntimeError(state.failed)
            # Treat worker-side execution errors as failures even when End never posts a result.
            async def history_status(worker_id, handle):
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
            errors = await asyncio.gather(*(history_status(wid, handle) for wid, handle in inflight.items()))
            failure = next((error for error in errors if isinstance(error, str)), None)
            if failure:
                raise RuntimeError(failure)
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
                       if now - handle["last_seen"] > timeout_seconds]
            if expired:
                raise TimeoutError(f"Image-list owned-prompt liveness timeout after {timeout_seconds:.1f}s: {', '.join(expired)}")
            await asyncio.sleep(0.25)

    async def cleanup(worker_id: str, handle: dict[str, Any]) -> None:
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
        results = await dynamic_schedule(len(state.items), [worker["id"] for worker in workers],
                                         dispatch, wait_next, cleanup, interrupted)
        return _collect_output_lists(results)
    finally:
        # Upload ownership exists before dynamic_schedule has an acknowledged
        # handle. Cover rejection/lost upload responses as well as cancellation.
        pending = list(owned_uploads.values())
        outcomes = await asyncio.gather(*(
            asyncio.wait_for(cleanup(handle["worker"]["id"], handle), CLEANUP_TIMEOUT_SECONDS)
            for handle in pending), return_exceptions=True)
        for handle, outcome in zip(pending, outcomes):
            if isinstance(outcome, BaseException):
                log(f"[Distributed Image List] Cleanup unconfirmed for worker {handle['worker']['id']}: {outcome}")
