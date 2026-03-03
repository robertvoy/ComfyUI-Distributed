import torch
import json
from typing import Any

from ..utils.logging import debug_log, log


def _chunk_bounds(total_items: int, n_splits: int) -> list[tuple[int, int]]:
    """Return contiguous [start, end) bounds for n_splits chunks."""
    split_count = max(1, int(n_splits))
    total = max(0, int(total_items))
    base, remainder = divmod(total, split_count)

    bounds: list[tuple[int, int]] = []
    start = 0
    for idx in range(split_count):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


class DistributedSeed:
    """
    Distributes seed values across multiple GPUs.
    On master: passes through the original seed.
    On workers: adds offset based on worker ID.
    """
    
    @classmethod
    def INPUT_TYPES(cls: type["DistributedSeed"]) -> dict[str, Any]:
        return {
            "required": {
                "seed": ("INT", {
                    "default": 1125899906842, 
                    "min": 0,
                    "max": 1125899906842624,
                    "forceInput": False  # Widget by default, can be converted to input
                }),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "distribute"
    CATEGORY = "utils"
    
    def distribute(
        self,
        seed: int,
        is_worker: bool = False,
        worker_id: str = "",
    ) -> tuple[int]:
        if not is_worker:
            # Master node: pass through original values
            debug_log(f"Distributor - Master: seed={seed}")
            return (seed,)
        else:
            # Worker node: apply offset based on worker index
            # Find worker index from enabled_worker_ids
            try:
                # Worker IDs are passed as "worker_0", "worker_1", etc.
                if worker_id.startswith("worker_"):
                    worker_index = int(worker_id.split("_")[1])
                else:
                    # Fallback: try to parse as direct index
                    worker_index = int(worker_id)
                
                offset = worker_index + 1
                new_seed = seed + offset
                debug_log(f"Distributor - Worker {worker_index}: seed={seed} → {new_seed}")
                return (new_seed,)
            except (ValueError, IndexError) as e:
                debug_log(f"Distributor - Error parsing worker_id '{worker_id}': {e}")
                # Fallback: return original seed
                return (seed,)


# Define ByPassTypeTuple for flexible return types
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


class DistributedValue:
    """
    Outputs a different value per worker.
    On master: returns default_value.
    On workers: looks up the worker-specific value from a JSON map,
    falling back to default_value if not set.
    """

    @classmethod
    def INPUT_TYPES(cls: type["DistributedValue"]) -> dict[str, Any]:
        return {
            "required": {
                "default_value": ("STRING", {"default": ""}),
                "worker_values": ("STRING", {"default": "{}"}),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "distribute"
    CATEGORY = "utils"

    @staticmethod
    def _coerce(value: Any, value_type: str) -> Any:
        """Convert a string value to the requested type."""
        if value_type == "INT":
            return int(float(value))
        if value_type == "FLOAT":
            return float(value)
        return value  # STRING and COMBO stay as strings

    @staticmethod
    def _coerce_safe(value: Any, value_type: str) -> Any:
        """Best-effort coercion with graceful fallback to original value."""
        try:
            return DistributedValue._coerce(value, value_type)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _infer_value_type(value: Any) -> str:
        """Infer coercion type from the provided default value."""
        if isinstance(value, bool):
            return "STRING"
        if isinstance(value, int):
            return "INT"
        if isinstance(value, float):
            return "FLOAT"
        return "STRING"

    def distribute(
        self,
        default_value: Any,
        worker_values: str | dict[str, Any] = "{}",
        is_worker: bool = False,
        worker_id: str = "",
    ) -> tuple[Any]:
        values = {}
        value_type = "STRING"

        try:
            values = json.loads(worker_values) if isinstance(worker_values, str) else worker_values
            if not isinstance(values, dict):
                values = {}
        except json.JSONDecodeError as e:
            debug_log(f"DistributedValue - Error parsing worker_values: {e}")
            values = {}

        inferred_type = self._infer_value_type(default_value)
        value_type = values.get("_type", inferred_type)
        if value_type not in {"STRING", "COMBO", "INT", "FLOAT"}:
            value_type = inferred_type
        values["_type"] = value_type
        coerced_default = self._coerce_safe(default_value, value_type)

        if not is_worker:
            debug_log(f"DistributedValue - Master: returning default '{coerced_default}'")
            return (coerced_default,)

        try:
            if worker_id.startswith("worker_"):
                idx = int(worker_id.split("_")[1])
            else:
                idx = int(worker_id)
            key = str(idx + 1)  # worker_0 → key "1" (1-indexed)
            raw = values.get(key, "")
            if raw:
                coerced = self._coerce(raw, value_type)
                debug_log(f"DistributedValue - Worker {idx}: returning '{coerced}'")
                return (coerced,)
        except (ValueError, IndexError) as e:
            debug_log(f"DistributedValue - Error: {e}")
        debug_log(f"DistributedValue - Worker fallback: returning default '{coerced_default}'")
        return (coerced_default,)

class DistributedModelName:
    @classmethod
    def INPUT_TYPES(cls: type["DistributedModelName"]) -> dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "log_input"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    def _stringify(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, indent=4)
        except Exception:
            return str(value)

    def _update_workflow(self, extra_pnginfo: Any, unique_id: Any, values: list[str]) -> None:
        if not extra_pnginfo:
            return
        info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) else extra_pnginfo
        if not isinstance(info, dict) or "workflow" not in info:
            return
        node_id = None
        if isinstance(unique_id, list) and unique_id:
            node_id = str(unique_id[0])
        elif unique_id is not None:
            node_id = str(unique_id)
        if not node_id:
            return
        workflow = info["workflow"]
        node = next((x for x in workflow["nodes"] if str(x.get("id")) == node_id), None)
        if node:
            node["widgets_values"] = [values]

    def log_input(
        self,
        text: Any,
        unique_id: Any = None,
        extra_pnginfo: Any = None,
    ) -> dict[str, Any]:
        values = []
        if isinstance(text, list):
            for val in text:
                values.append(self._stringify(val))
        else:
            values.append(self._stringify(text))

        # Keep widget display in workflow metadata if available.
        self._update_workflow(extra_pnginfo, unique_id, values)

        if isinstance(values, list) and len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],)}
        return {"ui": {"text": values}, "result": (values,)}

class ByPassTypeTuple(tuple):
    def __getitem__(self, index: int) -> Any:
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return any_type
        return item

class ImageBatchDivider:
    @classmethod
    def INPUT_TYPES(cls: type["ImageBatchDivider"]) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "divide_by": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of parts to divide the batch into"
                }),
            }
        }
    
    RETURN_TYPES = ByPassTypeTuple(("IMAGE", ))  # Flexible for variable outputs
    RETURN_NAMES = ByPassTypeTuple(tuple([f"batch_{i+1}" for i in range(10)]))
    FUNCTION = "divide_batch"
    OUTPUT_NODE = True
    CATEGORY = "image"
    
    def divide_batch(self, images: torch.Tensor, divide_by: int) -> tuple[torch.Tensor, ...]:
        total_splits = max(1, min(int(divide_by), 10))
        total_frames = images.shape[0]
        empty_tensor = images[:0]
        bounds = _chunk_bounds(total_frames, total_splits)
        outputs = [images[start:end] if end > start else empty_tensor for start, end in bounds]

        while len(outputs) < 10:
            outputs.append(empty_tensor)

        return tuple(outputs[:10])


class AudioBatchDivider:
    """Divides an audio waveform into multiple parts along the time/samples dimension."""

    @classmethod
    def INPUT_TYPES(cls: type["AudioBatchDivider"]) -> dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "divide_by": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of parts to divide the audio into"
                }),
            }
        }

    RETURN_TYPES = ByPassTypeTuple(("AUDIO",))  # Flexible for variable outputs
    RETURN_NAMES = ByPassTypeTuple(tuple([f"audio_{i+1}" for i in range(10)]))
    FUNCTION = "divide_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def divide_audio(self, audio: dict[str, Any], divide_by: int) -> tuple[dict[str, Any], ...]:
        import torch

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)

        if waveform is None or waveform.numel() == 0:
            # Return empty audio for all outputs
            empty_audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": sample_rate}
            return tuple([empty_audio] * 10)

        total_splits = max(1, min(int(divide_by), 10))
        total_samples = int(waveform.shape[-1])
        bounds = _chunk_bounds(total_samples, total_splits)

        outputs = []
        empty_waveform = waveform[..., :0]
        for start, end in bounds:
            split_waveform = waveform[..., start:end] if end > start else empty_waveform
            outputs.append({
                "waveform": split_waveform,
                "sample_rate": sample_rate
            })

        # Pad with empty audio up to max (10) to match RETURN_TYPES length
        empty_audio = {
            "waveform": empty_waveform,
            "sample_rate": sample_rate
        }

        while len(outputs) < 10:
            outputs.append(empty_audio)

        return tuple(outputs)


class DistributedEmptyImage:
    """Produces an empty IMAGE batch used when the master delegates all work."""

    @classmethod
    def INPUT_TYPES(cls: type["DistributedEmptyImage"]) -> dict[str, Any]:
        return {
            "required": {
                "height": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "width": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "channels": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create"
    CATEGORY = "image"

    def create(self, height: int, width: int, channels: int) -> tuple[torch.Tensor]:
        import torch

        shape = (0, height, width, channels)
        tensor = torch.zeros(shape, dtype=torch.float32)
        return (tensor,)
