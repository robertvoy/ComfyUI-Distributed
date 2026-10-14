"""Lossless, bounded CPU payloads shared with the standalone endpoint bridge.

Descriptors contain only kind-specific metadata. Tensor bytes are contiguous,
little-endian floating-point values, never a pickle or an image/audio container.
The optional byte budget can lower, but never raise, MAX_PAYLOAD_BYTES.
"""

import ctypes
import sys

import torch


MAX_PAYLOAD_BYTES = 256 * 1024 * 1024

_DTYPES = {
    "float16": (torch.float16, 2),
    "bfloat16": (torch.bfloat16, 2),
    "float32": (torch.float32, 4),
    "float64": (torch.float64, 8),
}
_DTYPE_NAMES = {dtype: name for name, (dtype, _) in _DTYPES.items()}
_LAYOUTS = {"IMAGE": "BHWC", "MASK": "BHW", "AUDIO": "BCT"}


def _validate_kind(kind):
    if type(kind) is not str or kind not in ("IMAGE", "MASK", "STRING", "AUDIO", "VALUE"):
        raise ValueError("Unsupported payload kind")


def _byte_limit(max_bytes):
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    return min(max_bytes, MAX_PAYLOAD_BYTES)


def _sample_rate(value):
    if type(value) is not int or value <= 0:
        raise ValueError("AUDIO sample_rate must be a positive integer")
    return value


def _tensor_size(kind, shape, itemsize, limit):
    if type(shape) is not list or len(shape) != len(_LAYOUTS[kind]):
        raise ValueError("Tensor shape has an invalid rank")
    if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError("Tensor shape dimensions must be positive integers")
    if kind == "IMAGE" and shape[-1] not in (1, 3, 4):
        raise ValueError("IMAGE shape must have 1, 3 or 4 channels")
    size = itemsize
    for dimension in shape:
        # Divide before multiplying, including for arbitrary-size Python ints.
        if dimension > limit // size:
            raise ValueError("Tensor payload exceeds byte limit")
        size *= dimension
    return size


def _native_little_endian(payload, itemsize):
    """Convert native to wire byte order, or vice versa (an involution)."""
    if sys.byteorder == "little":
        return payload
    swapped = bytearray(len(payload))
    for offset in range(itemsize):
        swapped[offset::itemsize] = payload[itemsize - 1 - offset::itemsize]
    return bytes(swapped)


def _encode_string(value, limit):
    if not isinstance(value, str):
        raise ValueError("STRING value must be text")
    # Check the exact UTF-8 size before allocating the encoded payload.
    if len(value) > limit:
        raise ValueError("STRING payload exceeds byte limit")
    size = 0
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError("STRING value must be valid UTF-8 text")
        size += 1 + (codepoint > 0x7F) + (codepoint > 0x7FF) + (codepoint > 0xFFFF)
        if size > limit:
            raise ValueError("STRING payload exceeds byte limit")
    return {"encoding": "utf8-v1"}, value.encode("utf-8")


def encode_payload(kind: str, value, *, max_bytes: int = MAX_PAYLOAD_BYTES) -> tuple[dict, bytes]:
    """Encode a typed value without dtype conversion, normalization or GPU use.

    IMAGE/MASK accept dense CPU tensors; AUDIO accepts a waveform CPU tensor
    plus a positive integer sample_rate. STRING accepts exact Unicode text.
    Invalid values and exceeded budgets raise nonsensitive ValueError messages.
    """
    _validate_kind(kind)
    limit = _byte_limit(max_bytes)
    if kind == "VALUE":
        from .value_codec import encode_value
        return encode_value(value, max_bytes=limit)
    if kind == "STRING":
        return _encode_string(value, limit)

    sample_rate = None
    if kind == "AUDIO":
        if not isinstance(value, dict) or set(value) != {"waveform", "sample_rate"}:
            raise ValueError("AUDIO value must contain waveform and sample_rate")
        sample_rate = _sample_rate(value["sample_rate"])
        value = value["waveform"]
    if not isinstance(value, torch.Tensor):
        raise ValueError("Tensor value must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError("Tensor value must be on CPU")
    if value.layout != torch.strided or value.is_nested:
        raise ValueError("Tensor value must have a dense strided layout")
    if value.dtype not in _DTYPE_NAMES:
        raise ValueError("Unsupported tensor dtype")

    dtype_name = _DTYPE_NAMES[value.dtype]
    _, itemsize = _DTYPES[dtype_name]
    shape = list(value.shape)
    size = _tensor_size(kind, shape, itemsize, limit)
    tensor = value.detach().resolve_neg().contiguous()
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Tensor values must be finite")
    # A uint16 view preserves BF16 bits without conversion through float32.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    # The live, contiguous CPU tensor owns this pointer for the entire copy.
    payload = _native_little_endian(ctypes.string_at(tensor.data_ptr(), size), itemsize)
    descriptor = {
        "encoding": "tensor-le-v1", "dtype": dtype_name,
        "shape": shape, "layout": _LAYOUTS[kind],
    }
    if kind == "AUDIO":
        descriptor["sample_rate"] = sample_rate
    return descriptor, payload


def decode_payload(kind: str, descriptor: dict, payload: bytes, *, max_bytes: int = MAX_PAYLOAD_BYTES):
    """Decode validated wire bytes into independent contiguous CPU tensors/text.

    All metadata, shape products and byte lengths are checked before tensor
    allocation. No GPU device discovery, transfers or object deserialization
    occur. AUDIO returns {'waveform': tensor, 'sample_rate': int}.
    """
    _validate_kind(kind)
    limit = _byte_limit(max_bytes)
    if not isinstance(descriptor, dict):
        raise ValueError("Payload descriptor must be a dictionary")
    if not isinstance(payload, bytes):
        raise ValueError("Payload must be bytes")
    if len(payload) > limit:
        raise ValueError("Payload exceeds byte limit")
    if kind == "VALUE":
        from .value_codec import decode_value
        return decode_value(descriptor, payload, max_bytes=limit)
    if kind == "IMAGE" and descriptor.get("encoding") == "image-transfer-v1":
        # Lazy and IMAGE-only: exact tensors and other kinds stay unchanged.
        from .image_codec import decode_image
        return decode_image(descriptor, payload, max_bytes=limit)
    if kind == "STRING":
        if set(descriptor) != {"encoding"} or descriptor["encoding"] != "utf8-v1":
            raise ValueError("Invalid STRING descriptor encoding or fields")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("STRING payload must be valid UTF-8 bytes") from None

    fields = {"encoding", "dtype", "shape", "layout"}
    if kind == "AUDIO":
        fields.add("sample_rate")
    if set(descriptor) != fields:
        raise ValueError("Invalid tensor descriptor fields")
    if descriptor["encoding"] != "tensor-le-v1":
        raise ValueError("Unsupported tensor descriptor encoding")
    if descriptor["layout"] != _LAYOUTS[kind]:
        raise ValueError("Invalid tensor descriptor layout")
    dtype_name = descriptor["dtype"]
    if type(dtype_name) is not str or dtype_name not in _DTYPES:
        raise ValueError("Unsupported tensor descriptor dtype")
    dtype, itemsize = _DTYPES[dtype_name]
    size = _tensor_size(kind, descriptor["shape"], itemsize, limit)
    if len(payload) != size:
        raise ValueError("Tensor payload length does not match shape and dtype")
    sample_rate = _sample_rate(descriptor["sample_rate"]) if kind == "AUDIO" else None

    # frombuffer is always CPU. Copy immutable wire bytes into a writable private
    # buffer, then clone so the returned tensor owns its storage independently.
    buffer = bytearray(_native_little_endian(payload, itemsize))
    storage_dtype = torch.uint16 if dtype == torch.bfloat16 else dtype
    tensor = torch.frombuffer(buffer, dtype=storage_dtype).view(dtype)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Tensor payload values must be finite")
    tensor = tensor.clone().reshape(descriptor["shape"])
    if kind == "AUDIO":
        return {"waveform": tensor, "sample_rate": sample_rate}
    return tensor
