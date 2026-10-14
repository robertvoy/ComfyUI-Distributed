"""Lossless transport of pinned native ComfyUI VIDEO representations.

No save_to/get_components fallback is allowed: an unknown class must not silently
transcode. Tensor components use the shared CPU codec. File bytes are opaque and
retain signed trim/crop parameters; decoded files own independent BytesIOs.
VideoFromList's native constructor flattens its inputs, so wire segments are a
bounded, flat list of exact VideoFromFile objects, never recursive playlists.

Offsets are relative to the containing VIDEO payload. Canonical part order is
images/alpha/audio, or segments/complete_audio. Every byte is covered exactly
once. All descriptor structure and tensor sizes are preflighted before decoding
any media storage. File containers are not parsed until a native consumer reads
them; this adapter does not attempt to validate or repair container codecs.
"""
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import stat

from .codec import MAX_PAYLOAD_BYTES, decode_payload, encode_payload


MAX_SEGMENTS = 64
MAX_METADATA_BYTES = 64 * 1024
MAX_METADATA_DEPTH = 16
_ENCODING = "native-video-v1"
_BASE_FIELDS = {"encoding", "representation", "byte_length"}
_COMPONENT_FIELDS = {"images", "alpha", "audio", "frame_rate", "metadata", "bit_depth", "color_space"}
_FILE_FIELDS = {"data", "start_time", "duration", "crop"}
_SEGMENT_FIELDS = {"segments", "complete_audio"}
_TENSOR_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


def _limit(max_bytes):
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("VIDEO max_bytes must be a nonnegative integer")
    return min(max_bytes, MAX_PAYLOAD_BYTES)


def _native_types():
    # Lazy imports keep non-VIDEO users independent of ComfyUI and PyAV.
    from comfy_api.latest._input_impl.video_types import (
        VideoFromComponents, VideoFromFile, VideoFromList,
    )
    from comfy_api.latest._util import VideoComponents
    return VideoFromComponents, VideoFromFile, VideoFromList, VideoComponents


def _fields(value, fields, label):
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"Invalid VIDEO {label} fields")


def _integer(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("VIDEO length, offset and dimensions must be integers in range")
    return value


def _number(value):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("VIDEO trim values must be finite numbers")
    return value


def _crop(value, *, wire=False):
    if value is None:
        return None
    if type(value) is not (list if wire else tuple) or len(value) != 4:
        raise ValueError("VIDEO crop must contain four integers")
    if any(type(part) is not int for part in value):
        raise ValueError("VIDEO crop must contain four integers")
    # Native crop normalization is a consumer concern, not a transport operation.
    return list(value) if not wire else tuple(value)


def _frame_rate(value):
    _fields(value, {"numerator", "denominator"}, "frame_rate")
    _integer(value["numerator"], positive=True)
    _integer(value["denominator"], positive=True)


def _encoding_options(bit_depth, color_space):
    _integer(bit_depth, positive=True)
    if type(color_space) is not str or color_space not in ("sRGB", "HDR", "HDR PQ"):
        raise ValueError("Unsupported native VIDEO color_space")


def _metadata(value):
    """Validate bounded JSON before allocating its serialized form; return a copy."""
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("VIDEO metadata must be a JSON object or null")
    remaining = MAX_METADATA_BYTES

    def charge(size):
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("VIDEO metadata exceeds byte limit")

    def visit(item, depth):
        if depth > MAX_METADATA_DEPTH:
            raise ValueError("VIDEO metadata exceeds depth limit")
        kind = type(item)
        if kind in (dict, list):
            # Each member costs at least one JSON byte. Also bounds iteration
            # before walking potentially huge containers or cyclic structures.
            if len(item) > remaining:
                raise ValueError("VIDEO metadata exceeds byte limit")
            charge(2 + max(0, len(item) - 1))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("VIDEO metadata keys must be strings")
                    visit(key, depth + 1)
                    charge(1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        elif kind in (str, int, float, bool) or item is None:
            if kind is str and len(item) > remaining:
                raise ValueError("VIDEO metadata exceeds byte limit")
            # Avoid converting an unbounded Python integer to decimal first.
            if kind is int and item.bit_length() > MAX_METADATA_BYTES * 4:
                raise ValueError("VIDEO metadata exceeds byte limit")
            if kind is float and not math.isfinite(item):
                raise ValueError("VIDEO metadata numbers must be finite")
            try:
                charge(len(json.dumps(item, ensure_ascii=True, allow_nan=False)))
            except (ValueError, OverflowError):
                raise ValueError("Invalid or oversized VIDEO metadata scalar") from None
        else:
            raise ValueError("VIDEO metadata must contain only JSON values")

    visit(value, 0)
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))


def _attrs(value, names):
    attributes = vars(value)
    if not all(name in attributes for name in names):
        raise ValueError("Native VIDEO attributes do not match the pinned adapter")
    return [attributes[name] for name in names]


def _roots(allowed_roots):
    if allowed_roots is None:
        import folder_paths
        allowed_roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(),
                         folder_paths.get_temp_directory()]
    if type(allowed_roots) not in (list, tuple):
        raise ValueError("VIDEO allowed_roots must be a list or tuple")
    try:
        roots = [Path(root).resolve() for root in allowed_roots]
    except (OSError, TypeError, ValueError, RuntimeError):
        raise ValueError("Invalid VIDEO allowed roots") from None
    return roots


def _read_path(value, limit, allowed_roots):
    """Resolve authorization, reject special files, and bound all reads.

    Open each resolved path component with NOFOLLOW using directory fds, so an
    intermediate symlink swapped in after resolve() cannot bypass authorization.
    NONBLOCK prevents an attacker swapping in a FIFO from hanging the worker.
    """
    roots = _roots(allowed_roots)
    directory_fd = file_fd = None
    try:
        path = Path(value).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError("VIDEO file is outside allowed roots")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(path.anchor, directory_flags)
        for part in path.parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("VIDEO source must be a regular file")
        if info.st_size > limit:
            raise ValueError("VIDEO file exceeds byte limit")
        # Read at most the stat size plus a sentinel; reject growth or truncation.
        chunks = []
        count = 0
        while count <= info.st_size:
            chunk = os.read(file_fd, min(1024 * 1024, info.st_size + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
        if count != info.st_size:
            raise ValueError("VIDEO source changed size during bounded read")
        return b"".join(chunks)
    except (OSError, RuntimeError):
        raise ValueError("VIDEO file cannot be read safely") from None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _file_bytes(value, limit, allowed_roots):
    if type(value) is io.BytesIO:
        try:
            with value.getbuffer() as view:
                if view.nbytes > limit:
                    raise ValueError("VIDEO file exceeds byte limit")
                return bytes(view)
        except ValueError:
            raise ValueError("VIDEO BytesIO is closed or exceeds byte limit") from None
    if isinstance(value, (str, os.PathLike)):
        return _read_path(value, limit, allowed_roots)
    raise ValueError("VIDEO file must be a local path or exact BytesIO")


def encode_video(value, *, max_bytes=MAX_PAYLOAD_BYTES, allowed_roots=None):
    """Encode an exact native VIDEO without opening/transcoding its container.

    Paths are authorized only under allowed_roots (when supplied), otherwise
    under ComfyUI input/output/temp roots. Budgets may lower, never raise, the
    shared maximum. Unsupported native state raises ValueError, not a fallback.
    Non-CPU tensors are rejected; runtime staging belongs to the bridge boundary.
    """
    return _encode_video(value, max_bytes=max_bytes, allowed_roots=allowed_roots, stage_cpu=False)


def encode_video_cpu(value, *, max_bytes=MAX_PAYLOAD_BYTES, allowed_roots=None):
    """Bridge-only boundary: detach/stage native tensors, then losslessly encode.

    Share the exact-type adapter with encode_video rather than reconstructing
    native videos, calling get_components, or duplicating private attributes.
    Images, alpha and both audio representations move without dtype conversion;
    metadata/options and opaque file bytes/trim/crop take the unchanged path.
    """
    return _encode_video(value, max_bytes=max_bytes, allowed_roots=allowed_roots, stage_cpu=True)


def _stage_tensor_cpu(value):
    import torch
    if not isinstance(value, torch.Tensor):
        raise ValueError("Tensor value must be a torch.Tensor")
    return value.detach().cpu()


def _encode_video(value, *, max_bytes, allowed_roots, stage_cpu):
    limit = _limit(max_bytes)
    components_class, file_class, list_class, components_type = _native_types()
    descriptor = {"encoding": _ENCODING}
    chunks = []
    size = 0

    def add(payload, nested=None):
        nonlocal size
        if len(payload) > limit - size:
            raise ValueError("VIDEO payload exceeds byte limit")
        slot = {"offset": size, "length": len(payload)}
        if nested is not None:
            slot["descriptor"] = nested
        chunks.append(payload)
        size += len(payload)
        return slot

    def tensor(kind, tensor_value):
        if stage_cpu:
            if kind == "AUDIO":
                if not isinstance(tensor_value, dict) or set(tensor_value) != {"waveform", "sample_rate"}:
                    raise ValueError("AUDIO value must contain waveform and sample_rate")
                tensor_value = {**tensor_value, "waveform": _stage_tensor_cpu(tensor_value["waveform"])}
            else:
                tensor_value = _stage_tensor_cpu(tensor_value)
        nested, payload = encode_payload(kind, tensor_value, max_bytes=limit - size)
        return add(payload, nested)

    if type(value) is components_class:
        component, bit_depth, color_space = _attrs(value, [
            "_VideoFromComponents__components", "_VideoFromComponents__bit_depth",
            "_VideoFromComponents__color_space",
        ])
        if type(component) is not components_type:
            raise ValueError("Unsupported native VIDEO components class")
        images, fps, audio, metadata, alpha = _attrs(component, [
            "images", "frame_rate", "audio", "metadata", "alpha",
        ])
        if type(fps) is not Fraction or fps <= 0:
            raise ValueError("VIDEO frame_rate must be a positive Fraction")
        _encoding_options(bit_depth, color_space)
        descriptor.update(representation="components", bit_depth=bit_depth, color_space=color_space,
                          frame_rate={"numerator": fps.numerator, "denominator": fps.denominator},
                          metadata=_metadata(metadata))
        descriptor["images"] = tensor("IMAGE", images)
        # Native get_components() returns BHW1 alpha; MASK nodes use BHW.
        # Select an existing lossless tensor layout without squeezing/converting.
        alpha_kind = "IMAGE" if getattr(alpha, "ndim", None) == 4 else "MASK"
        descriptor["alpha"] = tensor(alpha_kind, alpha) if alpha is not None else None
        if alpha is not None:
            _alpha_header(descriptor["alpha"], descriptor["images"])
        descriptor["audio"] = tensor("AUDIO", audio) if audio is not None else None
    elif type(value) is file_class:
        source, start, duration, crop = _attrs(value, [
            "_VideoFromFile__file", "_VideoFromFile__start_time", "_VideoFromFile__duration",
            "_VideoFromFile__crop",
        ])
        descriptor.update(representation="file", start_time=_number(start), duration=_number(duration),
                          crop=_crop(crop))
        descriptor["data"] = add(_file_bytes(source, limit, allowed_roots))
    elif type(value) is list_class:
        videos, complete_audio = _attrs(value, ["videos", "complete_audio"])
        if type(videos) is not list or not 1 <= len(videos) <= MAX_SEGMENTS:
            raise ValueError("VIDEO must have between 1 and 64 segments")
        if any(type(segment) is not file_class for segment in videos):
            raise ValueError("VIDEO segments must be exact native VideoFromFile objects")
        descriptor.update(representation="segments", segments=[])
        for segment in videos:
            nested, payload = encode_video(segment, max_bytes=limit - size, allowed_roots=allowed_roots)
            descriptor["segments"].append(add(payload, nested))
        descriptor["complete_audio"] = tensor("AUDIO", complete_audio) if complete_audio is not None else None
    else:
        raise ValueError("Unsupported native VIDEO class")
    descriptor["byte_length"] = size
    return descriptor, b"".join(chunks)


def _tensor_header(kind, descriptor, length):
    """Allocation-free preflight of the shared tensor-le-v1 descriptor.

    The shared decoder still validates the bytes/finite values. Doing its small
    structural check here first prevents allocation for an early part when a
    later part's descriptor is already known to be invalid.
    """
    fields = {"encoding", "dtype", "shape", "layout"}
    if kind == "AUDIO":
        fields.add("sample_rate")
    _fields(descriptor, fields, "tensor")
    layout = {"IMAGE": "BHWC", "MASK": "BHW", "AUDIO": "BCT"}[kind]
    if type(descriptor["encoding"]) is not str or descriptor["encoding"] != "tensor-le-v1":
        raise ValueError("Invalid VIDEO tensor encoding")
    if type(descriptor["layout"]) is not str or descriptor["layout"] != layout:
        raise ValueError("Invalid VIDEO tensor layout")
    dtype = descriptor["dtype"]
    if type(dtype) is not str or dtype not in _TENSOR_BYTES:
        raise ValueError("Invalid VIDEO tensor dtype")
    shape = descriptor["shape"]
    if type(shape) is not list or len(shape) != len(layout):
        raise ValueError("Invalid VIDEO tensor rank")
    size = _TENSOR_BYTES[dtype]
    for dimension in shape:
        _integer(dimension, positive=True)
        if dimension > length // size:
            raise ValueError("VIDEO tensor shape exceeds byte limit")
        size *= dimension
    if kind == "IMAGE" and shape[-1] not in (1, 3, 4):
        raise ValueError("Invalid VIDEO image channels")
    if size != length:
        raise ValueError("VIDEO tensor length does not match shape")
    if kind == "AUDIO":
        _integer(descriptor["sample_rate"], positive=True)


def _alpha_header(alpha, images):
    _fields(alpha, {"offset", "length", "descriptor"}, "alpha part")
    descriptor = alpha["descriptor"]
    if type(descriptor) is not dict:
        raise ValueError("Invalid VIDEO alpha descriptor")
    layout = descriptor.get("layout")
    if type(layout) is not str or layout not in ("BHW", "BHWC"):
        raise ValueError("Invalid VIDEO alpha layout")
    kind = "MASK" if layout == "BHW" else "IMAGE"
    _tensor_header(kind, descriptor, _integer(alpha["length"]))
    expected = images["descriptor"]["shape"][:3] + ([1] if kind == "IMAGE" else [])
    if descriptor["shape"] != expected:
        raise ValueError("VIDEO alpha shape must match images")
    return kind


def _validate_video(descriptor, size, *, file_only=False):
    if type(descriptor) is not dict:
        raise ValueError("VIDEO descriptor must be a dictionary")
    representation = descriptor.get("representation")
    if type(representation) is not str or representation not in ("components", "file", "segments"):
        raise ValueError("Unsupported VIDEO representation")
    if file_only and representation != "file":
        # Depth is at most one: native lists already flatten their segments.
        raise ValueError("VIDEO segments cannot contain nested lists or components")
    fields = {"components": _COMPONENT_FIELDS, "file": _FILE_FIELDS, "segments": _SEGMENT_FIELDS}[representation]
    _fields(descriptor, _BASE_FIELDS | fields, "descriptor")
    if type(descriptor["encoding"]) is not str or descriptor["encoding"] != _ENCODING:
        raise ValueError("Unsupported VIDEO encoding")
    if _integer(descriptor["byte_length"]) != size:
        raise ValueError("VIDEO payload length does not match descriptor")
    result = dict(descriptor)
    cursor = 0

    def part(slot, kind=None, video=False):
        nonlocal cursor
        fields = {"offset", "length", "descriptor"} if kind or video else {"offset", "length"}
        _fields(slot, fields, "part")
        offset = _integer(slot["offset"])
        length = _integer(slot["length"])
        if offset != cursor or length > size - cursor:
            raise ValueError("VIDEO parts must not overlap, leave gaps or exceed bounds")
        cursor += length
        if kind:
            _tensor_header(kind, slot["descriptor"], length)
        if video:
            return {**slot, "descriptor": _validate_video(slot["descriptor"], length, file_only=True)}
        return slot

    if representation == "components":
        _frame_rate(descriptor["frame_rate"])
        _encoding_options(descriptor["bit_depth"], descriptor["color_space"])
        result["metadata"] = _metadata(descriptor["metadata"])
        part(descriptor["images"], "IMAGE")
        if descriptor["alpha"] is not None:
            part(descriptor["alpha"], _alpha_header(descriptor["alpha"], descriptor["images"]))
        if descriptor["audio"] is not None:
            part(descriptor["audio"], "AUDIO")
    elif representation == "file":
        _number(descriptor["start_time"])
        _number(descriptor["duration"])
        _crop(descriptor["crop"], wire=True)
        part(descriptor["data"])
    else:
        segments = descriptor["segments"]
        if type(segments) is not list or not 1 <= len(segments) <= MAX_SEGMENTS:
            raise ValueError("VIDEO must have between 1 and 64 segments")
        result["segments"] = [part(segment, video=True) for segment in segments]
        if descriptor["complete_audio"] is not None:
            part(descriptor["complete_audio"], "AUDIO")
    if cursor != size:
        raise ValueError("VIDEO payload contains trailing bytes")
    return result


def _decode_validated(descriptor, payload, types):
    components_class, file_class, list_class, components_type = types

    def data(slot):
        return payload[slot["offset"]:slot["offset"] + slot["length"]]

    def tensor(kind, slot):
        if slot is None:
            return None
        return decode_payload(kind, slot["descriptor"], data(slot), max_bytes=slot["length"])

    if descriptor["representation"] == "file":
        # VideoFromFile retains the BytesIO, unlike a temporary filesystem path.
        return file_class(io.BytesIO(data(descriptor["data"])), start_time=descriptor["start_time"],
                          duration=descriptor["duration"], crop=_crop(descriptor["crop"], wire=True))
    if descriptor["representation"] == "components":
        fps = descriptor["frame_rate"]
        return components_class(components_type(
            images=tensor("IMAGE", descriptor["images"]),
            frame_rate=Fraction(fps["numerator"], fps["denominator"]),
            audio=tensor("AUDIO", descriptor["audio"]),
            metadata=descriptor["metadata"],
            alpha=tensor("IMAGE" if descriptor["alpha"] is not None and
                         descriptor["alpha"]["descriptor"]["layout"] == "BHWC" else "MASK",
                         descriptor["alpha"]),
        ), bit_depth=descriptor["bit_depth"], color_space=descriptor["color_space"])
    videos = [_decode_validated(slot["descriptor"], data(slot), types) for slot in descriptor["segments"]]
    return list_class(videos, complete_audio=tensor("AUDIO", descriptor["complete_audio"]))


def decode_video(descriptor, payload, *, max_bytes=MAX_PAYLOAD_BYTES):
    """Decode a bounded VIDEO into independently owned native CPU media objects."""
    limit = _limit(max_bytes)
    if type(payload) is not bytes:
        raise ValueError("VIDEO payload must be bytes")
    if len(payload) > limit:
        raise ValueError("VIDEO payload exceeds byte limit")
    validated = _validate_video(descriptor, len(payload))
    return _decode_validated(validated, payload, _native_types())
