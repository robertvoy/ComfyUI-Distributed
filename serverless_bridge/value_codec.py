"""Exact six-type values for internal sparse End exports; no pack imports.

Tensor/text payloads reuse the existing media codec. Scalars have distinct
encodings: BOOLEAN is one byte, INT is minimal signed little-endian, FLOAT is
IEEE binary64. No scalar passes through float() or a lossy JSON number parser.
"""
import math
import struct

import torch


KINDS = frozenset(("IMAGE", "MASK", "STRING", "BOOLEAN", "INT", "FLOAT"))
_SCALAR_ENCODINGS = {"BOOLEAN": "bool-v1", "INT": "int-le-v1", "FLOAT": "float64-le-v1"}


def value_kind(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 4 and value.shape[-1] in (3, 4):
            return "IMAGE"
        if value.ndim == 3:
            return "MASK"
        raise ValueError("VALUE tensor requires IMAGE [B,H,W,3|4] or MASK [B,H,W]")
    kind = {bool: "BOOLEAN", int: "INT", float: "FLOAT", str: "STRING"}.get(type(value))
    if kind is None:
        raise ValueError("VALUE requires IMAGE, MASK, STRING, INT, FLOAT or BOOLEAN")
    if kind == "FLOAT" and not math.isfinite(value):
        raise ValueError("VALUE numbers must be finite")
    return kind


def encode_value(value, *, max_bytes):
    from .codec import encode_payload
    kind = value_kind(value)
    if kind in ("IMAGE", "MASK", "STRING"):
        descriptor, payload = encode_payload(kind, value, max_bytes=max_bytes)
    else:
        descriptor = {"encoding": _SCALAR_ENCODINGS[kind]}
        size = ((value if value >= 0 else ~value).bit_length() + 8) // 8 if kind == "INT" else (8 if kind == "FLOAT" else 1)
        if size > max_bytes:
            raise ValueError("VALUE payload exceeds byte limit")
        if kind == "BOOLEAN":
            payload = b"\x01" if value else b"\x00"
        elif kind == "INT":
            payload = value.to_bytes(size, "little", signed=True)
        else:
            payload = struct.pack("<d", value)
    return {"encoding": "value-v1", "type": kind, "descriptor": descriptor}, payload


def decode_value(descriptor, payload, *, max_bytes):
    from .codec import decode_payload
    if (set(descriptor) != {"encoding", "type", "descriptor"}
            or descriptor["encoding"] != "value-v1"
            or type(descriptor["type"]) is not str or descriptor["type"] not in KINDS):
        raise ValueError("Invalid VALUE descriptor")
    kind, inner = descriptor["type"], descriptor["descriptor"]
    if kind in ("IMAGE", "MASK", "STRING"):
        # Generic values always stay exact even if End.images is compressed.
        if not isinstance(inner, dict) or inner.get("encoding") != (
                "utf8-v1" if kind == "STRING" else "tensor-le-v1"):
            raise ValueError("VALUE requires exact typed payloads")
        value = decode_payload(kind, inner, payload, max_bytes=max_bytes)
        if value_kind(value) != kind:
            raise ValueError("VALUE tensor/type mismatch")
        return value
    if inner != {"encoding": _SCALAR_ENCODINGS[kind]}:
        raise ValueError("Invalid VALUE scalar encoding")
    if kind == "BOOLEAN":
        if payload not in (b"\x00", b"\x01"):
            raise ValueError("Invalid BOOLEAN payload")
        return payload == b"\x01"
    if kind == "FLOAT":
        if len(payload) != 8:
            raise ValueError("Invalid FLOAT payload length")
        value = struct.unpack("<d", payload)[0]
        if not math.isfinite(value):
            raise ValueError("VALUE numbers must be finite")
        return value
    if not payload:
        raise ValueError("Invalid INT payload length")
    value = int.from_bytes(payload, "little", signed=True)
    if ((value if value >= 0 else ~value).bit_length() + 8) // 8 != len(payload):
        raise ValueError("Invalid INT payload length")
    return value
