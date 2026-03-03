import base64
import binascii
import os
from typing import Any

import numpy as np
import torch

from .image import ensure_contiguous


MAX_AUDIO_PAYLOAD_BYTES = int(
    os.environ.get("COMFYUI_MAX_AUDIO_PAYLOAD_BYTES", str(256 * 1024 * 1024))
)


def encode_audio_payload(audio_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialize an AUDIO dict into JSON-safe canonical envelope payload."""
    if not isinstance(audio_payload, dict):
        return None

    waveform = audio_payload.get("waveform")
    if waveform is None or not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        return None

    sample_rate = audio_payload.get("sample_rate", 44100)
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        sample_rate = 44100

    waveform_cpu = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    data_bytes = waveform_cpu.numpy().tobytes()
    if len(data_bytes) > MAX_AUDIO_PAYLOAD_BYTES:
        raise ValueError(
            f"Audio payload too large: {len(data_bytes)} bytes exceeds {MAX_AUDIO_PAYLOAD_BYTES}."
        )

    return {
        "sample_rate": sample_rate,
        "shape": [int(dim) for dim in waveform_cpu.shape],
        "dtype": "float32",
        "data": base64.b64encode(data_bytes).decode("ascii"),
    }


def decode_audio_payload(audio_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Decode canonical envelope audio payload into an AUDIO dict."""
    if audio_payload is None:
        return None
    if not isinstance(audio_payload, dict):
        raise ValueError("Field 'audio' must be an object when provided.")

    encoded = audio_payload.get("data")
    shape = audio_payload.get("shape")
    sample_rate = audio_payload.get("sample_rate", 44100)
    dtype = audio_payload.get("dtype", "float32")

    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError("Field 'audio.data' must be a non-empty base64 string.")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("Field 'audio.shape' must be a 3-item list [batch, channels, samples].")
    if dtype != "float32":
        raise ValueError("Field 'audio.dtype' must be 'float32'.")

    try:
        shape_tuple = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("Field 'audio.shape' must contain integers.") from exc

    if shape_tuple[0] <= 0 or shape_tuple[1] <= 0 or shape_tuple[2] < 0:
        raise ValueError(
            "Field 'audio.shape' must be [batch>0, channels>0, samples>=0]."
        )

    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("Field 'audio.sample_rate' must be an integer.") from exc
    if sample_rate <= 0:
        raise ValueError("Field 'audio.sample_rate' must be positive.")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Field 'audio.data' is not valid base64.") from exc

    if len(raw) > MAX_AUDIO_PAYLOAD_BYTES:
        raise ValueError(
            f"Field 'audio.data' too large: {len(raw)} bytes exceeds {MAX_AUDIO_PAYLOAD_BYTES}."
        )

    expected_bytes = int(np.prod(shape_tuple, dtype=np.int64)) * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Field 'audio.data' byte size mismatch: expected {expected_bytes}, got {len(raw)}."
        )

    array = np.frombuffer(raw, dtype=np.float32).reshape(shape_tuple)
    waveform = torch.from_numpy(array.copy())
    return {
        "waveform": ensure_contiguous(waveform),
        "sample_rate": sample_rate,
    }
