import importlib.util
import math
import struct
import types
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "serverless_bridge" / "codec.py"
spec = importlib.util.spec_from_file_location("dist_media_codec_test", MODULE_PATH)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
SHAPES = [
    ("IMAGE", (2, 3, 5, 1), "BHWC"),
    ("IMAGE", (2, 3, 5, 3), "BHWC"),
    ("IMAGE", (2, 3, 5, 4), "BHWC"),
    ("MASK", (2, 3, 5), "BHW"),
    ("AUDIO", (2, 1, 17), "BCT"),
    ("AUDIO", (2, 6, 17), "BCT"),
]


def audio(tensor, sample_rate=48000):
    return {"waveform": tensor, "sample_rate": sample_rate}


def descriptor(_kind="MASK", **changes):
    result = {
        "encoding": "tensor-le-v1", "dtype": "float32",
        "shape": [1, 1, 1], "layout": "BHW",
    }
    if _kind == "IMAGE":
        result.update(shape=[1, 1, 1, 3], layout="BHWC")
    elif _kind == "AUDIO":
        result.update(layout="BCT", sample_rate=48000)
    result.update(changes)
    return result


@pytest.fixture(autouse=True)
def forbid_gpu_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("codec must not query or initialize a GPU")
    for name in ("is_available", "init", "_lazy_init", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind,shape,layout", SHAPES)
def test_exact_roundtrip_preserves_dtype_shape_channels_and_bits(dtype, kind, shape, layout):
    count = math.prod(shape)
    tensor = (torch.arange(count, device="cpu", dtype=torch.float64) / 997 - 0.125).to(dtype).reshape(shape)
    tensor.flatten()[0] = -0.0
    tensor.requires_grad_(True)
    value = audio(tensor) if kind == "AUDIO" else tensor

    metadata, payload = codec.encode_payload(kind, value)
    expected = {
        "encoding": "tensor-le-v1", "dtype": str(dtype).removeprefix("torch."),
        "shape": list(shape), "layout": layout,
    }
    if kind == "AUDIO":
        expected["sample_rate"] = 48000
    assert metadata == expected
    assert type(payload) is bytes
    assert len(payload) == count * tensor.element_size()
    result = codec.decode_payload(kind, metadata, payload)
    if kind == "AUDIO":
        assert set(result) == {"waveform", "sample_rate"}
        assert result["sample_rate"] == 48000
        result = result["waveform"]
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    assert not result.requires_grad
    assert result.dtype == dtype
    assert result.shape == shape
    assert torch.equal(result.view(torch.uint8), tensor.detach().view(torch.uint8))


@pytest.mark.parametrize("dtype,fmt,expected_bits", [
    (torch.float16, "e", None), (torch.bfloat16, "H", [0x3F80, 0x8000, 0xC020]),
    (torch.float32, "f", None), (torch.float64, "d", None),
])
def test_wire_bytes_are_little_endian(dtype, fmt, expected_bits):
    values = [1.0, -0.0, -2.5]
    tensor = torch.tensor(values, dtype=dtype, device="cpu").reshape(1, 1, 3)
    metadata, payload = codec.encode_payload("MASK", tensor)
    expected = struct.pack("<3" + fmt, *(expected_bits if expected_bits else values))
    assert payload == expected
    assert torch.equal(codec.decode_payload("MASK", metadata, expected).view(torch.uint8), tensor.view(torch.uint8))


def test_bfloat16_extremes_and_subnormal_bits_are_lossless():
    bits = torch.tensor([0, 0x8000, 1, 0x8001, 0x3F81, 0x7F7F, 0xFF7F], dtype=torch.uint16, device="cpu")
    value = bits.view(torch.bfloat16).reshape(1, 1, -1)
    metadata, payload = codec.encode_payload("MASK", value)
    assert payload == struct.pack("<7H", *bits.tolist())
    assert torch.equal(codec.decode_payload("MASK", metadata, payload).view(torch.uint16).flatten(), bits)


def test_fractional_mask_is_not_quantized_clipped_or_normalized():
    value = torch.tensor([0.1234567890123, 1.0000000000001, -2.5], dtype=torch.float64, device="cpu").reshape(1, 1, 3)
    metadata, payload = codec.encode_payload("MASK", value)
    assert torch.equal(codec.decode_payload("MASK", metadata, payload), value)


@pytest.mark.parametrize("text", ["", " ", "\t\n  ", "café 雪 🎵", "e\u0301", "\x00\ufeff\r\n"])
def test_strings_preserve_exact_utf8(text):
    metadata, payload = codec.encode_payload("STRING", text)
    assert metadata == {"encoding": "utf8-v1"}
    assert payload == text.encode("utf-8")
    assert codec.decode_payload("STRING", metadata, payload) == text


def test_noncontiguous_and_lazy_negative_tensors_roundtrip():
    value = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4).transpose(1, 2)
    for tensor in (value, torch._neg_view(value)):
        metadata, payload = codec.encode_payload("MASK", tensor)
        result = codec.decode_payload("MASK", metadata, payload)
        assert result.is_contiguous()
        assert torch.equal(result, tensor)


def test_decode_is_independent_and_ignores_default_device(monkeypatch):
    source = torch.tensor([0.25], device="cpu").reshape(1, 1, 1)
    metadata, payload = codec.encode_payload("MASK", source)
    buffers = []
    original = torch.frombuffer
    def capture(buffer, *args, **kwargs):
        buffers.append(buffer)
        return original(buffer, *args, **kwargs)
    monkeypatch.setattr(torch, "frombuffer", capture)
    with torch.device("meta"):
        first = codec.decode_payload("MASK", metadata, payload)
        second = codec.decode_payload("MASK", metadata, payload)
    assert first.device.type == second.device.type == "cpu"
    first.fill_(9)
    source.fill_(7)
    for buffer in buffers:
        if isinstance(buffer, bytearray):
            buffer[:] = b"\x00" * len(buffer)
    assert second.item() == 0.25
    assert codec.decode_payload("MASK", metadata, payload).item() == 0.25


@pytest.mark.parametrize("kind", ["image", " MASK", "LATENT", "", None, 1, [], {}])
def test_unknown_kinds_raise_value_error(kind):
    with pytest.raises(ValueError, match="kind"):
        codec.encode_payload(kind, "")
    with pytest.raises(ValueError, match="kind"):
        codec.decode_payload(kind, {}, b"")


@pytest.mark.parametrize("kind,value", [
    ("STRING", None), ("STRING", b"text"), ("STRING", 123),
    ("IMAGE", [[[[0.5]]]]), ("MASK", None),
    ("AUDIO", torch.ones(1, 1, 1, device="cpu")), ("AUDIO", {}),
    ("AUDIO", {"waveform": "invalid", "sample_rate": 48000}),
    ("AUDIO", {"waveform": torch.ones(1, 1, 1, device="cpu")}),
])
def test_unsupported_values_raise_value_error(kind, value):
    with pytest.raises(ValueError):
        codec.encode_payload(kind, value)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool, torch.complex64])
def test_unsupported_tensor_dtype(dtype):
    with pytest.raises(ValueError, match="dtype"):
        codec.encode_payload("MASK", torch.ones(1, 1, 1, dtype=dtype, device="cpu"))


@pytest.mark.parametrize("kind,shape", [
    ("IMAGE", (1, 2, 3)), ("IMAGE", (1, 1, 1, 2)), ("IMAGE", (1, 1, 1, 5)),
    ("MASK", (1, 2)), ("MASK", (1, 2, 3, 1)), ("AUDIO", (2, 4)),
    ("IMAGE", (1, 0, 2, 3)), ("MASK", (0, 1, 1)), ("AUDIO", (1, 0, 4)),
    ("AUDIO", (1, 2, 0)),
])
def test_invalid_tensor_shape(kind, shape):
    tensor = torch.empty(shape, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="shape|rank|channel|dimension"):
        codec.encode_payload(kind, audio(tensor) if kind == "AUDIO" else tensor)


def test_rejects_non_cpu_and_sparse_without_transferring(monkeypatch):
    meta = torch.empty((1, 1, 1), device="meta")
    sparse = torch.ones((1, 1, 1), device="cpu").to_sparse()
    def forbidden(*args, **kwargs):
        pytest.fail("codec must not move tensors between devices")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    for tensor in (meta, sparse):
        with pytest.raises(ValueError, match="CPU|layout|strided"):
            codec.encode_payload("MASK", tensor)


@pytest.mark.parametrize("sample_rate", [True, False, 0, -1, 44100.0, "44100", None])
def test_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        codec.encode_payload("AUDIO", audio(torch.ones(1, 1, 1, device="cpu"), sample_rate))
    with pytest.raises(ValueError, match="sample_rate"):
        codec.decode_payload("AUDIO", descriptor("AUDIO", sample_rate=sample_rate), b"\x00" * 4)


@pytest.mark.parametrize("bad", [None, [], "descriptor", {}, {"encoding": "utf8-v1"}])
def test_malformed_tensor_descriptor(bad):
    with pytest.raises(ValueError, match="descriptor"):
        codec.decode_payload("MASK", bad, b"\x00" * 4)


@pytest.mark.parametrize("changes", [
    {"encoding": "pickle"}, {"encoding": None}, {"encoding": []},
    {"dtype": "uint8"}, {"dtype": torch.float32}, {"dtype": []},
    {"layout": "BHWC"}, {"layout": None}, {"layout": []},
    {"shape": (1, 1, 1)}, {"shape": None}, {"shape": "1,1,1"},
    {"shape": [1, 1]}, {"shape": [1, 1, 1, 1]}, {"shape": [1, 0, 1]},
    {"shape": [1, -1, 1]}, {"shape": [1, True, 1]}, {"shape": [1, 1.0, 1]},
    {"shape": [1, "1", 1]}, {"shape": [1, [], 1]}, {"kind": "MASK"},
])
def test_invalid_tensor_descriptor_fields(changes):
    with pytest.raises(ValueError):
        codec.decode_payload("MASK", descriptor(**changes), b"\x00" * 4)


@pytest.mark.parametrize("metadata", [{}, {"encoding": "tensor-le-v1"}, {"encoding": "utf8-v1", "kind": "STRING"}, {"encoding": []}])
def test_invalid_string_descriptor(metadata):
    with pytest.raises(ValueError):
        codec.decode_payload("STRING", metadata, b"")


@pytest.mark.parametrize("payload", [b"", b"\x00" * 3, b"\x00" * 5, b"\x00" * 8])
def test_shape_length_mismatch(payload):
    with pytest.raises(ValueError, match="length"):
        codec.decode_payload("MASK", descriptor(), payload)


@pytest.mark.parametrize("payload", [None, "text", [0, 0, 0, 0], 4])
def test_nonbyte_payload_rejected(payload):
    with pytest.raises(ValueError, match="bytes"):
        codec.decode_payload("MASK", descriptor(), payload)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_fail_on_encode_and_decode(dtype, value):
    tensor = torch.tensor([value], dtype=dtype, device="cpu").reshape(1, 1, 1)
    with pytest.raises(ValueError, match="finite"):
        codec.encode_payload("MASK", tensor)
    fmt = {torch.float16: "e", torch.float32: "f", torch.float64: "d"}
    if dtype == torch.bfloat16:
        payload = struct.pack("<H", 0x7FC0 if math.isnan(value) else (0x7F80 if value > 0 else 0xFF80))
    else:
        payload = struct.pack("<" + fmt[dtype], value)
    metadata = descriptor(dtype=str(dtype).removeprefix("torch."))
    with pytest.raises(ValueError, match="finite"):
        codec.decode_payload("MASK", metadata, payload)


def test_invalid_utf8_errors_do_not_echo_content():
    with pytest.raises(ValueError) as error:
        codec.encode_payload("STRING", "private-value\ud800")
    assert "private-value" not in str(error.value)
    with pytest.raises(ValueError) as error:
        codec.decode_payload("STRING", {"encoding": "utf8-v1"}, b"private-value\xff")
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("max_bytes", [True, False, -1, 4.0, "4", None])
def test_invalid_byte_budget(max_bytes):
    with pytest.raises(ValueError, match="max_bytes"):
        codec.encode_payload("STRING", "", max_bytes=max_bytes)
    with pytest.raises(ValueError, match="max_bytes"):
        codec.decode_payload("STRING", {"encoding": "utf8-v1"}, b"", max_bytes=max_bytes)


def test_exact_byte_budget_and_zero_budget_for_empty_string():
    tensor = torch.ones(1, 1, 1, device="cpu")
    metadata, payload = codec.encode_payload("MASK", tensor, max_bytes=4)
    assert codec.decode_payload("MASK", metadata, payload, max_bytes=4).item() == 1
    with pytest.raises(ValueError, match="limit|budget"):
        codec.encode_payload("MASK", tensor, max_bytes=3)
    with pytest.raises(ValueError, match="limit|budget"):
        codec.decode_payload("MASK", metadata, payload, max_bytes=3)
    metadata, payload = codec.encode_payload("STRING", "", max_bytes=0)
    assert codec.decode_payload("STRING", metadata, payload, max_bytes=0) == ""
    for text in ("a", "é", "🎵"):
        byte_count = len(text.encode("utf-8"))
        metadata, payload = codec.encode_payload("STRING", text, max_bytes=byte_count)
        assert codec.decode_payload("STRING", metadata, payload, max_bytes=byte_count) == text
        with pytest.raises(ValueError, match="limit|budget"):
            codec.encode_payload("STRING", text, max_bytes=byte_count - 1)
        with pytest.raises(ValueError, match="limit|budget"):
            codec.decode_payload("STRING", metadata, payload, max_bytes=byte_count - 1)


def test_oversized_shape_rejected_before_tensor_allocation(monkeypatch):
    assert codec.MAX_PAYLOAD_BYTES == 256 * 1024 * 1024
    def forbidden(*args, **kwargs):
        pytest.fail("oversized shape must fail before tensor allocation")
    for name in ("frombuffer", "empty", "tensor", "zeros", "isfinite"):
        monkeypatch.setattr(torch, name, forbidden)
    for shape in ([1, 1, codec.MAX_PAYLOAD_BYTES], [2**100, 2**100, 1]):
        with pytest.raises(ValueError, match="limit|budget"):
            codec.decode_payload("MASK", descriptor(shape=shape), b"\x00" * 4)


def test_oversized_encode_rejected_before_materialization(monkeypatch):
    huge_view = torch.ones(1, device="cpu").expand(1, 1, codec.MAX_PAYLOAD_BYTES)
    def forbidden(*args, **kwargs):
        pytest.fail("oversized tensor must fail before materialization or scanning")
    monkeypatch.setattr(torch.Tensor, "contiguous", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    with pytest.raises(ValueError, match="limit|budget"):
        codec.encode_payload("MASK", huge_view)


@pytest.mark.parametrize("itemsize", [2, 4, 8])
def test_native_little_endian_conversion_handles_big_endian(monkeypatch, itemsize):
    raw = bytes(range(itemsize * 3))
    monkeypatch.setattr(codec, "sys", types.SimpleNamespace(byteorder="little"))
    assert codec._native_little_endian(raw, itemsize) == raw
    monkeypatch.setattr(codec, "sys", types.SimpleNamespace(byteorder="big"))
    expected = b"".join(raw[i:i + itemsize][::-1] for i in range(0, len(raw), itemsize))
    assert codec._native_little_endian(raw, itemsize) == expected
    assert codec._native_little_endian(expected, itemsize) == raw
