"""Real CPU-only IMAGE transfers; no ComfyUI/bootstrap/provider imports."""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Load a private package, not Distributed's root __init__.
ROOT = Path(__file__).resolve().parents[1] / "serverless_bridge"
PACKAGE = "image_codec_tests_bridge"
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = pkg
codec = importlib.import_module(PACKAGE + ".codec")
images = importlib.import_module(PACKAGE + ".image_codec")
FORMATS = ("legacy_png", "png", "jpeg", "webp", "ffv1", "h264", "h265", "av1")
LOSSLESS = ("legacy_png", "png", "ffv1")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("GPU query or initialization is forbidden")
    for name in ("is_available", "init", "_lazy_init", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def available(fmt):
    if fmt in images._VIDEO_CODECS:
        av = pytest.importorskip("av")
        name = images._VIDEO_CODECS[fmt][0]
        if name not in av.codecs_available:
            pytest.skip("Software encoder unavailable: " + name)


def sample(channels=3, dtype=torch.float32, shape=(3, 65, 67)):
    g = torch.Generator(device="cpu").manual_seed(42)
    return torch.rand((*shape, channels), generator=g, device="cpu", dtype=dtype)


def quantized(value):
    return torch.from_numpy(np.clip(value.float().numpy() * 255, 0, 255).astype(np.uint8)).float() / 255


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_unchanged(dtype):
    value = sample(4, dtype, (2, 3, 5))
    value.flatten()[0] = -0.0
    expected = codec.encode_payload("IMAGE", value)
    assert images.encode_image(value) == expected
    result = images.decode_image(*expected)
    assert torch.equal(result.view(torch.uint8), value.view(torch.uint8))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_real_roundtrips_shape_order_quantization_alpha(fmt, channels):
    available(fmt)
    value = sample(channels)
    # Large inter-frame difference also verifies frame ordering.
    value[0] *= 0.2
    value[2] = value[2] * 0.2 + 0.8
    if channels == 4:
        value[..., 3] = torch.linspace(-0.1, 1.1, value[..., 3].numel()).reshape(value[..., 3].shape)
    descriptor, payload = images.encode_image(value, transfer_format=fmt)
    assert descriptor["encoding"] == "image-transfer-v1"
    result = codec.decode_payload("IMAGE", descriptor, payload)
    assert result.shape == value.shape
    assert result.device.type == "cpu" and result.is_contiguous()
    color = 3 if channels == 4 else channels
    expected = quantized(value[..., :color])
    if fmt in LOSSLESS:
        assert torch.equal(result[..., :color], expected)
    else:
        # Random RGB noise loses chroma detail in 4:2:0 even at high quality.
        assert (result[..., :color] - expected).abs().mean() < 0.12
        assert result[0, ..., :color].mean() < result[2, ..., :color].mean()
    if channels == 4:
        assert torch.equal(result[..., 3], value[..., 3])


@pytest.mark.parametrize("fmt", ["png", "ffv1", "jpeg", "webp", "h264", "h265", "av1"])
def test_float64_alpha_is_exact(fmt):
    available(fmt)
    value = sample(4, torch.float64, (1, 65, 67))
    descriptor, payload = images.encode_image(value, transfer_format=fmt)
    out = images.decode_image(descriptor, payload)
    assert out.dtype == torch.float64
    assert torch.equal(out[..., 3].contiguous().view(torch.uint8), value[..., 3].contiguous().view(torch.uint8))


@pytest.mark.parametrize("fmt", ["jpeg", "webp", "h264", "h265", "av1"])
def test_lossy_quality_changes_final_payload_size(fmt):
    available(fmt)
    value = sample(shape=(2, 96, 96))
    low = images.encode_image(value, transfer_format=fmt, transfer_quality=15)[1]
    high = images.encode_image(value, transfer_format=fmt, transfer_quality=95)[1]
    assert len(high) > len(low)


def test_truncation_clipping_grayscale_noncontiguous():
    value = torch.tensor([-0.1, 0.5, 1.1, 0.12345], device="cpu").reshape(1, 2, 2, 1).transpose(1, 2)
    out = images.decode_image(*images.encode_image(value, transfer_format="png"))
    assert torch.equal(out, quantized(value))
    assert out[0, 0, 1, 0] == 1


@pytest.mark.parametrize("value", [None, [], torch.ones(2, 3, 3), torch.ones(1, 2, 3, 2), torch.ones(1, 2, 3, 0), torch.ones(1, 2, 3, 3, dtype=torch.uint8), torch.ones(1, 2, 3, 3, device="meta"), torch.ones(1, 2, 3, 3).to_sparse()])
def test_invalid_values(value):
    with pytest.raises(ValueError):
        images.encode_image(value, transfer_format="png")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        images.encode_image(torch.full((1, 1, 1, 3), bad), transfer_format="png")


@pytest.mark.parametrize("options", [{"transfer_format": "PNG"}, {"transfer_format": []}, {"transfer_quality": True}, {"transfer_quality": 0}, {"transfer_quality": 101}, {"transfer_quality": 1.5}, {"max_bytes": True}, {"max_bytes": -1}])
def test_strict_options(options):
    with pytest.raises(ValueError):
        images.encode_image(sample(shape=(1, 1, 1)), **options)


@pytest.mark.parametrize("change", [{"extra": 1}, {"format": "exact"}, {"quality": True}, {"layout": "BCHW"}, {"shape": [1, 2**60, 1, 3]}, {"shape": [True, 1, 1, 3]}, {"parts": [0]}, {"parts": [True]}, {"parts": [2**60]}, {"alpha": {"extra": 0}}])
def test_bad_descriptors(change):
    descriptor, payload = images.encode_image(sample(shape=(1, 3, 5)), transfer_format="png")
    descriptor.update(change)
    with pytest.raises(ValueError):
        images.decode_image(descriptor, payload)


def test_truncated_appended_and_mismatched_parts():
    d, p = images.encode_image(sample(shape=(2, 3, 5)), transfer_format="png")
    for payload in (p[:-1], p + b"x"):
        with pytest.raises(ValueError):
            images.decode_image(d, payload)
    for change in ({"shape": [1, 3, 5, 3]}, {"parts": [len(p)]}, {"shape": [2, 2, 5, 3]}):
        bad = dict(d, **change)
        with pytest.raises(ValueError):
            images.decode_image(bad, p)


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp", "ffv1", "h264", "h265", "av1"])
def test_payload_and_decoded_budget(fmt):
    available(fmt)
    value = sample(shape=(1, 65, 67))
    d, p = images.encode_image(value, transfer_format=fmt)
    with pytest.raises(ValueError, match="limit|budget"):
        images.decode_image(d, p, max_bytes=100)
    with pytest.raises(ValueError, match="limit|budget"):
        images.encode_image(value, transfer_format=fmt, max_bytes=100)


def test_oversized_shape_before_materialization(monkeypatch):
    value = torch.ones(1).expand(1, 8192, 8192, 4)
    def forbidden(*a, **kw):
        pytest.fail("Materialized an oversized tensor")
    monkeypatch.setattr(torch.Tensor, "contiguous", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    with pytest.raises(ValueError):
        images.encode_image(value, transfer_format="png")


def test_header_dimensions_checked_before_pixels(monkeypatch):
    d, p = images.encode_image(sample(shape=(1, 9, 11)), transfer_format="png")
    d["shape"] = [1, 1, 1, 3]
    def forbidden(*a, **kw):
        pytest.fail("Pixels decoded before header dimensions were checked")
    monkeypatch.setattr(Image.Image, "load", forbidden)
    with pytest.raises(ValueError):
        images.decode_image(d, p)


def test_other_kinds_do_not_dispatch():
    d, p = images.encode_image(sample(shape=(1, 3, 5)), transfer_format="png")
    for kind in ("MASK", "AUDIO", "STRING"):
        with pytest.raises(ValueError):
            codec.decode_payload(kind, d, p)


def test_missing_video_encoder_is_explicit(monkeypatch):
    monkeypatch.setitem(images._VIDEO_CODECS, "h264", ("missing_software_encoder", 51, "yuv420p"))
    with pytest.raises(ValueError, match="encoder|unavailable"):
        images.encode_image(sample(shape=(1, 64, 64)), transfer_format="h264")


def test_bounded_sink_catches_final_flush_bytes():
    sink = images._BoundedIO(4)
    sink.write(b"1234")
    with pytest.raises(ValueError, match="limit"):
        sink.write(b"5")
    assert sink.getvalue() == b"1234"


def test_crf_mapping():
    assert images._quality_to_crf(90, 51) == 13
    assert images._quality_to_crf(90, 63) == 16
    assert images._quality_to_crf(100, 63) == 0


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_compressed_preserves_source_dtype_and_exact_alpha(fmt, dtype):
    available(fmt)
    value = sample(4, dtype, (1, 17, 19))
    value[..., 3].flatten()[0] = -0.0
    d, p = images.encode_image(value, transfer_format=fmt)
    with torch.device("meta"):
        result = images.decode_image(d, p)
    assert result.device.type == "cpu" and result.dtype == dtype
    assert d["dtype"] == str(dtype).removeprefix("torch.")
    assert torch.equal(result[..., 3].contiguous().view(torch.uint8), value[..., 3].contiguous().view(torch.uint8))
    if fmt in LOSSLESS:
        assert torch.equal(result[..., :3], quantized(value[..., :3]).to(dtype))


@pytest.mark.parametrize("fmt", ["ffv1", "h264", "h265", "av1"])
def test_video_count_and_header_fail_before_native_decoder(fmt, monkeypatch):
    available(fmt)
    d, p = images.encode_image(sample(shape=(2, 65, 67)), transfer_format=fmt)
    def forbidden(*a, **kw):
        pytest.fail("Native decoder reached before header validation")
    monkeypatch.setattr(images, "_decode_video", forbidden)
    for shape in ([1, 65, 67, 3], [3, 65, 67, 3], [2, 32, 32, 3]):
        bad = dict(d, shape=shape)
        with pytest.raises(ValueError):
            images.decode_image(bad, p)
    for payload in (p[:-1], p[:40], p + b"extra"):
        bad = dict(d, parts=[len(payload)])
        with pytest.raises(ValueError):
            images.decode_image(bad, payload)


@pytest.mark.parametrize("fmt", ["ffv1", "h264", "h265", "av1"])
def test_video_does_not_use_native_container_probe(fmt, monkeypatch):
    available(fmt)
    d, p = images.encode_image(sample(shape=(2, 65, 67)), transfer_format=fmt)
    av = images._import_av()
    def forbidden(*a, **kw):
        pytest.fail("Unbounded native stream probing is forbidden")
    monkeypatch.setattr(av, "open", forbidden)
    assert images.decode_image(d, p).shape == (2, 65, 67, 3)


@pytest.mark.parametrize("fmt", ["ffv1", "h264", "h265", "av1"])
def test_real_video_flush_bound(fmt):
    available(fmt)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    good = images._encode_video(iter(frames), fmt, 90, 64, 64, 100000)
    with pytest.raises(ValueError, match="limit"):
        images._encode_video(iter(frames), fmt, 90, 64, 64, len(good) - 1)


def test_alpha_invalid_dtype_nonfinite_and_extra_fields():
    import copy
    d, p = images.encode_image(sample(4, shape=(1, 3, 5)), transfer_format="png")
    for field, value in (("dtype", "float64"), ("shape", [1, 3, 6, 1]), ("extra", 1)):
        bad = copy.deepcopy(d)
        bad["alpha"]["descriptor"][field] = value
        with pytest.raises(ValueError):
            images.decode_image(bad, p)
    import struct
    bad = p[:-4] + struct.pack("<f", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        images.decode_image(d, bad)


def test_descriptor_structure_bound_and_frame_limit():
    d, p = images.encode_image(sample(shape=(1, 3, 5)), transfer_format="png")
    for bad in (dict(d, quality="x" * 70000), dict(d, parts=[1] * 241), dict(d, shape=[241, 1, 1, 3]), dict(d, dtype="x")):
        with pytest.raises(ValueError):
            images.decode_image(bad, p)
    with pytest.raises(ValueError):
        images.encode_image(torch.zeros(241, 1, 1, 3), transfer_format="png")


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_all_still_headers_checked_before_pillow_open(fmt, monkeypatch):
    d, p = images.encode_image(sample(shape=(1, 31, 33)), transfer_format=fmt)
    d["shape"] = [1, 2, 2, 3]
    def forbidden(*a, **kw):
        pytest.fail("Pillow opened a mismatched header")
    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(ValueError, match="dimensions"):
        images.decode_image(d, p)


@pytest.mark.parametrize("fmt", ["ffv1", "h264", "h265", "av1"])
def test_video_longer_batch_order_and_coded_allocation_bound(fmt):
    available(fmt)
    value = torch.linspace(0, 1, 16).reshape(16, 1, 1, 1).expand(16, 64, 64, 3)
    d, p = images.encode_image(value, transfer_format=fmt)
    header = images._video_header(p, fmt, 64, 64, 16)
    coded = images._coded_frame_bound(p, fmt, header)
    assert 16 <= coded <= images.MAX_FRAMES * 4
    result = images.decode_image(d, p)
    assert torch.all((result.mean(dim=(1, 2, 3)) - value.mean(dim=(1, 2, 3))).abs() < 0.03)


@pytest.mark.parametrize("fmt", ["h264", "h265", "av1"])
def test_packet_unit_bounds_before_native_decode(fmt):
    available(fmt)
    d, p = images.encode_image(sample(shape=(1, 64, 64)), transfer_format=fmt)
    config, bounds = images._video_header(p, fmt, 64, 64, 1)
    start, end = bounds[0]
    packet = p[start:end]
    with pytest.raises(ValueError):
        images._coded_frame_bound(packet[:-1], fmt, (config, [(0, len(packet) - 1)]))
    # An adversarial single block can contain many coded frames: bound it too.
    enormous = packet * (images.MAX_FRAMES * 4 + 1)
    with pytest.raises(ValueError, match="limit"):
        images._coded_frame_bound(enormous, fmt, (config, [(0, len(enormous))]))


@pytest.mark.parametrize("fmt", ["h264", "h265", "av1"])
def test_bitstream_resolution_cannot_bypass_native_pixel_limit(fmt, monkeypatch):
    available(fmt)
    d, p = images.encode_image(sample(shape=(1, 256, 256)), transfer_format=fmt)
    header = images._video_header(p, fmt, 256, 256, 1)
    av = images._import_av()
    # A lying container is not enough to disable max_pixels in the decoder.
    # The bitstream still contains 256x256 pixels but the budget permits 64x64.
    with pytest.raises((ValueError, av.error.FFmpegError)):
        list(images._decode_video(p, fmt, 64, 64, 1, header))
