import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _load_codec():
    path = Path(__file__).resolve().parents[1] / "utils" / "transfer_codec.py"
    spec = importlib.util.spec_from_file_location("dist_transfer_codec_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dist_transfer_codec_test"] = module
    spec.loader.exec_module(module)
    return module


tc = _load_codec()


def _smooth_frames(count=4, height=67, width=91):
    """Smooth moving gradient, closer to generated content than random noise."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = []
    for i in range(count):
        r = 0.5 + 0.5 * np.sin((xx + 3 * i) / 13.0)
        g = 0.5 + 0.5 * np.cos((yy - 2 * i) / 11.0)
        b = (xx + yy + 5 * i) / (width + height + 5 * count)
        frames.append(np.stack([r, g, b], axis=-1))
    return torch.from_numpy(np.stack(frames)).float()


def _legacy_master_tensor(frame_tensor):
    """What the unmodified pipeline delivered: tensor_to_pil -> PNG -> pil_to_tensor."""
    img = Image.fromarray((255 * frame_tensor.numpy()).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    with Image.open(io.BytesIO(buf.getvalue())) as decoded:
        return np.array(decoded.convert("RGB")).astype(np.float32) / 255.0


def _round_trip(frames_u8, fmt, quality=90, max_chunk_bytes=10**9):
    out = []
    chunks = list(tc.encode_chunks(frames_u8, fmt, quality, max_chunk_bytes))
    for chunk in chunks:
        envelope = {k: v for k, v in chunk.items() if k != "parts"}
        out.extend(tc.decode_chunk(envelope, chunk["parts"]))
    return chunks, out


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


@pytest.mark.parametrize("fmt", ["png", "ffv1"])
def test_lossless_formats_reproduce_legacy_master_tensors_exactly(fmt):
    images = _smooth_frames()
    frames = tc.tensor_batch_to_uint8(images)
    _, decoded = _round_trip(frames, fmt)

    assert len(decoded) == images.shape[0]
    for i, frame in enumerate(decoded):
        got = tc.uint8_frame_to_tensor_array(frame)[0]
        assert np.array_equal(got, _legacy_master_tensor(images[i]))


@pytest.mark.parametrize("fmt", ["jpeg", "webp", "h264", "h265", "av1"])
def test_lossy_formats_round_trip_odd_sizes_with_reasonable_quality(fmt):
    if fmt in tc.VIDEO_FORMATS and not tc.video_codec_available(fmt):
        pytest.skip(f"{fmt} encoder not available in this PyAV build")
    frames = tc.tensor_batch_to_uint8(_smooth_frames())
    _, decoded = _round_trip(frames, fmt, quality=90)

    assert len(decoded) == frames.shape[0]
    for i, frame in enumerate(decoded):
        assert frame.shape == frames[i].shape
        assert frame.dtype == np.uint8
        assert _psnr(frame, frames[i]) > 30


def test_h265_handles_frames_smaller_than_ctu():
    if not tc.video_codec_available("h265"):
        pytest.skip("h265 encoder not available")
    frames = tc.tensor_batch_to_uint8(_smooth_frames(count=3, height=17, width=23))
    _, decoded = _round_trip(frames, "h265")
    assert [f.shape for f in decoded] == [(17, 23, 3)] * 3


def test_image_formats_split_chunks_by_byte_cap_and_keep_order():
    frames = tc.tensor_batch_to_uint8(_smooth_frames(count=5))
    one = len(tc.encode_image_frame(frames[0], "png", 90))
    chunks, decoded = _round_trip(frames, "png", max_chunk_bytes=one * 2 + 1)

    assert len(chunks) >= 3
    assert [c["start_index"] for c in chunks] == sorted(c["start_index"] for c in chunks)
    assert sum(c["count"] for c in chunks) == 5
    assert all(len(c["parts"]) == c["count"] for c in chunks)
    assert np.array_equal(np.stack(decoded), frames)


def test_video_formats_split_chunks_by_frame_cap(monkeypatch):
    monkeypatch.setattr(tc, "MAX_VIDEO_CHUNK_FRAMES", 2)
    frames = tc.tensor_batch_to_uint8(_smooth_frames(count=5))
    chunks, decoded = _round_trip(frames, "ffv1")

    assert [(c["start_index"], c["count"]) for c in chunks] == [(0, 2), (2, 2), (4, 1)]
    assert all(len(c["parts"]) == 1 for c in chunks)
    assert np.array_equal(np.stack(decoded), frames)


def test_tensor_batch_to_uint8_clips_and_normalizes_channels():
    rgba = torch.tensor([[[[1.5, -0.2, 0.5, 0.3]]]])
    assert tc.tensor_batch_to_uint8(rgba).tolist() == [[[[255, 0, 127]]]]

    gray = torch.full((1, 2, 2, 1), 0.5)
    out = tc.tensor_batch_to_uint8(gray)
    assert out.shape == (1, 2, 2, 3)
    assert (out == 127).all()


def test_quality_to_crf_mapping():
    assert tc.quality_to_crf(100, 51) == 0
    assert tc.quality_to_crf(90, 51) == 13
    assert tc.quality_to_crf(80, 51) == 19
    assert tc.quality_to_crf(90, 63) == 16
    assert tc.quality_to_crf(1, 51) == 51
    assert tc.quality_to_crf("bogus", 51) == tc.quality_to_crf(tc.DEFAULT_TRANSFER_QUALITY, 51)


def test_normalize_format_rejects_unknown_values():
    assert tc.normalize_format(" H264 ") == "h264"
    assert tc.normalize_format(None) == tc.DEFAULT_TRANSFER_FORMAT
    with pytest.raises(tc.TransferCodecError):
        tc.normalize_format("gif")


def test_decode_chunk_validates_envelope_against_payload():
    frames = tc.tensor_batch_to_uint8(_smooth_frames(count=2))
    chunk = next(tc.encode_chunks(frames, "png", 90, 10**9))
    envelope = {k: v for k, v in chunk.items() if k != "parts"}

    with pytest.raises(tc.TransferCodecError, match="image parts"):
        tc.decode_chunk({**envelope, "count": 3}, chunk["parts"])
    with pytest.raises(tc.TransferCodecError, match="expected"):
        tc.decode_chunk({**envelope, "width": 10}, chunk["parts"])
    with pytest.raises(tc.TransferCodecError, match="count"):
        tc.decode_chunk({**envelope, "count": 0}, chunk["parts"])
    with pytest.raises(tc.TransferCodecError):
        tc.decode_chunk(envelope, [b"not an image", b"nope"])

    video = next(tc.encode_chunks(frames, "ffv1", 90, 10**9))
    video_env = {k: v for k, v in video.items() if k != "parts"}
    with pytest.raises(tc.TransferCodecError, match="frames"):
        tc.decode_chunk({**video_env, "count": 5}, video["parts"])


def test_legacy_format_is_not_chunk_encoded():
    frames = tc.tensor_batch_to_uint8(_smooth_frames(count=1))
    with pytest.raises(tc.TransferCodecError):
        list(tc.encode_chunks(frames, "legacy_png", 90, 10**9))
