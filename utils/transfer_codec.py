"""
Compressed worker -> master image transfer.

Workers encode their IMAGE batch with one of the formats below and POST it to
the master as multipart/form-data chunks; the master decodes each chunk back
into per-frame tensors. The goal is to cut network egress (e.g. cloud workers
returning results to a local master), so formats range from bit-exact
lossless to strongly compressed video.

Formats
-------
legacy_png  Original behaviour: one JSON request per frame, uncompressed PNG
            in base64. Kept for compatibility with unmodified masters.
png         Lossless PNG per frame, sent as binary.
ffv1        Lossless FFV1 video for the whole batch (good for frame sequences).
jpeg, webp  Lossy per-frame stills.
h264, h265, av1
            Lossy video for the whole batch. Smallest by far for video
            frame batches, where consecutive frames are similar.
exact       Original floating-point tensors (raw little-endian bytes, dtype,
            channels and values preserved). Largest; opt-in. Encoded by the
            Collector/job_complete route with serverless_bridge.codec, never by
            encode_chunks/decode_chunk below.

All formats except exact quantize to 8-bit exactly like the legacy path did,
so the lossless formats reproduce the legacy master tensors bit-for-bit.
"""
import io
import math
import os

import numpy as np
from PIL import Image

LEGACY_FORMAT = "legacy_png"
EXACT_FORMAT = "exact"
IMAGE_FORMATS = ("png", "jpeg", "webp")
VIDEO_FORMATS = ("ffv1", "h264", "h265", "av1")
LOSSLESS_FORMATS = (LEGACY_FORMAT, "png", "ffv1")
# Appended, so the pre-existing option order (and saved widget values) is unchanged.
TRANSFER_FORMATS = (LEGACY_FORMAT, "png", "ffv1", "jpeg", "webp", "h264", "h265", "av1", EXACT_FORMAT)

DEFAULT_TRANSFER_FORMAT = LEGACY_FORMAT
DEFAULT_TRANSFER_QUALITY = 90

TRANSFER_FORMAT_TOOLTIP = (
    "How remote workers send images back to the master; the master's own images are never re-encoded. "
    "legacy_png: original uncompressed PNG (largest). "
    "png / ffv1: lossless, identical result. "
    "jpeg / webp: lossy stills. "
    "h264 / h265 / av1: lossy video of the whole batch, smallest for frame sequences "
    "(h265 is experimental). "
    "exact: original floating-point values, no 8-bit rounding; much larger, opt-in. "
    "Serverless media pools send legacy_png as png (same pixels) and never substitute another format."
)
TRANSFER_QUALITY_TOOLTIP = (
    "Quality for lossy transfer formats (ignored by lossless ones). "
    "JPEG/WebP use it directly; video codecs map it to CRF "
    "(90 -> h264/h265 CRF 13, av1 CRF 16)."
)

# encoder name, max CRF, container-friendly pixel format
_VIDEO_CODECS = {
    "ffv1": ("ffv1", None, "bgr0"),
    "h264": ("libx264", 51, "yuv420p"),
    "h265": ("libx265", 51, "yuv420p"),
    "av1": ("libsvtav1", 63, "yuv420p"),
}
_VIDEO_RATE = 24
# Upper bound on frames per video chunk, so one request never holds an
# unbounded batch even when frames compress extremely well.
MAX_VIDEO_CHUNK_FRAMES = 240
# When sizing the next video chunk from observed bytes/frame, aim this far
# below the limit so ordinary variation does not force a re-encode.
VIDEO_CHUNK_SIZE_HEADROOM = 0.85


class TransferCodecError(ValueError):
    """Raised when a transfer payload cannot be encoded or decoded."""


class FrameTooLargeError(TransferCodecError):
    """A single encoded frame does not fit in one request.

    Raised by encode_chunks before anything has been yielded for that frame;
    callers that materialize all chunks first can therefore fail without
    having delivered a partial batch.
    """


def normalize_format(value):
    fmt = str(value or DEFAULT_TRANSFER_FORMAT).strip().lower()
    if fmt not in TRANSFER_FORMATS:
        raise TransferCodecError(
            f"Unknown transfer format '{value}'. Expected one of: {', '.join(TRANSFER_FORMATS)}"
        )
    return fmt


def normalize_quality(value):
    try:
        quality = int(value)
    except (TypeError, ValueError):
        quality = DEFAULT_TRANSFER_QUALITY
    return max(1, min(100, quality))


def quality_to_crf(quality, max_crf):
    """Map 1-100 quality onto a codec CRF scale (0 = lossless/best).

    The curve is chosen so the same quality number looks roughly the same
    across JPEG/WebP and video codecs: 90 -> CRF 13 (h264/h265), 16 (av1);
    80 -> 19 / 24; 100 -> 0.
    """
    quality = normalize_quality(quality)
    return int(round(max_crf * (1.0 - quality / 100.0) ** 0.6))


def tensor_batch_to_uint8(images):
    """IMAGE tensor [B,H,W,C] float 0..1 -> uint8 numpy [B,H,W,3].

    Uses the same truncating quantization as utils.image.tensor_to_pil so that
    lossless formats reproduce the legacy transfer exactly, but clips first so
    out-of-range values no longer wrap around.
    """
    arr = images.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise TransferCodecError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(arr.shape)}")
    channels = arr.shape[-1]
    if channels == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif channels >= 3:
        arr = arr[..., :3]
    else:
        raise TransferCodecError(f"Unsupported channel count {channels}")
    return np.ascontiguousarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))


def uint8_frame_to_tensor_array(frame):
    """uint8 [H,W,3] -> float32 [1,H,W,3] numpy, matching utils.image.pil_to_tensor."""
    return (frame.astype(np.float32) / 255.0)[None, ...]


# --------------------------------------------------------------------------- #
# Still images
# --------------------------------------------------------------------------- #

def encode_image_frame(frame, fmt, quality):
    img = Image.fromarray(frame)
    quality = normalize_quality(quality)
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", compress_level=4)  # ~same size as 6, 2x faster
    elif fmt == "jpeg":
        img.save(buf, format="JPEG", quality=quality, subsampling=0 if quality >= 90 else 2)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=quality, method=4)
    else:
        raise TransferCodecError(f"'{fmt}' is not a still-image transfer format")
    return buf.getvalue()


def decode_image_frame(data):
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise TransferCodecError(f"Failed to decode image frame: {exc}") from exc


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

def _import_av():
    # SVT-AV1 prints a banner per encoder instance unless told otherwise.
    os.environ.setdefault("SVT_LOG", "1")
    try:
        import av  # PyAV ships with ComfyUI
    except ImportError as exc:
        raise TransferCodecError("PyAV ('av') is required for video transfer formats") from exc
    return av


def video_codec_available(fmt):
    if fmt not in _VIDEO_CODECS:
        return False
    try:
        av = _import_av()
        av.codec.Codec(_VIDEO_CODECS[fmt][0], "w")
        return True
    except Exception:
        return False


def _padded_size(fmt, pix_fmt, width, height):
    """Encoder-friendly frame size; the decoder crops back to width x height.

    4:2:0 needs even dimensions, and libx265 crashes on frames smaller than
    its 64px CTU, so pad up to whichever is larger.
    """
    if pix_fmt == "yuv420p":
        width, height = width + width % 2, height + height % 2
    if fmt == "h265":
        width, height = max(width, 64), max(height, 64)
    return width, height


def _pad_to(frame, width, height):
    h, w = frame.shape[:2]
    if h == height and w == width:
        return frame
    return np.pad(frame, ((0, height - h), (0, width - w), (0, 0)), mode="edge")


class _VideoChunkWriter:
    def __init__(self, fmt, quality, width, height):
        av = _import_av()
        self._av = av
        codec_name, max_crf, pix_fmt = _VIDEO_CODECS[fmt]
        self.fmt = fmt
        self.pix_fmt = pix_fmt
        self.width, self.height = _padded_size(fmt, pix_fmt, width, height)
        self.buffer = io.BytesIO()
        self.container = av.open(self.buffer, mode="w", format="matroska")
        self.stream = self.container.add_stream(codec_name, rate=_VIDEO_RATE)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = pix_fmt
        options = {}
        if max_crf is not None:
            options["crf"] = str(quality_to_crf(quality, max_crf))
        if fmt == "h264":
            options["preset"] = "medium"
        elif fmt == "h265":
            options["preset"] = "fast"
            options["x265-params"] = "log-level=error"
        elif fmt == "av1":
            options["preset"] = "8"
        elif fmt == "ffv1":
            options["level"] = "3"
            options["slicecrc"] = "0"
        self.stream.options = options
        self.count = 0

    def add(self, frame):
        frame = _pad_to(frame, self.width, self.height)
        vf = self._av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(vf):
            self.container.mux(packet)
        self.count += 1

    def bytes_so_far(self):
        return self.buffer.tell()

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        return self.buffer.getvalue()


def decode_video(data, width, height, expected_frames=None):
    av = _import_av()
    frames = []
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            for vf in container.decode(video=0):
                arr = vf.to_ndarray(format="rgb24")
                frames.append(np.ascontiguousarray(arr[:height, :width]))
    except TransferCodecError:
        raise
    except Exception as exc:
        raise TransferCodecError(f"Failed to decode video chunk: {exc}") from exc
    if expected_frames is not None and len(frames) != expected_frames:
        raise TransferCodecError(
            f"Video chunk decoded to {len(frames)} frames, expected {expected_frames}"
        )
    return frames


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def encode_chunks(frames, fmt, quality, max_chunk_bytes, max_frame_bytes=None):
    """Yield transfer chunks for a uint8 frame batch [B,H,W,3].

    Each chunk is a dict:
        {"format", "start_index", "count", "width", "height", "parts": [bytes, ...]}
    Image formats carry one part per frame; video formats carry one part
    (a Matroska file) holding `count` frames.

    Size guarantees, measured on the finalized bytes (after encoder flush and
    muxer finalization, not on encoder buffers):

    * a chunk holding more than one frame never exceeds `max_chunk_bytes`;
    * a chunk holding a single frame may exceed `max_chunk_bytes` but never
      `max_frame_bytes` (defaults to `max_chunk_bytes`).

    A frame that cannot fit in `max_frame_bytes` even on its own raises
    FrameTooLargeError. Video chunks that come out too large are re-encoded
    with fewer frames.
    """
    fmt = normalize_format(fmt)
    if fmt == LEGACY_FORMAT:
        raise TransferCodecError("legacy_png is sent through the JSON path, not encode_chunks")
    if fmt == EXACT_FORMAT:
        raise TransferCodecError("exact tensors are not encoded by encode_chunks")
    total = int(frames.shape[0])
    if total == 0:
        return
    height, width = int(frames.shape[1]), int(frames.shape[2])
    max_chunk_bytes = max(1, int(max_chunk_bytes))
    max_frame_bytes = max_chunk_bytes if max_frame_bytes is None else max(max_chunk_bytes, int(max_frame_bytes))

    def _chunk(start, parts, count):
        return {
            "format": fmt,
            "start_index": start,
            "count": count,
            "width": width,
            "height": height,
            "parts": parts,
        }

    def _too_large(index, size):
        return FrameTooLargeError(
            f"Frame {index} encodes to {size} bytes as {fmt}, above the {max_frame_bytes}-byte request limit. "
            "Use a lossy format or lower quality, or raise the master's --max-upload-size together with "
            "COMFYUI_DISTRIBUTED_TRANSFER_MAX_REQUEST_BYTES."
        )

    if fmt in IMAGE_FORMATS:
        start, parts, size = 0, [], 0
        for i in range(total):
            data = encode_image_frame(frames[i], fmt, quality)
            if len(data) > max_frame_bytes:
                raise _too_large(i, len(data))
            if parts and size + len(data) > max_chunk_bytes:
                yield _chunk(start, parts, len(parts))
                start, parts, size = i, [], 0
            parts.append(data)
            size += len(data)
        if parts:
            yield _chunk(start, parts, len(parts))
        return

    start = 0
    budget = MAX_VIDEO_CHUNK_FRAMES
    while start < total:
        want = min(budget, total - start)
        data, count = _encode_video_range(frames, start, want, fmt, quality, width, height, max_chunk_bytes)
        if len(data) <= max_chunk_bytes or (count == 1 and len(data) <= max_frame_bytes):
            yield _chunk(start, [data], count)
            start += count
            per_frame = len(data) / count
            budget = max(1, min(MAX_VIDEO_CHUNK_FRAMES, int(max_chunk_bytes * VIDEO_CHUNK_SIZE_HEADROOM / per_frame)))
            continue
        if count == 1:
            raise _too_large(start, len(data))
        # Finalized chunk overshot: retry this range with proportionally fewer
        # frames. Strictly fewer than `count`, so this always terminates.
        estimate = int(count * max_chunk_bytes * VIDEO_CHUNK_SIZE_HEADROOM / len(data))
        budget = max(1, min(count - 1, estimate))


def _encode_video_range(frames, start, count, fmt, quality, width, height, stop_after_bytes):
    """Encode frames[start:start+count] as one finalized video.

    Stops feeding frames early once the muxed output already exceeds
    `stop_after_bytes`, since the chunk is then known to be too large and will
    be re-encoded smaller; this bounds wasted work. Encoder lookahead means the
    pre-flush size underestimates the final size, so the caller must still
    check the returned (finalized) bytes. Returns (bytes, frames_encoded).
    """
    writer = _VideoChunkWriter(fmt, quality, width, height)
    for i in range(start, start + count):
        writer.add(frames[i])
        if writer.bytes_so_far() > stop_after_bytes:
            break
    return writer.close(), writer.count


def decode_chunk(envelope, parts):
    """Decode one received chunk into a list of uint8 [H,W,3] frames."""
    fmt = normalize_format(envelope.get("format"))
    if fmt == LEGACY_FORMAT:
        raise TransferCodecError("legacy_png chunks are not sent as multipart")
    if fmt == EXACT_FORMAT:
        raise TransferCodecError("exact chunks are decoded as tensors, not frames")
    count = envelope.get("count")
    width = envelope.get("width")
    height = envelope.get("height")
    for name, value in (("count", count), ("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TransferCodecError(f"envelope.{name}: expected positive integer")
    if not parts:
        raise TransferCodecError("chunk has no media parts")

    if fmt in IMAGE_FORMATS:
        if len(parts) != count:
            raise TransferCodecError(f"expected {count} image parts, got {len(parts)}")
        frames = [decode_image_frame(p) for p in parts]
    else:
        if len(parts) != 1:
            raise TransferCodecError(f"expected 1 video part, got {len(parts)}")
        frames = decode_video(parts[0], width, height, expected_frames=count)

    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            raise TransferCodecError(
                f"decoded frame is {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}"
            )
    return frames


def describe_ratio(sent_bytes, raw_bytes):
    if sent_bytes <= 0:
        return "n/a"
    ratio = raw_bytes / sent_bytes
    return f"{ratio:.1f}x" if math.isfinite(ratio) else "n/a"
