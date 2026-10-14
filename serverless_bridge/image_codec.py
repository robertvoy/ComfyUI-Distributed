"""Bounded optional IMAGE returns, independent of ComfyUI and Distributed.

Exact mode is the existing tensor wire format, unchanged. All other modes clip
and truncate RGB/luma to uint8 (as the Collector does), normalize in float32,
then cast to the source dtype. RGBA always carries an exact tensor alpha
sidecar, including PNG/WebP. ``legacy_png`` means uncompressed binary PNG here,
not the old Collector's JSON/base64 transport.

The compressed wire format concatenates length-delimited stills or one Matroska
video, followed by optional alpha bytes. No archive, path, URL or pickle is
accepted. Limits cover wire bytes, descriptor structure, output tensor bytes,
and aggregate padded pixel buffers, not process RSS/native codec overhead.
Video preflight reads EBML track dimensions and block counts before creating a
decoder. We extract packets ourselves, avoiding native container stream probing
(which can decode before caller checks). FFmpeg max_pixels also bounds
bitstream-driven resolution changes, including native buffer alignment.
Software codecs only, one thread, no fallback to another encoder/format.
"""

import io
import struct
import zlib
from fractions import Fraction

import torch

from . import codec
from .codec import MAX_PAYLOAD_BYTES
from .image_transfer import DEFAULT_FORMAT, DEFAULT_QUALITY, validate_options

MAX_FRAMES = 240
MAX_DIMENSION = 8192
MAX_DESCRIPTOR_BYTES = 64 * 1024
MAX_CONTAINER_ELEMENTS = 16384
_VIDEO_CODECS = {
    "ffv1": ("ffv1", None, "bgr0"),
    "h264": ("libx264", 51, "yuv420p"),
    "h265": ("libx265", 51, "yuv420p"),
    "av1": ("libsvtav1", 63, "yuv420p"),
}
_CODEC_IDS = {"ffv1": b"V_FFV1", "h264": b"V_MPEG4/ISO/AVC",
              "h265": b"V_MPEGH/ISO/HEVC", "av1": b"V_AV1"}


class _BoundedIO(io.BytesIO):
    """Enforce the final file size on every write, including muxer flush/seek."""
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ValueError("Encoded IMAGE exceeds byte limit")
        return super().write(data)

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.tell() + offset
        elif whence == 2:
            position = self.getbuffer().nbytes + offset
        else:
            raise ValueError("Invalid image buffer seek")
        if not 0 <= position <= self.limit:
            raise ValueError("Encoded IMAGE exceeds byte limit")
        return super().seek(offset, whence)


def _quality_to_crf(quality, maximum):
    return int(round(maximum * (1.0 - quality / 100.0) ** 0.6))


def _padded_size(fmt, height, width):
    if fmt in ("h264", "h265", "av1"):
        height, width = height + height % 2, width + width % 2
    # libx265 / SVT-AV1 cannot safely encode arbitrarily small pictures.
    if fmt in ("h265", "av1"):
        height, width = max(64, height), max(64, width)
    return height, width


def _shape_budget(shape, fmt, limit, itemsize=4, alpha_size=0, coded_frames=None):
    size = codec._tensor_size("IMAGE", shape, itemsize, limit)
    count, height, width, _ = shape
    if count > MAX_FRAMES or max(height, width) > MAX_DIMENSION:
        raise ValueError("IMAGE frame count or dimensions exceed limit")
    ph, pw = _padded_size(fmt, height, width)
    # Reserve output and padded RGBA-sized decoded buffers BEFORE allocation.
    bh, bw = ((ph + 63) // 64 * 64, (pw + 63) // 64 * 64) if fmt in _VIDEO_CODECS else (ph, pw)
    # Slice/OBU counts conservatively bound native frames, including AV1 hidden
    # frames and multiple outputs from one packet. Reserve 16 bytes per pixel
    # for hostile high-bit-depth streams, not only our usual 8-bit RGB frames.
    native_count = max(count, coded_frames or count)
    if native_count > MAX_FRAMES * 4:
        raise ValueError("IMAGE coded frame count exceeds limit")
    pixels = codec._tensor_size("IMAGE", [native_count, bh, bw, 4], 4, limit)
    if size + pixels + alpha_size > limit:
        raise ValueError("IMAGE decoded aggregate exceeds byte limit")
    return ph, pw


def _descriptor_budget(value):
    """Bound the JSON-shaped tree without serializing attacker-sized values."""
    remaining = MAX_DESCRIPTOR_BYTES
    def walk(obj, depth):
        nonlocal remaining
        if depth > 5:
            raise ValueError("IMAGE descriptor nesting exceeds limit")
        remaining -= 8
        if remaining < 0:
            raise ValueError("IMAGE descriptor exceeds byte limit")
        if obj is None:
            return
        if type(obj) is str:
            remaining -= len(obj) * 6
        elif type(obj) is int:
            if obj.bit_length() > 63:
                raise ValueError("IMAGE descriptor integer exceeds limit")
            remaining -= 20
        elif type(obj) is list:
            if len(obj) > MAX_FRAMES:
                raise ValueError("IMAGE descriptor list exceeds limit")
            for item in obj:
                walk(item, depth + 1)
        elif type(obj) is dict:
            if len(obj) > 10:
                raise ValueError("IMAGE descriptor fields exceed limit")
            for key, item in obj.items():
                if type(key) is not str:
                    raise ValueError("Invalid IMAGE descriptor key")
                walk(key, depth + 1)
                walk(item, depth + 1)
        else:
            raise ValueError("Invalid IMAGE descriptor value")
        if remaining < 0:
            raise ValueError("IMAGE descriptor exceeds byte limit")
    walk(value, 0)


def _alpha_metadata(alpha, shape, limit):
    if shape[-1] != 4:
        if alpha is not None:
            raise ValueError("Unexpected IMAGE alpha sidecar")
        return 0, torch.float32
    if type(alpha) is not dict or set(alpha) != {"descriptor", "size"}:
        raise ValueError("Missing or invalid IMAGE alpha sidecar")
    d = alpha["descriptor"]
    if type(d) is not dict or set(d) != {"encoding", "dtype", "shape", "layout"}:
        raise ValueError("Invalid IMAGE alpha descriptor")
    if d["encoding"] != "tensor-le-v1" or d["layout"] != "BHWC" or d["shape"] != shape[:3] + [1]:
        raise ValueError("Invalid IMAGE alpha shape or encoding")
    if type(d["dtype"]) is not str or d["dtype"] not in codec._DTYPES:
        raise ValueError("Invalid IMAGE alpha dtype")
    dtype, itemsize = codec._DTYPES[d["dtype"]]
    size = codec._tensor_size("IMAGE", d["shape"], itemsize, limit)
    if type(alpha["size"]) is not int or alpha["size"] != size:
        raise ValueError("Invalid IMAGE alpha size")
    return size, torch.float64 if dtype == torch.float64 else torch.float32


def _import_av():
    try:
        import av
    except ImportError:
        raise ValueError("PyAV is required for IMAGE video transfer") from None
    return av


def _encode_still(frame, fmt, quality, limit):
    from PIL import Image
    sink = _BoundedIO(limit)
    if frame.shape[-1] == 1:
        frame = frame[..., 0]
    image = Image.fromarray(frame)
    if fmt in ("legacy_png", "png"):
        image.save(sink, format="PNG", compress_level=0 if fmt == "legacy_png" else 4)
    elif fmt == "jpeg":
        image.save(sink, format="JPEG", quality=quality, subsampling=0 if quality >= 90 else 2)
    else:
        image.save(sink, format="WEBP", quality=quality, method=4)
    return sink.getvalue()


def _encode_video(frames, fmt, quality, height, width, limit):
    import numpy as np
    av = _import_av()
    encoder, maximum, pix_fmt = _VIDEO_CODECS[fmt]
    try:
        av.codec.Codec(encoder, "w")
    except Exception:
        raise ValueError("IMAGE software encoder unavailable: " + encoder) from None
    sink = _BoundedIO(limit)
    options = {}
    if maximum is not None:
        options["crf"] = str(_quality_to_crf(quality, maximum))
    if fmt == "h264":
        options["preset"] = "medium"
    elif fmt == "h265":
        options.update(preset="fast", **{"x265-params": "log-level=error:pools=none:frame-threads=1"})
    elif fmt == "av1":
        options.update(preset="8", **{"svtav1-params": "lp=1"})
    else:
        options.update(level="3", slicecrc="1")
    try:
        with av.open(sink, mode="w", format="matroska") as container:
            stream = container.add_stream(encoder, rate=24)
            stream.width, stream.height = width, height
            stream.pix_fmt = pix_fmt
            stream.codec_context.thread_count = 1
            stream.options = options
            for index, frame in enumerate(frames):
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                h, w = frame.shape[:2]
                frame = np.pad(frame, ((0, height - h), (0, width - w), (0, 0)), mode="edge")
                vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
                vf.pts = index
                vf.time_base = Fraction(1, 24)
                for packet in stream.encode(vf):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        data = sink.getvalue()
        if len(data) > limit:
            raise ValueError("Encoded IMAGE exceeds byte limit")
        return data
    except av.error.FFmpegError:
        raise ValueError("IMAGE software video encode failed (codec, dimensions or byte limit)") from None
    except ValueError:
        raise
    except Exception:
        raise ValueError("IMAGE software video encode failed (codec, dimensions or byte limit)") from None


def encode_image(value, *, transfer_format=DEFAULT_FORMAT, transfer_quality=DEFAULT_QUALITY,
                 max_bytes=MAX_PAYLOAD_BYTES):
    """Return (descriptor, bytes); reject unsupported inputs, never fall back."""
    fmt, quality = validate_options(transfer_format, transfer_quality)
    if fmt == "exact":
        return codec.encode_payload("IMAGE", value, max_bytes=max_bytes)
    limit = codec._byte_limit(max_bytes)
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("IMAGE must be a CPU tensor")
    if value.layout != torch.strided or value.is_nested:
        raise ValueError("IMAGE must have dense strided layout")
    if value.dtype not in codec._DTYPE_NAMES:
        raise ValueError("Unsupported IMAGE dtype")
    shape = list(value.shape)
    codec._tensor_size("IMAGE", shape, value.element_size(), limit)
    alpha_size = shape[0] * shape[1] * shape[2] * value.element_size() if shape[-1] == 4 else 0
    itemsize = value.element_size()
    ph, pw = _shape_budget(shape, fmt, limit, itemsize, alpha_size)
    if not torch.isfinite(value).all().item():
        raise ValueError("IMAGE values must be finite")
    value = value.detach().resolve_neg()
    alpha, alpha_payload = None, b""
    if shape[-1] == 4:
        ad, alpha_payload = codec.encode_payload("IMAGE", value[..., 3:4], max_bytes=limit)
        alpha = {"descriptor": ad, "size": len(alpha_payload)}
    import numpy as np
    def frames():
        for frame in value:
            arr = frame[..., :3].float().numpy()
            with np.errstate(over="ignore", invalid="ignore"):
                yield np.ascontiguousarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    budget = limit - len(alpha_payload)
    if fmt in _VIDEO_CODECS:
        parts = [_encode_video(frames(), fmt, quality, ph, pw, budget)]
        header = _video_header(parts[0], fmt, ph, pw, shape[0])
        coded_frames = _coded_frame_bound(parts[0], fmt, header)
        _shape_budget(shape, fmt, limit, itemsize, alpha_size, coded_frames)
    else:
        parts = []
        for frame in frames():
            part = _encode_still(frame, fmt, quality, budget)
            budget -= len(part)
            parts.append(part)
    descriptor = {"encoding": "image-transfer-v1", "format": fmt, "quality": quality,
                  "shape": shape, "layout": "BHWC", "dtype": codec._DTYPE_NAMES[value.dtype],
                  "parts": [len(p) for p in parts], "alpha": alpha}
    _descriptor_budget(descriptor)
    payload = b"".join(parts) + alpha_payload
    if len(payload) > limit:
        raise ValueError("Encoded IMAGE exceeds byte limit")
    return descriptor, payload


def _still_header(data, fmt):
    """Read dimensions before Pillow (WebP may allocate pixels during open)."""
    if fmt in ("png", "legacy_png"):
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid IMAGE PNG header")
        offset, dimensions, saw_data, chunks = 8, None, False, 0
        while offset < len(data):
            chunks += 1
            if chunks > MAX_CONTAINER_ELEMENTS:
                raise ValueError("IMAGE PNG chunk count exceeds limit")
            if offset + 12 > len(data):
                raise ValueError("Truncated IMAGE PNG chunk")
            length = int.from_bytes(data[offset:offset + 4], "big")
            end = offset + 12 + length
            if end > len(data):
                raise ValueError("Truncated IMAGE PNG chunk")
            tag = data[offset + 4:offset + 8]
            body = data[offset + 8:end - 4]
            if zlib.crc32(data[offset + 4:end - 4]) != int.from_bytes(data[end - 4:end], "big"):
                raise ValueError("Invalid IMAGE PNG checksum")
            if tag == b"IHDR" and offset == 8 and length == 13:
                width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
                if depth != 8 or color not in (0, 2) or (compression, filtering, interlace) != (0, 0, 0):
                    raise ValueError("Unsupported IMAGE PNG layout")
                dimensions = (height, width)
            elif tag == b"IDAT" and dimensions:
                saw_data = True
            elif tag == b"IEND" and not length and end == len(data) and saw_data:
                return dimensions
            else:
                raise ValueError("Unsupported IMAGE PNG chunk")
            offset = end
        raise ValueError("Truncated IMAGE PNG")
    if fmt == "webp":
        # Our encoder writes a single lossy VP8 still. Reject animation/alpha/
        # extended WebP before WebPAnimDecoder can allocate its canvas.
        if len(data) < 30 or data[:4] != b"RIFF" or data[8:16] != b"WEBPVP8 ":
            raise ValueError("Unsupported IMAGE WebP header")
        size = int.from_bytes(data[16:20], "little")
        if int.from_bytes(data[4:8], "little") + 8 != len(data) or 20 + size + size % 2 != len(data):
            raise ValueError("Invalid IMAGE WebP length")
        if data[23:26] != b"\x9d\x01\x2a":
            raise ValueError("Invalid IMAGE WebP frame")
        width, height = struct.unpack("<HH", data[26:30])
        return height & 0x3fff, width & 0x3fff
    if data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ValueError("Invalid IMAGE JPEG header or trailer")
    offset, dimensions = 2, None
    while offset + 4 <= len(data):
        if data[offset] != 0xff:
            raise ValueError("Invalid IMAGE JPEG marker")
        marker = data[offset + 1]
        size = int.from_bytes(data[offset + 2:offset + 4], "big")
        end = offset + 2 + size
        if size < 2 or end > len(data):
            raise ValueError("Truncated IMAGE JPEG marker")
        if marker in (0xc0, 0xc2):
            if dimensions or size < 8 or data[offset + 4] != 8 or data[offset + 9] not in (1, 3):
                raise ValueError("Unsupported IMAGE JPEG layout")
            dimensions = struct.unpack(">HH", data[offset + 5:offset + 9])
        elif marker == 0xda:
            if dimensions is None:
                raise ValueError("Missing IMAGE JPEG dimensions")
            return dimensions
        offset = end
    raise ValueError("Missing IMAGE JPEG pixels")


def _video_header(data, fmt, height, width, count):
    """Strict bounded EBML preflight: one track, no lacing/attachments/encodings.

    Native stream probing is not the first dimension check: it may instantiate a
    decoder. Validate the Matroska header and count blocks without decompressing.
    """
    nodes = 0
    def vint(pos, end, identifier=False):
        if pos >= end or data[pos] == 0:
            raise ValueError("Invalid IMAGE EBML integer")
        length = 9 - data[pos].bit_length()
        if length > (4 if identifier else 8) or pos + length > end:
            raise ValueError("Truncated IMAGE EBML integer")
        value = int.from_bytes(data[pos:pos + length], "big")
        if not identifier:
            value &= (1 << (7 * length)) - 1
            if value == (1 << (7 * length)) - 1:
                raise ValueError("Unknown-sized IMAGE EBML elements are forbidden")
        return value, pos + length
    def elements(start, end):
        nonlocal nodes
        while start < end:
            nodes += 1
            if nodes > MAX_CONTAINER_ELEMENTS:
                raise ValueError("IMAGE container element count exceeds limit")
            tag, pos = vint(start, end, True)
            length, body = vint(pos, end)
            finish = body + length
            if finish > end:
                raise ValueError("Truncated IMAGE EBML element")
            yield tag, body, finish
            start = finish
    def integer(start, end):
        if not 1 <= end - start <= 8:
            raise ValueError("Invalid IMAGE EBML numeric field")
        return int.from_bytes(data[start:end], "big")
    roots = list(elements(0, len(data)))
    if len(roots) != 2 or [r[0] for r in roots] != [0x1a45dfa3, 0x18538067]:
        raise ValueError("Invalid IMAGE Matroska structure")
    tracks, blocks, packets = [], 0, []
    for tag, start, end in elements(roots[1][1], roots[1][2]):
        if tag == 0x1654ae6b:
            for entry, es, ee in elements(start, end):
                if entry in (0xbf, 0xec):
                    continue
                if entry != 0xae:
                    raise ValueError("Unexpected IMAGE track element")
                fields, dims = {}, {}
                for field, fs, fe in elements(es, ee):
                    if field in fields:
                        raise ValueError("Duplicate IMAGE track field")
                    fields[field] = (fs, fe)
                    if field == 0x6d80:
                        raise ValueError("IMAGE track encodings are forbidden")
                    if field == 0xe0:
                        for dim, ds, de in elements(fs, fe):
                            if dim in dims:
                                raise ValueError("Duplicate IMAGE video field")
                            dims[dim] = (ds, de)
                if any(key not in fields for key in (0xd7, 0x83, 0x86)) or any(key not in dims for key in (0xb0, 0xba)):
                    raise ValueError("Incomplete IMAGE video track")
                if integer(*fields[0xd7]) != 1 or integer(*fields[0x83]) != 1:
                    raise ValueError("Unsupported IMAGE video track")
                if fields[0x86][1] - fields[0x86][0] > 32:
                    raise ValueError("IMAGE codec identifier exceeds limit")
                codec_id = bytes(data[slice(*fields[0x86])])
                if codec_id != _CODEC_IDS[fmt] and not (fmt == "ffv1" and codec_id == b"V_MS/VFW/FOURCC"):
                    raise ValueError("IMAGE video codec mismatch")
                if integer(*dims[0xb0]) != width or integer(*dims[0xba]) != height:
                    raise ValueError("IMAGE video header dimensions mismatch")
                if codec_id == b"V_MS/VFW/FOURCC":
                    cs, ce = fields.get(0x63a2, (0, 0))
                    if ce - cs < 40 or data[cs + 16:cs + 20] != b"FFV1":
                        raise ValueError("Invalid IMAGE FFV1 codec header")
                    if struct.unpack("<ii", data[cs + 4:cs + 12]) != (width, height):
                        raise ValueError("IMAGE FFV1 header dimensions mismatch")
                tracks.append(fields)
        elif tag == 0x1f43b675:
            def block(bs, be):
                nonlocal blocks
                track, pos = vint(bs, be)
                if track != 1 or pos + 3 >= be or data[pos + 2] & 0x06:
                    raise ValueError("Invalid or laced IMAGE video block")
                blocks += 1
                if blocks > count:
                    raise ValueError("IMAGE video frame count exceeds descriptor")
                packets.append((pos + 3, be))
            for field, fs, fe in elements(start, end):
                if field == 0xa3:
                    block(fs, fe)
                elif field == 0xa0:
                    group = [e for e in elements(fs, fe) if e[0] == 0xa1]
                    if len(group) != 1:
                        raise ValueError("Invalid IMAGE video block group")
                    block(group[0][1], group[0][2])
                elif field not in (0xe7, 0xec, 0xbf):
                    raise ValueError("Unsupported IMAGE cluster field")
        elif tag not in (0x114d9b74, 0x1549a966, 0x1c53bb6b, 0x1254c367, 0xec, 0xbf):
            raise ValueError("Unsupported IMAGE Matroska element")
    if len(tracks) != 1 or blocks != count:
        raise ValueError("IMAGE video track or frame count mismatch")
    cs, ce = tracks[0].get(0x63a2, (0, 0))
    if not 0 < ce - cs <= MAX_DESCRIPTOR_BYTES:
        raise ValueError("IMAGE codec configuration exceeds limit or is missing")
    if bytes(data[slice(*tracks[0][0x86])]) == b"V_MS/VFW/FOURCC":
        cs += 40  # Skip BITMAPINFOHEADER; FFV1 receives only its codec config.
    return bytes(data[cs:ce]), packets


def _coded_frame_bound(data, fmt, header):
    """Conservative per-packet allocation bound before native decode.

    FFV1 has one picture per packet. Every AVC/HEVC picture needs at least one
    VCL NAL; every AV1 output (including show-existing) needs a frame/header OBU.
    Counting slices overestimates pictures, which is safe. Do not trust a
    Matroska block count alone: a malicious packet may contain several pictures.
    """
    config, packets = header
    if fmt == "ffv1":
        return len(packets)
    total, units = 0, 0
    if fmt in ("h264", "h265"):
        index = 4 if fmt == "h264" else 21
        if len(config) <= index or config[0] != 1:
            raise ValueError("Invalid IMAGE video configuration")
        length_size = (config[index] & 3) + 1
    for start, end in packets:
        while start < end:
            units += 1
            if units > MAX_CONTAINER_ELEMENTS:
                raise ValueError("IMAGE coded unit count exceeds limit")
            if fmt in ("h264", "h265"):
                if start + length_size > end:
                    raise ValueError("Truncated IMAGE NAL length")
                length = int.from_bytes(data[start:start + length_size], "big")
                start += length_size
                if length < (1 if fmt == "h264" else 2) or length > end - start:
                    raise ValueError("Truncated IMAGE NAL unit")
                kind = (data[start] & 31) if fmt == "h264" else (data[start] >> 1) & 63
                total += (1 <= kind <= 5) if fmt == "h264" else (kind <= 31)
            else:
                tag = data[start]
                start += 1
                if tag & 0x81 or not tag & 2:
                    raise ValueError("Unsupported IMAGE AV1 OBU header")
                kind = (tag >> 3) & 15
                if tag & 4:
                    start += 1
                length = 0
                for shift in range(0, 56, 7):
                    if start >= end:
                        raise ValueError("Truncated IMAGE AV1 OBU length")
                    value = data[start]
                    start += 1
                    length |= (value & 127) << shift
                    if not value & 128:
                        break
                else:
                    raise ValueError("IMAGE AV1 OBU length exceeds limit")
                if length > end - start:
                    raise ValueError("Truncated IMAGE AV1 OBU")
                total += kind in (3, 6)
            start += length
            if total > MAX_FRAMES * 4:
                raise ValueError("IMAGE coded frame count exceeds limit")
    if total < len(packets):
        raise ValueError("IMAGE video has missing coded frames")
    return total


def _decode_video(data, fmt, height, width, count, header):
    av = _import_av()
    maximum = ((height + 63) // 64 * 64) * ((width + 63) // 64 * 64)
    ctx = av.CodecContext.create("libdav1d" if fmt == "av1" else "hevc" if fmt == "h265" else fmt, "r")
    ctx.width, ctx.height = width, height
    ctx.extradata, packets = header
    ctx.thread_count = 1
    ctx.options = {"max_pixels": str(maximum), "threads": "1", "err_detect": "explode"}
    ctx.open()
    if "max_pixels" in ctx.options:
        raise ValueError("IMAGE decoder does not support max_pixels safety limit")
    if (ctx.width, ctx.height) != (width, height):
        raise ValueError("IMAGE bitstream dimensions mismatch")
    decoded = 0
    for bounds in packets + [None]:
        packet = av.Packet(data[slice(*bounds)]) if bounds is not None else None
        for frame in ctx.decode(packet):
            if decoded >= count or (frame.width, frame.height) != (width, height):
                raise ValueError("IMAGE decoded dimensions or frame count mismatch")
            decoded += 1
            yield frame.to_ndarray(format="rgb24")
    if decoded != count:
        raise ValueError("IMAGE decoded frame count mismatch")


def decode_image(descriptor, payload, *, max_bytes=MAX_PAYLOAD_BYTES):
    """Decode to an independent contiguous CPU tensor after bounded preflight."""
    limit = codec._byte_limit(max_bytes)
    if not isinstance(descriptor, dict):
        raise ValueError("IMAGE descriptor must be a dictionary")
    if descriptor.get("encoding") == "tensor-le-v1":
        return codec.decode_payload("IMAGE", descriptor, payload, max_bytes=limit)
    _descriptor_budget(descriptor)
    if set(descriptor) != {"encoding", "format", "quality", "shape", "layout", "dtype", "parts", "alpha"}:
        raise ValueError("Invalid IMAGE descriptor fields")
    if descriptor["encoding"] != "image-transfer-v1" or descriptor["layout"] != "BHWC":
        raise ValueError("Invalid IMAGE descriptor encoding or layout")
    fmt, _ = validate_options(descriptor["format"], descriptor["quality"])
    if fmt == "exact":
        raise ValueError("Exact IMAGE requires the tensor encoding")
    if not isinstance(payload, bytes):
        raise ValueError("IMAGE payload must be bytes")
    if len(payload) > limit:
        raise ValueError("IMAGE payload exceeds byte limit")
    shape = descriptor["shape"]
    if type(descriptor["dtype"]) is not str or descriptor["dtype"] not in codec._DTYPES:
        raise ValueError("Unsupported IMAGE descriptor dtype")
    dtype, itemsize = codec._DTYPES[descriptor["dtype"]]
    codec._tensor_size("IMAGE", shape, itemsize, limit)
    alpha_size, _ = _alpha_metadata(descriptor["alpha"], shape, limit)
    if alpha_size and descriptor["alpha"]["descriptor"]["dtype"] != descriptor["dtype"]:
        raise ValueError("IMAGE alpha dtype mismatch")
    ph, pw = _shape_budget(shape, fmt, limit, itemsize, alpha_size)
    count, height, width, channels = shape
    lengths = descriptor["parts"]
    if type(lengths) is not list or len(lengths) != (1 if fmt in _VIDEO_CODECS else count):
        raise ValueError("IMAGE part count mismatch")
    total = alpha_size
    for length in lengths:
        if type(length) is not int or not 0 < length <= limit - total:
            raise ValueError("IMAGE part length exceeds byte limit")
        total += length
    if total != len(payload):
        raise ValueError("IMAGE payload length does not match parts")
    parts, offset = [], 0
    for length in lengths:
        parts.append(memoryview(payload)[offset:offset + length])
        offset += length
    try:
        if fmt in _VIDEO_CODECS:
            video_header = _video_header(parts[0], fmt, ph, pw, count)
            coded_frames = _coded_frame_bound(parts[0], fmt, video_header)
            _shape_budget(shape, fmt, limit, itemsize, alpha_size, coded_frames)
        else:
            for part in parts:
                if _still_header(part, fmt) != (height, width):
                    raise ValueError("IMAGE header dimensions mismatch")
        # All headers and aggregate budgets are checked before output allocation.
        alpha = None
        if alpha_size:
            alpha = codec.decode_payload("IMAGE", descriptor["alpha"]["descriptor"], payload[offset:], max_bytes=limit)
        result = torch.empty(shape, dtype=dtype, device="cpu")
        if fmt in _VIDEO_CODECS:
            frames = _decode_video(parts[0], fmt, ph, pw, count, video_header)
        else:
            def stills():
                import numpy as np
                from PIL import Image
                for part in parts:
                    with Image.open(io.BytesIO(part)) as image:
                        if image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                            raise ValueError("IMAGE still dimensions or frame count mismatch")
                        if image.mode not in ("L", "RGB"):
                            raise ValueError("Unsupported IMAGE still pixel mode")
                        yield np.array(image.convert("RGB"), dtype=np.uint8)
            frames = stills()
        color = 3 if channels == 4 else channels
        for index, frame in enumerate(frames):
            # Float32 division followed by the source dtype cast reproduces
            # Collector normalized pixels without narrowing exact alpha.
            result[index, ..., :color] = torch.from_numpy(frame[:height, :width, :color].copy()).float().div_(255)
        if alpha is not None:
            result[..., 3:4] = alpha
        return result
    except ValueError:
        raise
    except Exception:
        raise ValueError("Malformed or unsupported IMAGE compressed payload") from None
