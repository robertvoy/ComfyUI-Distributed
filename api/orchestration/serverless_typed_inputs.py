"""Bounded local native inputs for comfy-media-v1 requests.

Only stock LoadAudio/LoadVideo schemas are adapted. Original nodes and links
remain in the private request graph for endpoint validation; only the existing
image preparer rewrites filenames. No upload, transcode, resample or provider
operation belongs here. The caller still enforces the complete request-body cap.
"""
import hashlib
import io
import json
from pathlib import Path

from ...serverless_bridge import codec, contract, video_codec
from . import serverless_media


_LOADERS = {"LoadAudio": ("audio", "AUDIO"), "LoadVideo": ("file", "VIDEO")}
Error = serverless_media.ServerlessMediaError


def _loader_references(workflow):
    if not isinstance(workflow, dict):
        raise Error("Serverless workflow must be an API graph")
    loaders = {}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or class_type not in _LOADERS:
            continue
        contract.identifier(node_id)
        key, kind = _LOADERS[class_type]
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != {key}:
            raise Error(f"{class_type} inputs must contain only its native filename; extra controls are unsupported")
        if not isinstance(inputs[key], str):
            raise Error(f"{class_type} requires a literal filename; linked inputs are unsupported")
        normalized, _suffix = serverless_media._normalize_media_reference(inputs[key])
        loaders[node_id] = (kind, normalized)

    references = []
    used = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        for key, value in node["inputs"].items():
            if not isinstance(value, (list, tuple)) or not value or not isinstance(value[0], str):
                continue
            if value[0] not in loaders:
                continue
            if type(value) is not list or len(value) != 2 or type(value[1]) is not int or value[1] != 0:
                raise Error("Native media loader links require output slot 0")
            contract.identifier(node_id)
            contract.identifier(key)
            references.append((node_id, key, value[0]))
            used.add(value[0])
    if used != set(loaders):
        raise Error("Native media loader has no consumer")
    return loaders, references


def _source_path(value, folder_paths):
    """Resolve native annotations; authorization/opening is owned by _read_path."""
    annotation = serverless_media._MEDIA_ANNOTATION.search(value)
    unannotated = value[:annotation.start()] if annotation else value
    if Path(unannotated).is_absolute():
        return unannotated
    try:
        return folder_paths.get_annotated_filepath(value)
    except Exception:
        raise Error("Native media input path is missing or invalid") from None


def _decode_audio(data, limit):
    """Match core nodes_audio.load, bounding each frame *before* accumulation.

    Float32 output bytes are charged even for narrow integer PCM. Native float64
    remains float64, as in f32_pcm (there is no silent dtype conversion). Raw
    retained arrays and each live AV plane allocation have a separate <= limit
    guard, so wider/padded raw frames cannot bypass the output budget. PyAV owns
    the one frame being decoded; it is checked before to_ndarray or torch.cat.
    Concatenation/conversion use bounded temporary copies, not an RSS guarantee.
    """
    if limit <= 0:
        raise Error("AUDIO decoded input exceeds byte limit")
    try:
        import av
        import torch
        from comfy_extras.nodes_audio import f32_pcm
    except ImportError:
        raise Error("Native AUDIO decoding is unavailable") from None

    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.audio:
                raise Error("Native AUDIO input has no audio stream")
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            channels = stream.channels
            if type(rate) is not int or rate <= 0 or type(channels) is not int or channels <= 0:
                raise Error("Native AUDIO stream shape or sample rate is invalid")
            frames = []
            output_bytes = raw_bytes = 0
            dtype = None
            for frame in container.decode(streams=stream.index):
                samples = frame.samples
                if (type(samples) is not int or samples <= 0 or frame.sample_rate != rate
                        or len(frame.layout.channels) != channels):
                    raise Error("Native AUDIO frame shape or sample rate changed")
                if frame.format.name not in {"s16", "s16p", "s32", "s32p", "flt", "fltp", "dbl", "dblp"}:
                    raise Error("Unsupported native AUDIO PCM format")
                itemsize = frame.format.bytes
                count = samples * channels
                # Native f32_pcm promotes integer PCM to float32, not float64.
                decoded_bytes = count * max(4, itemsize)
                frame_bytes = count * itemsize
                plane_bytes = sum(plane.buffer_size for plane in frame.planes)
                if (decoded_bytes > limit - output_bytes or frame_bytes > limit - raw_bytes
                        or plane_bytes > limit - raw_bytes):
                    raise Error("AUDIO decoded input or raw frame exceeds byte limit")
                output_bytes += decoded_bytes
                raw_bytes += frame_bytes
                array = frame.to_ndarray()
                if array.ndim != 2 or array.size != count or array.nbytes != frame_bytes:
                    raise Error("Native AUDIO frame shape is invalid")
                buf = torch.from_numpy(array)
                if buf.dtype not in (torch.int16, torch.int32, torch.float32, torch.float64):
                    raise Error("Unsupported native AUDIO PCM dtype")
                if dtype is not None and buf.dtype != dtype:
                    raise Error("Native AUDIO frame dtype changed")
                dtype = buf.dtype
                if buf.shape[0] != channels:
                    buf = buf.view(-1, channels).t()
                if tuple(buf.shape) != (channels, samples):
                    raise Error("Native AUDIO frame shape is invalid")
                frames.append(buf)
            if not frames:
                raise Error("Native AUDIO input has no decoded frames")
            waveform = f32_pcm(torch.cat(frames, dim=1)).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": rate}
    except Error:
        raise
    except Exception:
        raise Error("Native AUDIO input could not be decoded safely") from None


def _artifact_id(kind, descriptor, payload):
    digest = hashlib.sha256()
    digest.update(kind.encode("ascii") + b"\0")
    # Include interpretation (not just waveform bytes), notably sample_rate.
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    digest.update(b"\0")
    digest.update(payload)
    return "input-" + kind.lower() + "-" + digest.hexdigest()


async def prepare_media_inputs(workflow: dict) -> tuple[list, list, list]:
    """Return legacy images, inline typed artifacts, and every consumer reference.

    Paths are read once into bounded bytes through the video adapter's no-follow
    directory-fd reader. Decoders receive BytesIO, never a second path open.
    Source files and unique decoded artifacts have independent aggregate caps.
    Repeated content of the same kind is decoded once and charged once.
    """
    loaders, references = _loader_references(workflow)
    image_view = {node_id: node for node_id, node in workflow.items() if node_id not in loaders}
    # Retain the strict scan for every other loader before expensive decoding.
    serverless_media._iter_image_references(image_view)
    if not loaders:
        return await serverless_media.prepare_images(image_view), [], []
    try:
        import folder_paths
    except ImportError:
        raise Error("ComfyUI media directories are unavailable") from None
    roots = list(serverless_media._folder_roots(folder_paths).values())
    artifacts = {}
    source_artifacts = {}
    source_digests = set()
    filenames = {}
    loader_artifacts = {}
    file_total = decoded_total = 0
    for node_id, (kind, filename) in loaders.items():
        # Avoid retaining file buffers merely to deduplicate repeat references.
        if (kind, filename) in filenames:
            loader_artifacts[node_id] = filenames[kind, filename]
            continue
        path = _source_path(filename, folder_paths)
        data = video_codec._read_path(path, contract.MAX_ARTIFACT_BYTES, roots)
        source_digest = hashlib.sha256(data).hexdigest()
        if source_digest not in source_digests:
            if len(data) > contract.MAX_TOTAL_BYTES - file_total:
                raise Error("Native media source files exceed total byte limit")
            file_total += len(data)
            source_digests.add(source_digest)
        source_key = (kind, source_digest)
        if source_key not in source_artifacts:
            if len(artifacts) >= contract.MAX_ARTIFACTS:
                raise Error("Native media input artifact count exceeds limit")
            limit = min(contract.MAX_ARTIFACT_BYTES, contract.MAX_TOTAL_BYTES - decoded_total)
            if kind == "AUDIO":
                value = _decode_audio(data, limit)
                descriptor, payload = codec.encode_payload(kind, value, max_bytes=limit)
                del value
            else:
                from comfy_api.latest import InputImpl
                value = InputImpl.VideoFromFile(io.BytesIO(data))
                descriptor, payload = video_codec.encode_video(value, max_bytes=limit, allowed_roots=roots)
                del value
            aid = _artifact_id(kind, descriptor, payload)
            if aid not in artifacts:
                if len(payload) > limit:
                    raise Error("Native media decoded inputs exceed total byte limit")
                artifacts[aid] = contract.artifact_from_payload(
                    aid, kind, descriptor, payload, max_bytes=contract.MAX_ARTIFACT_BYTES)
                decoded_total += len(payload)
            source_artifacts[source_key] = aid
            del payload
        aid = source_artifacts[source_key]
        loader_artifacts[node_id] = filenames[kind, filename] = aid
        del data
    refs = [{"node_id": node_id, "input_name": key, "artifact_id": loader_artifacts[source]}
            for node_id, key, source in references]
    # Shared node dictionaries deliberately propagate only legacy image rewrites.
    images = await serverless_media.prepare_images(image_view)
    return images, list(artifacts.values()), refs
