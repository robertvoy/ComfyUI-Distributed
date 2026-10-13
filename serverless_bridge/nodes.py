"""Internal ComfyUI boundary nodes for the standalone media-v1 handler."""
import hashlib
import json
import uuid
from pathlib import Path

import folder_paths
from comfy_api.v0_0_2 import io

from .codec import decode_payload, encode_payload
from .contract import (
    MAX_ARTIFACT_BYTES, MAX_ITEMS, MAX_TOTAL_BYTES, SOCKET_KINDS,
    bounded_int, identifier, validate_binding, validate_json,
)


def _scalar(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Media bridge control must have exactly one value")
        return value[0]
    return value


def _json(text):
    if not isinstance(text, str) or len(text) > 256 * 1024:
        raise ValueError("Invalid media bridge JSON")
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid media bridge JSON") from None
    validate_json(value)
    return value


def _cpu(kind, value):
    # Runtime transfer only, never sampling or device discovery. Pure codecs
    # deliberately reject non-CPU storage rather than making hidden transfers.
    if kind in ("IMAGE", "MASK"):
        return value.detach().cpu()
    if kind == "AUDIO":
        return {"waveform": value["waveform"].detach().cpu(), "sample_rate": value["sample_rate"]}
    return value


def encode_value(kind, value):
    if kind == "VIDEO":
        from .video_codec import encode_video_cpu
        return encode_video_cpu(value)
    return encode_payload(kind, _cpu(kind, value))


def decode_value(kind, descriptor, payload):
    if kind == "VIDEO":
        from .video_codec import decode_video
        return decode_video(descriptor, payload)
    return decode_payload(kind, descriptor, payload)


def export_values(request_id, binding, values):
    """Write bounded owned payloads; history contains metadata, never base64."""
    identifier(request_id)
    validate_binding(binding)
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_ITEMS:
        raise ValueError("Connected media output has no values or exceeds item limit")
    request_key = hashlib.sha256(request_id.encode()).hexdigest()
    subfolder = f"distributed-media-v1/{request_key}"
    root = Path(folder_paths.get_temp_directory()).resolve()
    directory = (root / subfolder).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Unsafe media output directory")
    directory.mkdir(parents=True, exist_ok=True)
    written, artifacts, items = [], [], []
    total = 0
    try:
        for index, value in enumerate(values):
            descriptor, payload = encode_value(binding["kind"], value)
            validate_json(descriptor)
            total += len(payload)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Media output exceeds total byte limit")
            aid = uuid.uuid4().hex
            filename = aid + ".bin"
            path = directory / filename
            with path.open("xb") as stream:
                written.append(path)
                stream.write(payload)
            artifacts.append({"id": aid, "kind": binding["kind"], "descriptor": descriptor,
                              "byte_length": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                              "transport": {"type": "comfy-file", "filename": filename,
                                            "subfolder": subfolder, "folder_type": "temp"}})
            items.append({"binding_id": binding["binding_id"], "item_index": index, "artifact_id": aid})
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {"request_id": request_id, "binding": dict(binding, count=len(values)),
            "artifacts": artifacts, "items": items}


def import_value(artifact, filename):
    """Read a handler-staged, checksummed file within native media roots only."""
    fields = {"id", "kind", "descriptor", "byte_length", "sha256"}
    if not isinstance(artifact, dict) or set(artifact) != fields:
        raise ValueError("Invalid staged media metadata")
    identifier(artifact["id"])
    if artifact["kind"] not in SOCKET_KINDS.values():
        raise ValueError("Invalid staged media kind")
    size = bounded_int(artifact["byte_length"], MAX_ARTIFACT_BYTES)
    if not isinstance(filename, str):
        raise ValueError("Invalid staged media filename")
    roots = [Path(getter()).resolve() for getter in (
        folder_paths.get_input_directory, folder_paths.get_output_directory, folder_paths.get_temp_directory)]
    path = Path(folder_paths.get_annotated_filepath(filename)).resolve()
    if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
        raise ValueError("Staged media file is outside allowed roots or unavailable")
    if path.stat().st_size != size:
        raise ValueError("Staged media file length mismatch")
    with path.open("rb") as stream:
        payload = stream.read(size + 1)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise ValueError("Staged media file failed integrity validation")
    return decode_value(artifact["kind"], artifact["descriptor"], payload)


class MediaImportV1(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MediaImportV1", category="distributed/internal", is_deprecated=True,
            inputs=[io.String.Input("artifact_json"), io.String.Input("filename")],
            outputs=[io.Image.Output(), io.Audio.Output(), io.Mask.Output(), io.String.Output(), io.Video.Output()])

    @classmethod
    def execute(cls, artifact_json, filename):
        artifact = _json(artifact_json)
        value = import_value(artifact, filename)
        return io.NodeOutput(*(value if kind == artifact["kind"] else None for kind in SOCKET_KINDS.values()))


class MediaExportV1(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MediaExportV1", category="distributed/internal", is_deprecated=True,
            inputs=[io.String.Input("request_id"), io.String.Input("binding_json"),
                    io.Image.Input("images", optional=True), io.Audio.Input("audio", optional=True),
                    io.Mask.Input("masks", optional=True), io.String.Input("texts", optional=True, force_input=True),
                    io.Video.Input("video", optional=True)], outputs=[], is_output_node=True, is_input_list=True)

    @classmethod
    def execute(cls, request_id, binding_json, images=None, audio=None, masks=None, texts=None, video=None):
        binding = _json(_scalar(binding_json))
        validate_binding(binding)
        provided = dict(images=images, audio=audio, masks=masks, texts=texts, video=video)
        if {name for name, value in provided.items() if value is not None} != {binding["socket"]}:
            raise ValueError("Media export must receive exactly its declared typed socket")
        exported = export_values(_scalar(request_id), binding, provided[binding["socket"]])
        return io.NodeOutput(ui={"media_exports": [exported]})
