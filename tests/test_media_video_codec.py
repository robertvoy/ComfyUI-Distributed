"""Native ComfyUI VIDEO tests, isolated from the custom-node import graph.

The outer pytest process starts one bounded CPU-only subprocess. Inside it the
real pinned core is imported *after* cli_args.args.cpu=True; no fake VideoInput
classes, media transcodes, providers, server, or GPU queries are needed.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


if os.environ.get("_DISTRIBUTED_VIDEO_CODEC_CORE_TEST") != "1":
    def test_native_video_codec_in_cpu_subprocess():
        configured = os.environ.get("COMFYUI_SOURCE_ROOT", os.environ.get("COMFYUI_VIDEO_TEST_ROOT"))
        if configured is None:
            pytest.skip("Set COMFYUI_SOURCE_ROOT for CPU native VIDEO acceptance")
        core = Path(configured).expanduser().resolve()
        assert configured and (core / "comfy_api/latest/_input_impl/video_types.py").is_file(), (
            "Configured COMFYUI_SOURCE_ROOT (or COMFYUI_VIDEO_TEST_ROOT) must provide native video classes"
        )
        env = dict(os.environ)
        scratch = Path(os.environ.get("TMPDIR", str(Path.home() / ".hermes/cache/scratch")))
        scratch.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(scratch)
        for name in ("PYTHONPATH", "COMFYUI_SOURCE_ROOT"):
            env.pop(name, None)
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                   _DISTRIBUTED_VIDEO_CODEC_CORE_TEST="1", PYTHONDONTWRITEBYTECODE="1")
        bootstrap = """
import sys
sys.path.insert(0, sys.argv[1])
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('VIDEO tests must not query or initialize a GPU')
# Third-party core imports probe availability even in CPU mode: answer without
# executing a GPU query, then forbid even these probes during codec execution.
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
for name in ('init', '_lazy_init', 'current_device'):
    setattr(torch.cuda, name, forbidden)
import comfy_api.latest._input_impl.video_types
for name in ('is_available', 'device_count'):
    setattr(torch.cuda, name, forbidden)
import pytest
raise SystemExit(pytest.main([sys.argv[2], '-q', '-p', 'no:cacheprovider']))
"""
        result = subprocess.run(
            [sys.executable, "-c", bootstrap, str(core), str(Path(__file__).resolve())],
            env=env, text=True, capture_output=True, timeout=150,
            cwd=Path(__file__).resolve().parents[1],
        )
        print(result.stdout)
        assert result.returncode == 0, result.stdout + result.stderr
else:
    import copy
    from fractions import Fraction
    import importlib
    import io
    import json
    import math
    import types

    import av
    import numpy as np
    import torch
    from comfy_api.latest._input import VideoInput
    from comfy_api.latest._input_impl.video_types import (
        VideoFromComponents, VideoFromFile, VideoFromList,
    )
    from comfy_api.latest._util import VideoComponents

    # Only this sibling package is synthetic; all native media types are real.
    package_name = "_distributed_video_codec_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "serverless_bridge")]
    sys.modules[package_name] = package
    codec = importlib.import_module(package_name + ".video_codec")

    def audio(dtype=torch.float64):
        return {"waveform": torch.tensor([[[0.123456789, -0.0, -1.5, 2.0]]], dtype=dtype, device="cpu"),
                "sample_rate": 48000}

    def components(dtype=torch.float64, *, metadata=None, alpha=True, sound=True):
        image = torch.arange(36, device="cpu", dtype=torch.float64).reshape(2, 2, 3, 3) / 997
        image[0, 0, 0, 0] = -0.0
        return VideoFromComponents(VideoComponents(
            images=image.to(dtype), frame_rate=Fraction(30000, 1001),
            audio=audio(dtype) if sound else None,
            alpha=torch.full((2, 2, 3), 0.123456789, dtype=dtype, device="cpu") if alpha else None,
            metadata=metadata,
        ), bit_depth=10, color_space="HDR")

    def exact_tensor(actual, expected):
        assert actual.device.type == "cpu"
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    def file_value(data=b"opaque original media bytes", **kwargs):
        return VideoFromFile(io.BytesIO(data), **kwargs)

    def forbid(*args, **kwargs):
        raise AssertionError("Unexpected materialization/transcoding/allocation")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("alpha,sound", [(False, False), (True, False), (False, True), (True, True)])
    def test_components_exact_roundtrip(dtype, alpha, sound, monkeypatch):
        metadata = {"title": "雪 🎞", "flags": [True, None, -4, 0.125], "nested": {"empty": []}}
        source = components(dtype, metadata=metadata, alpha=alpha, sound=sound)
        expected = source.get_components()
        monkeypatch.setattr(VideoFromComponents, "get_components", forbid)
        monkeypatch.setattr(VideoFromComponents, "save_to", forbid)
        descriptor, payload = codec.encode_video(source)
        assert descriptor["encoding"] == "native-video-v1"
        assert descriptor["representation"] == "components"
        assert descriptor["frame_rate"] == {"numerator": 30000, "denominator": 1001}
        # This is the actual JSON wire descriptor, not a Python-only shortcut.
        descriptor = json.loads(json.dumps(descriptor, allow_nan=False))
        result = codec.decode_video(descriptor, payload)
        assert type(result) is VideoFromComponents
        assert isinstance(result, VideoInput)
        actual = vars(result)["_VideoFromComponents__components"]
        exact_tensor(actual.images, expected.images)
        assert actual.frame_rate == expected.frame_rate
        assert type(actual.frame_rate) is Fraction
        assert actual.metadata == metadata
        assert actual.metadata is not metadata
        assert vars(result)["_VideoFromComponents__bit_depth"] == 10
        assert vars(result)["_VideoFromComponents__color_space"] == "HDR"
        if alpha:
            exact_tensor(actual.alpha, expected.alpha)
        else:
            assert actual.alpha is None
        if sound:
            exact_tensor(actual.audio["waveform"], expected.audio["waveform"])
            assert actual.audio["sample_rate"] == expected.audio["sample_rate"]
        else:
            assert actual.audio is None
        actual.images.fill_(99)
        assert not torch.equal(actual.images, expected.images)
        actual.metadata["nested"]["empty"].append("new")
        assert descriptor["metadata"]["nested"]["empty"] == []
        assert metadata["nested"]["empty"] == []

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_native_bhw1_alpha_shape_is_preserved(dtype):
        source = components(dtype)
        original = source.get_components()
        vars(source)["_VideoFromComponents__components"].alpha = original.alpha.unsqueeze(-1)
        descriptor, payload = codec.encode_video(source)
        result = codec.decode_video(descriptor, payload).get_components()
        exact_tensor(result.alpha, original.alpha.unsqueeze(-1))
        assert descriptor["alpha"]["descriptor"]["shape"] == [2, 2, 3, 1]

    @pytest.mark.parametrize("start,duration,crop", [
        (-1.25, -2.5, (-3, 5, 7, -9)), (0, 0, None), (-0.0, 0.0, (0, 0, 2, 2)),
    ])
    def test_file_preserves_bytes_cursor_signed_trim_crop_and_ownership(start, duration, crop, monkeypatch):
        source = file_value(start_time=start, duration=duration, crop=crop)
        stream = vars(source)["_VideoFromFile__file"]
        stream.seek(7)
        monkeypatch.setattr(VideoFromFile, "get_components", forbid)
        monkeypatch.setattr(VideoFromFile, "save_to", forbid)
        descriptor, payload = codec.encode_video(source)
        assert stream.tell() == 7
        assert payload == stream.getvalue()
        assert descriptor["start_time"] == start
        assert descriptor["duration"] == duration
        result = codec.decode_video(json.loads(json.dumps(descriptor)), payload)
        second = codec.decode_video(descriptor, payload)
        attrs = vars(result)
        assert type(result) is VideoFromFile
        assert attrs["_VideoFromFile__start_time"] == start
        assert type(attrs["_VideoFromFile__start_time"]) is type(start)
        assert attrs["_VideoFromFile__duration"] == duration
        assert attrs["_VideoFromFile__crop"] == crop
        if type(start) is float:
            assert math.copysign(1, attrs["_VideoFromFile__start_time"]) == math.copysign(1, start)
        decoded = attrs["_VideoFromFile__file"]
        assert type(decoded) is io.BytesIO
        assert decoded is not stream
        stream.close()
        assert decoded.getvalue() == payload
        decoded.write(b"changed")
        assert vars(second)["_VideoFromFile__file"].getvalue() == payload

    def test_real_tiny_pyav_video_is_still_readable_after_source_closed():
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=Fraction(24, 1))
            # Width 32 avoids a native fillborders filter limitation on tiny
            # unaligned YUV frames; this is still a tiny CPU-only fixture.
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for value in (16, 64, 128):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        data = buffer.getvalue()
        source = VideoFromFile(buffer, crop=(0, 0, 2, 2))
        descriptor, payload = codec.encode_video(source)
        expected = source.get_components()
        result = codec.decode_video(descriptor, payload)
        buffer.close()
        assert payload == data
        actual = result.get_components()
        exact_tensor(actual.images, expected.images)
        assert actual.frame_rate == expected.frame_rate
        assert tuple(actual.images.shape) == (3, 2, 2, 3)

    def test_segments_order_complete_audio_no_transcode_and_independent_buffers(monkeypatch):
        values = [file_value(b"first", start_time=-2), file_value(b"second", duration=-4, crop=(1, 2, 3, 4))]
        source = VideoFromList([VideoFromList(values[:1]), values[1]], complete_audio=audio())
        monkeypatch.setattr(VideoFromFile, "get_components", forbid)
        monkeypatch.setattr(VideoFromFile, "save_to", forbid)
        monkeypatch.setattr(VideoFromList, "get_components", forbid)
        monkeypatch.setattr(VideoFromList, "save_to", forbid)
        descriptor, payload = codec.encode_video(source)
        assert descriptor["representation"] == "segments"
        assert payload.startswith(b"firstsecond")
        assert len(descriptor["segments"]) == 2
        decoded = codec.decode_video(json.loads(json.dumps(descriptor)), payload)
        assert type(decoded) is VideoFromList
        assert [vars(v)["_VideoFromFile__file"].getvalue() for v in decoded.videos] == [b"first", b"second"]
        exact_tensor(decoded.complete_audio["waveform"], source.complete_audio["waveform"])
        for old, new in zip(source.videos, decoded.videos):
            assert old is not new
            vars(old)["_VideoFromFile__file"].close()
            assert not vars(new)["_VideoFromFile__file"].closed
        assert vars(decoded.videos[0])["_VideoFromFile__start_time"] == -2
        assert vars(decoded.videos[1])["_VideoFromFile__crop"] == (1, 2, 3, 4)

    def test_path_roots_symlinks_and_no_path_in_wire(tmp_path, monkeypatch):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        media = allowed / "private-name.mkv"
        media.write_bytes(b"media bytes")
        sibling = tmp_path / "allowed-sibling"
        sibling.mkdir()
        secret = sibling / "secret.mkv"
        secret.write_bytes(b"private")
        (allowed / "escape.mkv").symlink_to(secret)
        (allowed / "inside.mkv").symlink_to(media)
        descriptor, payload = codec.encode_video(VideoFromFile(str(media)), allowed_roots=[allowed])
        assert payload == b"media bytes"
        assert "private-name" not in json.dumps(descriptor)
        assert codec.encode_video(VideoFromFile(str(allowed / "inside.mkv")), allowed_roots=[allowed])[1] == payload
        for path in (secret, allowed / "escape.mkv", allowed, allowed / "missing"):
            with pytest.raises(ValueError):
                codec.encode_video(VideoFromFile(str(path)), allowed_roots=[allowed])
        with pytest.raises(ValueError):
            codec.encode_video(VideoFromFile(str(media)), allowed_roots=[])
        import folder_paths
        for name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
            monkeypatch.setattr(folder_paths, name, lambda: str(allowed))
        assert codec.encode_video(VideoFromFile(str(media)))[1] == payload
        with pytest.raises(ValueError):
            codec.encode_video(VideoFromFile(str(secret)))
        result = codec.decode_video(descriptor, payload)
        media.unlink()
        assert vars(result)["_VideoFromFile__file"].getvalue() == b"media bytes"

    def test_path_special_files_and_symlink_swap_are_rejected(tmp_path, monkeypatch):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError):
            codec.encode_video(VideoFromFile(str(fifo)), allowed_roots=[tmp_path])
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        actual = allowed / "subdir"
        actual.mkdir()
        media = actual / "clip.mkv"
        media.write_bytes(b"allowed")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "clip.mkv").write_bytes(b"forbidden")
        real_resolve = codec.Path.resolve
        swapped = False

        def swap_after_resolution(path, *args, **kwargs):
            nonlocal swapped
            resolved = real_resolve(path, *args, **kwargs)
            if path == media and not swapped:
                swapped = True
                actual.rename(allowed / "old")
                actual.symlink_to(outside, target_is_directory=True)
            return resolved

        monkeypatch.setattr(codec.Path, "resolve", swap_after_resolution)
        with pytest.raises(ValueError):
            codec.encode_video(VideoFromFile(str(media)), allowed_roots=[allowed])
        assert swapped

    def test_file_size_budget_checked_before_read(tmp_path, monkeypatch):
        path = tmp_path / "large.mkv"
        path.write_bytes(b"12345")
        monkeypatch.setattr(codec.os, "read", forbid)
        with pytest.raises(ValueError, match="limit|budget"):
            codec.encode_video(VideoFromFile(str(path)), max_bytes=4, allowed_roots=[tmp_path])

    @pytest.mark.parametrize("limit", [True, False, -1, 1.5, None])
    def test_invalid_limits(limit):
        with pytest.raises(ValueError):
            codec.encode_video(file_value(), max_bytes=limit)
        with pytest.raises(ValueError):
            codec.decode_video({}, b"", max_bytes=limit)

    def test_aggregate_budget_and_exact_boundary():
        value = VideoFromList([file_value(b"12"), file_value(b"345")])
        descriptor, payload = codec.encode_video(value, max_bytes=5)
        assert payload == b"12345"
        codec.decode_video(descriptor, payload, max_bytes=5)
        with pytest.raises(ValueError):
            codec.encode_video(value, max_bytes=4)
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload, max_bytes=4)
        stream = vars(value.videos[0])["_VideoFromFile__file"]
        stream.seek(1)
        with pytest.raises(ValueError):
            codec.encode_video(value.videos[0], max_bytes=1)
        assert stream.tell() == 1

    @pytest.mark.parametrize("kind", ["components", "file", "segments"])
    def test_unknown_subclasses_never_materialized(kind):
        bases = {"components": VideoFromComponents, "file": VideoFromFile, "segments": VideoFromList}
        cls = type("UnknownVideo", (bases[kind],), {"get_components": forbid, "save_to": forbid})
        value = object.__new__(cls)
        with pytest.raises(ValueError, match="class|type|unsupported|Unsupported"):
            codec.encode_video(value)
        with pytest.raises(ValueError):
            codec.encode_video(object())

    @pytest.mark.parametrize("kind", ["components", "file", "segments"])
    def test_bridge_rejects_subclasses_without_materializing(kind, monkeypatch):
        nodes = importlib.import_module(package_name + ".nodes")
        bases = {"components": VideoFromComponents, "file": VideoFromFile, "segments": VideoFromList}
        cls = type("UnknownVideo", (bases[kind],), {"get_components": forbid, "save_to": forbid})
        monkeypatch.setattr(torch.Tensor, "cpu", forbid)
        with pytest.raises(ValueError, match="Unsupported native VIDEO class"):
            nodes.encode_value("VIDEO", object.__new__(cls))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("kind", ["components", "segments", "file"])
    def test_bridge_stages_cpu_tensors_without_materialization(dtype, kind, monkeypatch):
        nodes = importlib.import_module(package_name + ".nodes")
        if kind == "components":
            source = components(dtype, metadata={"nested": ["雪", None, True]})
            state = vars(source)["_VideoFromComponents__components"]
            state.alpha = state.alpha.unsqueeze(-1)
            tensors = [state.images, state.alpha, state.audio["waveform"]]
        elif kind == "segments":
            source = VideoFromList([file_value(start_time=-1.5, duration=-2, crop=(-1, 2, 3, 4))],
                                   complete_audio=audio(dtype))
            tensors = [source.complete_audio["waveform"]]
        else:
            source = file_value(start_time=-0.0, duration=-2, crop=(-1, 2, 3, 4))
            tensors = []
        for tensor in tensors:
            tensor.requires_grad_(True)
        expected = codec.encode_video(source)
        detached, staged = [], []
        real_detach, real_cpu = torch.Tensor.detach, torch.Tensor.cpu

        def detach(tensor):
            detached.append(id(tensor))
            return real_detach(tensor)

        def cpu(tensor, *args, **kwargs):
            assert not tensor.requires_grad
            staged.append(tensor)
            return real_cpu(tensor, *args, **kwargs)

        for cls in (VideoFromComponents, VideoFromFile, VideoFromList):
            monkeypatch.setattr(cls, "get_components", forbid)
            monkeypatch.setattr(cls, "save_to", forbid)
        monkeypatch.setattr(torch.Tensor, "detach", detach)
        monkeypatch.setattr(torch.Tensor, "cpu", cpu)
        actual = nodes.encode_value("VIDEO", source)
        assert len(staged) == len(tensors)
        assert all(id(tensor) in detached for tensor in tensors)
        assert all(tensor.requires_grad for tensor in tensors)
        assert actual == expected
        decoded = nodes.decode_value("VIDEO", *actual)
        assert type(decoded) is type(source)
        assert codec.encode_video(decoded) == expected
        assert len(staged) == len(tensors)  # Pure codec never stages implicitly.

    @pytest.mark.parametrize("field", ["images", "alpha", "audio", "complete_audio"])
    def test_pure_video_rejects_non_cpu_storage_without_transfer(field, monkeypatch):
        if field == "complete_audio":
            source = VideoFromList([file_value()], complete_audio=audio())
            target = source.complete_audio["waveform"]
        else:
            source = components()
            target = getattr(vars(source)["_VideoFromComponents__components"], field)
            if field == "audio":
                target = target["waveform"]
        # A property spy on real CPU tensors exercises the guard without a GPU
        # allocation, subclass, device discovery or CUDA initialization.
        real_device = torch.Tensor.device
        monkeypatch.setattr(torch.Tensor, "device", property(
            lambda tensor: types.SimpleNamespace(type="cuda") if tensor is target
            else real_device.__get__(tensor)))
        monkeypatch.setattr(torch.Tensor, "cpu", forbid)
        with pytest.raises(ValueError, match="must be on CPU"):
            codec.encode_video(source)

    @pytest.mark.parametrize("field,bad", [
        ("_VideoFromFile__start_time", True), ("_VideoFromFile__start_time", float("nan")),
        ("_VideoFromFile__duration", float("inf")), ("_VideoFromFile__crop", (1, 2, True, 4)),
        ("_VideoFromFile__crop", [1, 2, 3, 4]), ("_VideoFromFile__crop", (1, 2, 3)),
        ("_VideoFromFile__file", b"not a BytesIO"),
    ])
    def test_invalid_native_file_attributes(field, bad):
        value = file_value()
        setattr(value, field, bad)
        with pytest.raises(ValueError):
            codec.encode_video(value)

    def test_missing_attributes_and_invalid_component_state():
        value = file_value()
        del vars(value)["_VideoFromFile__duration"]
        with pytest.raises(ValueError):
            codec.encode_video(value)
        for field, bad in [("frame_rate", 24.0), ("frame_rate", Fraction(0)),
                           ("alpha", torch.ones(1, 1, 1, device="cpu"))]:
            value = components()
            setattr(vars(value)["_VideoFromComponents__components"], field, bad)
            with pytest.raises(ValueError):
                codec.encode_video(value)

    @pytest.mark.parametrize("metadata", [
        {"value": float("nan")}, {"value": float("inf")}, {1: "non-string key"},
        {"value": (1, 2)}, {"value": object()}, {"large": "x" * 65536},
    ])
    def test_invalid_metadata_rejected_both_directions(metadata):
        with pytest.raises(ValueError):
            codec.encode_video(components(metadata=metadata))
        descriptor, payload = codec.encode_video(components())
        descriptor["metadata"] = metadata
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)

    def test_metadata_depth_cycles_and_exact_size():
        metadata = {}
        current = metadata
        for _ in range(40):
            current["child"] = {}
            current = current["child"]
        for value in (metadata,):
            with pytest.raises(ValueError):
                codec.encode_video(components(metadata=value))
        cyclic = {}
        cyclic["self"] = cyclic
        with pytest.raises(ValueError):
            codec.encode_video(components(metadata=cyclic))
        exact = {"x": "a" * (65536 - 8)}
        assert len(json.dumps(exact, separators=(",", ":")).encode()) == 65536
        descriptor, payload = codec.encode_video(components(metadata=exact))
        assert codec.decode_video(descriptor, payload).get_components().metadata == exact

    @pytest.mark.parametrize("field,bad", [
        ("encoding", "other"), ("representation", "other"), ("byte_length", True),
        ("byte_length", -1), ("byte_length", 12345), ("start_time", False),
        ("duration", float("inf")), ("crop", [0, 0, True, 4]), ("crop", [1, 2, 3]),
    ])
    def test_malformed_file_descriptors(field, bad):
        descriptor, payload = codec.encode_video(file_value())
        descriptor[field] = bad
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)

    @pytest.mark.parametrize("field,bad", [("offset", True), ("offset", -1), ("offset", 1),
                                            ("length", True), ("length", -1), ("length", 2**80)])
    def test_invalid_offsets_lengths(field, bad):
        descriptor, payload = codec.encode_video(file_value())
        descriptor["data"][field] = bad
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)

    def test_truncation_trailing_gaps_overlap_and_unknown_fields():
        value = VideoFromList([file_value(b"123"), file_value(b"456")])
        original, payload = codec.encode_video(value)
        for wire in (payload[:-1], payload + b"x"):
            with pytest.raises(ValueError):
                codec.decode_video(original, wire)
        for offset in (2, 4):
            descriptor = copy.deepcopy(original)
            descriptor["segments"][1]["offset"] = offset
            with pytest.raises(ValueError):
                codec.decode_video(descriptor, payload)
        descriptor = copy.deepcopy(original)
        descriptor["surprise"] = "bad"
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)
        descriptor = copy.deepcopy(original)
        descriptor["segments"][0]["descriptor"]["byte_length"] += 1
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)
        descriptor = copy.deepcopy(original)
        descriptor["segments"][0]["descriptor"] = descriptor
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)

    def test_all_descriptors_validated_before_tensor_allocation(monkeypatch):
        original, payload = codec.encode_video(components())
        monkeypatch.setattr(codec, "decode_payload", forbid)
        cases = []
        for key, value in [("shape", [2**80, 2, 3]), ("shape", [True, 2, 3]),
                           ("dtype", "pickle"), ("layout", "BHWC")]:
            descriptor = copy.deepcopy(original)
            descriptor["alpha"]["descriptor"][key] = value
            cases.append(descriptor)
        descriptor = copy.deepcopy(original)
        descriptor["audio"]["descriptor"]["sample_rate"] = True
        cases.append(descriptor)
        descriptor = copy.deepcopy(original)
        descriptor["alpha"]["offset"] += 1
        cases.append(descriptor)
        for descriptor in cases:
            with pytest.raises(ValueError):
                codec.decode_video(descriptor, payload)

    @pytest.mark.parametrize("field,bad", [("numerator", True), ("denominator", True),
                                            ("denominator", 0), ("numerator", -3),
                                            ("denominator", -1001)])
    def test_invalid_fraction_fields(field, bad):
        descriptor, payload = codec.encode_video(components())
        descriptor["frame_rate"][field] = bad
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)

    def test_segment_limit_and_exact_maximum(monkeypatch):
        value = VideoFromList([file_value(b"x") for _ in range(64)])
        descriptor, payload = codec.encode_video(value)
        result = codec.decode_video(descriptor, payload)
        assert len(result.videos) == 64
        value.videos.append(file_value(b"x"))
        with pytest.raises(ValueError):
            codec.encode_video(value)
        descriptor["segments"].append(copy.deepcopy(descriptor["segments"][-1]))
        monkeypatch.setattr(codec, "decode_payload", forbid)
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)
        value.videos = []
        with pytest.raises(ValueError):
            codec.encode_video(value)
        value.videos = [components()]
        with pytest.raises(ValueError):
            codec.encode_video(value)

    @pytest.mark.parametrize("descriptor,payload", [(None, b""), ([], b""), ({}, bytearray()), ({}, "")])
    def test_invalid_top_level_types(descriptor, payload):
        with pytest.raises(ValueError):
            codec.decode_video(descriptor, payload)
