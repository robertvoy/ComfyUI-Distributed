"""Bounded input packing; native acceptance is opt-in and CPU-only."""
import asyncio
import base64
import copy
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import types
import wave
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_typed_inputs_tests"
for suffix in ("", ".api", ".api.orchestration", ".serverless_bridge"):
    package = types.ModuleType(PACKAGE + suffix)
    package.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(ROOT)]
    sys.modules[package.__name__] = package
packer = importlib.import_module(PACKAGE + ".api.orchestration.serverless_typed_inputs")
codec = importlib.import_module(PACKAGE + ".serverless_bridge.codec")


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def graph(kind="LoadAudio", filename="tone.wav"):
    key = "audio" if kind == "LoadAudio" else "file"
    socket = "audio" if kind == "LoadAudio" else "video"
    return {"source": node(kind, **{key: filename}),
            "consumer": node("Consumer", **{socket: ["source", 0]})}


def run(workflow):
    return asyncio.run(packer.prepare_media_inputs(workflow))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("input", "output", "temp")}
    for path in roots.values():
        path.mkdir()
    def resolve(value):
        for name in roots:
            marker = " [" + name + "]"
            if value.endswith(marker):
                return str(roots[name] / value[:-len(marker)])
        return str(roots["input"] / value)
    folders = types.SimpleNamespace(get_annotated_filepath=resolve)
    for name, path in roots.items():
        setattr(folders, "get_" + name + "_directory", lambda path=path: str(path))
    monkeypatch.setitem(sys.modules, "folder_paths", folders)
    return roots


@pytest.mark.parametrize("kind,key", [("LoadAudio", "audio"), ("LoadVideo", "file")])
@pytest.mark.parametrize("bad", [{}, None, [], {"filename": "a.wav"},
    {"audio": ["filename", 0]}, {"file": ["filename", 0]}])
def test_rejects_non_native_or_linked_inputs_before_read(kind, key, bad, monkeypatch):
    workflow = graph(kind)
    workflow["source"]["inputs"] = bad
    read = Mock(side_effect=AssertionError("must not read"))
    monkeypatch.setattr(packer.video_codec, "_read_path", read)
    with pytest.raises(ValueError):
        run(workflow)
    read.assert_not_called()


@pytest.mark.parametrize("kind,key", [("LoadAudio", "audio"), ("LoadVideo", "file")])
@pytest.mark.parametrize("extra", ["start_time", "duration", "sample_rate", "crop", "upload"])
def test_rejects_extra_loader_controls(kind, key, extra):
    workflow = graph(kind)
    workflow["source"]["inputs"][extra] = 0
    with pytest.raises(ValueError, match="controls|inputs"):
        run(workflow)


@pytest.mark.parametrize("slot", [1, -1, True, 0.0, "0", None])
def test_rejects_nonzero_or_noninteger_output_slot(slot):
    workflow = graph()
    workflow["consumer"]["inputs"]["audio"] = ["source", slot]
    with pytest.raises(ValueError, match="output|slot"):
        run(workflow)


def test_rejects_unused_loader_and_malformed_link():
    workflow = graph()
    workflow.pop("consumer")
    with pytest.raises(ValueError, match="consumer"):
        run(workflow)
    workflow = graph()
    workflow["consumer"]["inputs"]["audio"].append(0)
    with pytest.raises(ValueError):
        run(workflow)


@pytest.mark.parametrize("key,value", [("audio", "private.wav"), ("file", "private.mp4"), ("video", "private.mov")])
def test_unknown_media_loaders_still_use_strict_scan(key, value):
    with pytest.raises(ValueError):
        run({"source": node("CustomLoader", **{key: value})})


def test_images_still_rewrite_only_the_private_graph(paths):
    payload = b"legacy staging preserves the original file bytes"
    (paths["input"] / "picture.png").write_bytes(payload)
    source = {"image": node("LoadImage", image="picture.png"),
              "export": node("MediaExportV1", images=["image", 0])}
    workflow = copy.deepcopy(source)
    images, artifacts, refs = run(workflow)
    assert artifacts == refs == []
    assert len(images) == 1
    assert base64.b64decode(images[0]["image"]) == payload
    assert workflow["image"]["inputs"]["image"] == images[0]["name"]
    assert workflow["export"] == source["export"]
    assert source["image"]["inputs"]["image"] == "picture.png"


if os.environ.get("_DISTRIBUTED_TYPED_INPUT_CORE_TEST") != "1":
    def test_native_input_acceptance_cpu_subprocess():
        configured = os.environ.get("COMFYUI_SOURCE_ROOT")
        if configured is None:
            pytest.skip("Set COMFYUI_SOURCE_ROOT for CPU native input acceptance")
        core = Path(configured).expanduser().resolve()
        assert configured and (core / "comfy_extras/nodes_audio.py").is_file()
        env = dict(os.environ)
        scratch = Path(os.environ.get("TMPDIR", str(Path.home() / ".hermes/cache/scratch")))
        scratch.mkdir(parents=True, exist_ok=True)
        env.pop("PYTHONPATH", None)
        env.pop("COMFYUI_SOURCE_ROOT", None)
        env.update(TMPDIR=str(scratch), CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                   PYTHONDONTWRITEBYTECODE="1", _DISTRIBUTED_TYPED_INPUT_CORE_TEST="1")
        bootstrap = """
import sys
sys.path.insert(0, sys.argv[1])
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('Input packing must not query or initialize a GPU')
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
for name in ('init', '_lazy_init', 'current_device'):
    setattr(torch.cuda, name, forbidden)
import comfy_extras.nodes_audio
import comfy_api.latest._input_impl.video_types
for name in ('is_available', 'device_count'):
    setattr(torch.cuda, name, forbidden)
import pytest
raise SystemExit(pytest.main([sys.argv[2], '-q', '-p', 'no:cacheprovider']))
"""
        result = subprocess.run([sys.executable, "-c", bootstrap, str(core), str(Path(__file__).resolve())],
                                cwd=ROOT, env=env, text=True, capture_output=True, timeout=150)
        print(result.stdout)
        assert result.returncode == 0, result.stdout + result.stderr
else:
    import av
    import numpy as np
    import torch
    from comfy_extras import nodes_audio
    from comfy_api.latest._input_impl.video_types import VideoFromFile

    def wav_bytes(channels=2, sample_width=2, rate=22050, variant=0):
        values = np.array([-32768, -12001 + variant, -1, 0, 1, 8193, 32767] * channels,
                          dtype="<i2" if sample_width == 2 else "<i4")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(sample_width)
            out.setframerate(rate)
            out.writeframes(values.tobytes())
        return buffer.getvalue()

    def unpack(artifact):
        payload = base64.b64decode(artifact["transport"]["data_base64"])
        assert artifact["byte_length"] == len(payload)
        return payload

    @pytest.mark.parametrize("root", ["input", "output", "temp"])
    @pytest.mark.parametrize("channels,width", [(1, 2), (2, 2), (2, 4)])
    def test_exact_native_audio_and_every_consumer_ref(paths, root, channels, width):
        source = paths[root] / "tone.wav"
        source.write_bytes(wav_bytes(channels, width))
        workflow = graph(filename="tone.wav [" + root + "]")
        workflow["export"] = node("MediaExportV1", audio=["source", 0])
        workflow["consumer"]["inputs"]["second_audio"] = ["source", 0]
        before = copy.deepcopy(workflow)
        images, artifacts, refs = run(workflow)
        assert images == [] and len(artifacts) == 1
        artifact = artifacts[0]
        expected, rate = nodes_audio.load(str(source))
        actual = codec.decode_payload("AUDIO", artifact["descriptor"], unpack(artifact))
        assert actual["sample_rate"] == rate == 22050
        assert actual["waveform"].dtype == expected.dtype
        assert torch.equal(actual["waveform"].view(torch.uint8), expected.unsqueeze(0).contiguous().view(torch.uint8))
        assert refs == [{"node_id": "consumer", "input_name": "audio", "artifact_id": artifact["id"]},
                        {"node_id": "consumer", "input_name": "second_audio", "artifact_id": artifact["id"]},
                        {"node_id": "export", "input_name": "audio", "artifact_id": artifact["id"]}]
        assert workflow == before

    @pytest.mark.parametrize("root", ["input", "output", "temp"])
    def test_exact_video_bytes_without_container_decode(paths, root, monkeypatch):
        payload = b"opaque native video bytes\x00\xff"
        (paths[root] / "clip.mp4").write_bytes(payload)
        workflow = graph("LoadVideo", "clip.mp4 [" + root + "]")
        workflow["export"] = node("MediaExportV1", video=["source", 0])
        before = copy.deepcopy(workflow)
        def forbidden(*args, **kwargs):
            raise AssertionError("must not open, decode or transcode video")
        monkeypatch.setattr(av, "open", forbidden)
        monkeypatch.setattr(VideoFromFile, "get_components", forbidden)
        images, artifacts, refs = run(workflow)
        assert images == [] and len(artifacts) == 1
        artifact = artifacts[0]
        assert unpack(artifact) == payload
        assert artifact["descriptor"] == {"encoding": "native-video-v1", "representation": "file",
            "byte_length": len(payload), "start_time": 0, "duration": 0, "crop": None,
            "data": {"offset": 0, "length": len(payload)}}
        assert refs == [{"node_id": name, "input_name": "video", "artifact_id": artifact["id"]}
                        for name in ("consumer", "export")]
        native = packer.video_codec.decode_video(artifact["descriptor"], unpack(artifact))
        assert vars(native)["_VideoFromFile__file"].getvalue() == payload
        assert workflow == before

    def test_content_dedupe_kind_separation_and_mixed_images(paths):
        data = wav_bytes()
        (paths["input"] / "a.wav").write_bytes(data)
        (paths["output"] / "b.wav").write_bytes(data)
        (paths["input"] / "image.png").write_bytes(b"image bytes")
        workflow = graph(filename="a.wav")
        workflow["same"] = node("LoadAudio", audio="b.wav [output]")
        workflow["video"] = node("LoadVideo", file="a.wav")
        workflow["image"] = node("LoadImage", image="image.png")
        workflow["export"] = node("MediaExportV1", audio=["same", 0], video=["video", 0], images=["image", 0])
        before = copy.deepcopy(workflow)
        images, artifacts, refs = run(workflow)
        assert len(images) == 1 and len(artifacts) == 2 and len(refs) == 3
        assert refs[0]["artifact_id"] == refs[1]["artifact_id"] != refs[2]["artifact_id"]
        assert {a["kind"] for a in artifacts} == {"AUDIO", "VIDEO"}
        before["image"]["inputs"]["image"] = images[0]["name"]
        assert workflow == before
        second = graph(filename="b.wav [output]")
        assert run(second)[1][0]["id"] == artifacts[0]["id"]

    @pytest.mark.parametrize("kind", ["LoadAudio", "LoadVideo"])
    def test_path_rejections_and_per_file_bounds(paths, tmp_path, monkeypatch, kind):
        outside = tmp_path / "outside.wav"
        outside.write_bytes(wav_bytes())
        (paths["input"] / "escape.wav").symlink_to(outside)
        (paths["input"] / "dirlink").symlink_to(tmp_path, target_is_directory=True)
        os.mkfifo(paths["input"] / "fifo.wav")
        values = ["../outside.wav", str(outside), "escape.wav", "dirlink/outside.wav", "fifo.wav",
                  "absent.wav", "tone.wav [unknown]", "https://example.com/a.wav", "input [temp]", "C:\\a.wav"]
        for value in values:
            with pytest.raises(ValueError):
                run(graph(kind, value))
        (paths["input"] / "large.wav").write_bytes(wav_bytes())
        monkeypatch.setattr(packer.contract, "MAX_ARTIFACT_BYTES", 8)
        with pytest.raises(ValueError, match="limit|size"):
            run(graph(kind, "large.wav"))

    def test_aggregate_decoded_budget_and_duplicate_not_double_charged(paths, monkeypatch):
        payload = wav_bytes()
        (paths["input"] / "a.wav").write_bytes(payload)
        (paths["input"] / "b.wav").write_bytes(wav_bytes(variant=1))
        workflow = graph(filename="a.wav")
        workflow["other"] = node("LoadAudio", audio="a.wav")
        workflow["export"] = node("MediaExportV1", audio=["other", 0])
        # File source bytes have a separate aggregate bound; use a longer WAV
        # whose float output dominates its tiny header.
        with wave.open(io.BytesIO(payload), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setparams((2, 2, 22050, 0, "NONE", "not compressed"))
            wav.writeframes(frames * 100)
        (paths["input"] / "a.wav").write_bytes(buffer.getvalue())
        artifact = run(graph(filename="a.wav"))[1][0]
        monkeypatch.setattr(packer.contract, "MAX_TOTAL_BYTES", artifact["byte_length"])
        assert len(run(workflow)[1]) == 1
        workflow["other"]["inputs"]["audio"] = "b.wav"
        with pytest.raises(ValueError, match="limit|size|budget"):
            run(workflow)

    def test_audio_charged_before_ndarray_cat_or_conversion(monkeypatch):
        frame = types.SimpleNamespace(samples=100, sample_rate=22050,
            layout=types.SimpleNamespace(channels=[1, 2]),
            format=types.SimpleNamespace(name="s16", bytes=2),
            planes=[types.SimpleNamespace(buffer_size=400)], to_ndarray=Mock())
        stream = types.SimpleNamespace(index=0, channels=2,
                                      codec_context=types.SimpleNamespace(sample_rate=22050))
        container = Mock()
        container.__enter__ = Mock(return_value=container)
        container.__exit__ = Mock(return_value=False)
        container.streams = types.SimpleNamespace(audio=[stream])
        container.decode = Mock(return_value=iter([frame]))
        monkeypatch.setattr(av, "open", Mock(return_value=container))
        cat = Mock(side_effect=AssertionError("no accumulation allowed"))
        monkeypatch.setattr(torch, "cat", cat)
        with pytest.raises(ValueError, match="limit|size|budget"):
            packer._decode_audio(b"input", 799)
        frame.to_ndarray.assert_not_called()
        cat.assert_not_called()
        frame.samples = 1
        frame.planes[0].buffer_size = 800
        container.decode.return_value = iter([frame])
        with pytest.raises(ValueError, match="limit|size|budget"):
            packer._decode_audio(b"input", 799)
        frame.to_ndarray.assert_not_called()

    def test_audio_rejects_empty_invalid_and_channel_or_rate_drift(paths, monkeypatch):
        (paths["input"] / "bad.wav").write_bytes(b"not audio")
        with pytest.raises(ValueError):
            run(graph(filename="bad.wav"))
        stream = types.SimpleNamespace(index=0, channels=2, codec_context=types.SimpleNamespace(sample_rate=22050))
        for frames in ([], [types.SimpleNamespace(samples=1, sample_rate=44100,
                     layout=types.SimpleNamespace(channels=[1, 2]))],
                     [types.SimpleNamespace(samples=1, sample_rate=22050,
                     layout=types.SimpleNamespace(channels=[1]))]):
            container = Mock()
            container.__enter__ = Mock(return_value=container)
            container.__exit__ = Mock(return_value=False)
            container.streams = types.SimpleNamespace(audio=[stream])
            container.decode = Mock(return_value=iter(frames))
            monkeypatch.setattr(av, "open", Mock(return_value=container))
            with pytest.raises(ValueError):
                packer._decode_audio(b"input", 1000)
