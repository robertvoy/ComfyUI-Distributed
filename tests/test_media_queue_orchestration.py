"""Offline queue integration: real compilers, packers and wire validators.

Only host dispatch/probing/queueing and the manager lifecycle are mocked. No
provider requests, services or GPU work are permitted by these tests.
"""
import asyncio
import base64
import copy
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types
import uuid
import wave
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
# Preload before the shared integration loader snapshots sys.modules: extension
# modules cannot safely be unloaded and imported again between fixture contexts.
import torch
from PIL import Image

from test_serverless_integration import load_modules, module, PKG


PROFILE = "comfy-media-v1"


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def text_prompt():
    return {
        "text": node("PrimitiveString", value=" café雪\n"),
        "collector": node("DistributedMediaCollector", texts=["text", 0], load_balance=False),
        "save": node("TextSink", texts=["collector", 3]),
    }


def legacy_prompt():
    return {
        "source": node("EmptyImage", width=32, height=32, batch_size=1, color=0),
        "collector": node("DistributedCollector", images=["source", 0]),
        "save": node("SaveImage", images=["collector", 0], filename_prefix="local"),
    }


@pytest.fixture
def harness(monkeypatch):
    q, modules, ps = load_modules()
    with patch.dict(sys.modules, modules):
        pool = {"id": "pool", "name": "Media", "type": "serverless", "provider": "runpod",
                "endpoint_id": "offline-endpoint", "enabled": True, "jobs_per_run": 2,
                "timeout_seconds": 30, "media_profile": PROFILE}
        config = {"workers": [pool], "settings": {"master_delegate_only": True}}
        q.load_config = lambda: config
        q.select_active_workers = AsyncMock(side_effect=lambda workers, _ws, delegate, **kw: (workers, delegate))
        q.queue_prompt_payload = AsyncMock(return_value={"prompt_id": "master-prompt", "number": 7, "node_errors": {}})
        manager = types.SimpleNamespace(start=AsyncMock(), cancel=AsyncMock(), ensure_capacity=Mock())
        sys.modules[PKG + ".api.orchestration.serverless_runtime"] = module("runtime", get_manager=lambda: manager)
        # These are the actual modules, not permissive serializer/packing stubs.
        client = importlib.import_module(PKG + ".api.orchestration.runpod_client")
        compiler = importlib.import_module(PKG + ".api.orchestration.serverless_media_prompt")
        packer = importlib.import_module(PKG + ".api.orchestration.serverless_typed_inputs")
        validator = importlib.import_module(PKG + ".api.orchestration.serverless_typed_results")
        handler = importlib.import_module(PKG + ".serverless_bridge.handler")
        def forbidden(*args, **kwargs):
            raise AssertionError("Offline queue tests must not access a network or GPU")
        monkeypatch.setattr(aiohttp, "ClientSession", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        monkeypatch.setattr(socket.socket, "connect", forbidden)
        monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
        monkeypatch.setattr(torch.cuda, "init", forbidden)
        monkeypatch.setattr(torch.cuda, "current_device", forbidden)
        h = types.SimpleNamespace(q=q, ps=ps, pool=pool, config=config, manager=manager,
                                  client=client, compiler=compiler, packer=packer,
                                  validator=validator, handler=handler)
        yield h


@pytest.fixture
def media_paths(tmp_path, monkeypatch):
    paths = types.SimpleNamespace(get_annotated_filepath=lambda name: str(tmp_path / name))
    for kind in ("input", "output", "temp"):
        setattr(paths, "get_" + kind + "_directory", lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    return tmp_path


def run(h, prompt=None, **kwargs):
    return asyncio.run(h.q.orchestrate_distributed_execution(
        prompt if prompt is not None else text_prompt(), None, "client", **kwargs))


def assert_no_dispatch(h, *, no_probe=False):
    h.q.queue_prompt_payload.assert_not_awaited()
    h.q.dispatch_worker_prompt.assert_not_awaited()
    h.manager.start.assert_not_awaited()
    h.manager.cancel.assert_not_awaited()
    assert h.ps.distributed_pending_jobs == {}
    if no_probe:
        h.q.select_active_workers.assert_not_awaited()
        h.q.select_least_busy_worker.assert_not_awaited()
        h.q.fetch_worker_has_node.assert_not_awaited()
        h.q.sync_worker_media.assert_not_awaited()


@pytest.mark.parametrize("delegate", [True, False])
def test_real_compile_pack_serialize_and_canonical_bindings(harness, media_paths, delegate):
    h = harness
    image_path = media_paths / "picture.png"
    Image.new("RGBA", (2, 3), color=(17, 29, 41, 53)).save(image_path)
    image_bytes = image_path.read_bytes()
    prompt = text_prompt()
    prompt["image"] = node("LoadImage", image="picture.png")
    prompt["collector"]["inputs"].update(images=["image", 0], masks=["image", 1])
    prompt["collector2"] = node("DistributedMediaCollector", texts=["text", 0])
    prompt["unused"] = node("PreviewImage", images=["image", 0])
    source = copy.deepcopy(prompt)
    with patch.object(h.compiler, "compile_media_prompt", wraps=h.compiler.compile_media_prompt) as compile_call, \
         patch.object(h.packer, "prepare_media_inputs", wraps=h.packer.prepare_media_inputs) as pack_call, \
         patch.object(h.client, "serialize_request", wraps=h.client.serialize_request) as serialize_call, \
         patch.object(h.q, "prune_prompt_for_worker", side_effect=AssertionError("typed graph must bypass legacy pruning")):
        result = run(h, prompt, delegate_master=delegate)
    assert result == ("master-prompt", 7, 2, {})
    assert compile_call.call_count == pack_call.await_count == 2
    assert serialize_call.call_count >= 2
    assert prompt == source
    h.q.dispatch_worker_prompt.assert_not_awaited()
    jobs = h.manager.start.call_args.args[2]
    assert [j["worker"]["id"] for j in jobs] == ["serverless:pool:1", "serverless:pool:2"]
    assert len({j["media"]["request_id"] for j in jobs}) == 2
    master = h.q.queue_prompt_payload.call_args.args[0]
    for job in jobs:
        media = job["media"]
        assert str(uuid.UUID(media["request_id"])) == media["request_id"]
        assert uuid.UUID(media["request_id"]).version == 4
        assert media["profile"] == PROFILE and media["version"] == 1
        assert media["outputs"] == job["manifest"]
        assert media["outputs"] is not job["manifest"]
        assert media["artifacts"] == media["inputs"] == []
        assert {b["socket"] for b in job["manifest"]} == {"texts", "images", "masks"}
        assert len(job["manifest"]) == 4
        assert base64.b64decode(job["images"][0]["image"]) == image_bytes
        assert job["workflow"]["image"]["inputs"]["image"] == job["images"][0]["name"]
        assert not ({"save", "unused", "collector", "collector2"} & set(job["workflow"]))
        body = h.client.serialize_request(job["workflow"], job["images"], 30, media=media)
        wire = json.loads(body)
        assert wire["input"] == {"workflow": job["workflow"], "images": job["images"], "media": media}
        assert wire["policy"] == {"executionTimeout": 30000, "ttl": 30000}
        _, _, _, targets = h.handler._validate_request(wire["input"])
        assert targets == [b["node_id"] for b in job["manifest"]]
        assert h.validator.validate_prepared_job(job) is True
        for binding in job["manifest"]:
            assert binding["participant_id"] == job["worker"]["id"]
            assert binding["job_id"] == master[binding["collector_id"]]["inputs"]["multi_job_id"]
            controls = job["workflow"][binding["node_id"]]["inputs"]
            assert json.loads(controls["binding_json"]) == binding
            assert controls["request_id"] == media["request_id"]
    assert master["collector"]["inputs"]["delegate_only"] is delegate
    assert json.loads(master["collector"]["inputs"]["expected_media"]) == ["images", "masks", "texts"]
    if delegate:
        assert "text" not in master and "image" not in master
        assert not any(name in master["collector"]["inputs"] for name in ("images", "audio", "masks", "texts", "video"))
        assert not any(n["class_type"] == "DistributedEmptyImage" for n in master.values())
    else:
        assert master["collector"]["inputs"]["texts"] == ["text", 0]
        assert "text" in master and "image" in master


def test_delegate_disconnects_every_media_socket(harness):
    prompt = text_prompt()
    for index, name in enumerate(("images", "audio", "masks", "video")):
        source_id = "source-" + name
        prompt[source_id] = node("GeneratedMedia", seed=index)
        prompt["collector"]["inputs"][name] = [source_id, 0]
    original = copy.deepcopy(prompt)
    run(harness, prompt)
    master = harness.q.queue_prompt_payload.call_args.args[0]
    inputs = master["collector"]["inputs"]
    sockets = {"images", "audio", "masks", "texts", "video"}
    assert set(json.loads(inputs["expected_media"])) == sockets
    assert not sockets.intersection(inputs)
    assert set(master) == {"collector", "save"}
    for job in harness.manager.start.call_args.args[2]:
        assert {b["socket"] for b in job["manifest"]} == sockets
        assert {"text", "source-images", "source-audio", "source-masks", "source-video"} <= set(job["workflow"])
    assert prompt == original


@pytest.mark.parametrize("selected", ["master", "worker"])
def test_typed_load_balance_selects_master_or_one_serverless_slot(harness, selected):
    h = harness
    prompt = text_prompt()
    prompt["collector"]["inputs"]["load_balance"] = "true"
    h.q.select_least_busy_worker.side_effect = lambda candidates, **kw: next(
        w for w in candidates if (w["id"] == "master") == (selected == "master"))
    result = run(h, prompt, delegate_master=False)
    h.q.select_least_busy_worker.assert_awaited_once()
    master = h.q.queue_prompt_payload.call_args.args[0]
    if selected == "master":
        assert result[2] == 0
        h.manager.start.assert_not_awaited()
        assert master["collector"]["inputs"]["delegate_only"] is False
        assert "text" in master
    else:
        assert result[2] == 1
        assert len(h.manager.start.call_args.args[2]) == 1
        assert master["collector"]["inputs"]["delegate_only"] is True
        assert "text" not in master


@pytest.mark.parametrize("profile", [None, "legacy-image-v1", PROFILE])
def test_explicit_profile_compatibility_rejected_before_probe(harness, profile):
    h = harness
    prompt = legacy_prompt() if profile == PROFILE else text_prompt()
    if profile is None:
        h.pool.pop("media_profile")
    else:
        h.pool["media_profile"] = profile
    with pytest.raises(ValueError, match="requires|require"):
        run(h, prompt)
    assert_no_dispatch(h, no_probe=True)


@pytest.mark.parametrize("profile", [None, "auto", "comfy-media-v2", False])
def test_invalid_explicit_profiles_fail_closed(harness, profile):
    harness.pool["media_profile"] = profile
    with pytest.raises(ValueError, match="media_profile"):
        run(harness)
    assert_no_dispatch(harness, no_probe=True)


@pytest.mark.parametrize("direct_type", ["local", "remote", "cloud"])
@pytest.mark.parametrize("explicit", [False, True])
def test_direct_workers_never_silently_skipped_or_fallback(harness, direct_type, explicit):
    h = harness
    h.config["workers"].append({"id": "direct", "type": direct_type, "host": "offline.invalid",
                                "enabled": not explicit, "port": 8188})
    prompt = text_prompt()
    prompt["collector"]["inputs"]["load_balance"] = True
    kwargs = {"enabled_worker_ids": ["pool", "direct"]} if explicit else {}
    with pytest.raises(ValueError, match="direct workers"):
        run(h, prompt, delegate_master=False, **kwargs)
    assert_no_dispatch(h, no_probe=True)


def test_disabled_or_unselected_incompatible_workers_do_not_block(harness):
    h = harness
    h.config["workers"] += [
        {"id": "direct", "type": "local", "enabled": False},
        dict(h.pool, id="legacy", media_profile="legacy-image-v1", enabled=False),
    ]
    assert run(h)[2] == 2
    h.pool["enabled"] = False
    assert run(h, enabled_worker_ids=["pool"])[2] == 2


@pytest.mark.parametrize("other", ["DistributedCollector", "UltimateSDUpscaleDistributed", "DistributedImageListStart"])
@pytest.mark.parametrize("workers", [True, False])
def test_mixed_collectors_or_image_list_rejected_even_master_only(harness, other, workers):
    h = harness
    if not workers:
        h.config["workers"] = []
    prompt = text_prompt()
    prompt["incompatible"] = node(other)
    with pytest.raises(ValueError, match="cannot be mixed"):
        run(h, prompt, delegate_master=False)
    assert_no_dispatch(h, no_probe=True)


def test_mixed_pool_profiles_rejected_before_load_balancing(harness):
    h = harness
    h.config["workers"].append(dict(h.pool, id="legacy", media_profile="legacy-image-v1"))
    prompt = text_prompt()
    prompt["collector"]["inputs"]["load_balance"] = True
    with pytest.raises(ValueError, match="requires"):
        run(h, prompt, delegate_master=False)
    assert_no_dispatch(h, no_probe=True)


def test_typed_master_only_is_supported_without_serverless_work(harness):
    h = harness
    h.config["workers"] = []
    assert run(h, delegate_master=False)[2] == 0
    master = h.q.queue_prompt_payload.call_args.args[0]
    assert master["collector"]["inputs"]["enabled_worker_ids"] == "[]"
    assert master["collector"]["inputs"]["texts"] == ["text", 0]
    h.manager.start.assert_not_awaited()
    h.q.dispatch_worker_prompt.assert_not_awaited()


@pytest.mark.parametrize("offline", [False, True])
def test_typed_delegate_requires_available_worker_without_master_fallback(harness, offline):
    h = harness
    if offline:
        h.q.select_active_workers.side_effect = lambda *a, **kw: ([], False)
    else:
        h.config["workers"] = []
    with pytest.raises(RuntimeError, match="comfy-media-v1"):
        run(h)
    assert_no_dispatch(h)


def test_unknown_explicit_typed_worker_is_rejected(harness):
    with pytest.raises(ValueError, match="Unknown explicitly selected"):
        run(harness, enabled_worker_ids=["unknown"], delegate_master=False)
    assert_no_dispatch(harness, no_probe=True)


@pytest.mark.parametrize("bad", [" pool ", "pool white", "pool\n"])
def test_pool_media_identity_is_not_repaired(harness, bad):
    harness.pool["id"] = bad
    with pytest.raises(ValueError, match="identifier"):
        run(harness)
    assert_no_dispatch(harness, no_probe=True)


def test_integer_source_node_identity_is_not_normalized(harness):
    prompt = text_prompt()
    prompt[1] = prompt.pop("text")
    prompt["collector"]["inputs"]["texts"] = [1, 0]
    with pytest.raises(ValueError, match="identifier"):
        run(harness, prompt)
    assert_no_dispatch(harness, no_probe=True)


@pytest.mark.parametrize("broken", ["dangling-unused", "literal", "missing-file", "bad-loader-controls", "oversize"])
def test_compilation_packing_and_size_errors_precede_all_dispatch(harness, media_paths, broken):
    h = harness
    prompt = text_prompt()
    if broken == "dangling-unused":
        prompt["unused"] = node("PreviewImage", images=["missing", 0])
    elif broken == "literal":
        prompt["collector"]["inputs"]["texts"] = "must use a linked string node"
    elif broken in ("missing-file", "bad-loader-controls"):
        prompt["audio"] = node("LoadAudio", audio="missing.wav")
        prompt["collector"]["inputs"]["audio"] = ["audio", 0]
        if broken == "bad-loader-controls":
            prompt["audio"]["inputs"]["duration"] = 5
    else:
        prompt["text"]["inputs"]["value"] = "雪" * 1000
    original = copy.deepcopy(prompt)
    with patch.object(h.client, "MAX_REQUEST_BYTES", 100 if broken == "oversize" else h.client.MAX_REQUEST_BYTES):
        with pytest.raises((ValueError, OSError)):
            run(h, prompt)
    assert prompt == original
    assert_no_dispatch(h)


@pytest.mark.parametrize("corruption", ["manifest", "participant", "request-control", "binding-control", "profile", "input-ref"])
def test_whole_batch_identity_preflight_before_master_and_manager(harness, corruption):
    h = harness
    real_prepare = h.q._prepare_worker_payload
    async def corrupt_later(*args, **kwargs):
        worker, prepared = await real_prepare(*args, **kwargs)
        if worker["id"].endswith(":2"):
            binding = prepared["manifest"][0]
            if corruption == "manifest":
                binding["job_id"] = "wrong-job"
            elif corruption == "participant":
                prepared["worker"] = dict(worker, id="different-participant")
            elif corruption == "request-control":
                prepared["workflow"][binding["node_id"]]["inputs"]["request_id"] = "different-request"
            elif corruption == "binding-control":
                prepared["workflow"][binding["node_id"]]["inputs"]["binding_json"] = "{}"
            elif corruption == "profile":
                prepared["worker"] = dict(worker, media_profile="legacy-image-v1")
            else:
                prepared["media"]["inputs"] = [{"node_id": "missing", "input_name": "texts", "artifact_id": "unknown"}]
        return worker, prepared
    with patch.object(h.q, "_prepare_worker_payload", side_effect=corrupt_later):
        with pytest.raises(ValueError):
            run(h)
    h.manager.ensure_capacity.assert_not_called()
    assert_no_dispatch(h)


def test_batch_preflight_also_precedes_regular_direct_dispatch(harness):
    h = harness
    h.pool.pop("media_profile")
    h.config["workers"].append({"id": "direct", "type": "local", "enabled": True})
    real_prepare = h.q._prepare_worker_payload
    async def corrupt_later(*args, **kwargs):
        worker, prepared = await real_prepare(*args, **kwargs)
        if worker["id"] == "serverless:pool:2":
            prepared["worker"] = dict(worker, media_profile=PROFILE)
        return worker, prepared
    with patch.object(h.q, "_prepare_worker_payload", side_effect=corrupt_later):
        with pytest.raises(ValueError, match="mismatch"):
            run(h, legacy_prompt())
    assert_no_dispatch(h)


def test_capacity_preflight_precedes_collector_state_and_master(harness):
    h = harness
    h.manager.ensure_capacity.side_effect = ValueError("capacity exhausted")
    with pytest.raises(ValueError, match="capacity"):
        run(h)
    assert_no_dispatch(h)


def test_master_queue_failure_cleans_queues_and_never_starts_manager(harness):
    h = harness
    h.q.queue_prompt_payload.side_effect = ValueError("invalid master graph")
    with pytest.raises(ValueError, match="invalid master"):
        run(h)
    assert h.ps.distributed_pending_jobs == {}
    h.manager.start.assert_not_awaited()
    h.q.dispatch_worker_prompt.assert_not_awaited()


def test_manager_setup_failure_emits_one_terminal_per_collector_participant(harness):
    h = harness
    prompt = text_prompt()
    prompt["image"] = node("EmptyImage", width=2, height=2, batch_size=1, color=0)
    prompt["collector"]["inputs"]["images"] = ["image", 0]
    h.manager.start.side_effect = ValueError("journal failure")
    with pytest.raises(ValueError, match="journal failure"):
        run(h, prompt)
    h.manager.cancel.assert_awaited_once()
    queued = next(iter(h.ps.distributed_pending_jobs.values()))
    assert queued.qsize() == 2  # two participants, not four per-socket completions
    messages = [queued.get_nowait(), queued.get_nowait()]
    assert {m["worker_id"] for m in messages} == {"serverless:pool:1", "serverless:pool:2"}
    assert all(m["is_last"] is True and m["serverless_error"] for m in messages)


@pytest.mark.parametrize("profile", [None, "legacy-image-v1"])
def test_legacy_envelope_and_compiler_are_unchanged(harness, profile):
    h = harness
    if profile is None:
        h.pool.pop("media_profile")
    else:
        h.pool["media_profile"] = profile
    prompt = legacy_prompt()
    original = copy.deepcopy(prompt)
    with patch.object(h.compiler, "compile_media_prompt", side_effect=AssertionError("legacy must not use typed compiler")), \
         patch.object(h.packer, "prepare_media_inputs", side_effect=AssertionError("legacy must not use typed packer")):
        assert run(h, prompt)[2] == 2
    for job in h.manager.start.call_args.args[2]:
        assert "media" not in job
        assert any(n["class_type"] == "SaveImage" for n in job["workflow"].values())
        expected = {"input": {"workflow": job["workflow"], "images": job["images"]},
                    "policy": {"executionTimeout": 30000, "ttl": 30000}}
        assert h.client.serialize_request(job["workflow"], job["images"], 30) == json.dumps(
            expected, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()
    assert prompt == original


def test_legacy_no_worker_delegate_fallback_is_preserved(harness):
    h = harness
    h.config["workers"] = []
    assert run(h, legacy_prompt())[2] == 0
    assert h.q.queue_prompt_payload.call_args.args[0]["collector"]["inputs"]["delegate_only"] is False


def test_native_audio_compile_pack_serialized_input_cpu(harness, media_paths):
    """Exercise real PyAV + core PCM conversion, not a mocked packing result."""
    if os.environ.get("_MEDIA_QUEUE_CORE_TEST") != "1":
        core = os.environ.get("COMFYUI_SOURCE_ROOT")
        if not core:
            pytest.skip("Set COMFYUI_SOURCE_ROOT for CPU native audio queue integration")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("COMFYUI_SOURCE_ROOT", None)
        scratch = Path.home() / ".hermes/cache/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        env.update(_MEDIA_QUEUE_CORE_TEST="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                   PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch))
        bootstrap = """
import sys
sys.path.insert(0, sys.argv[1])
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch

def forbidden(*args, **kwargs):
    raise AssertionError('CPU audio integration may not initialize a GPU')
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.init = forbidden
torch.cuda.current_device = forbidden
torch.cuda._lazy_init = forbidden
import pytest
raise SystemExit(pytest.main([sys.argv[2], '-q', '-p', 'no:cacheprovider', '-k',
                             'test_native_audio_compile_pack_serialized_input_cpu']))
"""
        result = subprocess.run([sys.executable, "-c", bootstrap, core, str(Path(__file__).resolve())],
                                env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout
        return
    h = harness
    h.pool["jobs_per_run"] = 1
    path = media_paths / "tone.wav"
    pcm = torch.tensor([-100, 0, 100, 32767, -32768, 7], dtype=torch.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(pcm.numpy().tobytes())
    original_bytes = path.read_bytes()
    prompt = text_prompt()
    prompt["audio"] = node("LoadAudio", audio="tone.wav")
    prompt["collector"]["inputs"]["audio"] = ["audio", 0]
    original = copy.deepcopy(prompt)
    run(h, prompt)
    job = h.manager.start.call_args.args[2][0]
    wire = json.loads(h.client.serialize_request(job["workflow"], job["images"], 30, media=job["media"]))
    artifacts = wire["input"]["media"]["artifacts"]
    refs = wire["input"]["media"]["inputs"]
    assert len(artifacts) == len(refs) == 1
    binding = next(b for b in job["manifest"] if b["socket"] == "audio")
    assert refs == [{"node_id": binding["node_id"], "input_name": "audio", "artifact_id": artifacts[0]["id"]}]
    contract = importlib.import_module(PKG + ".serverless_bridge.contract")
    codec = importlib.import_module(PKG + ".serverless_bridge.codec")
    payload = asyncio.run(contract.resolve_artifact(artifacts[0]))
    value = codec.decode_payload("AUDIO", artifacts[0]["descriptor"], payload)
    assert value["sample_rate"] == 8000
    assert value["waveform"].shape == (1, 1, 6)
    assert torch.equal(value["waveform"], (pcm.to(torch.float32) / 32768).reshape(1, 1, -1))
    assert path.read_bytes() == original_bytes
    assert prompt == original
    assert h.validator.validate_prepared_job(job) is True
