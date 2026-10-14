"""CPU-only image-list compilation and lifecycle tests; providers are mocked."""
import asyncio
import base64
import copy
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_image_list_serverless_tests"
for suffix in ("", ".api", ".api.orchestration", ".image_list", ".serverless_bridge"):
    module = types.ModuleType(PACKAGE + suffix)
    module.__path__ = [str(ROOT.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(ROOT)]
    sys.modules[module.__name__] = module
adapter = importlib.import_module(PACKAGE + ".image_list.serverless")
runtime = adapter.runtime
manager_module = importlib.import_module(PACKAGE + ".api.orchestration.serverless_runtime")
handler = importlib.import_module(PACKAGE + ".serverless_bridge.handler")
codec, contract = adapter.codec, adapter.contract


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def pool(wid="pool", jobs=1, enabled=True, **overrides):
    return dict(id=wid, type="serverless", endpoint_id="ep", timeout_seconds=30,
                jobs_per_run=jobs, enabled=enabled, media_profile=contract.PROFILE, **overrides)


def source():
    return {
        "upstream": node("LoadImage", image="never-read.png"),
        "ns/start": node(runtime.START_CLASS, images=["upstream", 0]),
        "model": node("CheckpointLoaderSimple", ckpt_name="fixed.safetensors"),
        "process": node("Process", image=["ns/start", 0], mask=["ns/start", 1],
                        index=["ns/start", 2], text=["ns/start", 4], integer=["ns/start", 5],
                        floating=["ns/start", 6], boolean=["ns/start", 7], model=["model", 0]),
        "ns/end": node(runtime.END_CLASS, context=["ns/start", 3], images=["process", 0],
                       masks=["process", 1], texts=["ns/start", 4], worker_result_token="drop-me"),
        "downstream": node("SaveImage", images=["ns/end", 0]),
    }


def region(prompt=None):
    return {"start_id": "ns/start", "end_id": "ns/end",
            "node_ids": set(prompt or source()) - {"upstream", "downstream"}}


def item(index=0, dtype=torch.float32):
    return {"index": index, "image": torch.arange(24, dtype=dtype).reshape(1, 2, 3, 4) / 24,
            "mask": torch.arange(6, dtype=dtype).reshape(1, 2, 3) / 6,
            "values": {"value_1": " \n雪", "value_2": 123, "value_3": 0.25, "value_4": False}}


async def prepare(prompt=None, value=None, worker=None):
    prompt = prompt or source()
    return await adapter.prepare_serverless_item(prompt, region(prompt), value or item(), worker or pool(),
                                                  request_id="request:1", job_id="queue:1")


async def decoded_inputs(job):
    return {artifact["id"]: codec.decode_payload(artifact["kind"], artifact["descriptor"],
                    await contract.resolve_artifact(artifact)) for artifact in job["media"]["artifacts"]}


def test_selection_is_exact_no_expansion_or_profile_fallback():
    a, b = pool("a", jobs=16), pool("b", enabled=False)
    direct = {"id": "direct", "type": "remote", "enabled": True}
    config = {"workers": [a, b, direct]}
    assert [w["id"] for w in adapter.resolve_serverless_workers(config, None)] == ["a"]
    resolved = adapter.resolve_serverless_workers(config, ["b", "direct"])
    assert [w["id"] for w in resolved] == ["b"]
    assert resolved[0]["pool_id"] == "b"
    resolved[0]["jobs_per_run"] = 2
    assert b["jobs_per_run"] == 1
    assert adapter.resolve_serverless_workers(config, []) == []
    legacy = pool("legacy", enabled=False)
    legacy.pop("media_profile")
    config["workers"].append(legacy)
    assert [w["id"] for w in adapter.resolve_serverless_workers(config, None)] == ["a"]
    with pytest.raises(ValueError, match="comfy-media-v1"):
        adapter.resolve_serverless_workers(config, ["legacy"])
    with pytest.raises(ValueError, match="Unknown"):
        adapter.resolve_serverless_workers(config, ["a:1"])
    with pytest.raises(ValueError, match="Invalid"):
        adapter.resolve_serverless_workers(config, [" a"])
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.resolve_serverless_workers({"workers": [a, a]}, ["a"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_deterministic_graph_exact_codecs_and_source_immutable(dtype):
    async def check():
        prompt, value = source(), item(dtype=dtype)
        original, saved_image = copy.deepcopy(prompt), value["image"].clone()
        one = await prepare(prompt, value)
        two = await prepare(dict(reversed(list(prompt.items()))), value)
        assert one == two
        assert prompt == original and torch.equal(value["image"], saved_image)
        assert "upstream" not in one["workflow"] and "downstream" not in one["workflow"]
        assert one["workflow"]["model"] == prompt["model"]
        assert not any("distributed" in n["class_type"].lower() for n in one["workflow"].values())
        proc = one["workflow"]["process"]["inputs"]
        for name, expected in [("index", 0), ("integer", 123), ("floating", 0.25), ("boolean", False), ("text", " \n雪")]:
            assert proc[name] == expected and type(proc[name]) is type(expected)
        output = next(b for b in one["manifest"] if b["socket"] == "texts")
        text_source = one["workflow"][output["node_id"]]["inputs"]["texts"][0]
        assert one["workflow"][text_source] == node("PrimitiveString", value=" \n雪")
        values = await decoded_inputs(one)
        for ref in one["media"]["inputs"]:
            result = values[ref["artifact_id"]]
            assert result.dtype == dtype
            assert torch.equal(result, value[ref["input_name"]])
        assert len(values) == 2
        handler._validate_request({key: one[key] for key in ("workflow", "images", "media")})
    asyncio.run(check())


def test_direct_passthrough_exports_and_tensor_generic_values_deduplicate():
    async def check():
        prompt, value = source(), item()
        prompt["process"]["inputs"]["reference"] = ["ns/start", 8]
        prompt["process"]["inputs"]["reference_mask"] = ["ns/start", 9]
        value["values"].update(value_5=value["image"], value_6=value["mask"])
        prompt["ns/end"]["inputs"]["images"] = ["ns/start", 0]
        job = await prepare(prompt, value)
        assert "process" in job["workflow"]  # mask still depends on processing
        assert len(job["media"]["artifacts"]) == 2
        refs = job["media"]["inputs"]
        assert len(refs) == 5
        image_binding = next(b for b in job["manifest"] if b["socket"] == "images")
        assert any(r["node_id"] == image_binding["node_id"] and r["input_name"] == "images" for r in refs)
        decoded = await decoded_inputs(job)
        for ref in refs:
            tensor = decoded[ref["artifact_id"]]
            assert torch.equal(tensor, value["mask"] if "mask" in ref["input_name"] else value["image"])
    asyncio.run(check())


@pytest.mark.parametrize("bad", [None, object(), torch.ones(2), torch.ones(2, 3)])
def test_connected_unsupported_generic_values_fail(bad):
    value = item()
    value["values"]["value_1"] = bad
    with pytest.raises((ValueError, TypeError)):
        asyncio.run(prepare(value=value))


@pytest.mark.parametrize("bad", [True, -1, "0", 0.0])
def test_start_slot_identity_not_coerced(bad):
    prompt = source()
    prompt["process"]["inputs"]["image"] = ["ns/start", bad]
    with pytest.raises(ValueError, match="link"):
        asyncio.run(prepare(prompt))


@pytest.mark.parametrize("bad", ["../bad", "ns//bad", "ns/./bad", "bad ", 42])
def test_node_identity_not_normalized_even_on_pruned_branch(bad):
    prompt = source()
    prompt[bad] = node("PrimitiveString", value="unused")
    with pytest.raises(ValueError):
        asyncio.run(prepare(prompt))


def test_placeholder_namespace_collision_and_empty_text():
    async def check():
        prompt, value = source(), item()
        prompt["image-list:input:0"] = node("PrimitiveInt", value=5)
        prompt["process"]["inputs"]["fixed"] = ["image-list:input:0", 0]
        value["values"]["value_1"] = ""
        job = await prepare(prompt, value)
        assert job["workflow"]["image-list:input:0"] == prompt["image-list:input:0"]
        assert job["workflow"]["process"]["inputs"]["image"] == ["image-list:input:0:1", 0]
        assert any(n == node("PrimitiveString", value="") for n in job["workflow"].values())
    asyncio.run(check())


def test_no_connected_none_and_payload_budget_before_submission():
    value = item()
    value["mask"] = None
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(prepare(value=value))
    client = importlib.import_module(PACKAGE + ".api.orchestration.runpod_client")
    with patch.object(client, "MAX_REQUEST_BYTES", 10):
        with pytest.raises(ValueError, match="Serialized"):
            asyncio.run(prepare())


def test_fixed_file_inputs_are_prepared_on_private_copy_and_merge(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "temp").mkdir()
    payload = b"bounded original file bytes"
    (input_dir / "fixed.png").write_bytes(payload)
    folders = types.SimpleNamespace(get_input_directory=lambda: str(input_dir),
        get_output_directory=lambda: str(tmp_path / "output"), get_temp_directory=lambda: str(tmp_path / "temp"),
        get_annotated_filepath=lambda name: str(input_dir / name))
    monkeypatch.setitem(sys.modules, "folder_paths", folders)
    prompt = source()
    prompt["fixed"] = node("LoadImage", image="fixed.png")
    prompt["process"]["inputs"]["fixed"] = ["fixed", 0]
    original = copy.deepcopy(prompt)
    job = asyncio.run(prepare(prompt))
    assert prompt == original
    assert len(job["images"]) == 1
    assert base64.b64decode(job["images"][0]["image"]) == payload
    assert job["workflow"]["fixed"]["inputs"]["image"] == job["images"][0]["name"]
    assert len(job["media"]["artifacts"]) == 2


def test_merge_rejects_conflicting_artifact_ids_and_combined_caps():
    async def check():
        job = await prepare()
        artifact = job["media"]["artifacts"][0]
        assert adapter._merge_artifacts([artifact], [copy.deepcopy(artifact)]) == [artifact]
        bad = copy.deepcopy(artifact)
        bad["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="identity"):
            adapter._merge_artifacts([artifact], [bad])
        with patch.object(contract, "MAX_TOTAL_BYTES", 1):
            with pytest.raises(ValueError, match="limits"):
                adapter._merge_artifacts([artifact])
    asyncio.run(check())


def test_handler_injects_exact_native_values_and_prunes_placeholders():
    async def check():
        value = item(dtype=torch.float64)
        prompt = source()
        prompt["ns/end"]["inputs"].update(images=["ns/start", 0], masks=["ns/start", 1])
        job = await prepare(prompt, value)
        workflow, media, legacy, targets = handler._validate_request(
            {key: job[key] for key in ("workflow", "images", "media")})
        buffers = {}
        def write(payload):
            name = f"owned-{len(buffers)}.bin"
            buffers[name] = payload
            return name
        class Submitted(Exception):
            pass
        async def submit(native_graph):
            assert all(n["class_type"] in {"MediaExportV1", "MediaImportV1", "PrimitiveString"}
                       for n in native_graph.values())
            for binding in job["manifest"]:
                link = native_graph[binding["node_id"]]["inputs"][binding["socket"]]
                native = native_graph[link[0]]
                if binding["kind"] == "STRING":
                    assert native == node("PrimitiveString", value=value["values"]["value_1"])
                    continue
                assert native["class_type"] == "MediaImportV1"
                metadata = json.loads(native["inputs"]["artifact_json"])
                decoded = codec.decode_payload(metadata["kind"], metadata["descriptor"],
                                                buffers[native["inputs"]["filename"]])
                assert link[1] == {"IMAGE": 0, "MASK": 2}[binding["kind"]]
                assert torch.equal(decoded, value["image" if binding["kind"] == "IMAGE" else "mask"])
            # Stop at the mocked HTTP boundary; no endpoint/process/provider exists.
            raise Submitted()
        execution = types.SimpleNamespace(media=media, targets=targets, remaining=lambda: 30,
                                           stage=types.SimpleNamespace(write=write), submit_and_wait=submit)
        with pytest.raises(Submitted):
            await handler._run(execution, workflow, legacy, None, None)
    asyncio.run(check())


def test_fixed_typed_inputs_merge_with_start_refs_without_losing_consumers():
    async def check():
        prompt = source()
        prompt["process"]["inputs"]["fixed_audio"] = 0
        audio = {"waveform": torch.ones(1, 2, 3), "sample_rate": 48000}
        descriptor, payload = codec.encode_payload("AUDIO", audio)
        artifact = contract.artifact_from_payload("fixed-audio", "AUDIO", descriptor, payload)
        fixed = AsyncMock(return_value=([], [artifact], [
            {"node_id": "process", "input_name": "fixed_audio", "artifact_id": "fixed-audio"}]))
        with patch.object(adapter, "prepare_media_inputs", fixed):
            job = await prepare(prompt)
        assert len(job["media"]["artifacts"]) == 3
        assert len(job["media"]["inputs"]) == 3
        assert any(ref["input_name"] == "fixed_audio" for ref in job["media"]["inputs"])
        decoded = await decoded_inputs(job)
        assert decoded["fixed-audio"]["sample_rate"] == 48000
        assert torch.equal(decoded["fixed-audio"]["waveform"], audio["waveform"])
    asyncio.run(check())


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = types.SimpleNamespace(distributed_pending_jobs={}, distributed_jobs_lock=asyncio.Lock())
        self.manager = manager_module.ServerlessManager(self.server,
            types.SimpleNamespace(get=lambda _wid: "fake-key"), Path(self.temp.name) / "jobs.json", AsyncMock())
        self.manager.poll_seconds = 0
        self.state = runtime.ImageListJob("context", "ns/start", "ns/end", source(),
            [item(i) for i in range(40)], ["pool"], "", prompt_id="master-prompt")
        self.gates, self.jobs, self.mutations = {}, {}, {}
        self.started = asyncio.Event()
        async def submit(workflow, images, timeout, *, media):
            jid = "provider-" + str(len(self.jobs))
            self.jobs[jid] = dict(workflow=workflow, images=images, media=media)
            self.gates[jid] = asyncio.Event()
            self.started.set()
            return jid
        async def status(jid):
            await self.gates[jid].wait()
            output = self.output(self.jobs[jid])
            if jid in self.mutations:
                self.mutations[jid](output)
            return {"status": "COMPLETED", "output": output}
        self.api = types.SimpleNamespace(submit=AsyncMock(side_effect=submit), status=AsyncMock(side_effect=status),
                                         cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        self.patches = [patch.object(manager_module, "RunpodClient", return_value=self.api),
                        patch.object(adapter, "get_manager", return_value=self.manager)]
        for p in self.patches:
            p.start()
        self.event = asyncio.Event()
        self.controllers = []
        self.controller = await self.controller_for()

    async def controller_for(self, workers=None, state=None):
        ctl = await adapter.create_serverless_slots(state or self.state, region(),
                    {"workers": workers or [pool()]}, self.event)
        self.controllers.append(ctl)
        return ctl

    async def asyncTearDown(self):
        for ctl in self.controllers:
            await ctl.close()
        await self.manager.cancel()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def output(self, job):
        media = job["media"]
        output = dict(version=1, profile=contract.PROFILE, request_id=media["request_id"],
                      bindings=[], artifacts=[], items=[])
        index = int(media["request_id"].rsplit(":", 1)[1])
        data = item(index)
        values = {"images": data["image"], "masks": data["mask"], "texts": str(index)}
        values.update(getattr(self, "override_values", {}))
        for n, binding in enumerate(media["outputs"]):
            descriptor, payload = codec.encode_payload(binding["kind"], values[binding["socket"]])
            aid = f"result:{n}"
            output["bindings"].append(dict(binding, count=1))
            output["artifacts"].append(contract.artifact_from_payload(aid, binding["kind"], descriptor, payload))
            output["items"].insert(0, dict(binding_id=binding["binding_id"], item_index=0, artifact_id=aid))
        return {"media": output}

    async def dispatch(self, index=0, ctl=None, slot=None):
        ctl = ctl or self.controller
        slot = slot or ctl.slots[index]
        handle = await ctl.dispatch(slot, index)
        return slot, handle

    async def finish(self, handle):
        # Submission has run by the time dispatch's shield completes, but yield
        # explicitly rather than assuming provider status has already been entered.
        while len(self.jobs) <= handle["index"]:
            await asyncio.sleep(0)
        jid = next(j for j, job in self.jobs.items()
                   if job["media"]["request_id"] == handle["execution_id"])
        self.gates[jid].set()
        await handle["task"]

    async def test_no_pool_does_not_initialize_manager_or_credentials(self):
        state = copy.copy(self.state)
        state.worker_ids = []
        with patch.object(adapter, "get_manager", side_effect=AssertionError("must stay lazy")):
            ctl = await adapter.create_serverless_slots(state, region(), {"workers": [pool()]}, self.event)
            assert ctl.slots == []
            await ctl.close()

    async def test_slots_are_bounded_not_jobs_per_run_and_cannot_collide(self):
        assert len(self.controller.slots) == manager_module.PER_WORKER_JOB_LIMIT == 16
        ctl = await self.controller_for([pool(jobs=16)])
        assert ctl.slots == self.controller.slots
        assert all(isinstance(slot, tuple) and len(slot) == 3 for slot in ctl.slots)
        assert len(set(ctl.slots)) == 16 and "pool" not in ctl.slots
        many = [pool(f"p{i}") for i in range(8)]
        state = copy.copy(self.state)
        state.worker_ids = [w["id"] for w in many]
        ctl = await self.controller_for(many, state)
        assert len(ctl.slots) == 64
        assert all(sum(s[1] == w["id"] for s in ctl.slots) <= 16 for w in many)

    async def test_reverse_completion_order_and_slot_reuse(self):
        first, second = await self.dispatch(0), await self.dispatch(1)
        inflight = dict([first, second])
        assert self.controller.poll(inflight) is None
        await self.finish(second[1])
        await asyncio.sleep(0)
        assert self.event.is_set()
        slot, index, result = self.controller.poll(inflight)
        assert (slot, index) == (second[0], 1) and result["text"] == "1"
        assert torch.equal(result["image"], item(1)["image"])
        assert second[1]["job_id"] not in self.server.distributed_pending_jobs
        del inflight[slot]
        third = await self.dispatch(2, slot=slot)
        assert third[1]["execution_id"] != second[1]["execution_id"]
        await self.finish(first[1])
        assert self.controller.poll(inflight)[1] == 0
        await self.finish(third[1])
        assert self.controller.poll(dict([third]))[1] == 2

    async def test_many_items_submit_once_each_through_bounded_reusable_slots(self):
        ctl = await self.controller_for([pool(jobs=16)])
        first_wave = asyncio.Event()
        active = 0
        peak = 0
        original_submit = self.api.submit.side_effect
        async def submit(*args, **kwargs):
            jid = await original_submit(*args, **kwargs)
            if len(self.jobs) == 16:
                first_wave.set()
            return jid
        async def status(jid):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await first_wave.wait()
                await asyncio.sleep(0.001)
                return {"status": "COMPLETED", "output": self.output(self.jobs[jid])}
            finally:
                active -= 1
        self.api.submit.side_effect = submit
        self.api.status.side_effect = status
        async def next_result(inflight):
            while True:
                self.event.clear()
                completed = ctl.poll(inflight)
                if completed is not None:
                    return completed
                await self.event.wait()
        async def cleanup(*_):
            pass
        results = await asyncio.wait_for(runtime.dynamic_schedule(
            len(self.state.items), ctl.slots, ctl.dispatch, next_result, cleanup), 10)
        assert self.api.submit.await_count == len(self.state.items)
        assert [row["text"] for row in results] == [str(i) for i in range(len(self.state.items))]
        assert peak == 16
        assert not self.server.distributed_pending_jobs
        assert all(task.done() for task in self.manager.tasks.values())

    async def test_waits_for_task_drain_not_provider_terminal_or_queue_message(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.manager._notify_capacity
        async def delayed():
            entered.set()
            await release.wait()
            await original()
        with patch.object(self.manager, "_notify_capacity", delayed):
            slot, handle = await self.dispatch()
            self.gates["provider-0"].set()
            await entered.wait()
            assert self.server.distributed_pending_jobs[handle["job_id"]].qsize() == 1
            assert self.controller.poll({slot: handle}) is None
            with pytest.raises(RuntimeError, match="occupied"):
                await self.controller.dispatch(slot, 1)
            release.set()
            await handle["task"]
        assert self.controller.poll({slot: handle})[1] == 0

    async def test_unknown_reservations_survive_and_prevent_provider_admission(self):
        for i in range(16):
            self.manager.ledger.update(f"unknown-{i}", worker_id="pool", status="UNKNOWN", job_id=None)
        slot, handle = await self.dispatch()
        await asyncio.sleep(0)
        self.api.submit.assert_not_awaited()
        assert self.controller.poll({slot: handle}) is None
        await self.controller.close()
        assert sum(r["status"] == "UNKNOWN" for r in self.manager.ledger.all()) == 16
        assert not self.server.distributed_pending_jobs
        assert self.manager.pool_is_active("pool")

    async def test_local_pending_cap_checked_before_journal(self):
        waiting = [asyncio.create_task(asyncio.Event().wait()) for _ in range(64)]
        self.manager.tasks.update({f"unrelated-{i}": t for i, t in enumerate(waiting)})
        self.manager.runs.update({f"unrelated-{i}": {"execution_id": "unrelated"} for i in range(64)})
        try:
            with pytest.raises(ValueError, match="limit 64"):
                await self.dispatch()
            assert not self.manager.ledger.all()
            self.api.submit.assert_not_awaited()
            assert not self.server.distributed_pending_jobs
        finally:
            for i, task in enumerate(waiting):
                self.manager.tasks.pop(f"unrelated-{i}")
                self.manager.runs.pop(f"unrelated-{i}")
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)

    async def test_missing_corrupt_or_mismatched_results_are_atomic_errors(self):
        mutations = [lambda o: o["media"]["items"].pop(),
                     lambda o: o["media"]["artifacts"][-1].update(sha256="0" * 64),
                     lambda o: o["media"].update(request_id="wrong-request"),
                     lambda o: o["media"]["bindings"][-1].update(participant_id="wrong-worker")]
        for index, mutation in enumerate(mutations):
            slot, handle = await self.dispatch(index)
            self.mutations[f"provider-{index}"] = mutation
            await self.finish(handle)
            queue = self.server.distributed_pending_jobs[handle["job_id"]]
            assert queue.qsize() == 1
            with pytest.raises(RuntimeError, match="failed"):
                self.controller.poll({slot: handle})

    async def test_terminal_queue_identity_completeness_and_multiplicity(self):
        for index, kind in enumerate(("missing", "duplicate", "worker", "mask", "text", "multiple", "empty")):
            slot, handle = await self.dispatch(index)
            await self.finish(handle)
            queue = self.server.distributed_pending_jobs[handle["job_id"]]
            message = queue.get_nowait()
            if kind == "duplicate":
                queue.put_nowait(message)
            if kind == "worker":
                message["worker_id"] = "unrelated"
            if kind in ("mask", "text"):
                message["media"].pop("masks" if kind == "mask" else "texts")
            if kind == "multiple":
                message["media"]["images"].append(item()["image"])
            if kind == "empty":
                message["media"]["texts"] = [None]
            if kind != "missing":
                queue.put_nowait(message)
            with pytest.raises(RuntimeError):
                self.controller.poll({slot: handle})

    async def test_wrong_frame_count_channels_and_unpaired_mask_fail(self):
        for index, kind in enumerate(("batch", "channels", "grid")):
            slot, handle = await self.dispatch(index)
            await self.finish(handle)
            queue = self.server.distributed_pending_jobs[handle["job_id"]]
            message = queue.get_nowait()
            if kind == "batch":
                message["media"]["images"] = [torch.ones(2, 2, 3, 4)]
            elif kind == "channels":
                message["media"]["images"] = [torch.ones(1, 2, 3, 1)]
            else:
                message["media"]["masks"] = [torch.ones(1, 5, 5)]
            queue.put_nowait(message)
            with pytest.raises(ValueError):
                self.controller.poll({slot: handle})

    async def test_close_cancels_only_owned_execution_and_exact_queues(self):
        state = copy.copy(self.state)
        state.context_id = "unrelated"
        unrelated = await self.controller_for(state=state)
        other_slot, other = await self.dispatch(0, unrelated)
        slot, handle = await self.dispatch(0)
        queue = self.server.distributed_pending_jobs[handle["job_id"]]
        replacement = asyncio.Queue()
        self.server.distributed_pending_jobs[handle["job_id"]] = replacement
        await self.controller.close()
        assert other["task"].done() is False
        assert self.server.distributed_pending_jobs[handle["job_id"]] is replacement
        assert other["job_id"] in self.server.distributed_pending_jobs
        assert handle["task"].done()
        assert len(self.api.cancel.await_args_list) == 1
        assert queue.empty()
        self.server.distributed_pending_jobs.pop(handle["job_id"])
        # A late manager callback skips a retired destination instead of recreating it.
        run = next(r for r in self.manager.runs.values() if r["execution_id"] == handle["execution_id"])
        await self.manager._deliver(run, error="late")
        assert handle["job_id"] not in self.server.distributed_pending_jobs

    async def test_cancel_dispatch_after_start_loses_handle_but_not_ownership(self):
        entered = asyncio.Event()
        original = self.manager.start
        async def delayed(*args):
            ids = await original(*args)
            entered.set()
            await asyncio.Event().wait()
            return ids
        with patch.object(self.manager, "start", delayed):
            task = asyncio.create_task(self.controller.dispatch(self.controller.slots[0], 0))
            await entered.wait()
            await self.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await self.controller.close()
        assert all(t.done() for t in self.manager.tasks.values())
        assert not self.server.distributed_pending_jobs
        self.api.cancel.assert_awaited_once_with("provider-0")

    async def test_close_during_preparation_prevents_late_start(self):
        entered = asyncio.Event()
        prepared = await prepare()
        async def resistant(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return prepared
        with patch.object(adapter, "prepare_serverless_item", resistant):
            task = asyncio.create_task(self.controller.dispatch(self.controller.slots[0], 0))
            await entered.wait()
            await self.controller.close()
            with pytest.raises(RuntimeError, match="closed"):
                await task
        assert not self.manager.ledger.all()
        assert not self.server.distributed_pending_jobs
        self.api.submit.assert_not_awaited()

    async def test_cancel_during_provider_submission_preserves_unknown(self):
        entered = asyncio.Event()
        async def ambiguous(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        self.api.submit.side_effect = ambiguous
        slot, handle = await self.dispatch()
        await entered.wait()
        await self.controller.close()
        assert next(iter(self.manager.ledger.all()))["status"] == "UNKNOWN"
        assert self.manager.pool_is_active("pool")
        self.api.cancel.assert_not_awaited()
        assert not self.server.distributed_pending_jobs

    async def test_cancelled_wait_closes_owned_work(self):
        entered = asyncio.Event()
        async def job():
            try:
                await self.dispatch()
                entered.set()
                await asyncio.Event().wait()
            finally:
                await self.controller.close()
        task = asyncio.create_task(job())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(t.done() for t in self.manager.tasks.values())
        assert not self.server.distributed_pending_jobs

    async def test_repeated_cancellation_does_not_abandon_close(self):
        slot, handle = await self.dispatch()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.manager.cancel
        async def delayed(**kwargs):
            entered.set()
            await release.wait()
            return await original(**kwargs)
        with patch.object(self.manager, "cancel", delayed):
            closing = asyncio.create_task(self.controller.close())
            await entered.wait()
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
            assert not self.server.distributed_pending_jobs
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
        assert handle["task"].done()
        assert self.controller._close_task.done()

    async def test_serialization_failure_never_journals_or_submits(self):
        with patch.object(adapter, "serialize_request", side_effect=ValueError("budget")):
            with pytest.raises(ValueError, match="budget"):
                await self.dispatch()
        assert not self.manager.ledger.all()
        assert not self.server.distributed_pending_jobs
        self.api.submit.assert_not_awaited()

    async def test_sparse_overrides_overlay_master_values_without_forwarding(self):
        from test_image_list_serverless_values import assert_exact
        self.override_values = {"value_2": 2**63 + 333, "value_10": False}
        self.state.original_prompt["ns/end"]["inputs"].update(
            value_2=["process", 2], value_10=["process", 3])
        slot, handle = await self.dispatch()
        await self.finish(handle)
        result = self.controller.poll({slot: handle})[2]
        assert set(result["values"]) == set(self.override_values)
        for name, expected in self.override_values.items():
            assert_exact(result["values"][name], expected)
        state = copy.copy(self.state)
        state.items = [self.state.items[0]]
        columns = runtime._collect_output_lists([result], state=state, include_values=True)
        assert columns[3][0] == {**state.items[0]["values"], **self.override_values}
        assert self.state.items[0]["values"]["value_2"] == 123

    async def test_override_missing_extra_wrong_nonfinite_multiple_results_fail(self):
        self.override_values = {"value_2": 42}
        self.state.original_prompt["ns/end"]["inputs"]["value_2"] = ["process", 2]
        for index, bad in enumerate(("missing", "extra", "wrong", "nonfinite", "multiple", "none")):
            slot, handle = await self.dispatch(index)
            await self.finish(handle)
            queue = self.server.distributed_pending_jobs[handle["job_id"]]
            message = queue.get_nowait()
            media = message["media"]
            if bad == "missing":
                media.pop("value_2")
            elif bad == "extra":
                media["value_3"] = [1]
            else:
                media["value_2"] = {"wrong": [{}], "nonfinite": [float("nan")],
                                    "multiple": [1, 2], "none": [None]}[bad]
            queue.put_nowait(message)
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                self.controller.poll({slot: handle})

    async def test_no_override_results_preserve_original_return_shape(self):
        slot, handle = await self.dispatch()
        await self.finish(handle)
        result = self.controller.poll({slot: handle})[2]
        assert set(result) == {"image", "mask", "text"}
        assert not any(binding["kind"] == "VALUE" for binding in self.jobs["provider-0"]["media"]["outputs"])

    async def test_duplicate_index_and_mismatched_handle_fail(self):
        slot, handle = await self.dispatch()
        with pytest.raises(ValueError, match="duplicate"):
            await self.controller.dispatch(self.controller.slots[1], 0)
        with pytest.raises(RuntimeError, match="Mismatched"):
            self.controller.poll({slot: dict(handle)})
        await self.finish(handle)
        self.controller.poll({slot: handle})
        with pytest.raises(RuntimeError, match="duplicate"):
            self.controller.poll({slot: handle})
