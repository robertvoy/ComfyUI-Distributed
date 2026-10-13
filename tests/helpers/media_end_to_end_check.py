"""Real CPU Comfy V3 + HTTP proof; invoked only by the opt-in parent test.

The only harness pieces are deterministic source/assertion nodes and a bounded
queue worker. /prompt, /history and /view are Comfy's real handlers; compilation,
packing, staging, codecs, bridge execution, response decoding and Collector are
unmodified production code. No normal instance or Runpod client is contacted.
"""
import asyncio
import copy
import importlib
import io as bytes_io
import json
import os
from pathlib import Path
import socket
import sys
import types
import wave
from contextlib import asynccontextmanager
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[2]


def bootstrap():
    if len(sys.argv) != 3:
        raise SystemExit("Use test_media_end_to_end.py with explicit COMFYUI_SOURCE_ROOT")
    core, run_root = (Path(value).resolve() for value in sys.argv[1:])
    assert run_root.is_dir() and run_root != core and not run_root.is_relative_to(core)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert os.environ.get(name) == "", f"CPU isolation missing: {name}"
    assert not any(name in sys.modules for name in ("torch", "execution", "folder_paths"))
    sys.path.insert(0, str(core))
    import comfy.cli_args
    args = comfy.cli_args.args
    # Set CPU and all writable roots BEFORE any framework/tensor import.
    args.cpu = True
    args.disable_assets = True
    args.disable_all_custom_nodes = True
    args.enable_manager = False
    args.offline = True
    args.base_directory = str(run_root)
    args.front_end_root = str(run_root)
    args.database_url = "sqlite:///:memory:"
    for name in ("input", "output", "temp", "user"):
        (run_root / name).mkdir()
        setattr(args, name + "_directory", str(run_root / name))
    return run_root


async def check(base):
    import av
    import numpy as np
    import torch
    import execution
    import folder_paths
    import nodes
    import server
    from aiohttp import web
    from app.assets.manager import default_asset_manager
    from comfy_api.v0_0_2 import io
    from comfy_api.latest._input_impl.video_types import (
        VideoFromComponents, VideoFromFile, VideoFromList,
    )
    from comfy_api.latest._util import VideoComponents
    from comfy_extras import nodes_audio

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    assert execution.comfy.model_management.get_torch_device().type == "cpu"
    for name in ("input", "output", "temp", "user"):
        assert Path(getattr(folder_paths, "get_" + name + "_directory")()).is_relative_to(base)

    # Namespace scaffolding avoids the unrelated pack bootstrap (service/config
    # side effects). Every imported module below is the real source on disk.
    package_name = "_media_e2e_acceptance"
    for suffix in ("", ".api", ".api.orchestration", ".nodes"):
        package = types.ModuleType(package_name + suffix)
        package.__path__ = [str(ROOT.joinpath(*suffix.lstrip(".").split(".")))] if suffix else [str(ROOT)]
        sys.modules[package.__name__] = package
    def load(name):
        return importlib.import_module(package_name + "." + name)

    asset_manager = default_asset_manager()
    ps = server.PromptServer(asyncio.get_running_loop(), asset_manager)
    ps.distributed_pending_jobs = {}
    ps.distributed_jobs_lock = asyncio.Lock()
    compiler = load("api.orchestration.serverless_media_prompt")
    packer = load("api.orchestration.serverless_typed_inputs")
    decoder = load("api.orchestration.serverless_typed_results")
    serializer = load("api.orchestration.runpod_client").serialize_request
    handler = load("serverless_bridge.handler")
    contract = load("serverless_bridge.contract")
    video_codec = load("serverless_bridge.video_codec")
    bridge = load("serverless_bridge")
    collector = load("nodes.media_collector").DistributedMediaCollector
    extension = await bridge.comfy_entrypoint()
    bridge_classes = await extension.get_node_list()
    assert {cls.GET_SCHEMA().node_id for cls in bridge_classes} == {"MediaImportV1", "MediaExportV1"}
    sockets = list(contract.SOCKET_KINDS)
    kinds = list(contract.SOCKET_KINDS.values())
    assert kinds == ["IMAGE", "AUDIO", "MASK", "STRING", "VIDEO"]
    io_types = [io.Image, io.Audio, io.Mask, io.String, io.Video]
    fixtures, expected_sinks, seen_sinks, source_calls = {}, {}, {}, []

    class AcceptanceMediaSource(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="AcceptanceMediaSource", inputs=[io.String.Input("key")],
                             outputs=[kind.Output(is_output_list=True) for kind in io_types])

        @classmethod
        def execute(cls, key):
            source_calls.append(key)
            return io.NodeOutput(*(fixtures[key][socket] for socket in sockets))

    def tensor_equal(actual, expected):
        assert actual.device.type == "cpu" and actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        # Byte comparison preserves signed zero and catches dtype narrowing.
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    def exact(kind, actual, expected):
        if kind in ("IMAGE", "MASK"):
            tensor_equal(actual, expected)
        elif kind == "AUDIO":
            assert actual["sample_rate"] == expected["sample_rate"]
            tensor_equal(actual["waveform"], expected["waveform"])
        elif kind == "STRING":
            assert type(actual) is str and actual.encode("utf-8") == expected.encode("utf-8")
        else:
            assert video_codec.encode_video(actual) == expected
            if type(actual) is VideoFromFile:
                buffer = vars(actual)["_VideoFromFile__file"]
                assert type(buffer) is bytes_io.BytesIO and not buffer.closed
                position = buffer.tell()
                buffer.seek(0)
                with av.open(buffer) as container:
                    assert len(list(container.decode(video=0))) == 3
                buffer.seek(position)
            elif type(actual) is VideoFromList:
                for item in actual.videos:
                    exact("VIDEO", item, video_codec.encode_video(item))
            else:
                assert type(actual) is VideoFromComponents
                components = actual.get_components()
                assert type(components.frame_rate) is Fraction
                assert components.images.dtype == torch.float64
                assert components.alpha.dtype == torch.bfloat16
                assert components.audio["waveform"].dtype == torch.float64
                assert actual.get_bit_depth() == 10 and actual.get_color_space() == "HDR PQ"

    class AcceptanceMediaAssert(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="AcceptanceMediaAssert", inputs=[io.String.Input("tag"),
                *[kind.Input(socket, optional=True, **({"force_input": True} if socket == "texts" else {}))
                  for socket, kind in zip(sockets, io_types)]], outputs=[],
                is_input_list=True, is_output_node=True)

        @classmethod
        def execute(cls, tag, **media):
            assert len(tag) == 1 and tag[0] not in seen_sinks
            expected = expected_sinks[tag[0]]
            assert set(media) == set(expected)
            for socket in expected:
                assert len(media[socket]) == len(expected[socket]), (tag, socket)
                # Collector must preserve native objects and ordered item lists,
                # not merge batches, resample, convert types, or encode again.
                assert all(a is e for a, e in zip(media[socket], expected[socket])), (tag, socket)
            counts = {socket: len(media[socket]) for socket in expected}
            seen_sinks[tag[0]] = counts
            return io.NodeOutput(ui={"accepted": [counts]})

    for cls in [*bridge_classes, collector, AcceptanceMediaSource, AcceptanceMediaAssert]:
        assert issubclass(cls, io.ComfyNode)
        cls.GET_NODE_INFO_V1()
        nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls

    def node(class_type, **inputs):
        return {"class_type": class_type, "inputs": inputs}

    def tiny_video(variant):
        buffer = bytes_io.BytesIO()
        with av.open(buffer, "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=Fraction(24, 1))
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for level in (16 + variant, 64 + variant, 128 + variant):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), level, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return buffer.getvalue()

    def native_fixtures(variant):
        image = torch.arange(48, dtype=torch.float64).reshape(2, 2, 3, 4) / 997 + variant / 100
        image[0, 0, 0, 0] = -0.0
        mask = torch.tensor([[[-0.0, 0.125, 0.875]]], dtype=torch.bfloat16)
        audio = {"waveform": torch.arange(38, dtype=torch.float64).reshape(1, 2, 19) / 997,
                 "sample_rate": 44100 + variant}
        audio["waveform"][0, 0, 0] = -0.0
        component = VideoFromComponents(VideoComponents(
            images=image[..., :3], frame_rate=Fraction(30000, 1001), alpha=mask.reshape(1, 1, 3, 1).repeat(2, 2, 1, 1),
            audio=audio, metadata={"title": "雪", "nested": [True, None, -4, 0.125]}),
            bit_depth=10, color_space="HDR PQ")
        payload = tiny_video(variant)
        file = VideoFromFile(bytes_io.BytesIO(payload), start_time=-0.0, duration=0, crop=(0, 0, 8, 8))
        sequence = VideoFromList([VideoFromFile(bytes_io.BytesIO(payload)), file], complete_audio=audio)
        return {"images": [image, image[:1].clone()], "audio": [audio], "masks": [mask],
                "texts": [f"  café 🎥 {variant}\r\n雪  ", ""], "video": [component, sequence]}

    for key, variant in (("local", 0), ("worker-a", 1), ("worker-b", 2)):
        fixtures[key] = native_fixtures(variant)
    native_root = base / "master-input"
    native_root.mkdir()
    wav = native_root / "stereo.wav"
    pcm = np.array([-32768, -12001, -1, 0, 1, 8193, 32767, 17] * 2, dtype="<i2")
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(pcm.tobytes())
    waveform, sample_rate = nodes_audio.load(str(wav))
    clip = native_root / "native.mkv"
    clip.write_bytes(tiny_video(3))
    native_expected = copy.copy(fixtures["worker-a"])
    native_expected["audio"] = [{"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}]
    native_expected["video"] = [VideoFromFile(str(clip))]
    expectations = {}
    for key, values in (("worker-a", native_expected), ("worker-b", fixtures["worker-b"])):
        expectations[key] = {socket: [video_codec.encode_video(v, allowed_roots=[native_root]) if socket == "video" else v
                                      for v in values[socket]] for socket in sockets}

    # Native routes only, bound to a new ephemeral loopback listener. No fake
    # history, permissive validator, bridge/codec mock, or existing service.
    allowed_paths = {"/prompt", "/history/{prompt_id}", "/view", "/queue"}
    ps.app.add_routes([route for route in ps.routes if route.path in allowed_paths])
    completed = []
    stop = asyncio.Event()

    async def queue_worker():
        while not stop.is_set():
            item = await asyncio.to_thread(ps.prompt_queue.get, 0.1)
            if item is None:
                continue
            queued, item_id = item
            assert len(queued) in (5, 6), "Unsupported core PromptQueue tuple; adapt acceptance worker explicitly"
            executor = execution.PromptExecutor(ps, cache_args={"ram": 0, "ram_inactive": 0}, asset_manager=asset_manager)
            await asyncio.to_thread(executor.execute, queued[2], queued[1], queued[3], queued[4])
            status = execution.PromptQueue.ExecutionStatus(
                status_str="success" if executor.success else "error", completed=executor.success,
                messages=executor.status_messages)
            ps.prompt_queue.task_done(item_id, executor.history_result, status=status)
            completed.append((queued, executor.success, executor.status_messages))

    @asynccontextmanager
    async def serve():
        runner = web.AppRunner(ps.app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        ALLOWED_CONNECTIONS.add(("127.0.0.1", port))
        worker = asyncio.create_task(queue_worker())
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            stop.set()
            await asyncio.wait_for(worker, 10)
            await runner.cleanup()
            ALLOWED_CONNECTIONS.clear()

    grouped_results = {}
    async with serve() as url:
        for key in ("worker-a", "worker-b"):
            participant = "serverless:" + key
            request_id = "acceptance-" + key
            graph = {"source": node("AcceptanceMediaSource", key=key),
                     "collector": node("DistributedMediaCollector", load_balance=False,
                         multi_job_id="job-main", worker_id=participant,
                         **{socket: ["source", index] for index, socket in enumerate(sockets)}),
                     "text-collector": node("DistributedMediaCollector", load_balance=False,
                         multi_job_id="job-text", worker_id=participant, texts=["source", 3])}
            if key == "worker-a":
                graph.update({"wav": node("LoadAudio", audio=wav.name), "clip": node("LoadVideo", file=clip.name)})
                graph["collector"]["inputs"].update(audio=["wav", 0], video=["clip", 0])
            before = copy.deepcopy(graph)
            workflow, manifest = compiler.compile_media_prompt(graph, request_id)
            assert graph == before
            assert len(manifest) == 6 and {b["kind"] for b in manifest} == set(kinds)
            folder_paths.set_input_directory(str(native_root))
            workflow_before = copy.deepcopy(workflow)
            images, artifacts, refs = await packer.prepare_media_inputs(workflow)
            assert images == [] and workflow == workflow_before
            if key == "worker-a":
                assert {a["kind"] for a in artifacts} == {"AUDIO", "VIDEO"} and len(artifacts) == 2
                assert len(refs) == 2
                assert {ref["input_name"] for ref in refs} == {"audio", "video"}
                assert all(workflow[ref["node_id"]]["class_type"] == "MediaExportV1" for ref in refs)
                # Endpoint cannot fall back to the master's local files/loaders.
                wav.unlink()
                clip.unlink()
                print("NATIVE_INPUTS_AND_DIRECT_EXPORT_REFS_OK WAV + native file VIDEO")
            else:
                assert not artifacts and not refs
            folder_paths.set_input_directory(str(base / "input"))
            media = {"version": 1, "profile": contract.PROFILE, "request_id": request_id,
                     "artifacts": artifacts, "inputs": refs, "outputs": manifest}
            body = json.loads(serializer(workflow, images, 30, media=media))
            assert decoder.validate_prepared_job({"worker": {"id": participant, "media_profile": contract.PROFILE,
                       "timeout_seconds": 30}, "workflow": workflow, "images": images,
                       "media": media, "manifest": manifest}) is True
            before = copy.deepcopy(body["input"])
            result = await handler.execute_media_request(body["input"], comfy_url=url,
                        input_dir=base / "input", temp_dir=base / "temp", timeout_seconds=30)
            assert body["input"] == before
            assert completed and completed[-1][1], completed
            queued = completed[-1][0]
            assert set(queued[4]) == {b["node_id"] for b in manifest}
            submitted = queued[2]
            if key == "worker-a":
                assert "wav" not in submitted and "clip" not in submitted
                imports = [n for n in submitted.values() if n["class_type"] == "MediaImportV1"]
                assert len(imports) == 2
                assert {json.loads(n["inputs"]["artifact_json"])["kind"] for n in imports} == {"AUDIO", "VIDEO"}
            assert not list((base / "input").rglob("*.bin"))
            assert not list((base / "temp").rglob("*.bin"))
            if key == "worker-b":
                # Retire the originals as well as handler-owned staging/export
                # files before constructing the result's lazy native videos.
                for original in fixtures[key]["video"][1].videos:
                    vars(original)["_VideoFromFile__file"].close()
            # Exercise the wire format, not an in-memory handler-only object.
            result = json.loads(json.dumps(result, allow_nan=False))
            grouped = await decoder.collect_typed_results(result, media, manifest)
            assert set(grouped) == {"job-main", "job-text"}
            assert set(grouped["job-main"]) == set(sockets)
            assert grouped["job-text"]["texts"] == expectations[key]["texts"]
            for socket, kind in zip(sockets, kinds):
                values, expected = grouped["job-main"][socket], expectations[key][socket]
                assert len(values) == len(expected), (key, socket)
                for actual, wanted in zip(values, expected):
                    exact(kind, actual, wanted)
            if key == "worker-b":
                original = fixtures[key]["video"][0].get_components()
                decoded = grouped["job-main"]["video"][0].get_components()
                for field in ("images", "alpha"):
                    assert getattr(decoded, field).data_ptr() != getattr(original, field).data_ptr()
                assert decoded.audio["waveform"].data_ptr() != original.audio["waveform"].data_ptr()
                assert decoded.metadata == original.metadata and decoded.metadata is not original.metadata
            grouped_results[participant] = grouped
            # Late invalid artifact must reject the entire grouped response.
            corrupt = copy.deepcopy(result)
            corrupt["media"]["artifacts"][-1]["sha256"] = "0" * 64
            try:
                await decoder.collect_typed_results(corrupt, media, manifest)
            except ValueError:
                pass
            else:
                raise AssertionError("Result decoder accepted a corrupt final artifact")
            assert ps.distributed_pending_jobs == {}
        print("REAL_HTTP_BRIDGE_RESULTS_OK all five kinds; 6 bindings/request; native VIDEO remains readable after cleanup")
        print("ATOMIC_DECODE_REJECTION_OK corrupt final artifact rejected before delivery")

    participants = list(grouped_results)
    for delegate in (False, True):
        tag = "delegate" if delegate else "master"
        job = "collect-" + tag
        queue = asyncio.Queue()
        for participant in reversed(participants):
            await queue.put({"worker_id": participant, "media": grouped_results[participant]["job-main"], "is_last": True})
        ps.distributed_pending_jobs[job] = queue
        ordered = ([] if delegate else [fixtures["local"]]) + [grouped_results[p]["job-main"] for p in participants]
        expected_sinks[tag] = {socket: [item for values in ordered for item in values[socket]] for socket in sockets}
        graph = {"collector": node("DistributedMediaCollector", load_balance=False, multi_job_id=job,
                    enabled_worker_ids=json.dumps(participants), expected_media=json.dumps(sockets), delegate_only=delegate),
                 "assert": node("AcceptanceMediaAssert", tag=tag,
                    **{socket: ["collector", index] for index, socket in enumerate(sockets)})}
        if not delegate:
            graph["source"] = node("AcceptanceMediaSource", key="local")
            graph["collector"]["inputs"].update({socket: ["source", index] for index, socket in enumerate(sockets)})
        # The second group from the very same complete response must reach its
        # own Collector, rather than being folded into the first collector job.
        text_job, text_tag = job + "-text", tag + "-text"
        text_queue = asyncio.Queue()
        for participant in reversed(participants):
            await text_queue.put({"worker_id": participant,
                "media": grouped_results[participant]["job-text"], "is_last": True})
        ps.distributed_pending_jobs[text_job] = text_queue
        text_ordered = ([] if delegate else [fixtures["local"]]) + [grouped_results[p]["job-text"] for p in participants]
        expected_sinks[text_tag] = {"texts": [item for values in text_ordered for item in values["texts"]]}
        graph["text-collector"] = node("DistributedMediaCollector", load_balance=False, multi_job_id=text_job,
            enabled_worker_ids=json.dumps(participants), expected_media='["texts"]', delegate_only=delegate)
        graph["text-assert"] = node("AcceptanceMediaAssert", tag=text_tag, texts=["text-collector", 3])
        if not delegate:
            graph["text-collector"]["inputs"]["texts"] = ["source", 3]
        valid = await execution.validate_prompt(job, graph, None)
        assert valid[0] and not valid[3], valid
        executor = execution.PromptExecutor(ps, cache_args={"ram": 0, "ram_inactive": 0}, asset_manager=asset_manager)
        await asyncio.to_thread(executor.execute, graph, job, {}, valid[2])
        assert executor.success, executor.status_messages
        assert executor.history_result["outputs"]["assert"]["accepted"] == [seen_sinks[tag]]
        assert executor.history_result["outputs"]["text-assert"]["accepted"] == [seen_sinks[text_tag]]
        assert job not in ps.distributed_pending_jobs and text_job not in ps.distributed_pending_jobs
    assert source_calls == ["worker-a", "worker-b", "local"], source_calls
    assert set(seen_sinks) == {"master", "delegate", "master-text", "delegate-text"}
    assert not list((base / "output").rglob("*"))
    assert not torch.cuda.is_initialized()
    print("MASTER_AND_DELEGATE_ORDER_OK", json.dumps(seen_sinks, sort_keys=True))
    print("MEDIA_END_TO_END_CPU_OK real PromptExecutor/HTTP; no queue-orchestration or provider acceptance claimed")


ALLOWED_CONNECTIONS = set()


def network_guard(event, args):
    """Fail closed on provider/normal-instance traffic, including during imports."""
    if event == "socket.connect":
        sock, address = args
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            assert address in ALLOWED_CONNECTIONS, "Acceptance forbids non-owned network connections"
    elif event == "socket.bind":
        sock, address = args
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            assert address == ("127.0.0.1", 0), "Acceptance binds only ephemeral loopback"


if __name__ == "__main__":
    sys.addaudithook(network_guard)
    asyncio.run(asyncio.wait_for(check(bootstrap()), timeout=120))
