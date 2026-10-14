"""Exact Collector transport through real serializers, HTTP, route and queue.

Only ComfyUI's process-wide server/model-management and unrelated orchestration
are replaced. Tensor/audio codecs, multipart bodies and aiohttp are real; no
running ComfyUI service, GPU, provider or external network access is required.
"""
import asyncio
from contextlib import asynccontextmanager
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import AsyncMock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = "dist_collector_exact_testpkg"
JOB_ID = "exact-job"
WORKER_ID = "worker-1"


@pytest.fixture
def transport(monkeypatch):
    # Other legacy test loaders may leave aiohttp stubs behind. Replace only
    # those entries, never unload/reload real extension-backed dependencies.
    for name, loaded_module in list(sys.modules.items()):
        if (name == "aiohttp" or name.startswith("aiohttp.")) and not getattr(loaded_module, "__file__", None):
            monkeypatch.delitem(sys.modules, name)
    aiohttp = importlib.import_module("aiohttp")
    web = importlib.import_module("aiohttp.web")

    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    def load(relative):
        name = f"{PKG}.{relative}"
        spec = importlib.util.spec_from_file_location(name, ROOT.joinpath(*relative.split(".")).with_suffix(".py"))
        assert spec is not None and spec.loader is not None
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, result)
        return result

    module(PKG, __path__=[str(ROOT)])
    for package in ("nodes", "api", "utils", "serverless_bridge"):
        module(f"{PKG}.{package}", __path__=[str(ROOT / package)])
    server = types.SimpleNamespace(routes=web.RouteTableDef(), distributed_pending_jobs={})
    module("server", PromptServer=types.SimpleNamespace(instance=server))
    mm = module("comfy.model_management", throw_exception_if_processing_interrupted=lambda: None)
    progress = module("comfy.utils", ProgressBar=lambda _total: types.SimpleNamespace(update=lambda _n: None))
    module("comfy", __path__=[], model_management=mm, utils=progress)
    module(f"{PKG}.utils.logging", debug_log=lambda *_a, **_kw: None, log=lambda *_a, **_kw: None)
    module(f"{PKG}.utils.config", get_worker_timeout_seconds=lambda: 0.1,
           load_config=lambda: {"workers": []}, is_master_delegate_only=lambda: False)

    async def handle_api_error(_request, error, status=500):
        return web.json_response({"error": str(error)}, status=status)

    module(f"{PKG}.utils.network", handle_api_error=handle_api_error,
           get_client_session=AsyncMock(), build_worker_url=lambda _worker: "http://unused.invalid",
           probe_worker=AsyncMock(return_value=None))
    module(f"{PKG}.utils.async_helpers", run_async_in_server_loop=asyncio.run)
    module(f"{PKG}.api.queue_orchestration", ensure_distributed_state=lambda: None,
           orchestrate_distributed_execution=AsyncMock())
    module(f"{PKG}.api.queue_request", parse_queue_request_payload=lambda value: value)
    for name in ("utils.constants", "utils.image", "utils.audio_payload", "utils.transfer_codec"):
        load(name)
    bridge_codec = load("serverless_bridge.codec")
    load("serverless_bridge.image_transfer")
    image_codec = load("serverless_bridge.image_codec")
    collector = load("nodes.collector")
    routes = load("api.job_routes")
    return types.SimpleNamespace(aiohttp=aiohttp, web=web, server=server,
                                 collector=collector, routes=routes, bridge_codec=bridge_codec,
                                 image_codec=image_codec)


@asynccontextmanager
async def endpoint(transport, monkeypatch, *, request_limit=2 * 1024 * 1024,
                   chunk_target=1024 * 1024, client_max_size=None, handler=None):
    """A disposable loopback endpoint; record actual HTTP lengths, not estimates."""
    queue = asyncio.Queue()
    transport.server.distributed_jobs_lock = asyncio.Lock()
    transport.server.distributed_pending_jobs = {JOB_ID: queue}
    requests = []
    web = transport.web

    @web.middleware
    async def record_request(request, handler):
        record = {"bytes": request.content_length, "content_type": request.content_type}
        requests.append(record)
        response = await handler(request)
        record["status"] = response.status
        if response.status < 400 and request.content_type.startswith("multipart/"):
            form = await request.post()
            record["envelope"] = json.loads(form["envelope"])
        return response

    app = web.Application(client_max_size=client_max_size or request_limit, middlewares=[record_request])
    app.router.add_post("/distributed/job_complete", handler or transport.routes.job_complete_endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        async with transport.aiohttp.ClientSession() as session:
            monkeypatch.setattr(transport.collector, "get_client_session", AsyncMock(return_value=session))
            monkeypatch.setattr(transport.collector, "TRANSFER_MAX_REQUEST_BYTES", request_limit)
            monkeypatch.setattr(transport.collector, "TRANSFER_MAX_CHUNK_BYTES", chunk_target)
            yield types.SimpleNamespace(
                session=session, url=f"http://127.0.0.1:{port}", requests=requests, queue=queue,
                node=transport.collector.DistributedCollectorNode(),
            )
    finally:
        await runner.cleanup()


def frames(dtype=torch.float32, channels=3, count=5, height=7, width=9):
    # Non-contiguous, out-of-range and non-8-bit-representable values exercise
    # detach/contiguous without permitting clipping, alpha loss or quantization.
    values = torch.arange(count * height * width * channels, dtype=torch.float64)
    values = (values.remainder(41) - 11) / 17
    images = values.reshape(count, width, height, channels).transpose(1, 2).to(dtype)
    images[0, 0, 0, 0] = -0.0
    return images.requires_grad_(True)


def assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    # Compare raw bits as well as values, including BF16 and signed zero.
    assert torch.equal(actual, expected)
    assert torch.equal(actual.view(torch.uint8), expected.detach().contiguous().view(torch.uint8))


def queued_items(queue):
    return [queue.get_nowait() for _ in range(queue.qsize())]


async def send(ctx, images, *, audio=None, fmt="exact"):
    await ctx.node.send_batch_to_master(images, audio, JOB_ID, ctx.url, WORKER_ID, fmt, 17)


@pytest.mark.parametrize("delegate_only", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_exact_sender_route_and_collector_preserve_dtype_values_and_channels(
        transport, monkeypatch, dtype, channels, delegate_only):
    images = frames(dtype, channels)
    frame_bytes = images[0].numel() * images.element_size()
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=reserve + 2 * frame_bytes,
                            chunk_target=2 * frame_bytes) as ctx:
            await send(ctx, images)
            items = queued_items(ctx.queue)
            assert [item["image_index"] for item in items] == list(range(len(images)))
            assert [item["is_last"] for item in items] == [False] * 4 + [True]
            assert all(item["worker_id"] == WORKER_ID and item["format"] == "exact" for item in items)
            for item, expected in zip(items, images.split(1)):
                assert_exact(item["tensor"], expected)
                await ctx.queue.put(item)
            envelopes = [record["envelope"] for record in ctx.requests]
            assert [(e["start_index"], e["count"]) for e in envelopes] == [(0, 2), (2, 2), (4, 1)]
            assert all(record["bytes"] <= reserve + 2 * frame_bytes for record in ctx.requests)
            master_images = None if delegate_only else images[:1].detach().clone().add(0.03125)
            actual, _audio = await ctx.node.execute(
                master_images, None, multi_job_id=JOB_ID, enabled_worker_ids=json.dumps([WORKER_ID]),
                delegate_only=delegate_only, transfer_format="exact",
            )
            expected = images if delegate_only else torch.cat([master_images, images])
            assert_exact(actual, expected)
            assert JOB_ID not in transport.server.distributed_pending_jobs

    asyncio.run(scenario())


def test_exact_single_frame_can_exceed_chunk_target_but_not_request_limit(transport, monkeypatch):
    images = frames(count=3)
    frame_bytes = images[0].numel() * images.element_size()
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=reserve + frame_bytes,
                            chunk_target=frame_bytes // 2) as ctx:
            await send(ctx, images)
            assert [record["envelope"]["count"] for record in ctx.requests] == [1, 1, 1]
            assert all(record["bytes"] <= reserve + frame_bytes for record in ctx.requests)
            assert_exact(torch.cat([item["tensor"] for item in queued_items(ctx.queue)]), images)

    asyncio.run(scenario())


def test_exact_oversized_frame_is_rejected_before_encoding_or_post(transport, monkeypatch):
    images = frames(count=3)
    frame_bytes = images[0].numel() * images.element_size()
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES
    encode = AsyncMock(side_effect=AssertionError("oversized frame must not be encoded"))
    monkeypatch.setattr(transport.collector, "_encode_exact_frames", encode)

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=reserve + frame_bytes - 1) as ctx:
            with pytest.raises(transport.collector.transfer_codec.FrameTooLargeError, match="request limit"):
                await send(ctx, images)
            assert ctx.requests == []
            assert ctx.queue.empty()
            encode.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("shape", [(0, 2, 3, 3), (1, 0, 3, 3), (1, 2, 0, 3), (1, 2, 3, 0),
                                    (1, 2, 3, 2), (1, 2, 3, 5), (1, 2, 3), (2, 3)])
def test_exact_sender_rejects_invalid_geometry_before_post(transport, monkeypatch, shape):
    image = torch.empty(shape)
    with pytest.raises(transport.collector.transfer_codec.TransferCodecError):
        transport.collector._exact_frame_bytes(image)

    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            # An empty batch is also a no-media completion error (ValueError).
            with pytest.raises(ValueError):
                await send(ctx, image)
            assert ctx.requests == []
            assert ctx.queue.empty()

    asyncio.run(scenario())


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.bool, torch.complex64])
def test_exact_sender_rejects_unsupported_dtype_before_post(transport, monkeypatch, dtype):
    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            with pytest.raises(transport.collector.transfer_codec.TransferCodecError, match="dtype"):
                await send(ctx, torch.zeros(1, 2, 3, 3, dtype=dtype))
            assert ctx.requests == []

    asyncio.run(scenario())


def test_exact_frame_planning_enforces_bridge_byte_limit(transport, monkeypatch):
    monkeypatch.setattr(transport.bridge_codec, "MAX_PAYLOAD_BYTES", 32)
    with pytest.raises(transport.collector.transfer_codec.TransferCodecError, match="[Bb]yte|limit"):
        transport.collector._exact_frame_bytes(torch.zeros(1, 2, 3, 3))


def test_exact_frame_planning_rejects_non_dense_layout(transport):
    image = torch.zeros(1, 2, 3, 3).to_sparse()
    with pytest.raises(transport.collector.transfer_codec.TransferCodecError, match="dense|layout|strided"):
        transport.collector._exact_frame_bytes(image)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_exact_sender_rejects_nonfinite_payload(transport, monkeypatch, value):
    images = torch.full((1, 2, 3, 3), value)

    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            with pytest.raises(transport.collector.transfer_codec.TransferCodecError, match="finite"):
                await send(ctx, images)
            assert ctx.requests == []
            assert ctx.queue.empty()

    asyncio.run(scenario())


def exact_chunk(transport, count=2):
    images = frames(count=count).detach()
    encoded = [transport.bridge_codec.encode_payload("IMAGE", frame) for frame in images.split(1)]
    return {
        "job_id": JOB_ID, "worker_id": WORKER_ID, "format": "exact",
        "start_index": 0, "count": count, "width": images.shape[2], "height": images.shape[1],
        "is_last": True, "descriptors": [descriptor for descriptor, _ in encoded],
    }, [payload for _, payload in encoded]


@pytest.mark.parametrize("case", [
    "zero_count", "bool_width", "negative_height", "negative_index", "bool_index",
    "missing_descriptor", "extra_descriptor", "missing_part", "extra_part", "not_a_descriptor",
    "wrong_height", "wrong_batch", "bad_channels", "bad_dtype", "bad_layout", "bad_encoding",
    "extra_descriptor_field", "truncated", "trailing_bytes", "nonfinite", "compressed_spoof",
])
def test_exact_route_rejects_bad_geometry_and_payload_atomically(transport, monkeypatch, case):
    envelope, parts = exact_chunk(transport)
    descriptor = envelope["descriptors"][-1]
    if case == "zero_count":
        envelope["count"] = 0
    elif case == "bool_width":
        envelope["width"] = True
    elif case == "negative_height":
        envelope["height"] = -1
    elif case == "negative_index":
        envelope["start_index"] = -1
    elif case == "bool_index":
        envelope["start_index"] = True
    elif case == "missing_descriptor":
        envelope["descriptors"].pop()
    elif case == "extra_descriptor":
        envelope["descriptors"].append(dict(descriptor))
    elif case == "missing_part":
        parts.pop()
    elif case == "extra_part":
        parts.append(parts[-1])
    elif case == "not_a_descriptor":
        envelope["descriptors"][-1] = None
    elif case == "wrong_height":
        descriptor["shape"][1] += 1
    elif case == "wrong_batch":
        descriptor["shape"][0] = 2
    elif case == "bad_channels":
        descriptor["shape"][-1] = 2
    elif case == "bad_dtype":
        descriptor["dtype"] = "uint8"
    elif case == "bad_layout":
        descriptor["layout"] = "BCHW"
    elif case == "bad_encoding":
        descriptor["encoding"] = "pickle"
    elif case == "extra_descriptor_field":
        descriptor["unknown"] = True
    elif case == "truncated":
        parts[-1] = parts[-1][:-1]
    elif case == "trailing_bytes":
        parts[-1] += b"\0"
    elif case == "nonfinite":
        bad = torch.full(tuple(descriptor["shape"]), float("nan"))
        parts[-1] = bad.numpy().tobytes()
    elif case == "compressed_spoof":
        # A valid serverless image-transfer descriptor must NOT be accepted as
        # exact by the direct route, even though the shared bridge can decode it.
        envelope["descriptors"][-1], parts[-1] = transport.image_codec.encode_image(
            frames(count=1).detach(), transfer_format="png", transfer_quality=90,
        )
        # Prove the spoof is a valid compressed bridge payload, not simply junk.
        decoded = transport.bridge_codec.decode_payload("IMAGE", envelope["descriptors"][-1], parts[-1])
        assert list(decoded.shape) == descriptor["shape"]

    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            async with ctx.session.post(
                ctx.url + "/distributed/job_complete",
                data=ctx.node._build_multipart(envelope, parts),
            ) as response:
                body = await response.text()
                assert response.status == 400, body
            # A corrupt second frame must not leave the first in the queue.
            assert ctx.queue.empty()

    asyncio.run(scenario())


@pytest.mark.parametrize("status,body", [(400, "Invalid JSON payload: old master"),
                                        (413, "upload limit"), (500, "unexpected failure")])
def test_exact_master_rejection_never_falls_back(transport, monkeypatch, status, body):
    async def reject(_request):
        return transport.web.Response(status=status, text=body)

    async def scenario():
        async with endpoint(transport, monkeypatch, handler=reject, chunk_target=1) as ctx:
            legacy = AsyncMock(side_effect=AssertionError("exact must never use legacy transport"))
            compressed = AsyncMock(side_effect=AssertionError("exact must never use compressed transport"))
            monkeypatch.setattr(ctx.node, "_send_legacy_batch", legacy)
            monkeypatch.setattr(ctx.node, "_send_compressed_batch", compressed)
            with pytest.raises(RuntimeError, match=f"HTTP {status}") as raised:
                await send(ctx, frames(count=3))
            assert "never" in str(raised.value) and "8-bit" in str(raised.value)
            assert len(ctx.requests) == 1
            assert ctx.requests[0]["content_type"].startswith("multipart/")
            assert ctx.queue.empty()
            legacy.assert_not_called()
            compressed.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("fmt", ["legacy_png", "png"])
def test_exact_collector_rejects_old_worker_8bit_result(transport, monkeypatch, fmt):
    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            await send(ctx, frames(count=1).detach(), fmt=fmt)
            assert ctx.queue.qsize() == 1
            with pytest.raises(RuntimeError, match="requested exact"):
                await ctx.node.execute(
                    None, None, multi_job_id=JOB_ID, enabled_worker_ids=json.dumps([WORKER_ID]),
                    delegate_only=True, transfer_format="exact",
                )
            assert JOB_ID not in transport.server.distributed_pending_jobs

    asyncio.run(scenario())


def test_exact_collector_shape_mismatch_raises_instead_of_returning_local_images(transport, monkeypatch):
    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            # Channel mixing is supported; incompatible spatial sizes still fail.
            await send(ctx, frames(count=1, channels=4, width=10))
            with pytest.raises(RuntimeError, match="spatial dimensions"):
                await ctx.node.execute(
                    frames(count=1, channels=3), None, multi_job_id=JOB_ID,
                    enabled_worker_ids=json.dumps([WORKER_ID]), transfer_format="exact",
                )
            assert JOB_ID not in transport.server.distributed_pending_jobs

    asyncio.run(scenario())


@pytest.mark.parametrize("local_channels,worker_channels", [(3, 4), (4, 3)])
def test_exact_collector_mixed_rgb_rgba_preserves_every_value(transport, monkeypatch, local_channels, worker_channels):
    local = frames(count=1, channels=local_channels)
    worker = frames(count=1, channels=worker_channels)

    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            await send(ctx, worker)
            result, _audio = await ctx.node.execute(
                local, None, multi_job_id=JOB_ID,
                enabled_worker_ids=json.dumps([WORKER_ID]), transfer_format="exact",
            )
            assert result.shape == (2, 7, 9, 4)
            assert result.is_contiguous()
            # RGB views into RGBA have a four-channel stride; materialize the
            # sliced comparison without changing the output-contiguity check.
            assert_exact(result[:1, ..., :local_channels].contiguous(), local)
            assert_exact(result[1:, ..., :worker_channels].contiguous(), worker)
            if local_channels == 3:
                assert torch.all(result[:1, ..., 3] == 1)
            if worker_channels == 3:
                assert torch.all(result[1:, ..., 3] == 1)
            assert JOB_ID not in transport.server.distributed_pending_jobs

    asyncio.run(scenario())


def test_exact_incomplete_worker_raises_and_cleans_queue(transport, monkeypatch):
    """A timed-out worker's first frame is not a successful partial result."""
    envelope, parts = exact_chunk(transport, count=1)
    envelope["is_last"] = False

    async def scenario():
        async with endpoint(transport, monkeypatch) as ctx:
            async with ctx.session.post(
                ctx.url + "/distributed/job_complete", data=ctx.node._build_multipart(envelope, parts),
            ) as response:
                assert response.status == 200, await response.text()
            assert ctx.queue.qsize() == 1
            with pytest.raises(RuntimeError, match="Exact transfer incomplete"):
                await asyncio.wait_for(ctx.node.execute(
                    frames(count=1).detach(), None, multi_job_id=JOB_ID,
                    enabled_worker_ids=json.dumps([WORKER_ID]), transfer_format="exact",
                ), timeout=2)
            assert JOB_ID not in transport.server.distributed_pending_jobs

    asyncio.run(scenario())


def test_exact_oversized_multipart_metadata_is_rejected_before_post(transport, monkeypatch):
    images = frames(count=1)
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES
    limit = reserve + images.numel() * images.element_size()

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=limit) as ctx:
            with pytest.raises(transport.collector.transfer_codec.FrameTooLargeError, match="request limit"):
                await ctx.node.send_batch_to_master(
                    images, None, JOB_ID, ctx.url, "w" * limit, "exact", 90,
                )
            assert ctx.requests == []
            assert ctx.queue.empty()

    asyncio.run(scenario())


def test_exact_actual_server_upload_rejection_is_413_without_retry(transport, monkeypatch):
    async def scenario():
        async with endpoint(transport, monkeypatch, client_max_size=1024) as ctx:
            with pytest.raises(RuntimeError, match="HTTP 413"):
                await send(ctx, frames(count=3, height=64, width=64))
            assert len(ctx.requests) == 1
            assert ctx.requests[0]["status"] == 413
            assert ctx.queue.empty()

    asyncio.run(scenario())


def test_exact_oversized_audio_is_rejected_before_any_image_post(transport, monkeypatch):
    images = frames(count=3)
    audio = {"waveform": torch.zeros(1, 1, 100000), "sample_rate": 48000}
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=reserve + 1024) as ctx:
            with pytest.raises(transport.collector.transfer_codec.FrameTooLargeError, match="audio.*request limit"):
                await send(ctx, images, audio=audio)
            assert ctx.requests == []
            assert ctx.queue.empty()

    asyncio.run(scenario())


def test_exact_tiny_frames_cannot_overflow_multipart_metadata_budget(transport, monkeypatch):
    images = torch.full((2048, 1, 1, 1), 0.125, dtype=torch.float16)
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES
    limit = reserve + images.numel() * images.element_size()

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=limit, chunk_target=limit) as ctx:
            await send(ctx, images)
            assert len(ctx.requests) > 1
            assert all(record["bytes"] <= limit for record in ctx.requests)
            assert_exact(torch.cat([item["tensor"] for item in queued_items(ctx.queue)]), images)

    asyncio.run(scenario())


@pytest.mark.parametrize("separate", [False, True])
def test_exact_audio_is_preserved_and_only_final_completion_is_terminal(transport, monkeypatch, separate):
    images = frames(count=3)
    audio = {"waveform": torch.linspace(-0.9, 0.9, 60).reshape(1, 1, -1), "sample_rate": 48000}
    frame_bytes = images[0].numel() * images.element_size()
    reserve = transport.collector.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES
    media_budget = frame_bytes if separate else frame_bytes + 8192
    limit = reserve + media_budget

    async def scenario():
        async with endpoint(transport, monkeypatch, request_limit=limit, chunk_target=frame_bytes) as ctx:
            await send(ctx, images, audio=audio)
            items = queued_items(ctx.queue)
            assert len(ctx.requests) == 3 + int(separate)
            assert [item["is_last"] for item in items] == [False] * (len(items) - 1) + [True]
            assert all(item["audio"] is None for item in items[:-1])
            assert items[-1]["audio"]["sample_rate"] == audio["sample_rate"]
            assert_exact(items[-1]["audio"]["waveform"], audio["waveform"])
            assert_exact(torch.cat([item["tensor"] for item in items if item["tensor"] is not None]), images)
            assert all(record["bytes"] <= limit for record in ctx.requests)
            if separate:
                assert items[-1]["tensor"] is None
                assert ctx.requests[-1]["content_type"] == "application/json"

    asyncio.run(scenario())
