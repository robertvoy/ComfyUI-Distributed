"""End-to-end: real worker send path -> real aiohttp server -> job_complete endpoint."""
import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS_DIR = Path(__file__).resolve().parents[1]


def _load_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_modules():
    real_aiohttp = sys.modules["aiohttp"]  # imported inside the isolated test
    queue_tests = _load_from(_TESTS_DIR / "api" / "test_distributed_queue.py", "dist_transfer_it_queue")
    collector_tests = _load_from(_TESTS_DIR / "test_collector_list_inputs.py", "dist_transfer_it_collector")
    job_routes = queue_tests._load_job_routes_module()
    collector = collector_tests._load_collector_module()

    # Error responses must be real HTTP responses here (the unit-test harness
    # returns plain objects); mirror utils.network.handle_api_error.
    from aiohttp import web

    async def handle_api_error(_request, error, status=500):
        if isinstance(error, list):
            return web.json_response({"errors": [str(item) for item in error]}, status=status)
        return web.json_response({"error": str(error)}, status=status)

    job_routes.handle_api_error = handle_api_error
    # The collector harness stubs aiohttp; this test needs the real client.
    sys.modules["aiohttp"] = real_aiohttp
    collector.aiohttp = real_aiohttp
    return job_routes, collector


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """The loaders below install stub modules (server, comfy, aiohttp...); undo that."""
    saved = dict(sys.modules)
    # Earlier test modules may have left a stub `aiohttp` behind; drop it so the
    # real package is imported for this test (restored afterwards).
    for name in [n for n in sys.modules if n == "aiohttp" or n.startswith("aiohttp.")]:
        if getattr(sys.modules[name], "__file__", None) is None:
            del sys.modules[name]
    yield
    sys.modules.clear()
    sys.modules.update(saved)


async def _run_transfer(job_routes, collector_module, images, fmt, *, client_max_size,
                        request_limit, chunk_target, audio=None):
    """Send `images` from a real worker node to the real completion route.

    Returns (queued items, sizes of every request body the server received,
    exception raised by the worker or None).
    """
    import aiohttp
    from aiohttp import web

    queue = asyncio.Queue()
    job_routes.prompt_server.distributed_jobs_lock = asyncio.Lock()
    job_routes.prompt_server.distributed_pending_jobs = {"job-it": queue}
    body_sizes = []

    @web.middleware
    async def record_size(request, handler):
        body_sizes.append(request.content_length)
        return await handler(request)

    app = web.Application(client_max_size=client_max_size, middlewares=[record_size])
    app.router.add_post("/distributed/job_complete", job_routes.job_complete_endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    error = None
    try:
        async with aiohttp.ClientSession() as session:
            async def _get_session():
                return session

            collector_module.get_client_session = _get_session
            collector_module.TRANSFER_MAX_REQUEST_BYTES = request_limit
            collector_module.TRANSFER_MAX_CHUNK_BYTES = chunk_target
            node = collector_module.DistributedCollectorNode()
            try:
                await node.send_batch_to_master(
                    images, audio, "job-it", f"http://127.0.0.1:{port}", "worker-1", fmt, 90
                )
            except Exception as exc:  # surfaced to the caller for assertions
                error = exc
    finally:
        await runner.cleanup()

    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items, body_sizes, error


def _noise_images(count, height=360, width=640, seed=97):
    return torch.from_numpy(np.random.default_rng(seed).random((count, height, width, 3), dtype=np.float32))


@pytest.mark.parametrize("fmt", ["h264", "ffv1"])
def test_high_entropy_video_stays_under_server_upload_limit(fmt):
    """Encoder flush used to push finalized chunks past the limit (HTTP 400)."""
    pytest.importorskip("aiohttp")
    job_routes, collector_module = _load_modules()
    images = _noise_images(24)
    limit = 2 * 1024 * 1024
    reserve = collector_module.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES

    items, body_sizes, error = asyncio.run(_run_transfer(
        job_routes, collector_module, images, fmt,
        client_max_size=limit, request_limit=limit, chunk_target=limit - reserve))

    assert error is None, error
    assert len(body_sizes) > 1 and max(body_sizes) <= limit
    assert [item["image_index"] for item in items] == list(range(24))
    assert [item["is_last"] for item in items] == [False] * 23 + [True]


def test_server_rejection_for_size_is_reported_not_retried_as_legacy():
    """A worker configured above the master's limit fails loudly instead of
    silently resending the whole batch as uncompressed PNG."""
    pytest.importorskip("aiohttp")
    job_routes, collector_module = _load_modules()
    images = _noise_images(8)

    items, body_sizes, error = asyncio.run(_run_transfer(
        job_routes, collector_module, images, "ffv1",
        client_max_size=1024 * 1024, request_limit=8 * 1024 * 1024, chunk_target=4 * 1024 * 1024))

    assert isinstance(error, RuntimeError)
    assert "HTTP 413" in str(error) and "--max-upload-size" in str(error)
    assert len(body_sizes) == 1
    assert items == []


def test_frame_that_cannot_fit_sends_nothing():
    pytest.importorskip("aiohttp")
    job_routes, collector_module = _load_modules()
    images = _noise_images(3)
    reserve = collector_module.DistributedCollectorNode.TRANSFER_ENVELOPE_RESERVE_BYTES

    items, body_sizes, error = asyncio.run(_run_transfer(
        job_routes, collector_module, images, "png",
        client_max_size=100 * 1024 * 1024, request_limit=reserve + 64 * 1024, chunk_target=64 * 1024))

    assert isinstance(error, collector_module.transfer_codec.FrameTooLargeError)
    assert body_sizes == []
    assert items == []


@pytest.mark.parametrize("fmt", ["png", "ffv1", "h264", "jpeg"])
def test_worker_to_master_over_real_http(fmt):
    # Import here, not at module level: other test harnesses only install their
    # aiohttp stubs when the real package has not been imported yet.
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    job_routes, collector_module = _load_modules()
    images = torch.rand(5, 24, 36, 3)

    async def scenario():
        queue = asyncio.Queue()
        job_routes.prompt_server.distributed_jobs_lock = asyncio.Lock()
        job_routes.prompt_server.distributed_pending_jobs = {"job-it": queue}

        app = web.Application()
        app.router.add_post("/distributed/job_complete", job_routes.job_complete_endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        async with aiohttp.ClientSession() as session:
            async def _get_session():
                return session

            collector_module.get_client_session = _get_session
            collector_module.TRANSFER_MAX_CHUNK_BYTES = 2000  # several requests for still formats
            node = collector_module.DistributedCollectorNode()
            await node.send_batch_to_master(
                images, None, "job-it", f"http://127.0.0.1:{port}", "worker-1", fmt, 90
            )
        await runner.cleanup()

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    items = asyncio.run(scenario())

    assert [item["image_index"] for item in items] == [0, 1, 2, 3, 4]
    assert [item["is_last"] for item in items] == [False, False, False, False, True]
    expected = np.clip(255.0 * images.numpy(), 0, 255).astype(np.uint8).astype(np.float32) / 255.0
    got = torch.cat([item["tensor"] for item in items]).numpy()
    assert got.shape == expected.shape
    if fmt in ("png", "ffv1"):
        assert np.array_equal(got, expected)
    else:
        assert np.abs(got - expected).mean() < 0.25  # random noise: only sanity-check lossy codecs
