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
