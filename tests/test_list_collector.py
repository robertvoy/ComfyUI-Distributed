import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch
from PIL import Image


def _bootstrap_package(package_name):
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    nodes_pkg = types.ModuleType(f"{package_name}.nodes")
    nodes_pkg.__path__ = []
    sys.modules[f"{package_name}.nodes"] = nodes_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.get_worker_timeout_seconds = lambda: 0.2
    config_module.is_master_delegate_only = lambda: False
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.HEARTBEAT_INTERVAL = 1.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    image_module = types.ModuleType(f"{package_name}.utils.image")

    def ensure_contiguous(tensor):
        return tensor.contiguous()

    def tensor_to_pil(image_batch, index):
        tensor = image_batch[index].detach().cpu().clamp(0, 1)
        array = (tensor.numpy() * 255.0).astype("uint8")
        return Image.fromarray(array)

    image_module.ensure_contiguous = ensure_contiguous
    image_module.tensor_to_pil = tensor_to_pil
    sys.modules[f"{package_name}.utils.image"] = image_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def get_client_session():
        raise RuntimeError("test should patch get_client_session")

    network_module.get_client_session = get_client_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda coro: asyncio.run(coro)
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module


def _load_module(package_name, module_rel_path, module_name):
    module_path = Path(__file__).resolve().parents[1] / module_rel_path
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_list_collector_module():
    package_name = "dist_list_collector_testpkg"
    _bootstrap_package(package_name)
    return _load_module(package_name, "nodes/list_collector.py", "nodes.list_collector")


collector_module = _load_list_collector_module()


class _FakeResponseCtx:
    def __init__(self, recorder, payload, url):
        self._recorder = recorder
        self._payload = payload
        self._url = url

    async def __aenter__(self):
        self._recorder.append({"url": self._url, "payload": self._payload})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        return _FakeResponseCtx(self.calls, json, url)


class _FakePromptServer:
    def __init__(self):
        self.distributed_pending_jobs = {}
        self.distributed_jobs_lock = asyncio.Lock()


class DistributedListCollectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.node = collector_module.DistributedListCollector()

    async def test_worker_mode_sends_images_to_master(self):
        fake_session = _FakeSession()

        async def _get_session():
            return fake_session

        collector_module.get_client_session = _get_session

        images = [
            torch.zeros((1, 4, 4, 3), dtype=torch.float32),
            torch.ones((1, 4, 4, 3), dtype=torch.float32),
        ]

        result = await self.node.execute(
            images,
            multi_job_id="job-list",
            is_worker=True,
            master_url="http://master.local:8188",
            worker_id="worker-a",
        )

        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(fake_session.calls), 2)
        self.assertEqual(fake_session.calls[0]["url"], "http://master.local:8188/distributed/job_complete")
        self.assertEqual(fake_session.calls[0]["payload"]["batch_idx"], 0)
        self.assertFalse(fake_session.calls[0]["payload"]["is_last"])
        self.assertEqual(fake_session.calls[1]["payload"]["batch_idx"], 1)
        self.assertTrue(fake_session.calls[1]["payload"]["is_last"])

    async def test_master_mode_collects_results_as_list(self):
        fake_server = _FakePromptServer()
        collector_module._get_prompt_server_instance = lambda: fake_server

        queue = asyncio.Queue()
        worker_a = torch.full((1, 4, 4, 3), 0.25, dtype=torch.float32)
        worker_b = torch.full((1, 4, 4, 3), 0.75, dtype=torch.float32)
        await queue.put({"tensor": worker_a, "worker_id": "worker-a", "image_index": 0, "is_last": True})
        await queue.put({"tensor": worker_b, "worker_id": "worker-b", "image_index": 0, "is_last": True})
        fake_server.distributed_pending_jobs["job-list"] = queue

        master_images = [torch.zeros((1, 4, 4, 3), dtype=torch.float32)]

        result = await self.node.execute(
            master_images,
            multi_job_id="job-list",
            is_worker=False,
            enabled_worker_ids=json.dumps(["worker-a", "worker-b"]),
        )

        self.assertEqual(len(result[0]), 3)
        self.assertTrue(torch.equal(result[0][0], master_images[0]))
        self.assertTrue(torch.equal(result[0][1], worker_a))
        self.assertTrue(torch.equal(result[0][2], worker_b))

    async def test_delegate_only_mode_returns_worker_images_only(self):
        fake_server = _FakePromptServer()
        collector_module._get_prompt_server_instance = lambda: fake_server

        queue = asyncio.Queue()
        worker_img = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
        await queue.put({"tensor": worker_img, "worker_id": "worker-a", "image_index": 0, "is_last": True})
        fake_server.distributed_pending_jobs["job-list"] = queue

        master_images = [torch.zeros((1, 4, 4, 3), dtype=torch.float32)]

        result = await self.node.execute(
            master_images,
            multi_job_id="job-list",
            is_worker=False,
            enabled_worker_ids=json.dumps(["worker-a"]),
            delegate_only=True,
        )

        self.assertEqual(len(result[0]), 1)
        self.assertTrue(torch.equal(result[0][0], worker_img))


if __name__ == "__main__":
    unittest.main()
