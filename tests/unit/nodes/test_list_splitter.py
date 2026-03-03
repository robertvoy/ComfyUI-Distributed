import asyncio
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import torch


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

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda coro: asyncio.run(coro)
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _get_client_session():
        raise RuntimeError("test should patch get_client_session")

    network_module.get_client_session = _get_client_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    job_models_module = types.ModuleType(f"{package_name}.upscale.job_models")

    class BaseJobState:
        pass

    @dataclass
    class ImageJobState(BaseJobState):
        multi_job_id: str
        mode: str = field(default="dynamic", init=False)
        queue: asyncio.Queue = field(default_factory=asyncio.Queue)
        pending_images: asyncio.Queue = field(default_factory=asyncio.Queue)
        completed_images: dict = field(default_factory=dict)
        worker_status: dict = field(default_factory=dict)
        assigned_to_workers: dict = field(default_factory=dict)

        @property
        def pending_tasks(self):
            return self.pending_images

    job_models_module.BaseJobState = BaseJobState
    job_models_module.ImageJobState = ImageJobState
    sys.modules[f"{package_name}.upscale.job_models"] = job_models_module

    prompt_server_holder = {
        "value": types.SimpleNamespace(
            distributed_tile_jobs_lock=asyncio.Lock(),
            distributed_pending_tile_jobs={},
        )
    }

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")

    def ensure_tile_jobs_initialized():
        return prompt_server_holder["value"]

    async def init_dynamic_job(multi_job_id, batch_size, enabled_workers, all_indices=None):
        prompt_server = prompt_server_holder["value"]
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                return
            job_data = ImageJobState(multi_job_id=multi_job_id)
            for idx in (all_indices if all_indices is not None else list(range(batch_size))):
                await job_data.pending_images.put(idx)
            prompt_server.distributed_pending_tile_jobs[multi_job_id] = job_data

    job_store_module.ensure_tile_jobs_initialized = ensure_tile_jobs_initialized
    job_store_module.init_dynamic_job = init_dynamic_job
    job_store_module._prompt_server_holder = prompt_server_holder
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module


def _load_module(package_name, module_rel_path, module_name):
    module_path = Path(__file__).resolve().parents[3] / module_rel_path
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_list_splitter_module():
    package_name = "dist_list_splitter_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "nodes/utilities.py", "nodes.utilities")
    module = _load_module(package_name, "nodes/list_splitter.py", "nodes.list_splitter")
    return module


splitter_module = _load_list_splitter_module()


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        if not self.responses:
            return _FakeResponse({"item_idx": None}, status=200)
        payload = self.responses.pop(0)
        return _FakeResponse(payload, status=200)


class DistributedListSplitterTests(unittest.TestCase):
    def setUp(self):
        self.node = splitter_module.DistributedListSplitter()
        prompt_server = splitter_module.ensure_tile_jobs_initialized()
        prompt_server.distributed_pending_tile_jobs = {}

    def _images(self, count):
        return [torch.full((1, 4, 4, 3), float(idx), dtype=torch.float32) for idx in range(count)]

    def test_single_participant_returns_full_list(self):
        images = self._images(5)
        out = self.node.split(images, mode="static", participant_index=0, total_participants=1)[0]
        self.assertEqual(len(out), 5)
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in out], [0, 1, 2, 3, 4])

    def test_two_participants_split_six_images_evenly(self):
        images = self._images(6)
        p0 = self.node.split(images, mode="static", participant_index=0, total_participants=2)[0]
        p1 = self.node.split(images, mode="static", participant_index=1, total_participants=2)[0]
        self.assertEqual(len(p0), 3)
        self.assertEqual(len(p1), 3)
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p0], [0, 1, 2])
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p1], [3, 4, 5])

    def test_three_participants_split_seven_images(self):
        images = self._images(7)
        p0 = self.node.split(images, mode="static", participant_index=0, total_participants=3)[0]
        p1 = self.node.split(images, mode="static", participant_index=1, total_participants=3)[0]
        p2 = self.node.split(images, mode="static", participant_index=2, total_participants=3)[0]
        self.assertEqual([len(p0), len(p1), len(p2)], [3, 2, 2])

    def test_more_participants_than_images_yields_empty_slices(self):
        images = self._images(2)
        outputs = [self.node.split(images, mode="static", participant_index=i, total_participants=5)[0] for i in range(5)]
        self.assertEqual([len(chunk) for chunk in outputs], [1, 1, 0, 0, 0])

    def test_mixed_dimensions_are_preserved(self):
        images = [
            torch.zeros((1, 4, 4, 3), dtype=torch.float32),
            torch.zeros((1, 8, 6, 3), dtype=torch.float32),
            torch.zeros((1, 3, 9, 3), dtype=torch.float32),
        ]
        p0 = self.node.split(images, mode="static", participant_index=0, total_participants=2)[0]
        p1 = self.node.split(images, mode="static", participant_index=1, total_participants=2)[0]
        self.assertEqual([tuple(t.shape) for t in p0], [(1, 4, 4, 3), (1, 8, 6, 3)])
        self.assertEqual([tuple(t.shape) for t in p1], [(1, 3, 9, 3)])

    def test_dynamic_mode_master_pulls_from_local_queue(self):
        images = self._images(4)

        pulled = self.node.split(
            images,
            mode="dynamic",
            multi_job_id="job-dynamic",
            is_worker=False,
            worker_id="master",
        )[0]

        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in pulled], [0, 1, 2, 3])

    def test_dynamic_mode_worker_pulls_via_http(self):
        images = self._images(5)
        fake_session = _FakeSession([
            {"item_idx": 1, "estimated_remaining": 3},
            {"item_idx": 3, "estimated_remaining": 2},
            {"item_idx": None},
        ])

        async def _get_session():
            return fake_session

        splitter_module.get_client_session = _get_session

        pulled = self.node.split(
            images,
            mode="dynamic",
            multi_job_id="job-dynamic",
            is_worker=True,
            master_url="http://master.local:8188",
            worker_id="worker-a",
        )[0]

        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in pulled], [1, 3])
        self.assertGreaterEqual(len(fake_session.calls), 3)

    def test_dynamic_mode_exhausted_queue_returns_empty(self):
        images = self._images(3)

        first_pull = self.node.split(
            images,
            mode="dynamic",
            multi_job_id="job-dynamic",
            is_worker=False,
            worker_id="master",
        )[0]
        second_pull = self.node.split(
            images,
            mode="dynamic",
            multi_job_id="job-dynamic",
            is_worker=False,
            worker_id="master",
        )[0]

        self.assertEqual(len(first_pull), 3)
        self.assertEqual(second_pull, [])


if __name__ == "__main__":
    unittest.main()
