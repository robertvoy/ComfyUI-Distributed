import asyncio
import importlib.util
import sys
import time
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path


def _load_job_timeout_module():
    module_path = Path(__file__).resolve().parents[3] / "upscale" / "job_timeout.py"
    package_name = "dist_job_timeout_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    config_holder = {"value": {"settings": {}, "workers": []}}
    probe_holder = {"fn": None}
    prompt_server_holder = {
        "value": types.SimpleNamespace(
            distributed_tile_jobs_lock=asyncio.Lock(),
            distributed_pending_tile_jobs={},
        )
    }

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: config_holder["value"]
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.HEARTBEAT_TIMEOUT = 60
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_worker_url = lambda worker: f"http://{worker.get('host', '127.0.0.1')}:{worker.get('port', 8188)}"

    async def _probe_worker(url, timeout=2.0):
        fn = probe_holder["fn"]
        if fn is None:
            return None
        return await fn(url, timeout)

    network_module.probe_worker = _probe_worker
    sys.modules[f"{package_name}.utils.network"] = network_module

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.ensure_tile_jobs_initialized = lambda: prompt_server_holder["value"]
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

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
        batch_size: int = 0
        num_tiles_per_image: int = 0
        batched_static: bool = False

        @property
        def pending_tasks(self):
            return self.pending_images

        @property
        def completed_tasks(self):
            return self.completed_images

    @dataclass
    class TileJobState(BaseJobState):
        multi_job_id: str
        mode: str = field(default="static", init=False)
        queue: asyncio.Queue = field(default_factory=asyncio.Queue)
        pending_tasks: asyncio.Queue = field(default_factory=asyncio.Queue)
        completed_tasks: dict = field(default_factory=dict)
        worker_status: dict = field(default_factory=dict)
        assigned_to_workers: dict = field(default_factory=dict)
        batch_size: int = 0
        num_tiles_per_image: int = 0
        batched_static: bool = False

    job_models_module.BaseJobState = BaseJobState
    job_models_module.ImageJobState = ImageJobState
    job_models_module.TileJobState = TileJobState
    sys.modules[f"{package_name}.upscale.job_models"] = job_models_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.upscale.job_timeout", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    module._config_holder = config_holder
    module._probe_holder = probe_holder
    module._prompt_server_holder = prompt_server_holder
    module._ImageJobState = ImageJobState
    module._TileJobState = TileJobState
    return module


jt = _load_job_timeout_module()


class JobTimeoutRequeueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        jt._prompt_server_holder["value"] = types.SimpleNamespace(
            distributed_tile_jobs_lock=asyncio.Lock(),
            distributed_pending_tile_jobs={},
        )
        jt._config_holder["value"] = {
            "settings": {"worker_timeout_seconds": 5},
            "workers": [{"id": "worker-1", "host": "worker.local", "port": 8188}],
        }

    async def test_requeues_only_incomplete_dynamic_tasks_for_timed_out_worker(self):
        async def _offline_probe(_url, _timeout):
            return None

        jt._probe_holder["fn"] = _offline_probe
        prompt_server = jt._prompt_server_holder["value"]
        job_data = jt._ImageJobState("job-1")
        job_data.worker_status["worker-1"] = time.time() - 60.0
        job_data.assigned_to_workers["worker-1"] = [0, 1]
        job_data.completed_images[1] = "done"
        prompt_server.distributed_pending_tile_jobs["job-1"] = job_data

        requeued = await jt._check_and_requeue_timed_out_workers("job-1", total_tasks=2)

        self.assertEqual(requeued, 1)
        self.assertEqual(await job_data.pending_images.get(), 0)
        self.assertNotIn("worker-1", job_data.worker_status)
        self.assertEqual(job_data.assigned_to_workers["worker-1"], [])

    async def test_busy_probe_graces_worker_and_skips_requeue(self):
        async def _busy_probe(_url, _timeout):
            return {"exec_info": {"queue_remaining": 3}}

        jt._probe_holder["fn"] = _busy_probe
        prompt_server = jt._prompt_server_holder["value"]
        job_data = jt._ImageJobState("job-2")
        old_heartbeat = time.time() - 60.0
        job_data.worker_status["worker-1"] = old_heartbeat
        job_data.assigned_to_workers["worker-1"] = [0]
        prompt_server.distributed_pending_tile_jobs["job-2"] = job_data

        requeued = await jt._check_and_requeue_timed_out_workers("job-2", total_tasks=1)

        self.assertEqual(requeued, 0)
        self.assertIn("worker-1", job_data.worker_status)
        self.assertGreaterEqual(job_data.worker_status["worker-1"], old_heartbeat)
        self.assertTrue(job_data.pending_images.empty())

    async def test_completed_dynamic_task_is_not_requeued(self):
        async def _offline_probe(_url, _timeout):
            return None

        jt._probe_holder["fn"] = _offline_probe
        prompt_server = jt._prompt_server_holder["value"]
        job_data = jt._ImageJobState("job-3")
        job_data.worker_status["worker-1"] = time.time() - 60.0
        job_data.assigned_to_workers["worker-1"] = [7]
        job_data.completed_images[7] = "complete"
        prompt_server.distributed_pending_tile_jobs["job-3"] = job_data

        requeued = await jt._check_and_requeue_timed_out_workers("job-3", total_tasks=1)

        self.assertEqual(requeued, 0)
        self.assertTrue(job_data.pending_images.empty())


if __name__ == "__main__":
    unittest.main()
