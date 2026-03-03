import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _bootstrap_package(package_name):
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace())
    sys.modules["server"] = server_module


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


def _load_job_store_modules():
    package_name = "dist_job_store_testpkg"
    _bootstrap_package(package_name)
    job_models = _load_module(package_name, "upscale/job_models.py", "upscale.job_models")
    job_store = _load_module(package_name, "upscale/job_store.py", "upscale.job_store")
    return job_models, job_store


job_models_module, job_store_module = _load_job_store_modules()


class JobStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.prompt_server = job_store_module.server.PromptServer.instance
        self.prompt_server.distributed_pending_tile_jobs = {}
        self.prompt_server.distributed_tile_jobs_lock = asyncio.Lock()

    async def test_ensure_tile_jobs_initialized_prunes_invalid_entries(self):
        self.prompt_server.distributed_pending_tile_jobs["valid"] = job_models_module.TileJobState("job-valid")
        self.prompt_server.distributed_pending_tile_jobs["invalid"] = {"not": "state"}

        job_store_module.ensure_tile_jobs_initialized()

        self.assertIn("valid", self.prompt_server.distributed_pending_tile_jobs)
        self.assertNotIn("invalid", self.prompt_server.distributed_pending_tile_jobs)

    async def test_init_dynamic_job_populates_pending_images(self):
        await job_store_module.init_dynamic_job(
            multi_job_id="job-dynamic",
            batch_size=3,
            enabled_workers=["worker-a"],
            all_indices=[2, 0, 1],
        )

        job_data = self.prompt_server.distributed_pending_tile_jobs["job-dynamic"]
        self.assertIsInstance(job_data, job_models_module.ImageJobState)
        self.assertEqual(job_data.batch_size, 3)
        self.assertIn("worker-a", job_data.worker_status)

        pulled = [job_data.pending_images.get_nowait() for _ in range(3)]
        self.assertEqual(pulled, [2, 0, 1])

    async def test_init_static_job_batched_queues_tile_ids(self):
        await job_store_module.init_static_job_batched(
            multi_job_id="job-static",
            batch_size=2,
            num_tiles_per_image=4,
            enabled_workers=["worker-a", "worker-b"],
        )

        job_data = self.prompt_server.distributed_pending_tile_jobs["job-static"]
        self.assertIsInstance(job_data, job_models_module.TileJobState)
        self.assertTrue(job_data.batched_static)

        pulled = [job_data.pending_tasks.get_nowait() for _ in range(4)]
        self.assertEqual(pulled, [0, 1, 2, 3])

    async def test_drain_results_queue_collects_images_tiles_and_completion(self):
        job_data = job_models_module.TileJobState("job-drain")
        job_data.worker_status = {"worker-a": 1.0, "worker-b": 1.0}
        self.prompt_server.distributed_pending_tile_jobs["job-drain"] = job_data

        await job_data.queue.put(
            {
                "worker_id": "worker-a",
                "is_last": True,
                "image_idx": 5,
                "image": "img-5",
            }
        )
        await job_data.queue.put(
            {
                "worker_id": "worker-b",
                "is_last": False,
                "tiles": [
                    {"tile_idx": 0, "global_idx": 0, "tensor": "tile-0"},
                    {"tile_idx": 1, "global_idx": 1, "tensor": "tile-1"},
                ],
            }
        )

        drained = await job_store_module._drain_results_queue("job-drain")

        self.assertEqual(drained, 3)
        self.assertEqual(job_data.completed_tasks[5], "img-5")
        self.assertEqual(job_data.completed_tasks[0]["tensor"], "tile-0")
        self.assertEqual(job_data.completed_tasks[1]["tensor"], "tile-1")
        self.assertNotIn("worker-a", job_data.worker_status)
        self.assertIn("worker-b", job_data.worker_status)


if __name__ == "__main__":
    unittest.main()
