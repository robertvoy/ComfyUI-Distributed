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


def _load_job_state_modules():
    package_name = "dist_job_state_testpkg"
    _bootstrap_package(package_name)
    job_models = _load_module(package_name, "upscale/job_models.py", "upscale.job_models")

    prompt_server = types.SimpleNamespace(
        distributed_pending_tile_jobs={},
        distributed_tile_jobs_lock=asyncio.Lock(),
    )
    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.ensure_tile_jobs_initialized = lambda: prompt_server
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    call_log = {"args": None}
    job_timeout_module = types.ModuleType(f"{package_name}.upscale.job_timeout")

    async def _requeue(multi_job_id, batch_size):
        call_log["args"] = (multi_job_id, batch_size)
        return 3

    job_timeout_module._check_and_requeue_timed_out_workers = _requeue
    sys.modules[f"{package_name}.upscale.job_timeout"] = job_timeout_module

    job_state = _load_module(package_name, "upscale/job_state.py", "upscale.job_state")
    return job_models, job_state, prompt_server, call_log


job_models_module, job_state_module, prompt_server, timeout_call_log = _load_job_state_modules()


class _Node(job_state_module.JobStateMixin):
    pass


class JobStateMixinTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        prompt_server.distributed_pending_tile_jobs = {}
        prompt_server.distributed_tile_jobs_lock = asyncio.Lock()
        self.node = _Node()

    async def test_get_all_completed_tasks_supports_tile_and_image_modes(self):
        tile_state = job_models_module.TileJobState("tile-job")
        tile_state.completed_tasks[4] = "tile-4"
        image_state = job_models_module.ImageJobState("image-job")
        image_state.completed_images[2] = "image-2"
        prompt_server.distributed_pending_tile_jobs = {
            "tile-job": tile_state,
            "image-job": image_state,
        }

        self.assertEqual(await self.node._get_all_completed_tasks("tile-job"), {4: "tile-4"})
        self.assertEqual(await self.node._get_all_completed_tasks("image-job"), {2: "image-2"})

    async def test_next_index_and_pending_count_from_queues(self):
        image_state = job_models_module.ImageJobState("image-job")
        await image_state.pending_images.put(7)
        tile_state = job_models_module.TileJobState("tile-job")
        await tile_state.pending_tasks.put(9)
        prompt_server.distributed_pending_tile_jobs = {
            "image-job": image_state,
            "tile-job": tile_state,
        }

        self.assertEqual(await self.node._get_next_image_index("image-job"), 7)
        self.assertEqual(await self.node._get_next_tile_index("tile-job"), 9)
        self.assertEqual(await self.node._get_pending_count("image-job"), 0)
        self.assertEqual(await self.node._get_pending_count("tile-job"), 0)

    async def test_drain_worker_results_queue_collects_completed_images(self):
        image_state = job_models_module.ImageJobState("image-job")
        await image_state.queue.put({"worker_id": "w1", "image_idx": 1, "image": "img1"})
        await image_state.queue.put({"worker_id": "w2", "image_idx": 2, "image": "img2"})
        prompt_server.distributed_pending_tile_jobs = {"image-job": image_state}

        drained = await self.node._drain_worker_results_queue("image-job")

        self.assertEqual(drained, 2)
        self.assertEqual(image_state.completed_images[1], "img1")
        self.assertEqual(image_state.completed_images[2], "img2")

    async def test_check_and_requeue_delegates_to_timeout_helper(self):
        count = await self.node._check_and_requeue_timed_out_workers("job-abc", 5)
        self.assertEqual(count, 3)
        self.assertEqual(timeout_call_log["args"], ("job-abc", 5))


if __name__ == "__main__":
    unittest.main()
