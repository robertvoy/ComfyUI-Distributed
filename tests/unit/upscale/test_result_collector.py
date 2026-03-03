import asyncio
import importlib.util
import sys
import time
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
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.get_worker_timeout_seconds = lambda: 0.05
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.DYNAMIC_MODE_MAX_POLL_TIMEOUT = 10.0
    constants_module.HEARTBEAT_INTERVAL = 0.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    comfy_module = types.ModuleType("comfy")
    model_mgmt = types.ModuleType("comfy.model_management")

    class _InterruptProcessingException(Exception):
        pass

    model_mgmt.processing_interrupted = lambda: False
    model_mgmt.InterruptProcessingException = _InterruptProcessingException
    comfy_module.model_management = model_mgmt
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_mgmt

    prompt_server = types.SimpleNamespace(
        distributed_pending_tile_jobs={},
        distributed_tile_jobs_lock=asyncio.Lock(),
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=prompt_server)
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


def _load_result_collector_modules():
    package_name = "dist_result_collector_testpkg"
    _bootstrap_package(package_name)
    job_models = _load_module(package_name, "upscale/job_models.py", "upscale.job_models")
    job_store = _load_module(package_name, "upscale/job_store.py", "upscale.job_store")

    job_timeout = types.ModuleType(f"{package_name}.upscale.job_timeout")

    async def _noop_requeue(_multi_job_id, _batch_size):
        return 0

    job_timeout._check_and_requeue_timed_out_workers = _noop_requeue
    job_timeout.check_and_requeue_timed_out_workers = _noop_requeue
    sys.modules[f"{package_name}.upscale.job_timeout"] = job_timeout

    result_collector = _load_module(
        package_name,
        "upscale/result_collector.py",
        "upscale.result_collector",
    )
    return job_models, job_store, result_collector


job_models_module, job_store_module, result_collector_module = _load_result_collector_modules()


class _Node(result_collector_module.ResultCollectorMixin):
    def __init__(self):
        self.requeue_calls = []

    async def _check_and_requeue_timed_out_workers(self, multi_job_id, batch_size):
        self.requeue_calls.append((multi_job_id, batch_size))
        return 1


class _WrapperNode(result_collector_module.ResultCollectorMixin):
    def __init__(self):
        self.calls = []

    async def _async_collect_results(
        self,
        multi_job_id,
        num_workers,
        mode="static",
        remaining_to_collect=None,
        batch_size=None,
    ):
        self.calls.append(
            (multi_job_id, num_workers, mode, remaining_to_collect, batch_size)
        )
        return {"mode": mode}


class ResultCollectorMixinTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.prompt_server = job_store_module.server.PromptServer.instance
        self.prompt_server.distributed_pending_tile_jobs = {}
        self.prompt_server.distributed_tile_jobs_lock = asyncio.Lock()
        self.node = _Node()

    async def test_static_collection_collects_tiles_and_cleans_up_job(self):
        job_data = job_models_module.TileJobState("job-static")
        self.prompt_server.distributed_pending_tile_jobs["job-static"] = job_data
        await job_data.queue.put(
            {
                "worker_id": "worker-a",
                "is_last": True,
                "tiles": [
                    {"tile_idx": 3},
                    {
                        "tile_idx": 2,
                        "x": 10,
                        "y": 20,
                        "extracted_width": 64,
                        "extracted_height": 64,
                        "padding": 8,
                        "batch_idx": 1,
                        "global_idx": 9,
                        "tensor": "tile-9",
                    },
                ],
            }
        )

        collected = await self.node._async_collect_results(
            "job-static",
            num_workers=1,
            mode="static",
        )

        self.assertEqual(list(collected.keys()), [9])
        self.assertEqual(collected[9]["tensor"], "tile-9")
        self.assertEqual(collected[9]["worker_id"], "worker-a")
        self.assertNotIn("job-static", self.prompt_server.distributed_pending_tile_jobs)

    async def test_dynamic_collection_stops_at_remaining_target_and_cleans_up(self):
        job_data = job_models_module.ImageJobState("job-dynamic")
        self.prompt_server.distributed_pending_tile_jobs["job-dynamic"] = job_data
        await job_data.queue.put(
            {
                "worker_id": "worker-a",
                "is_last": False,
                "image_idx": 2,
                "image": "img-2",
            }
        )

        completed = await self.node._async_collect_results(
            "job-dynamic",
            num_workers=2,
            mode="dynamic",
            remaining_to_collect=1,
            batch_size=3,
        )

        self.assertEqual(completed[2], "img-2")
        self.assertNotIn("job-dynamic", self.prompt_server.distributed_pending_tile_jobs)

    async def test_dynamic_timeout_calls_requeue_helper(self):
        job_data = job_models_module.ImageJobState("job-timeout")
        job_data.worker_status = {"worker-a": time.time() - 5}
        self.prompt_server.distributed_pending_tile_jobs["job-timeout"] = job_data

        old_timeout = result_collector_module.get_worker_timeout_seconds
        result_collector_module.get_worker_timeout_seconds = lambda: 0.01
        try:
            completed = await self.node._async_collect_results(
                "job-timeout",
                num_workers=1,
                mode="dynamic",
                batch_size=4,
            )
        finally:
            result_collector_module.get_worker_timeout_seconds = old_timeout

        self.assertEqual(completed, {})
        self.assertEqual(self.node.requeue_calls, [("job-timeout", 4)])
        self.assertNotIn("job-timeout", self.prompt_server.distributed_pending_tile_jobs)

    async def test_log_worker_timeout_status_handles_non_job_state(self):
        worker_ids = self.node._log_worker_timeout_status(
            {"worker_status": {"worker-a": time.time()}},
            current_time=time.time(),
            multi_job_id="job-invalid",
        )
        self.assertEqual(worker_ids, [])

    async def test_log_worker_timeout_status_returns_worker_ids_for_job_state(self):
        job_data = job_models_module.ImageJobState("job-state")
        job_data.worker_status = {"worker-a": time.time() - 1}

        worker_ids = self.node._log_worker_timeout_status(
            job_data,
            current_time=time.time(),
            multi_job_id="job-state",
        )
        self.assertEqual(worker_ids, ["worker-a"])

    async def test_mode_mismatch_and_unsupported_mode_raise(self):
        self.prompt_server.distributed_pending_tile_jobs["job-mismatch"] = (
            job_models_module.TileJobState("job-mismatch")
        )
        with self.assertRaises(RuntimeError):
            await self.node._async_collect_results(
                "job-mismatch",
                num_workers=1,
                mode="dynamic",
                batch_size=1,
            )

        self.prompt_server.distributed_pending_tile_jobs["job-unsupported"] = (
            job_models_module.TileJobState("job-unsupported")
        )
        with self.assertRaises(RuntimeError):
            await self.node._async_collect_results(
                "job-unsupported",
                num_workers=1,
                mode="unsupported",
            )

    async def test_mark_image_completed_updates_job_state(self):
        job_data = job_models_module.ImageJobState("job-mark")
        self.prompt_server.distributed_pending_tile_jobs["job-mark"] = job_data

        await self.node._mark_image_completed("job-mark", 7, "img-7")

        self.assertEqual(job_data.completed_images[7], "img-7")

    async def test_wrapper_methods_delegate_to_async_collect_results(self):
        node = _WrapperNode()

        tile_result = await node._async_collect_worker_tiles("job-wrap", 3)
        image_result = await node._async_collect_dynamic_images(
            "job-wrap",
            remaining_to_collect=2,
            num_workers=4,
            batch_size=8,
            master_processed_count=0,
        )

        self.assertEqual(tile_result["mode"], "static")
        self.assertEqual(image_result["mode"], "dynamic")
        self.assertEqual(
            node.calls,
            [
                ("job-wrap", 3, "static", None, None),
                ("job-wrap", 4, "dynamic", 2, 8),
            ],
        )


if __name__ == "__main__":
    unittest.main()
