import importlib.util
import sys
import types
import unittest
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

    modes_pkg = types.ModuleType(f"{package_name}.upscale.modes")
    modes_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale.modes"] = modes_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    worker_ids_module = types.ModuleType(f"{package_name}.utils.worker_ids")
    worker_ids_module.coerce_enabled_worker_ids = lambda value: (
        [str(v) for v in value]
        if isinstance(value, list)
        else []
    )
    sys.modules[f"{package_name}.utils.worker_ids"] = worker_ids_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda coro, timeout=None: (_ for _ in ()).throw(
        RuntimeError(f"run_async_in_server_loop should not be used in these tests (timeout={timeout})")
    )
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.ensure_tile_jobs_initialized = lambda: None
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    tile_ops_module = types.ModuleType(f"{package_name}.upscale.tile_ops")

    class _TileOpsMixin:
        def round_to_multiple(self, value):
            return int(value)

        def calculate_tiles(self, width, height, tile_width, tile_height, force_uniform_tiles):
            _ = force_uniform_tiles
            tiles = []
            for y in range(0, int(height), max(1, int(tile_height))):
                for x in range(0, int(width), max(1, int(tile_width))):
                    tiles.append((x, y))
            return tiles or [(0, 0)]

    tile_ops_module.TileOpsMixin = _TileOpsMixin
    sys.modules[f"{package_name}.upscale.tile_ops"] = tile_ops_module

    result_collector_module = types.ModuleType(f"{package_name}.upscale.result_collector")
    result_collector_module.ResultCollectorMixin = type("ResultCollectorMixin", (), {})
    sys.modules[f"{package_name}.upscale.result_collector"] = result_collector_module

    worker_comms_module = types.ModuleType(f"{package_name}.upscale.worker_comms")
    worker_comms_module.WorkerCommsMixin = type("WorkerCommsMixin", (), {})
    sys.modules[f"{package_name}.upscale.worker_comms"] = worker_comms_module

    job_state_module = types.ModuleType(f"{package_name}.upscale.job_state")
    job_state_module.JobStateMixin = type("JobStateMixin", (), {})
    sys.modules[f"{package_name}.upscale.job_state"] = job_state_module

    single_gpu_module = types.ModuleType(f"{package_name}.upscale.modes.single_gpu")

    class _SingleGpuModeMixin:
        def process_single_gpu(self, *_args, **_kwargs):
            return ("single_gpu_result",)

    single_gpu_module.SingleGpuModeMixin = _SingleGpuModeMixin
    sys.modules[f"{package_name}.upscale.modes.single_gpu"] = single_gpu_module

    static_mode_module = types.ModuleType(f"{package_name}.upscale.modes.static")

    class _StaticModeMixin:
        def _process_worker_static_sync(self, *_args, **_kwargs):
            return ("worker_static_result",)

        def _process_master_static_sync(self, *_args, **_kwargs):
            return ("master_static_result",)

    static_mode_module.StaticModeMixin = _StaticModeMixin
    sys.modules[f"{package_name}.upscale.modes.static"] = static_mode_module

    dynamic_mode_module = types.ModuleType(f"{package_name}.upscale.modes.dynamic")

    class _DynamicModeMixin:
        def process_worker_dynamic(self, *_args, **_kwargs):
            return ("worker_dynamic_result",)

        def process_master_dynamic(self, *_args, **_kwargs):
            return ("master_dynamic_result",)

    dynamic_mode_module.DynamicModeMixin = _DynamicModeMixin
    sys.modules[f"{package_name}.upscale.modes.dynamic"] = dynamic_mode_module

    comfy_module = types.ModuleType("comfy")
    samplers_module = types.ModuleType("comfy.samplers")

    class _KSampler:
        SAMPLERS = ("euler",)
        SCHEDULERS = ("normal",)

    samplers_module.KSampler = _KSampler
    comfy_module.samplers = samplers_module
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.samplers"] = samplers_module


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


def _load_distributed_upscale_module():
    package_name = "dist_distributed_upscale_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "nodes/hidden_inputs.py", "nodes.hidden_inputs")
    _load_module(package_name, "upscale/mode_contexts.py", "upscale.mode_contexts")
    _load_module(package_name, "upscale/processing_args.py", "upscale.processing_args")
    return _load_module(package_name, "nodes/distributed_upscale.py", "nodes.distributed_upscale")


upscale_module = _load_distributed_upscale_module()


class DistributedUpscaleTests(unittest.TestCase):
    def setUp(self):
        self.node = upscale_module.UltimateSDUpscaleDistributed()
        self.base_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        self.common_kwargs = {
            "model": object(),
            "positive": object(),
            "negative": object(),
            "vae": object(),
            "seed": 1,
            "steps": 10,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.5,
            "tile_width": 32,
            "tile_height": 32,
            "padding": 16,
            "mask_blur": 4,
            "force_uniform_tiles": True,
            "tiled_decode": False,
        }

    def test_parse_enabled_worker_ids_supports_json_and_lists(self):
        self.assertEqual(
            upscale_module._parse_enabled_worker_ids(["worker-a", 2]),
            ["worker-a", "2"],
        )
        self.assertEqual(
            upscale_module._parse_enabled_worker_ids('["worker-a","worker-b"]'),
            ["worker-a", "worker-b"],
        )
        self.assertEqual(upscale_module._parse_enabled_worker_ids("invalid-json"), [])
        self.assertEqual(upscale_module._parse_enabled_worker_ids(None), [])

    def test_determine_processing_mode(self):
        self.assertEqual(self.node._determine_processing_mode(batch_size=1, num_workers=0, dynamic_threshold=8), "single_gpu")
        self.assertEqual(self.node._determine_processing_mode(batch_size=9, num_workers=2, dynamic_threshold=8), "static")

    def test_run_raises_for_non_4n_plus_1_master_batches(self):
        bad_batch = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.node.run(
                upscaled_image=bad_batch,
                multi_job_id="job-1",
                is_worker=False,
                **self.common_kwargs,
            )

    def test_run_dispatches_single_gpu_when_not_distributed(self):
        self.node.process_single_gpu = lambda *_args, **_kwargs: ("single",)
        result = self.node.run(
            upscaled_image=self.base_image,
            multi_job_id="",
            is_worker=False,
            **self.common_kwargs,
        )
        self.assertEqual(result, ("single",))

    def test_run_dispatches_worker_or_master_paths(self):
        self.node.process_worker = lambda *_args, **_kwargs: ("worker",)
        self.node.process_master = lambda *_args, **_kwargs: ("master",)

        worker_result = self.node.run(
            upscaled_image=self.base_image,
            multi_job_id="job-2",
            is_worker=True,
            master_url="http://master.local:8188",
            enabled_worker_ids='["worker-a"]',
            worker_id="worker-a",
            tile_indices="",
            dynamic_threshold=8,
            **self.common_kwargs,
        )
        master_result = self.node.run(
            upscaled_image=self.base_image,
            multi_job_id="job-2",
            is_worker=False,
            enabled_worker_ids='["worker-a"]',
            worker_id="",
            tile_indices="",
            dynamic_threshold=8,
            **self.common_kwargs,
        )

        self.assertEqual(worker_result, ("worker",))
        self.assertEqual(master_result, ("master",))


if __name__ == "__main__":
    unittest.main()
