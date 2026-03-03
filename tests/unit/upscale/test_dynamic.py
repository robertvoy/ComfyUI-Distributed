import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency in CI/runtime
    torch = None


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

    modes_pkg = types.ModuleType(f"{package_name}.upscale.modes")
    modes_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale.modes"] = modes_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    image_module = types.ModuleType(f"{package_name}.utils.image")

    def tensor_to_pil(image_batch, index):
        tensor = image_batch[index].detach().cpu().clamp(0, 1)
        array = (tensor.numpy() * 255.0).astype("uint8")
        return Image.fromarray(array)

    def pil_to_tensor(image):
        if torch is None:
            raise RuntimeError("torch is required for this test helper")
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    image_module.tensor_to_pil = tensor_to_pil
    image_module.pil_to_tensor = pil_to_tensor
    sys.modules[f"{package_name}.utils.image"] = image_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda _coro, timeout=None: (_ for _ in ()).throw(
        RuntimeError(f"run_async_in_server_loop unexpectedly called with timeout={timeout}")
    )
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.get_worker_timeout_seconds = lambda: 5.0
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.TILE_WAIT_TIMEOUT = 5.0
    constants_module.TILE_SEND_TIMEOUT = 5.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    if torch is None:
        torch_module = types.ModuleType("torch")
        torch_module.cat = lambda *_args, **_kwargs: None
        sys.modules["torch"] = torch_module

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.ensure_tile_jobs_initialized = lambda: None

    async def init_dynamic_job(*_args, **_kwargs):
        return None

    job_store_module.init_dynamic_job = init_dynamic_job
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    comfy_module = types.ModuleType("comfy")
    model_mgmt = types.ModuleType("comfy.model_management")
    model_mgmt.throw_exception_if_processing_interrupted = lambda: None
    comfy_module.model_management = model_mgmt
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_mgmt


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


def _load_dynamic_mode_module():
    package_name = "dist_dynamic_mode_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "upscale/processing_args.py", "upscale.processing_args")
    return _load_module(package_name, "upscale/modes/dynamic.py", "upscale.modes.dynamic")


dynamic_module = _load_dynamic_mode_module()


def _run_sync(value, timeout=None):
    _ = timeout
    if asyncio.iscoroutine(value):
        return asyncio.run(value)
    return value


dynamic_module.run_async_in_server_loop = _run_sync


class _DummyDynamicNode(dynamic_module.DynamicModeMixin):
    def __init__(self, *, poll_ready=True, assignments=None):
        self._poll_ready = poll_ready
        self._assignments = list(assignments or [])
        self.sent_images = []
        self.heartbeat_count = 0
        self.completion_sent = False

    def round_to_multiple(self, value):
        return int(value)

    def calculate_tiles(self, _width, _height, _tile_width, _tile_height, _force_uniform_tiles):
        return [(0, 0)]

    def _poll_job_ready(self, *_args, **_kwargs):
        return self._poll_ready

    async def _request_image_from_master(self, *_args, **_kwargs):
        if self._assignments:
            return self._assignments.pop(0)
        return (None, 0)

    def _slice_conditioning(self, positive, negative, _image_idx):
        return positive, negative

    def _process_and_blend_tile(self, _tile_idx, _pos, _source_tensor, local_image, *_args, **_kwargs):
        return local_image

    async def _send_heartbeat_to_master(self, *_args, **_kwargs):
        self.heartbeat_count += 1

    async def _send_full_image_to_master(self, local_image, image_idx, _multi_job_id, _master_url, _worker_id, is_last):
        self.sent_images.append((image_idx, is_last, local_image.size))

    async def _send_worker_complete_signal(self, *_args, **_kwargs):
        self.completion_sent = True


@unittest.skipIf(torch is None, "torch is not installed")
class DynamicModeTests(unittest.TestCase):
    def setUp(self):
        self.image_batch = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        self.core_args = dynamic_module.UpscaleCoreArgs(
            model=None,
            positive=None,
            negative=None,
            vae=None,
            seed=1,
            steps=10,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=0.5,
            tiled_decode=False,
        )

    def test_worker_dynamic_returns_early_if_job_not_ready(self):
        node = _DummyDynamicNode(poll_ready=False)

        result = node.process_worker_dynamic(
            upscaled_image=self.image_batch,
            core_args=self.core_args,
            tile_width=8,
            tile_height=8,
            padding=4,
            mask_blur=2,
            force_uniform_tiles=True,
            multi_job_id="job-1",
            master_url="http://master.local:8188",
            worker_id="worker-a",
            enabled_worker_ids='["worker-a"]',
            dynamic_threshold=8,
        )

        self.assertEqual(result[0].shape, self.image_batch.shape)
        self.assertEqual(node.sent_images, [])
        self.assertFalse(node.completion_sent)

    def test_worker_dynamic_processes_assigned_image_and_sends_completion(self):
        node = _DummyDynamicNode(poll_ready=True, assignments=[(0, 0)])

        result = node.process_worker_dynamic(
            upscaled_image=self.image_batch,
            core_args=self.core_args,
            tile_width=8,
            tile_height=8,
            padding=4,
            mask_blur=2,
            force_uniform_tiles=True,
            multi_job_id="job-2",
            master_url="http://master.local:8188",
            worker_id="worker-a",
            enabled_worker_ids='["worker-a"]',
            dynamic_threshold=8,
        )

        self.assertEqual(result[0].shape, self.image_batch.shape)
        self.assertEqual(len(node.sent_images), 1)
        self.assertEqual(node.sent_images[0][0], 0)
        self.assertTrue(node.sent_images[0][1])
        self.assertTrue(node.completion_sent)
        self.assertGreaterEqual(node.heartbeat_count, 1)


if __name__ == "__main__":
    unittest.main()
