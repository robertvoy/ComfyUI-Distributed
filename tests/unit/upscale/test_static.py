import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_static_mode_module():
    module_path = Path(__file__).resolve().parents[3] / "upscale" / "modes" / "static.py"
    package_name = "dist_static_mode_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    modes_pkg = types.ModuleType(f"{package_name}.upscale.modes")
    modes_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale.modes"] = modes_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    created_comfy_stub = False
    if "comfy" not in sys.modules:
        created_comfy_stub = True
        comfy_module = types.ModuleType("comfy")
        model_mgmt = types.ModuleType("comfy.model_management")

        class _InterruptProcessingException(Exception):
            pass

        model_mgmt.processing_interrupted = lambda: False
        model_mgmt.throw_exception_if_processing_interrupted = lambda: None
        model_mgmt.InterruptProcessingException = _InterruptProcessingException

        comfy_module.model_management = model_mgmt
        sys.modules["comfy"] = comfy_module
        sys.modules["comfy.model_management"] = model_mgmt

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    image_module = types.ModuleType(f"{package_name}.utils.image")
    from PIL import Image as PILImage
    import numpy as np

    def _tensor_to_pil(img_tensor, batch_index=0):
        return PILImage.fromarray((255 * img_tensor[batch_index].cpu().numpy()).astype(np.uint8))

    def _pil_to_tensor(image):
        arr = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _blend_processed_batch_item(
        result_images,
        processed_batch,
        batch_index,
        blend_fn,
        x1,
        y1,
        ew,
        eh,
        tile_mask,
        padding,
    ):
        tile_pil = _tensor_to_pil(processed_batch, batch_index)
        if tile_pil.size != (ew, eh):
            tile_pil = tile_pil.resize((ew, eh), PILImage.LANCZOS)
        result_images[batch_index] = blend_fn(
            result_images[batch_index],
            tile_pil,
            x1,
            y1,
            (ew, eh),
            tile_mask,
            padding,
        )

    image_module.tensor_to_pil = _tensor_to_pil
    image_module.pil_to_tensor = _pil_to_tensor
    image_module.blend_processed_batch_item = _blend_processed_batch_item
    sys.modules[f"{package_name}.utils.image"] = image_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")

    def _run_async_in_server_loop(coro, timeout=None):
        if timeout is not None:
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        return asyncio.run(coro)

    async_helpers_module.run_async_in_server_loop = _run_async_in_server_loop
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.get_worker_timeout_seconds = lambda: 60
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.HEARTBEAT_INTERVAL = 10.0
    constants_module.JOB_POLL_INTERVAL = 0.0
    constants_module.JOB_POLL_MAX_ATTEMPTS = 3
    constants_module.MAX_BATCH = 20
    constants_module.TILE_SEND_TIMEOUT = 1.0
    constants_module.TILE_WAIT_TIMEOUT = 1.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")

    async def _noop(*_args, **_kwargs):
        return None

    job_store_module.ensure_tile_jobs_initialized = lambda: types.SimpleNamespace(
        distributed_tile_jobs_lock=asyncio.Lock(),
        distributed_pending_tile_jobs={},
    )
    job_store_module.init_static_job_batched = _noop
    job_store_module.mark_task_completed = _noop
    job_store_module.cleanup_job = _noop
    job_store_module.drain_results_queue = _noop
    job_store_module.get_completed_count = _noop
    job_store_module._mark_task_completed = _noop
    job_store_module._cleanup_job = _noop
    job_store_module._drain_results_queue = _noop
    job_store_module._get_completed_count = _noop
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    job_models_module = types.ModuleType(f"{package_name}.upscale.job_models")

    class _TileJobState:
        pass

    job_models_module.TileJobState = _TileJobState
    sys.modules[f"{package_name}.upscale.job_models"] = job_models_module

    processing_args_path = Path(__file__).resolve().parents[3] / "upscale" / "processing_args.py"
    processing_args_spec = importlib.util.spec_from_file_location(
        f"{package_name}.upscale.processing_args",
        processing_args_path,
    )
    processing_args_module = importlib.util.module_from_spec(processing_args_spec)
    assert processing_args_spec is not None and processing_args_spec.loader is not None
    sys.modules[processing_args_spec.name] = processing_args_module
    processing_args_spec.loader.exec_module(processing_args_module)

    tile_processing_path = Path(__file__).resolve().parents[3] / "upscale" / "tile_processing.py"
    tile_processing_spec = importlib.util.spec_from_file_location(
        f"{package_name}.upscale.tile_processing",
        tile_processing_path,
    )
    tile_processing_module = importlib.util.module_from_spec(tile_processing_spec)
    assert tile_processing_spec is not None and tile_processing_spec.loader is not None
    sys.modules[tile_processing_spec.name] = tile_processing_module
    tile_processing_spec.loader.exec_module(tile_processing_module)

    mode_contexts_path = Path(__file__).resolve().parents[3] / "upscale" / "mode_contexts.py"
    mode_contexts_spec = importlib.util.spec_from_file_location(
        f"{package_name}.upscale.mode_contexts",
        mode_contexts_path,
    )
    mode_contexts_module = importlib.util.module_from_spec(mode_contexts_spec)
    assert mode_contexts_spec is not None and mode_contexts_spec.loader is not None
    sys.modules[mode_contexts_spec.name] = mode_contexts_module
    mode_contexts_spec.loader.exec_module(mode_contexts_module)

    spec = importlib.util.spec_from_file_location(f"{package_name}.upscale.modes.static", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    if created_comfy_stub:
        sys.modules.pop("comfy.model_management", None)
        sys.modules.pop("comfy", None)

    return module


static_mode = _load_static_mode_module()


class _FakeStaticWorker(static_mode.StaticModeMixin):
    def __init__(self):
        self.sent_batches = []
        self.request_calls = 0
        self.heartbeat_calls = 0
        self.job_ready = True
        self.tile_sequence = [(0, 0, True), (None, 0, True)]

    def round_to_multiple(self, value):
        return value

    def calculate_tiles(self, _width, _height, _tile_width, _tile_height, _force_uniform_tiles):
        return [(0, 0)]

    def _poll_job_ready(self, *_args, **_kwargs):
        return self.job_ready

    async def request_assignment(self, *_args, **_kwargs):
        self.request_calls += 1
        tile_idx, estimated_remaining, batched_static = self.tile_sequence.pop(0)
        return SimpleNamespace(
            kind="tile" if tile_idx is not None else "none",
            task_idx=tile_idx,
            estimated_remaining=estimated_remaining,
            batched_static=batched_static,
        )

    async def send_heartbeat(self, *_args, **_kwargs):
        self.heartbeat_calls += 1

    async def send_tiles_batch(
        self,
        processed_tiles,
        _multi_job_id,
        _master_url,
        _padding,
        _worker_id,
        is_final_flush=False,
    ):
        self.sent_batches.append(
            {
                "tiles": list(processed_tiles),
                "is_final_flush": bool(is_final_flush),
            }
        )

    def _extract_and_process_tile(self, upscaled_image, *_args, **_kwargs):
        batch_size = upscaled_image.shape[0]
        processed_batch = torch.zeros((batch_size, 2, 2, 3), dtype=torch.float32)
        return processed_batch, 0, 0, 2, 2

    def create_tile_mask(self, *_args, **_kwargs):
        from PIL import Image
        return Image.new("L", (4, 4), 255)

    def blend_tile(self, base_image, *_args, **_kwargs):
        return base_image


def _call_worker_static(fake_worker):
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    return fake_worker._process_worker_static_sync(
        image,
        model=None,
        positive=None,
        negative=None,
        vae=None,
        seed=1,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.5,
        tile_width=4,
        tile_height=4,
        padding=8,
        mask_blur=4,
        force_uniform_tiles=True,
        tiled_decode=False,
        multi_job_id="job-1",
        master_url="http://master:8188",
        worker_id="worker-1",
        enabled_workers=["worker-1"],
    )


class StaticModeWorkerFlowTests(unittest.TestCase):
    def test_worker_static_aborts_when_job_not_ready(self):
        worker = _FakeStaticWorker()
        worker.job_ready = False

        result = _call_worker_static(worker)

        self.assertEqual(result[0].shape[0], 1)
        self.assertEqual(worker.request_calls, 0)
        self.assertEqual(worker.heartbeat_calls, 0)
        self.assertEqual(worker.sent_batches, [])

    def test_worker_static_requests_tiles_and_flushes_final_batch(self):
        worker = _FakeStaticWorker()

        _call_worker_static(worker)

        self.assertEqual(worker.request_calls, 2)  # one tile, then sentinel
        self.assertEqual(worker.heartbeat_calls, 1)
        self.assertEqual(len(worker.sent_batches), 1)
        self.assertTrue(worker.sent_batches[0]["is_final_flush"])
        tiles = worker.sent_batches[0]["tiles"]
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0]["tile_idx"], 0)
        self.assertEqual(tiles[0]["global_idx"], 0)
        self.assertEqual(tiles[0]["batch_idx"], 0)

    def test_flush_empty_final_still_sends_completion_signal(self):
        worker = _FakeStaticWorker()

        returned = worker._flush_tiles_to_master(
            [],
            "job-1",
            "http://master:8188",
            8,
            "worker-1",
            is_final_flush=True,
        )

        self.assertEqual(returned, [])
        self.assertEqual(len(worker.sent_batches), 1)
        self.assertEqual(worker.sent_batches[0]["tiles"], [])
        self.assertTrue(worker.sent_batches[0]["is_final_flush"])


if __name__ == "__main__":
    unittest.main()
