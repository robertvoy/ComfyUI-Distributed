import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def _load_static_mode_module():
    module_path = Path(__file__).resolve().parents[1] / "upscale" / "modes" / "static.py"
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

    image_module.tensor_to_pil = _tensor_to_pil
    image_module.pil_to_tensor = _pil_to_tensor
    sys.modules[f"{package_name}.utils.image"] = image_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")

    def _run_async_in_server_loop(coro, timeout=None, *, cancel_on_timeout=True):
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

    async def _request_tile_from_master(self, *_args, **_kwargs):
        self.request_calls += 1
        return self.tile_sequence.pop(0)

    async def _send_heartbeat_to_master(self, *_args, **_kwargs):
        self.heartbeat_calls += 1

    async def send_tiles_batch_to_master(
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


class _MasterInterrupted(Exception):
    """Stands in for comfy's InterruptProcessingException in these tests."""


class _FakeStaticMaster(static_mode.StaticModeMixin):
    def __init__(self, tile_sequence, collected=None, fail_on_tile_call=None):
        self.tile_sequence = list(tile_sequence)
        self.collected = dict(collected or {})
        self.fail_on_tile_call = fail_on_tile_call
        self.tile_calls = 0
        self.collect_calls = 0

    async def _get_next_tile_index(self, _multi_job_id):
        return self.tile_sequence.pop(0) if self.tile_sequence else None

    async def _get_all_completed_tasks(self, _multi_job_id):
        return dict(self.collected)

    async def _async_collect_and_monitor_static(self, *_args, **_kwargs):
        self.collect_calls += 1
        return dict(self.collected)

    def _master_process_one_tile(self, *_args, **_kwargs):
        self.tile_calls += 1
        if self.fail_on_tile_call == self.tile_calls:
            raise RuntimeError("tile processing failed")
        return 1  # batch_size == 1

    def create_tile_mask(self, *_args, **_kwargs):
        from PIL import Image
        return Image.new("L", (4, 4), 255)

    def blend_tile(self, base_image, *_args, **_kwargs):
        return base_image


def _call_master_static(fake_master, num_tiles_per_image=2):
    image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    return fake_master._process_master_static_sync(
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
        enabled_workers=["worker-1"],
        all_tiles=[(0, 0)] * num_tiles_per_image,
        num_tiles_per_image=num_tiles_per_image,
    )


class StaticModeMasterCleanupTests(unittest.TestCase):
    """The master must drop its job state on every exit once the job is initialised."""

    def setUp(self):
        self.cleaned = []
        self.initialised = []

        async def _record_cleanup(multi_job_id):
            self.cleaned.append(multi_job_id)

        async def _record_init(multi_job_id, *_args, **_kwargs):
            self.initialised.append(multi_job_id)

        model_management = static_mode.comfy.model_management
        patchers = [
            patch.object(static_mode, "_cleanup_job", _record_cleanup),
            patch.object(static_mode, "init_static_job_batched", _record_init),
            patch.object(
                model_management,
                "throw_exception_if_processing_interrupted",
                lambda: None,
                create=True,
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _interrupt_on_check(self, check_number):
        calls = {"count": 0}

        def _check():
            calls["count"] += 1
            if calls["count"] >= check_number:
                raise _MasterInterrupted()

        return patch.object(
            static_mode.comfy.model_management,
            "throw_exception_if_processing_interrupted",
            _check,
            create=True,
        )

    def test_interrupt_during_master_tile_loop_cleans_up_job_once(self):
        master = _FakeStaticMaster(tile_sequence=[0, 1])

        # First check passes (tile 0 is rendered), the second one sees the interrupt.
        with self._interrupt_on_check(2), self.assertRaises(_MasterInterrupted):
            _call_master_static(master)

        self.assertEqual(master.tile_calls, 1)
        self.assertEqual(self.initialised, ["job-1"])
        self.assertEqual(self.cleaned, ["job-1"])

    def test_exception_during_master_tile_loop_cleans_up_job_once(self):
        master = _FakeStaticMaster(tile_sequence=[0, 1], fail_on_tile_call=1)

        with self.assertRaises(RuntimeError):
            _call_master_static(master)

        self.assertEqual(self.cleaned, ["job-1"])

    def test_interrupt_during_worker_collection_cleans_up_job_once(self):
        master = _FakeStaticMaster(tile_sequence=[None, None])

        async def _interrupted_collection(*_args, **_kwargs):
            raise _MasterInterrupted()

        master._async_collect_and_monitor_static = _interrupted_collection

        with self.assertRaises(_MasterInterrupted):
            _call_master_static(master)

        self.assertEqual(self.cleaned, ["job-1"])

    def test_exception_during_local_fallback_cleans_up_job_once(self):
        # Master gets no tiles at first, workers return nothing, then the local
        # fallback loop picks up tile 0 and fails while rendering it.
        master = _FakeStaticMaster(tile_sequence=[None, None, 0], fail_on_tile_call=1)

        with self.assertRaises(RuntimeError):
            _call_master_static(master)

        self.assertEqual(master.collect_calls, 1)
        self.assertEqual(self.cleaned, ["job-1"])

    def test_cleanup_failure_does_not_mask_interrupt(self):
        master = _FakeStaticMaster(tile_sequence=[0, 1])

        async def _failing_cleanup(multi_job_id):
            self.cleaned.append(multi_job_id)
            raise TimeoutError("cleanup timed out")

        with patch.object(static_mode, "_cleanup_job", _failing_cleanup), \
                self._interrupt_on_check(2), self.assertRaises(_MasterInterrupted):
            _call_master_static(master)

        self.assertEqual(self.cleaned, ["job-1"])

    def test_cleanup_is_never_cancelled_by_its_own_timeout(self):
        # A slow server loop must not abandon the job: it still gets removed late.
        master = _FakeStaticMaster(tile_sequence=[0, 1])
        bridge = static_mode.run_async_in_server_loop
        cleanup_calls = []

        def _recording_bridge(coro, timeout=None, **kwargs):
            if getattr(coro, "cr_code", None) is static_mode._cleanup_job.__code__:
                cleanup_calls.append(kwargs)
            return bridge(coro, timeout, **kwargs)

        with patch.object(static_mode, "run_async_in_server_loop", _recording_bridge):
            _call_master_static(master)

        self.assertEqual(cleanup_calls, [{"cancel_on_timeout": False}])
        self.assertEqual(self.cleaned, ["job-1"])

    def test_tile_claims_are_cancelled_when_the_wait_times_out(self):
        # A claim the master stopped waiting for must not still pop a tile later.
        master = _FakeStaticMaster(tile_sequence=[0, 1])
        bridge = static_mode.run_async_in_server_loop
        claim_calls = []

        def _recording_bridge(coro, timeout=None, **kwargs):
            if getattr(coro, "cr_code", None) is _FakeStaticMaster._get_next_tile_index.__code__:
                claim_calls.append(kwargs)
            return bridge(coro, timeout, **kwargs)

        with patch.object(static_mode, "run_async_in_server_loop", _recording_bridge):
            _call_master_static(master)

        self.assertEqual(claim_calls, [{"cancel_on_timeout": True}] * 2)

    def test_success_path_returns_result_and_cleans_up_once(self):
        master = _FakeStaticMaster(tile_sequence=[0, 1])

        result = _call_master_static(master)

        self.assertEqual(master.tile_calls, 2)
        self.assertEqual(master.collect_calls, 0)
        self.assertEqual(tuple(result[0].shape), (1, 4, 4, 3))
        self.assertEqual(self.cleaned, ["job-1"])


if __name__ == "__main__":
    unittest.main()
