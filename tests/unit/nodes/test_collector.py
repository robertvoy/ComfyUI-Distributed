import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
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
    config_module.load_config = lambda: {"workers": []}
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

    def pil_to_tensor(image):
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    image_module.ensure_contiguous = ensure_contiguous
    image_module.tensor_to_pil = tensor_to_pil
    image_module.pil_to_tensor = pil_to_tensor
    sys.modules[f"{package_name}.utils.image"] = image_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_worker_url = lambda _worker: "http://worker.local:8188"
    network_module.get_client_session = lambda: None

    async def probe_worker(_url, timeout=2.0):
        _ = timeout
        return None

    network_module.probe_worker = probe_worker
    sys.modules[f"{package_name}.utils.network"] = network_module

    audio_payload_module = types.ModuleType(f"{package_name}.utils.audio_payload")
    audio_payload_module.encode_audio_payload = lambda payload: payload
    sys.modules[f"{package_name}.utils.audio_payload"] = audio_payload_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda coro: asyncio.run(coro)
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    if "aiohttp" not in sys.modules:
        try:
            import aiohttp as _aiohttp  # noqa: F401
        except Exception:
            aiohttp_module = types.ModuleType("aiohttp")

            class _ClientTimeout:
                def __init__(self, total=None):
                    self.total = total

            aiohttp_module.ClientTimeout = _ClientTimeout
            sys.modules["aiohttp"] = aiohttp_module

    class _FakePromptServer:
        def __init__(self):
            self.distributed_pending_jobs = {}
            self.distributed_jobs_lock = asyncio.Lock()

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=_FakePromptServer())
    sys.modules["server"] = server_module

    comfy_module = types.ModuleType("comfy")
    model_mgmt = types.ModuleType("comfy.model_management")

    class _InterruptProcessingException(Exception):
        pass

    model_mgmt.throw_exception_if_processing_interrupted = lambda: None
    model_mgmt.InterruptProcessingException = _InterruptProcessingException
    comfy_module.model_management = model_mgmt
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_mgmt

    comfy_utils_module = types.ModuleType("comfy.utils")

    class _ProgressBar:
        def __init__(self, _total):
            self.total = _total

        def update(self, _step):
            return None

    comfy_utils_module.ProgressBar = _ProgressBar
    sys.modules["comfy.utils"] = comfy_utils_module


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


def _load_collector_module():
    package_name = "dist_collector_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "nodes/hidden_inputs.py", "nodes.hidden_inputs")
    _load_module(package_name, "utils/worker_ids.py", "utils.worker_ids")
    return _load_module(package_name, "nodes/collector.py", "nodes.collector")


collector_module = _load_collector_module()


class DistributedCollectorTests(unittest.TestCase):
    def setUp(self):
        self.node = collector_module.DistributedCollectorNode()

    def test_store_worker_result_tracks_by_worker_and_index(self):
        worker_images = {}
        stored = self.node._store_worker_result(
            worker_images,
            {
                "worker_id": "worker-a",
                "image_index": 2,
                "tensor": torch.full((1, 1, 1, 1), 0.25),
            },
        )

        self.assertEqual(stored, 1)
        self.assertIn("worker-a", worker_images)
        self.assertIn(2, worker_images["worker-a"])

    def test_combine_audio_honors_master_then_worker_order(self):
        master_audio = {"waveform": torch.ones((1, 2, 1)), "sample_rate": 48000}
        worker_audio = {
            "worker-b": {"waveform": torch.full((1, 2, 1), 3.0), "sample_rate": 48000},
            "worker-a": {"waveform": torch.full((1, 2, 2), 2.0), "sample_rate": 48000},
        }
        empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100}

        combined = self.node._combine_audio(
            master_audio=master_audio,
            worker_audio=worker_audio,
            empty_audio=empty_audio,
            worker_order=["worker-a", "worker-b"],
        )

        expected = torch.cat(
            [
                master_audio["waveform"],
                worker_audio["worker-a"]["waveform"],
                worker_audio["worker-b"]["waveform"],
            ],
            dim=-1,
        )
        self.assertEqual(combined["sample_rate"], 48000)
        self.assertTrue(torch.equal(combined["waveform"], expected))

    def test_reorder_and_combine_tensors_uses_enabled_worker_priority(self):
        worker_images = {
            "worker-b": {
                1: torch.full((1, 1, 1, 1), 4.0),
                0: torch.full((1, 1, 1, 1), 3.0),
            },
            "worker-a": {
                0: torch.full((1, 1, 1, 1), 2.0),
            },
        }
        master_images = torch.full((1, 1, 1, 1), 1.0)

        combined = self.node._reorder_and_combine_tensors(
            worker_images=worker_images,
            worker_order=["worker-a", "worker-b"],
            master_batch_size=1,
            images_on_cpu=master_images,
            delegate_mode=False,
            fallback_images=None,
        )

        self.assertEqual(combined.shape, (4, 1, 1, 1))
        self.assertEqual(combined[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(combined[1, 0, 0, 0].item(), 2.0)
        self.assertEqual(combined[2, 0, 0, 0].item(), 3.0)
        self.assertEqual(combined[3, 0, 0, 0].item(), 4.0)


if __name__ == "__main__":
    unittest.main()
