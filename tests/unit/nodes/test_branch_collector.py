import asyncio
import base64
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path

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

    def encode_tensor_png_data_url(image_batch, batch_index=0):
        image = tensor_to_pil(image_batch, batch_index)
        byte_io = io.BytesIO()
        image.save(byte_io, format="PNG", compress_level=0)
        encoded = base64.b64encode(byte_io.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    image_module.ensure_contiguous = ensure_contiguous
    image_module.tensor_to_pil = tensor_to_pil
    image_module.encode_tensor_png_data_url = encode_tensor_png_data_url
    sys.modules[f"{package_name}.utils.image"] = image_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def get_client_session():
        raise RuntimeError("test should patch get_client_session")

    network_module.get_client_session = get_client_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda coro: asyncio.run(coro)
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module


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


def _load_branch_collector_module():
    package_name = "dist_join_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "nodes/utilities.py", "nodes.utilities")
    _load_module(package_name, "nodes/hidden_inputs.py", "nodes.hidden_inputs")
    _load_module(package_name, "nodes/runtime_helpers.py", "nodes.runtime_helpers")
    _load_module(package_name, "nodes/queue_wait.py", "nodes.queue_wait")
    _load_module(package_name, "utils/worker_ids.py", "utils.worker_ids")
    return _load_module(package_name, "nodes/branch_collector.py", "nodes.branch_collector")


branch_collector_module = _load_branch_collector_module()


def _context(**overrides):
    defaults = {
        "multi_job_id": "",
        "is_worker": False,
        "master_url": "",
        "enabled_worker_ids": "[]",
        "worker_id": "",
        "assigned_branch": -1,
        "delegate_only": False,
    }
    defaults.update(overrides)
    return branch_collector_module.BranchRunContext(**defaults)


class _FakeResponseCtx:
    def __init__(self, recorder, payload, url):
        self._recorder = recorder
        self._payload = payload
        self._url = url

    async def __aenter__(self):
        self._recorder.append({"url": self._url, "payload": self._payload})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        return _FakeResponseCtx(self.calls, json, url)


class _FakePromptServer:
    def __init__(self):
        self.distributed_pending_jobs = {}
        self.distributed_jobs_lock = asyncio.Lock()


class DistributedBranchCollectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.node = branch_collector_module.DistributedBranchCollector()

    async def test_worker_mode_sends_result_with_assigned_branch_index(self):
        fake_session = _FakeSession()

        async def _get_session():
            return fake_session

        branch_collector_module.get_client_session = _get_session

        tensor = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        outputs = await self.node.execute(
            [None, tensor, None, None, None, None, None, None, None, None],
            num_branches=2,
            context=_context(
                multi_job_id="join-job",
                is_worker=True,
                master_url="http://master.local:8188",
                worker_id="worker-a",
                assigned_branch=1,
            ),
        )

        self.assertEqual(len(fake_session.calls), 1)
        payload = fake_session.calls[0]["payload"]
        self.assertEqual(payload["batch_idx"], 1)
        self.assertTrue(payload["is_last"])
        self.assertEqual(payload["worker_id"], "worker-a")
        self.assertTrue(torch.equal(outputs[1], tensor))

    async def test_master_mode_collects_worker_results_into_branch_slots(self):
        fake_server = _FakePromptServer()
        branch_collector_module._get_prompt_server_instance = lambda: fake_server

        queue = asyncio.Queue()
        worker_1 = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
        worker_2 = torch.full((1, 4, 4, 3), 0.8, dtype=torch.float32)
        await queue.put({"tensor": worker_1, "worker_id": "worker-a", "image_index": 1, "is_last": True})
        await queue.put({"tensor": worker_2, "worker_id": "worker-b", "image_index": 2, "is_last": True})
        fake_server.distributed_pending_jobs["join-job"] = queue

        master_tensor = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        outputs = await self.node.execute(
            [master_tensor, None, None, None, None, None, None, None, None, None],
            num_branches=3,
            context=_context(
                multi_job_id="join-job",
                enabled_worker_ids=json.dumps(["worker-a", "worker-b"]),
                assigned_branch=0,
            ),
        )

        self.assertTrue(torch.equal(outputs[0], master_tensor))
        self.assertTrue(torch.equal(outputs[1], worker_1))
        self.assertTrue(torch.equal(outputs[2], worker_2))

    async def test_unassigned_slots_remain_none(self):
        fake_server = _FakePromptServer()
        branch_collector_module._get_prompt_server_instance = lambda: fake_server

        queue = asyncio.Queue()
        worker_1 = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
        await queue.put({"tensor": worker_1, "worker_id": "worker-a", "image_index": 1, "is_last": True})
        fake_server.distributed_pending_jobs["join-job"] = queue

        master_tensor = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        outputs = await self.node.execute(
            [master_tensor, None, None, None, None, None, None, None, None, None],
            num_branches=2,
            context=_context(
                multi_job_id="join-job",
                enabled_worker_ids=json.dumps(["worker-a"]),
                assigned_branch=0,
            ),
        )

        self.assertIsNone(outputs[2])
        self.assertIsNone(outputs[9])

    async def test_missing_worker_results_use_safe_fallback_for_expected_slots(self):
        fake_server = _FakePromptServer()
        branch_collector_module._get_prompt_server_instance = lambda: fake_server

        queue = asyncio.Queue()
        # No worker payloads enqueued; master should timeout and fill expected worker slots.
        fake_server.distributed_pending_jobs["join-job"] = queue

        master_tensor = torch.full((1, 4, 4, 3), 0.7, dtype=torch.float32)
        outputs = await self.node.execute(
            [master_tensor, None, None, None, None, None, None, None, None, None],
            num_branches=3,
            context=_context(
                multi_job_id="join-job",
                enabled_worker_ids=json.dumps(["worker-a", "worker-b"]),
                assigned_branch=0,
            ),
        )

        self.assertTrue(torch.equal(outputs[0], master_tensor))
        self.assertIsInstance(outputs[1], torch.Tensor)
        self.assertIsInstance(outputs[2], torch.Tensor)
        self.assertEqual(float(outputs[1].abs().sum().item()), 0.0)
        self.assertEqual(float(outputs[2].abs().sum().item()), 0.0)

    async def test_assigned_slot_can_be_filled_from_single_mismatched_local_input(self):
        fake_server = _FakePromptServer()
        branch_collector_module._get_prompt_server_instance = lambda: fake_server

        # No worker results expected for this check.
        fake_server.distributed_pending_jobs["join-job"] = asyncio.Queue()

        local_tensor = torch.full((1, 4, 4, 3), 0.33, dtype=torch.float32)
        outputs = await self.node.execute(
            # Local value arrives on slot 1 even though participant is assigned slot 0.
            [None, local_tensor, None, None, None, None, None, None, None, None],
            num_branches=3,
            context=_context(
                multi_job_id="join-job",
                enabled_worker_ids=json.dumps([]),
                assigned_branch=0,
            ),
        )

        self.assertTrue(torch.equal(outputs[0], local_tensor))


if __name__ == "__main__":
    unittest.main()
