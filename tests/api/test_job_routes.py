import importlib.util
import sys
import types
import unittest
import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from tests.api.harness import (
    bootstrap_test_package,
    cleanup_optional_module,
    install_aiohttp_stub,
    install_request_guards_stub,
    install_server_stub,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeRequest:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    async def json(self):
        return self._payload


def _load_job_routes_module():
    module_path = Path(__file__).resolve().parents[2] / "api" / "job_routes.py"
    package_name = "dist_api_queue_testpkg"

    bootstrap_test_package(package_name, with_api=True, with_utils=True)

    # aiohttp.web stub
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )

    # server module stub with route decorators
    prompt_server_instance = types.SimpleNamespace(
        distributed_jobs_lock=None,
        distributed_pending_jobs={},
    )
    install_server_stub(prompt_server_instance)

    # torch stub (only needed to satisfy import)
    created_torch_stub = False
    if "torch" not in sys.modules:
        created_torch_stub = True
        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
        )
        sys.modules["torch"] = torch_module

    # PIL stub (only needed to satisfy import)
    created_pil_stub = False
    if "PIL" not in sys.modules:
        created_pil_stub = True
        pil_module = types.ModuleType("PIL")
        image_module = types.ModuleType("PIL.Image")
        pil_module.Image = image_module
        sys.modules["PIL"] = pil_module
        sys.modules["PIL.Image"] = image_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    image_module = types.ModuleType(f"{package_name}.utils.image")
    image_module.pil_to_tensor = lambda *_args, **_kwargs: None
    image_module.ensure_contiguous = lambda tensor: tensor
    sys.modules[f"{package_name}.utils.image"] = image_module

    audio_payload_module = types.ModuleType(f"{package_name}.utils.audio_payload")

    def _decode_audio_payload(payload):
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("Field 'audio' must be an object when provided.")

        encoded = payload.get("data")
        shape = payload.get("shape")
        dtype = payload.get("dtype", "float32")
        sample_rate = payload.get("sample_rate", 44100)
        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError("Field 'audio.data' must be a non-empty base64 string.")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError("Field 'audio.shape' must be a 3-item list.")
        if dtype != "float32":
            raise ValueError("Field 'audio.dtype' must be 'float32'.")
        try:
            shape_tuple = tuple(int(dim) for dim in shape)
        except Exception as exc:
            raise ValueError("Field 'audio.shape' must contain integers.") from exc

        raw = base64.b64decode(encoded, validate=True)
        expected_bytes = int(np.prod(shape_tuple, dtype=np.int64)) * 4
        if len(raw) != expected_bytes:
            raise ValueError("Field 'audio.data' byte size mismatch.")

        waveform = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(shape_tuple).copy())
        return {"waveform": waveform, "sample_rate": int(sample_rate)}

    audio_payload_module.decode_audio_payload = _decode_audio_payload
    sys.modules[f"{package_name}.utils.audio_payload"] = audio_payload_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status=status)

    network_module.handle_api_error = _handle_api_error
    sys.modules[f"{package_name}.utils.network"] = network_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.MEMORY_CLEAR_DELAY = 0.0
    constants_module.JOB_INIT_GRACE_PERIOD = 10.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: {"settings": {}}
    sys.modules[f"{package_name}.utils.config"] = config_module

    runtime_state_module = types.ModuleType(f"{package_name}.utils.runtime_state")

    def _ensure_distributed_runtime_state(server_instance=None):
        ps = server_instance or prompt_server_instance
        if not hasattr(ps, "distributed_pending_jobs"):
            ps.distributed_pending_jobs = {}
        if not hasattr(ps, "distributed_jobs_lock") or ps.distributed_jobs_lock is None:
            ps.distributed_jobs_lock = asyncio.Lock()
        if not hasattr(ps, "distributed_job_allowed_workers"):
            ps.distributed_job_allowed_workers = {}
        if not hasattr(ps, "distributed_pending_tile_jobs"):
            ps.distributed_pending_tile_jobs = {}
        if not hasattr(ps, "distributed_tile_jobs_lock"):
            ps.distributed_tile_jobs_lock = asyncio.Lock()
        return types.SimpleNamespace(
            distributed_pending_jobs=ps.distributed_pending_jobs,
            distributed_jobs_lock=ps.distributed_jobs_lock,
            distributed_job_allowed_workers=ps.distributed_job_allowed_workers,
            distributed_pending_tile_jobs=ps.distributed_pending_tile_jobs,
            distributed_tile_jobs_lock=ps.distributed_tile_jobs_lock,
        )

    runtime_state_module.ensure_distributed_runtime_state = _ensure_distributed_runtime_state
    runtime_state_module.get_prompt_server_instance = lambda: prompt_server_instance
    sys.modules[f"{package_name}.utils.runtime_state"] = runtime_state_module

    schemas_module = types.ModuleType(f"{package_name}.api.schemas")
    schemas_module.is_authorized_request = lambda _request, _config: True
    schemas_module.require_bool_literal = (
        lambda value, field_name="value": value
        if isinstance(value, bool)
        else (_ for _ in ()).throw(ValueError(f"Field '{field_name}' must be a boolean literal."))
    )
    sys.modules[f"{package_name}.api.schemas"] = schemas_module

    install_request_guards_stub(package_name)

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.queue_prompt_payload = AsyncMock(return_value="prompt_local")
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    queue_orchestration_module = types.ModuleType(f"{package_name}.api.queue_orchestration")
    queue_orchestration_module.orchestrate_distributed_execution = AsyncMock(return_value=("prompt_dist", 1))
    sys.modules[f"{package_name}.api.queue_orchestration"] = queue_orchestration_module

    @dataclass(frozen=True)
    class _QueuePayload:
        prompt: dict
        workflow_meta: object
        client_id: str
        delegate_master: object
        enabled_worker_ids: list
        trace_execution_id: object

    def _parse_queue_request_payload(data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object body")
        prompt = data.get("prompt")
        if not isinstance(prompt, dict):
            raise ValueError("Field 'prompt' must be an object")
        enabled = data.get("enabled_worker_ids")
        if not isinstance(enabled, list):
            raise ValueError("enabled_worker_ids required")
        client_id = data.get("client_id")
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("client_id required")
        return _QueuePayload(
            prompt=prompt,
            workflow_meta=data.get("workflow"),
            client_id=client_id,
            delegate_master=data.get("delegate_master"),
            enabled_worker_ids=enabled,
            trace_execution_id=data.get("trace_execution_id"),
        )

    queue_request_module = types.ModuleType(f"{package_name}.api.queue_request")
    queue_request_module.parse_queue_request_payload = _parse_queue_request_payload
    sys.modules[f"{package_name}.api.queue_request"] = queue_request_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.job_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    cleanup_optional_module("torch", created_torch_stub)
    if created_pil_stub:
        cleanup_optional_module("PIL.Image", True)
        cleanup_optional_module("PIL", True)

    return module


job_routes = _load_job_routes_module()


class DistributedQueueEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_distributed_queue_happy_path_returns_prompt_id(self):
        request = _FakeRequest(
            {
                "prompt": {"1": {"class_type": "Node"}},
                "enabled_worker_ids": ["w1"],
                "client_id": "client-1",
                "auto_prepare": True,
            }
        )
        with patch.object(
            job_routes,
            "orchestrate_distributed_execution",
            new=AsyncMock(return_value=("prompt_123", 2)),
        ):
            response = await job_routes.distributed_queue_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("prompt_id"), "prompt_123")

    async def test_distributed_queue_missing_prompt_returns_400(self):
        request = _FakeRequest(
            {
                "enabled_worker_ids": ["w1"],
                "client_id": "client-1",
            }
        )
        response = await job_routes.distributed_queue_endpoint(request)
        self.assertEqual(response.status, 400)
        self.assertIn("prompt", response.payload.get("message", "").lower())

    async def test_distributed_queue_missing_enabled_worker_ids_returns_400(self):
        request = _FakeRequest(
            {
                "prompt": {"1": {"class_type": "Node"}},
                "client_id": "client-1",
            }
        )
        response = await job_routes.distributed_queue_endpoint(request)
        self.assertEqual(response.status, 400)
        self.assertIn("enabled_worker_ids", response.payload.get("message", "").lower())


class JobCompleteAudioPayloadTests(unittest.IsolatedAsyncioTestCase):
    def _encoded_audio_payload(self):
        waveform = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        return {
            "sample_rate": 44100,
            "shape": [1, 2, 4],
            "dtype": "float32",
            "data": base64.b64encode(waveform.tobytes()).decode("ascii"),
        }

    async def test_job_complete_accepts_audio_payload(self):
        queue = asyncio.Queue()
        prompt_server = job_routes.server.PromptServer.instance
        prompt_server.distributed_jobs_lock = asyncio.Lock()
        prompt_server.distributed_pending_jobs = {"job-1": queue}
        prompt_server.distributed_job_allowed_workers = {"job-1": {"worker-1"}}
        request = _FakeRequest(
            {
                "job_id": "job-1",
                "worker_id": "worker-1",
                "batch_idx": 0,
                "image": "data:image/png;base64,AAAA",
                "audio": self._encoded_audio_payload(),
                "is_last": True,
            }
        )

        with patch.object(job_routes, "_decode_canonical_png_tensor", return_value="tensor-data"):
            response = await job_routes.job_complete_endpoint(request)

        self.assertEqual(response.status, 200)
        queued = await queue.get()
        self.assertEqual(queued["worker_id"], "worker-1")
        self.assertTrue(queued["is_last"])
        self.assertIsNotNone(queued["audio"])
        self.assertEqual(queued["audio"]["sample_rate"], 44100)
        self.assertEqual(tuple(queued["audio"]["waveform"].shape), (1, 2, 4))

    async def test_job_complete_rejects_worker_not_in_job_allowlist(self):
        queue = asyncio.Queue()
        prompt_server = job_routes.server.PromptServer.instance
        prompt_server.distributed_jobs_lock = asyncio.Lock()
        prompt_server.distributed_pending_jobs = {"job-allow": queue}
        prompt_server.distributed_job_allowed_workers = {"job-allow": {"worker-expected"}}
        request = _FakeRequest(
            {
                "job_id": "job-allow",
                "worker_id": "worker-unexpected",
                "batch_idx": 0,
                "image": "data:image/png;base64,AAAA",
                "is_last": True,
            }
        )

        with patch.object(job_routes, "_decode_canonical_png_tensor", return_value="tensor-data"):
            response = await job_routes.job_complete_endpoint(request)

        self.assertEqual(response.status, 403)
        self.assertIn("unauthorized", response.payload.get("message", "").lower())

    def test_decode_audio_payload_rejects_bad_shape(self):
        bad = {
            "sample_rate": 44100,
            "shape": [1, 2],
            "dtype": "float32",
            "data": base64.b64encode(b"\x00\x00\x00\x00").decode("ascii"),
        }
        with self.assertRaises(ValueError):
            job_routes._decode_audio_payload(bad)

    def test_decode_audio_payload_rejects_bad_dtype(self):
        payload = {
            "sample_rate": 44100,
            "shape": [1, 2, 4],
            "dtype": "float16",
            "data": base64.b64encode((np.zeros((1, 2, 4), dtype=np.float32)).tobytes()).decode("ascii"),
        }
        with self.assertRaises(ValueError):
            job_routes._decode_audio_payload(payload)


if __name__ == "__main__":
    unittest.main()
