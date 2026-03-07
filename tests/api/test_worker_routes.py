import importlib.util
import os
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

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
    def __init__(self, payload=None, match_info=None, query=None, headers=None):
        self._payload = payload
        self.match_info = match_info or {}
        self.query = query or {}
        self.headers = headers or {}

    async def json(self):
        return self._payload


class _FakeHTTPClientResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeHTTPClientSession:
    def __init__(self, payload, status=200):
        self._payload = payload
        self._status = status
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return _FakeHTTPClientResponse(self._payload, status=self._status)


class _DummyWorkerManager:
    def __init__(self):
        self.processes = {}

    def launch_worker(self, worker):
        worker_id = str(worker["id"])
        self.processes[worker_id] = {
            "pid": 12345,
            "log_file": f"/tmp/distributed_worker_{worker_id}.log",  # nosec B108 - deterministic fake test path
            "process": None,
        }
        return 12345

    def is_process_running(self, _pid):
        return False

    def save_processes(self):
        return None

    def stop_worker(self, _worker_id):
        return True, "Stopped"

    def get_managed_workers(self):
        return []


class _ImmediateLoop:
    async def run_in_executor(self, _executor, func, *args):
        return func(*args)


def _load_worker_routes_module():
    module_path = Path(__file__).resolve().parents[2] / "api" / "worker_routes.py"
    package_name = "dist_api_worker_testpkg"

    bootstrap_test_package(package_name, with_api=True, with_utils=True, with_workers=True)

    workers_pkg = types.ModuleType(f"{package_name}.workers")
    workers_pkg.__path__ = []
    workers_pkg.get_worker_manager = lambda: _DummyWorkerManager()
    sys.modules[f"{package_name}.workers"] = workers_pkg

    detection_module = types.ModuleType(f"{package_name}.workers.detection")
    detection_module.is_local_worker = lambda *_args, **_kwargs: True
    detection_module.is_same_physical_host = lambda *_args, **_kwargs: True
    detection_module.get_machine_id = lambda: "machine-id"
    detection_module.is_docker_environment = lambda: False
    detection_module.is_runpod_environment = lambda: False
    detection_module.get_comms_channel = lambda *_args, **_kwargs: "lan"
    sys.modules[f"{package_name}.workers.detection"] = detection_module

    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    install_server_stub()

    created_torch_stub = False
    if "torch" not in sys.modules:
        created_torch_stub = True
        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            current_device=lambda: 0,
            device_count=lambda: 0,
        )
        sys.modules["torch"] = torch_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: {"workers": []}
    sys.modules[f"{package_name}.utils.config"] = config_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status=status)

    network_module.handle_api_error = _handle_api_error
    network_module.normalize_host = lambda value: value
    network_module.build_worker_url = lambda worker, endpoint="": f"http://localhost:{worker.get('port', 8188)}{endpoint}"

    async def _probe_worker(*_args, **_kwargs):
        return None

    network_module.probe_worker = _probe_worker

    async def _get_client_session():
        raise RuntimeError("not used in these tests")

    network_module.get_client_session = _get_client_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.CHUNK_SIZE = 8192
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")

    async def _queue_prompt_payload(*_args, **_kwargs):
        return "prompt-id"

    async_helpers_module.queue_prompt_payload = _queue_prompt_payload

    class _PromptValidationError(RuntimeError):
        def __init__(self, message="invalid prompt", validation_error=None, node_errors=None):
            super().__init__(message)
            self.validation_error = validation_error if isinstance(validation_error, dict) else {}
            self.node_errors = node_errors if isinstance(node_errors, dict) else {}

    async_helpers_module.PromptValidationError = _PromptValidationError
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    schemas_module = types.ModuleType(f"{package_name}.api.schemas")

    def _require_fields(data, *fields):
        missing = []
        for field in fields:
            value = data.get(field) if isinstance(data, dict) else None
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        return missing

    def _validate_worker_id(worker_id, config):
        return any(str(worker.get("id")) == str(worker_id) for worker in config.get("workers", []))

    def _require_worker_id(worker_id, config, field_name="worker_id"):
        _ = field_name
        worker_id_str = str(worker_id).strip()
        if any(str(worker.get("id")) == worker_id_str for worker in config.get("workers", [])):
            return worker_id_str
        raise ValueError(f"Worker {worker_id_str} not found")

    schemas_module.require_fields = _require_fields
    schemas_module.validate_worker_id = _validate_worker_id
    schemas_module.require_worker_id = _require_worker_id
    schemas_module.is_authorized_request = lambda _request, _config: True
    schemas_module.distributed_auth_headers = lambda _config: {}
    sys.modules[f"{package_name}.api.schemas"] = schemas_module

    install_request_guards_stub(package_name)

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.worker_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    cleanup_optional_module("torch", created_torch_stub)

    return module


worker_routes = _load_worker_routes_module()


class WorkerRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_launch_worker_valid_id_returns_200(self):
        manager = _DummyWorkerManager()
        config = {"workers": [{"id": "worker-a", "name": "Worker A", "port": 8188}]}
        request = _FakeRequest({"worker_id": "worker-a"})

        with patch.object(worker_routes, "get_worker_manager", return_value=manager), patch.object(
            worker_routes, "load_config", return_value=config
        ), patch.object(
            worker_routes.asyncio, "get_running_loop", return_value=_ImmediateLoop()
        ):
            response = await worker_routes.launch_worker_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "success")
        self.assertEqual(response.payload.get("pid"), 12345)

    async def test_launch_worker_unknown_id_returns_404(self):
        manager = _DummyWorkerManager()
        config = {"workers": [{"id": "worker-a", "name": "Worker A", "port": 8188}]}
        request = _FakeRequest({"worker_id": "missing-worker"})

        with patch.object(worker_routes, "get_worker_manager", return_value=manager), patch.object(
            worker_routes, "load_config", return_value=config
        ):
            response = await worker_routes.launch_worker_endpoint(request)

        self.assertEqual(response.status, 404)
        self.assertIn("not found", response.payload.get("message", "").lower())

    async def test_worker_log_returns_content_json(self):
        manager = _DummyWorkerManager()
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("line-1\nline-2\nline-3\n")
            log_path = handle.name

        manager.processes["worker-a"] = {
            "pid": 9999,
            "log_file": log_path,
            "process": None,
        }

        request = _FakeRequest(match_info={"worker_id": "worker-a"}, query={"lines": "2"})
        try:
            with patch.object(worker_routes, "get_worker_manager", return_value=manager), patch.object(
                worker_routes.asyncio, "get_running_loop", return_value=_ImmediateLoop()
            ):
                response = await worker_routes.get_worker_log_endpoint(request)
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "success")
        self.assertIn("content", response.payload)
        self.assertIn("line-3", response.payload["content"])

    async def test_local_log_reads_memory_buffer(self):
        request = _FakeRequest(query={"lines": "2"})
        fake_logs = deque(
            [
                {"m": "line-1\n"},
                {"m": "line-2\n"},
                {"m": "line-3\n"},
            ],
            maxlen=300,
        )
        app_module = types.ModuleType("app")
        app_module.__path__ = []
        logger_module = types.ModuleType("app.logger")
        logger_module.get_logs = lambda: fake_logs
        app_module.logger = logger_module

        with patch.dict(sys.modules, {"app": app_module, "app.logger": logger_module}):
            response = await worker_routes.get_local_log_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "success")
        self.assertEqual(response.payload.get("source"), "memory")
        self.assertEqual(response.payload.get("entries"), 2)
        self.assertIn("line-3", response.payload.get("content", ""))

    async def test_remote_worker_log_proxies_to_worker_local_log_endpoint(self):
        config = {
            "workers": [
                {
                    "id": "worker-remote",
                    "name": "Remote Worker",
                    "host": "worker.example.com",
                    "port": 8188,
                    "type": "remote",
                }
            ]
        }
        request = _FakeRequest(match_info={"worker_id": "worker-remote"}, query={"lines": "120"})
        proxied_payload = {
            "status": "success",
            "content": "remote-log-content\n",
            "entries": 1,
            "source": "memory",
            "truncated": False,
            "lines_shown": 1,
        }
        fake_session = _FakeHTTPClientSession(proxied_payload)

        async def _fake_get_client_session():
            return fake_session

        with patch.object(worker_routes, "load_config", return_value=config), patch.object(
            worker_routes, "get_client_session", side_effect=_fake_get_client_session
        ):
            response = await worker_routes.get_remote_worker_log_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("content"), "remote-log-content\n")
        self.assertEqual(len(fake_session.calls), 1)
        self.assertEqual(fake_session.calls[0]["params"], {"lines": "120"})
        self.assertTrue(fake_session.calls[0]["url"].endswith("/distributed/local_log"))

    async def test_remote_worker_log_rejects_local_workers(self):
        config = {"workers": [{"id": "worker-local", "name": "Local Worker", "port": 8188}]}
        request = _FakeRequest(match_info={"worker_id": "worker-local"})

        with patch.object(worker_routes, "load_config", return_value=config):
            response = await worker_routes.get_remote_worker_log_endpoint(request)

        self.assertEqual(response.status, 400)
        self.assertIn("local", response.payload.get("message", "").lower())


if __name__ == "__main__":
    unittest.main()
