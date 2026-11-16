import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeRequest:
    def __init__(self, payload=None, match_info=None, query=None):
        self._payload = payload
        self.match_info = match_info or {}
        self.query = query or {}

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

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeHTTPClientResponse(self._payload, status=self._status)


class _DummyWorkerManager:
    def __init__(self, log_dir=None):
        self.processes = {}
        self.log_dir = log_dir

    def get_worker_log_dir(self):
        return self.log_dir

    def _check_worker_process(self, _worker_id, _proc_info):
        return False, False

    def launch_worker(self, worker):
        worker_id = str(worker["id"])
        self.processes[worker_id] = {
            "pid": 12345,
            "log_file": f"/tmp/distributed_worker_{worker_id}.log",
            "process": None,
        }
        return 12345

    def _is_process_running(self, _pid):
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

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    api_pkg = types.ModuleType(f"{package_name}.api")
    api_pkg.__path__ = []
    sys.modules[f"{package_name}.api"] = api_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    workers_pkg = types.ModuleType(f"{package_name}.workers")
    workers_pkg.__path__ = [str(module_path.parents[1] / "workers")]
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

    created_aiohttp_stub = False
    if "aiohttp" not in sys.modules:
        created_aiohttp_stub = True
        aiohttp_module = types.ModuleType("aiohttp")

        class _ClientTimeout:
            def __init__(self, total=None):
                self.total = total

        class _WSMsgType:
            TEXT = "TEXT"
            ERROR = "ERROR"
            CLOSED = "CLOSED"

        class _WebSocketResponse:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            async def prepare(self, _request):
                return None

            async def send_json(self, _payload):
                return None

            def __aiter__(self):
                async def _empty():
                    if False:
                        yield None
                return _empty()

        aiohttp_module.ClientTimeout = _ClientTimeout
        aiohttp_module.WSMsgType = _WSMsgType
        aiohttp_module.web = types.SimpleNamespace(
            json_response=lambda payload, status=200: _FakeResponse(payload, status=status),
            WebSocketResponse=_WebSocketResponse,
        )
        sys.modules["aiohttp"] = aiohttp_module

    class _Routes:
        def get(self, _path):
            def _decorator(fn):
                return fn
            return _decorator

        def post(self, _path):
            def _decorator(fn):
                return fn
            return _decorator

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    sys.modules["server"] = server_module

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
    network_module.get_server_port = lambda: 8189
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

    schemas_module.require_fields = _require_fields
    schemas_module.validate_worker_id = _validate_worker_id
    sys.modules[f"{package_name}.api.schemas"] = schemas_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.worker_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    # Keep this module's response double independent of aiohttp import order.
    module.web = types.SimpleNamespace(json_response=lambda payload, status=200: _FakeResponse(payload, status))

    if created_aiohttp_stub:
        sys.modules.pop("aiohttp", None)
    if created_torch_stub:
        sys.modules.pop("torch", None)

    return module


worker_routes = _load_worker_routes_module()


class WebsocketExecutionTargetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from test_async_helpers import _load_async_helpers_module
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.helper, self.server = _load_async_helpers_module()
        sys.modules['dist_api_worker_testpkg.utils.async_helpers'] = self.helper
        self.prompt = {'9': {'class_type': 'Output'}, '10': {'class_type': 'Output'}}

    async def receive(self, payload):
        import json
        sent = []
        class Socket:
            async def prepare(self, request):
                pass
            async def send_json(self, value):
                sent.append(value)
            def __aiter__(self):
                async def messages():
                    yield types.SimpleNamespace(type=worker_routes.aiohttp.WSMsgType.TEXT,
                                                data=json.dumps(payload))
                return messages()
        with patch.object(worker_routes, 'web', types.SimpleNamespace(WebSocketResponse=lambda **kw: Socket())), \
             patch.object(worker_routes, 'queue_prompt_payload', self.helper.queue_prompt_payload), \
             patch.object(worker_routes, 'PromptValidationError', self.helper.PromptValidationError):
            await worker_routes.worker_ws_endpoint(_FakeRequest())
        return sent[0]

    def payload(self, targets):
        return {'type': 'dispatch_prompt', 'request_id': 'req', 'prompt': self.prompt,
                'partial_execution_targets': targets}

    async def test_explicit_targets_reach_queue_and_acknowledgement(self):
        ack = await self.receive(self.payload(['9']))
        self.assertTrue(ack['ok'], ack)
        self.assertEqual(ack['request_id'], 'req')
        self.assertEqual(self.server.prompt_queue.items[0][4], ['9'])

    async def test_malformed_targets_and_messages_are_rejected(self):
        for targets in (None, [], '9', [True], [9], ['absent'], ['9', '9']):
            with self.subTest(targets=targets):
                ack = await self.receive(self.payload(targets))
                self.assertFalse(ack['ok'])
                self.assertIn('partial_execution_targets', ack['error'])
        self.assertFalse((await self.receive([]))['ok'])
        self.assertEqual(self.server.prompt_queue.items, [])

    async def test_validation_cannot_silently_queue_a_partial_subset(self):
        ack = await self.receive(self.payload(['9', '10']))
        self.assertFalse(ack['ok'])
        self.assertEqual(ack['validation_error']['type'], 'partial_execution_targets_failed_validation')
        self.assertEqual(self.server.prompt_queue.items, [])

    async def test_missing_target_field_keeps_legacy_queue_call(self):
        from unittest.mock import AsyncMock
        queued = AsyncMock(return_value='legacy')
        original = self.helper.queue_prompt_payload
        self.helper.queue_prompt_payload = queued
        try:
            payload = self.payload(['9'])
            payload.pop('partial_execution_targets')
            self.assertTrue((await self.receive(payload))['ok'])
            self.assertNotIn('partial_execution_targets', queued.call_args.kwargs)
        finally:
            self.helper.queue_prompt_payload = original


class WorkerRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_info_reports_actual_master_and_available_ports(self):
        with patch.object(worker_routes, "_get_cuda_info", return_value=(0, 1, 4)), \
             patch.object(worker_routes, "get_network_ips", return_value=["127.0.0.1"]), \
             patch.object(worker_routes, "load_config", return_value={"workers": []}), \
             patch.object(worker_routes, "get_server_port", return_value=8189), \
             patch.object(worker_routes, "allocate_worker_ports", return_value=[8191, 8192, 8193]) as allocate:
            response = await worker_routes.get_network_info_endpoint(_FakeRequest())
        self.assertEqual(response.payload["master_port"], 8189)
        self.assertEqual(response.payload["local_worker_ports"], [8191, 8192, 8193])
        self.assertEqual(response.payload["cuda_device_count"], 4)
        allocate.assert_called_once_with(8189, [], 3)

    async def test_network_info_does_not_reallocate_existing_configuration(self):
        with patch.object(worker_routes, "_get_cuda_info", return_value=(0, 4, 4)), \
             patch.object(worker_routes, "get_network_ips", return_value=[]), \
             patch.object(worker_routes, "load_config", return_value={"workers": [{"id": "existing", "port": 8189}]}), \
             patch.object(worker_routes, "allocate_worker_ports") as allocate:
            response = await worker_routes.get_network_info_endpoint(_FakeRequest())
        self.assertEqual(response.payload["local_worker_ports"], [])
        allocate.assert_not_called()

    async def test_launch_conflict_is_a_clear_client_error(self):
        manager = _DummyWorkerManager()
        config = {"workers": [{"id": "worker-a", "name": "Worker A", "port": 8189}]}
        with patch.object(worker_routes, "get_worker_manager", return_value=manager), \
             patch.object(worker_routes, "load_config", return_value=config), \
             patch.object(manager, "launch_worker", side_effect=ValueError("Worker port 8189 conflicts with the master port 8189")):
            response = await worker_routes.launch_worker_endpoint(_FakeRequest({"worker_id": "worker-a"}))
        self.assertEqual(response.status, 400)
        self.assertIn("master port 8189", response.payload["message"])
        self.assertEqual(manager.processes, {})

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
        log_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, log_dir, True)
        manager = _DummyWorkerManager(log_dir=log_dir)
        log_path = os.path.join(log_dir, "Worker_A_20260101.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("line-1\nline-2\nline-3\n")

        manager.processes["worker-a"] = {
            "pid": 9999,
            "log_file": log_path,
            "process": None,
        }

        request = _FakeRequest(match_info={"worker_id": "worker-a"}, query={"lines": "2"})
        with patch.object(worker_routes, "get_worker_manager", return_value=manager), patch.object(
            worker_routes.asyncio, "get_running_loop", return_value=_ImmediateLoop()
        ):
            response = await worker_routes.get_worker_log_endpoint(request)

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


class WorkerLogAndLaunchSafetyTests(unittest.IsolatedAsyncioTestCase):
    def _log_dir(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        log_dir = os.path.join(root, "logs", "workers")
        os.makedirs(log_dir)
        return root, log_dir

    async def _read_log(self, manager, lines="1000"):
        request = _FakeRequest(match_info={"worker_id": "worker-a"}, query={"lines": lines})
        with patch.object(worker_routes, "get_worker_manager", return_value=manager), patch.object(
            worker_routes.asyncio, "get_running_loop", return_value=_ImmediateLoop()
        ):
            return await worker_routes.get_worker_log_endpoint(request)

    async def test_worker_log_refuses_files_outside_the_worker_log_directory(self):
        root, log_dir = self._log_dir()
        secret = os.path.join(root, "secret.txt")
        with open(secret, "w", encoding="utf-8") as handle:
            handle.write("top secret\n")
        for path in (secret, os.path.join(log_dir, "..", "..", "secret.txt"), log_dir):
            with self.subTest(path=path):
                manager = _DummyWorkerManager(log_dir=log_dir)
                manager.processes["worker-a"] = {"pid": 1, "log_file": path, "process": None}
                response = await self._read_log(manager)
                self.assertEqual(response.status, 403)
                self.assertNotIn("top secret", str(response.payload))

    async def test_worker_log_without_a_known_log_directory_is_refused(self):
        _root, log_dir = self._log_dir()
        log_path = os.path.join(log_dir, "w.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("line\n")
        manager = _DummyWorkerManager(log_dir=None)
        manager.processes["worker-a"] = {"pid": 1, "log_file": log_path, "process": None}
        self.assertEqual((await self._read_log(manager)).status, 403)

    async def test_large_log_tail_with_multibyte_text_has_no_duplicate_lines(self):
        _root, log_dir = self._log_dir()
        log_path = os.path.join(log_dir, "w.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            for index in range(30000):
                handle.write(f"line {index:06d} " + "\u2588" * 20 + "\n")
        self.assertGreater(os.path.getsize(log_path), 1024 * 1024)
        manager = _DummyWorkerManager(log_dir=log_dir)
        manager.processes["worker-a"] = {"pid": 1, "log_file": log_path, "process": None}
        response = await self._read_log(manager, lines="300")
        self.assertEqual(response.status, 200)
        lines = response.payload["content"].split("\n")
        expected = [f"line {index:06d} " + "\u2588" * 20 for index in range(29700, 30000)]
        self.assertEqual(lines, expected)
        self.assertTrue(response.payload["truncated"])

    def _read_tail_counting_bytes(self, log_path, lines):
        real_open = open
        read_bytes = [0]

        class _Counting:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.handle.close()

            def read(self, size=-1):
                data = self.handle.read(size)
                read_bytes[0] += len(data)
                return data

            def __getattr__(self, name):
                return getattr(self.handle, name)

        with patch.object(worker_routes, "open", lambda *a, **k: _Counting(real_open(*a, **k)), create=True):
            result = worker_routes._read_worker_log_sync(log_path, lines)
        return result, read_bytes[0]

    def test_large_log_of_progress_bar_updates_reads_only_its_tail(self):
        # tqdm separates updates with a bare \r; each one is a line, as before.
        _root, log_dir = self._log_dir()
        log_path = os.path.join(log_dir, "w.log")
        with open(log_path, "wb") as handle:
            handle.write(b"".join(b"\r step %06d/100000 |#####     |" % index for index in range(60000)))
            handle.write(b"\r\ndone\r\n")
        self.assertGreater(os.path.getsize(log_path), 1024 * 1024)
        result, read_bytes = self._read_tail_counting_bytes(log_path, 300)
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 300)
        self.assertEqual(lines[-1], "done")
        self.assertEqual(lines[-2], " step 059999/100000 |#####     |")
        self.assertTrue(result["truncated"])
        self.assertLess(read_bytes, 64 * 1024, "the whole log was read")

    def test_large_log_without_line_breaks_reads_a_bounded_tail(self):
        _root, log_dir = self._log_dir()
        log_path = os.path.join(log_dir, "w.log")
        with open(log_path, "wb") as handle:
            handle.write(b"x" * (3 * 1024 * 1024))
        with patch.object(worker_routes, "MAX_LOG_TAIL_BYTES", 256 * 1024):
            result, read_bytes = self._read_tail_counting_bytes(log_path, 300)
        self.assertLessEqual(read_bytes, 256 * 1024 + worker_routes.CHUNK_SIZE)
        self.assertTrue(result["truncated"])

    async def test_launch_is_not_blocked_by_a_reused_pid_of_a_restored_worker(self):
        manager = _DummyWorkerManager()
        manager.processes["worker-a"] = {"pid": 4242, "process": None, "log_file": None}
        config = {"workers": [{"id": "worker-a", "name": "Worker A", "port": 8188}]}
        # The PID exists (reused by another program) but is not the launched worker.
        with patch.object(manager, "_is_process_running", return_value=True), \
             patch.object(manager, "_check_worker_process", return_value=(False, False)), \
             patch.object(worker_routes, "get_worker_manager", return_value=manager), \
             patch.object(worker_routes, "load_config", return_value=config), \
             patch.object(worker_routes.asyncio, "get_running_loop", return_value=_ImmediateLoop()):
            response = await worker_routes.launch_worker_endpoint(_FakeRequest({"worker_id": "worker-a"}))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("pid"), 12345)

    async def test_worker_websocket_accepts_prompts_up_to_comfyui_upload_limit(self):
        created = []

        class _Socket:
            async def prepare(self, _request):
                return None

            def __aiter__(self):
                async def _messages():
                    if False:
                        yield None
                return _messages()

        def _websocket_response(**kwargs):
            created.append(kwargs)
            return _Socket()

        with patch.object(worker_routes, "web", types.SimpleNamespace(WebSocketResponse=_websocket_response)):
            await worker_routes.worker_ws_endpoint(_FakeRequest())
        self.assertEqual(created[0].get("max_msg_size"), 100 * 1024 * 1024)
        self.assertEqual(created[0].get("heartbeat"), 30)
