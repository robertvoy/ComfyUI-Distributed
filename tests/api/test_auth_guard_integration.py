import asyncio
import copy
import importlib.util
import sys
import types
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from tests.api.harness import (
    bootstrap_test_package,
    cleanup_optional_module,
    install_aiohttp_stub,
    install_server_stub,
)


AUTH_TOKEN = "integration-secret"


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeRequest:
    def __init__(self, *, headers=None, json_payload=None, post_payload=None, query=None):
        self.headers = headers or {}
        self._json_payload = json_payload or {}
        self._post_payload = post_payload or {}
        self.query = query or {}

    async def json(self):
        return self._json_payload

    async def post(self):
        return self._post_payload


def _load_module(package_name: str, rel_path: str, module_name: str):
    module_path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(f"{package_name}.{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_common_utils(package_name: str, config_data: dict):
    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: copy.deepcopy(config_data)
    config_module.save_config = lambda _cfg: True

    @asynccontextmanager
    async def _config_transaction():
        yield config_module.load_config()

    config_module.config_transaction = _config_transaction
    sys.modules[f"{package_name}.utils.config"] = config_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status=status)

    network_module.handle_api_error = _handle_api_error
    network_module.normalize_host = lambda value: value
    network_module.build_worker_url = lambda worker, path="": f"http://{worker.get('host', '127.0.0.1')}{path}"
    network_module.get_client_session = lambda: None
    network_module.probe_worker = lambda *_args, **_kwargs: {"ok": True}
    sys.modules[f"{package_name}.utils.network"] = network_module


def _load_real_guard_stack(package_name: str):
    _load_module(package_name, "utils/auth.py", "utils.auth")
    _load_module(package_name, "utils/parsing.py", "utils.parsing")
    _load_module(package_name, "api/schemas.py", "api.schemas")
    _load_module(package_name, "api/request_guards.py", "api.request_guards")
    _load_module(package_name, "api/endpoint_policy.py", "api.endpoint_policy")


def _load_request_guards_only_module():
    package_name = "dist_auth_guard_testpkg"
    config_data = {"settings": {"distributed_api_token": AUTH_TOKEN}}
    bootstrap_test_package(package_name, with_api=True, with_utils=True)
    install_server_stub()
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    _install_common_utils(package_name, config_data)
    _load_real_guard_stack(package_name)
    request_guards = sys.modules[f"{package_name}.api.request_guards"]
    schemas = sys.modules[f"{package_name}.api.schemas"]
    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    return request_guards, schemas


def _load_config_routes_real_guard_module():
    package_name = "dist_config_auth_integration_testpkg"
    config_data = {
        "workers": [],
        "master": {"host": "127.0.0.1"},
        "settings": {"distributed_api_token": AUTH_TOKEN},
        "tunnel": {},
    }
    bootstrap_test_package(package_name, with_api=True, with_utils=True)
    install_server_stub()
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    _install_common_utils(package_name, config_data)
    _load_real_guard_stack(package_name)
    module = _load_module(package_name, "api/config_routes.py", "api.config_routes")
    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    return module


def _load_tunnel_routes_real_guard_module():
    package_name = "dist_tunnel_auth_integration_testpkg"
    config_data = {
        "workers": [],
        "master": {"host": "127.0.0.1"},
        "settings": {"distributed_api_token": AUTH_TOKEN},
        "tunnel": {},
    }
    bootstrap_test_package(package_name, with_api=True, with_utils=True)
    install_server_stub()
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    _install_common_utils(package_name, config_data)
    _load_real_guard_stack(package_name)

    cloudflare_module = types.ModuleType(f"{package_name}.utils.cloudflare")
    cloudflare_module.cloudflare_tunnel_manager = types.SimpleNamespace(
        get_status=lambda: {"active": False},
        start_tunnel=lambda: {"active": True},
        stop_tunnel=lambda: {"active": False},
    )
    sys.modules[f"{package_name}.utils.cloudflare"] = cloudflare_module

    module = _load_module(package_name, "api/tunnel_routes.py", "api.tunnel_routes")
    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    return module


def _load_usdu_routes_real_guard_module():
    package_name = "dist_usdu_auth_integration_testpkg"
    config_data = {"settings": {"distributed_api_token": AUTH_TOKEN}}
    bootstrap_test_package(package_name, with_api=True, with_utils=True, with_upscale=True)
    install_server_stub()
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    _install_common_utils(package_name, config_data)
    _load_real_guard_stack(package_name)

    usdu_management_module = types.ModuleType(f"{package_name}.utils.usdu_management")
    usdu_management_module.MAX_PAYLOAD_SIZE = 1024 * 1024
    sys.modules[f"{package_name}.utils.usdu_management"] = usdu_management_module

    prompt_server_holder = {
        "value": types.SimpleNamespace(
            distributed_tile_jobs_lock=asyncio.Lock(),
            distributed_pending_tile_jobs={},
        )
    }

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.ensure_tile_jobs_initialized = lambda: prompt_server_holder["value"]
    job_store_module.init_dynamic_job = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    job_models_module = types.ModuleType(f"{package_name}.upscale.job_models")

    class _BaseJobState:
        pass

    class _ImageJobState(_BaseJobState):
        def __init__(self):
            self.mode = "dynamic"
            self.queue = asyncio.Queue()
            self.pending_images = asyncio.Queue()
            self.completed_images = {}
            self.worker_status = {}
            self.assigned_to_workers = {}

    class _TileJobState(_BaseJobState):
        def __init__(self):
            self.mode = "static"
            self.queue = asyncio.Queue()
            self.pending_tasks = asyncio.Queue()
            self.completed_tasks = {}
            self.worker_status = {}
            self.assigned_to_workers = {}
            self.batched_static = False

    job_models_module.BaseJobState = _BaseJobState
    job_models_module.ImageJobState = _ImageJobState
    job_models_module.TileJobState = _TileJobState
    sys.modules[f"{package_name}.upscale.job_models"] = job_models_module

    parsers_module = types.ModuleType(f"{package_name}.upscale.payload_parsers")
    parsers_module.parse_tiles_from_form = lambda _data: []
    sys.modules[f"{package_name}.upscale.payload_parsers"] = parsers_module

    module = _load_module(package_name, "api/usdu_routes.py", "api.usdu_routes")
    module._prompt_server_holder = prompt_server_holder
    module._ImageJobState = _ImageJobState
    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    return module


def _install_torch_stub_if_missing():
    if "torch" in sys.modules:
        return False

    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
    )
    sys.modules["torch"] = torch_module
    return True


def _load_worker_routes_real_guard_module():
    package_name = "dist_worker_auth_integration_testpkg"
    config_data = {"settings": {"distributed_api_token": AUTH_TOKEN}, "workers": []}
    bootstrap_test_package(package_name, with_api=True, with_utils=True, with_workers=True)
    install_server_stub()
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    created_torch_stub = _install_torch_stub_if_missing()

    _install_common_utils(package_name, config_data)
    _load_real_guard_stack(package_name)

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.CHUNK_SIZE = 65536
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    workers_module = types.ModuleType(f"{package_name}.workers")
    workers_module.get_worker_manager = lambda: types.SimpleNamespace(
        processes={},
        save_processes=lambda: None,
        is_process_running=lambda _pid: False,
    )
    sys.modules[f"{package_name}.workers"] = workers_module

    detection_module = types.ModuleType(f"{package_name}.workers.detection")
    detection_module.get_machine_id = lambda: "machine-id"
    detection_module.is_docker_environment = lambda: False
    detection_module.is_runpod_environment = lambda: False
    sys.modules[f"{package_name}.workers.detection"] = detection_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")

    class _PromptValidationError(Exception):
        def __init__(self, message="validation error"):
            super().__init__(message)
            self.validation_error = {}
            self.node_errors = {}

    async def _queue_prompt_payload(*_args, **_kwargs):
        return "prompt-id"

    async_helpers_module.PromptValidationError = _PromptValidationError
    async_helpers_module.queue_prompt_payload = _queue_prompt_payload
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    module = _load_module(package_name, "api/worker_routes.py", "api.worker_routes")
    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    if created_torch_stub:
        sys.modules.pop("torch", None)
    return module


def _load_job_routes_real_guard_module():
    package_name = "dist_job_auth_integration_testpkg"
    config_data = {"settings": {"distributed_api_token": AUTH_TOKEN}}
    bootstrap_test_package(package_name, with_api=True, with_utils=True)
    install_server_stub()
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    created_torch_stub = _install_torch_stub_if_missing()

    _install_common_utils(package_name, config_data)
    _load_real_guard_stack(package_name)

    image_module = types.ModuleType(f"{package_name}.utils.image")
    image_module.pil_to_tensor = lambda _img: None
    image_module.ensure_contiguous = lambda tensor: tensor
    sys.modules[f"{package_name}.utils.image"] = image_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.JOB_INIT_GRACE_PERIOD = 0.1
    constants_module.MEMORY_CLEAR_DELAY = 0.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    runtime_state_module = types.ModuleType(f"{package_name}.utils.runtime_state")
    runtime_state = types.SimpleNamespace(
        distributed_jobs_lock=asyncio.Lock(),
        distributed_pending_jobs={},
    )
    runtime_state_module.ensure_distributed_runtime_state = lambda: runtime_state
    sys.modules[f"{package_name}.utils.runtime_state"] = runtime_state_module

    orchestration_module = types.ModuleType(f"{package_name}.api.queue_orchestration")
    orchestration_module.orchestrate_distributed_execution = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.api.queue_orchestration"] = orchestration_module

    queue_request_module = types.ModuleType(f"{package_name}.api.queue_request")
    queue_request_module.parse_queue_request_payload = lambda data: data
    sys.modules[f"{package_name}.api.queue_request"] = queue_request_module

    module = _load_module(package_name, "api/job_routes.py", "api.job_routes")
    cleanup_optional_module("aiohttp", created_aiohttp_stub)
    if created_torch_stub:
        sys.modules.pop("torch", None)
    return module


request_guards_module, schemas_module = _load_request_guards_only_module()
config_routes_module = _load_config_routes_real_guard_module()
tunnel_routes_module = _load_tunnel_routes_real_guard_module()
usdu_routes_module = _load_usdu_routes_real_guard_module()
worker_routes_module = _load_worker_routes_real_guard_module()
job_routes_module = _load_job_routes_real_guard_module()


class AuthGuardContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_is_authorized_request_missing_token_fails(self):
        request = _FakeRequest(headers={})
        config = {"settings": {"distributed_api_token": AUTH_TOKEN}}
        self.assertFalse(schemas_module.is_authorized_request(request, config))

    async def test_is_authorized_request_matching_token_succeeds(self):
        request = _FakeRequest(headers={schemas_module.AUTH_HEADER_NAME: AUTH_TOKEN})
        config = {"settings": {"distributed_api_token": AUTH_TOKEN}}
        self.assertTrue(schemas_module.is_authorized_request(request, config))

    async def test_authorization_error_or_none_returns_403_for_invalid_token(self):
        request = _FakeRequest(headers={schemas_module.AUTH_HEADER_NAME: "wrong-token"})
        response = await request_guards_module.authorization_error_or_none(request)
        self.assertIsNotNone(response)
        self.assertEqual(response.status, 403)

    async def test_authorization_error_or_none_returns_none_for_valid_token(self):
        request = _FakeRequest(headers={schemas_module.AUTH_HEADER_NAME: AUTH_TOKEN})
        response = await request_guards_module.authorization_error_or_none(request)
        self.assertIsNone(response)


class RouteAuthIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def _auth_headers(self):
        return {schemas_module.AUTH_HEADER_NAME: AUTH_TOKEN}

    async def test_config_endpoint_enforces_real_guard(self):
        unauthorized = await config_routes_module.get_config_endpoint(_FakeRequest(headers={}))
        self.assertEqual(unauthorized.status, 403)

        authorized = await config_routes_module.get_config_endpoint(_FakeRequest(headers=self._auth_headers()))
        self.assertEqual(authorized.status, 200)
        self.assertIn("settings", authorized.payload)

    async def test_tunnel_endpoint_enforces_real_guard(self):
        unauthorized = await tunnel_routes_module.tunnel_status_endpoint(_FakeRequest(headers={}))
        self.assertEqual(unauthorized.status, 403)

        authorized = await tunnel_routes_module.tunnel_status_endpoint(_FakeRequest(headers=self._auth_headers()))
        self.assertEqual(authorized.status, 200)
        self.assertEqual(authorized.payload.get("status"), "success")

    async def test_usdu_endpoint_enforces_real_guard(self):
        unauthorized = await usdu_routes_module.job_status_endpoint(_FakeRequest(headers={}))
        self.assertEqual(unauthorized.status, 403)

        authorized = await usdu_routes_module.job_status_endpoint(
            _FakeRequest(headers=self._auth_headers(), query={"multi_job_id": "job-1"})
        )
        self.assertEqual(authorized.status, 200)
        self.assertIn("ready", authorized.payload)

    async def test_worker_endpoint_enforces_real_guard(self):
        worker_routes_module._collect_network_info_sync = lambda: {
            "interfaces": [],
            "recommended_host": "127.0.0.1",
        }
        unauthorized = await worker_routes_module.get_network_info_endpoint(_FakeRequest(headers={}))
        self.assertEqual(unauthorized.status, 403)

        authorized = await worker_routes_module.get_network_info_endpoint(
            _FakeRequest(headers=self._auth_headers())
        )
        self.assertEqual(authorized.status, 200)
        self.assertEqual(authorized.payload.get("status"), "success")

    async def test_job_endpoint_enforces_real_guard(self):
        unauthorized = await job_routes_module.prepare_job_endpoint(
            _FakeRequest(headers={}, json_payload={"multi_job_id": "job-1"})
        )
        self.assertEqual(unauthorized.status, 403)

        authorized = await job_routes_module.prepare_job_endpoint(
            _FakeRequest(headers=self._auth_headers(), json_payload={"multi_job_id": "job-1"})
        )
        self.assertEqual(authorized.status, 200)
        self.assertEqual(authorized.payload.get("status"), "success")


if __name__ == "__main__":
    unittest.main()
