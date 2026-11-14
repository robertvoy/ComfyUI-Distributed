import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeRequest:
    def __init__(self, payload=None):
        self._payload = payload

    async def json(self):
        return self._payload


def _load_config_routes_module():
    module_path = Path(__file__).resolve().parents[2] / "api" / "config_routes.py"
    package_name = "dist_api_config_testpkg"

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
    utils_pkg.__path__ = [str(module_path.parents[1] / "utils")]
    sys.modules[f"{package_name}.utils"] = utils_pkg

    import aiohttp  # Load the real dependency; isolate the fake response at the tested module.

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

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status=status)

    network_module.handle_api_error = _handle_api_error
    network_module.normalize_host = lambda value: value
    sys.modules[f"{package_name}.utils.network"] = network_module

    default_config = {
        "workers": [],
        "master": {"host": ""},
        "settings": {"debug": False},
        "tunnel": {},
    }

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: copy.deepcopy(default_config)
    config_module.save_config = lambda _cfg: True
    sys.modules[f"{package_name}.utils.config"] = config_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.config_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    class _EmptySecretStore:
        def has(self, _worker_id):
            return False

    module.web = types.SimpleNamespace(
        json_response=lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    module._serverless_secret_store = lambda: _EmptySecretStore()

    return module


config_routes = _load_config_routes_module()


class ConfigRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_config_returns_core_sections(self):
        cfg = {"workers": [], "master": {}, "settings": {}, "tunnel": {}}
        with patch.object(config_routes, "load_config", return_value=cfg):
            response = await config_routes.get_config_endpoint(_FakeRequest())

        self.assertEqual(response.status, 200)
        self.assertIn("workers", response.payload)
        self.assertIn("master", response.payload)
        self.assertIn("settings", response.payload)

    async def test_update_config_valid_field_persists(self):
        cfg = {"workers": [], "master": {}, "settings": {"debug": False}, "tunnel": {}}
        with patch.object(config_routes, "load_config", return_value=cfg), patch.object(
            config_routes, "save_config", return_value=True
        ):
            response = await config_routes.update_config_endpoint(_FakeRequest({"debug": True}))

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["status"], "success")
        self.assertTrue(response.payload["config"]["settings"]["debug"])

    async def test_update_config_unknown_field_returns_400(self):
        cfg = {"workers": [], "master": {}, "settings": {"debug": False}, "tunnel": {}}
        with patch.object(config_routes, "load_config", return_value=cfg):
            response = await config_routes.update_config_endpoint(_FakeRequest({"unknown_field": 1}))

        self.assertEqual(response.status, 400)
        self.assertIn("unknown_field", " ".join(response.payload.get("error", [])).lower())

    async def test_update_config_wrong_type_returns_400(self):
        cfg = {"workers": [], "master": {}, "settings": {"debug": False}, "tunnel": {}}
        with patch.object(config_routes, "load_config", return_value=cfg):
            response = await config_routes.update_config_endpoint(_FakeRequest({"debug": "true"}))

        self.assertEqual(response.status, 400)
        self.assertIn("debug", " ".join(response.payload.get("error", [])).lower())


def _server_config():
    return {
        "workers": [],
        "master": {"host": ""},
        "settings": {"debug": False, "worker_timeout_seconds": 60},
        "tunnel": {"status": "running", "pid": 4242, "log_file": "logs/tunnel.log"},
        "managed_processes": {
            "w1": {"pid": 5151, "started_at": 1.0, "config": {}, "log_file": "logs/w1.log"},
        },
    }


class BulkConfigServerOwnedStateTests(unittest.IsolatedAsyncioTestCase):
    """Bulk POST /distributed/config must not let clients plant PIDs or log paths."""

    async def _post(self, cfg, payload):
        with patch.object(config_routes, "load_config", return_value=cfg), patch.object(
            config_routes, "save_config", return_value=True
        ):
            return await config_routes.update_config_endpoint(_FakeRequest(payload))

    async def test_client_managed_processes_and_tunnel_are_ignored(self):
        cfg = _server_config()
        expected_tunnel = copy.deepcopy(cfg["tunnel"])
        expected_processes = copy.deepcopy(cfg["managed_processes"])

        response = await self._post(cfg, {
            "debug": True,
            "managed_processes": {"evil": {"pid": 4, "log_file": "C:/Windows/System32/x.log"}},
            "tunnel": {"status": "running", "pid": 4, "log_file": "C:/Windows/win.ini"},
        })

        self.assertEqual(response.status, 200)
        self.assertTrue(cfg["settings"]["debug"])
        self.assertEqual(cfg["tunnel"], expected_tunnel)
        self.assertEqual(cfg["managed_processes"], expected_processes)
        self.assertEqual(response.payload["config"]["tunnel"], expected_tunnel)
        self.assertEqual(response.payload["config"]["managed_processes"], expected_processes)

    async def test_round_tripped_get_payload_keeps_server_state(self):
        cfg = _server_config()
        echoed = copy.deepcopy(cfg)
        echoed["tunnel"]["pid"] = 1
        echoed["managed_processes"]["w1"]["pid"] = 1

        response = await self._post(cfg, echoed)

        self.assertEqual(response.status, 200)
        self.assertEqual(cfg["tunnel"]["pid"], 4242)
        self.assertEqual(cfg["managed_processes"]["w1"]["pid"], 5151)

    async def test_settings_object_with_wrong_type_is_rejected(self):
        cfg = _server_config()

        response = await self._post(cfg, {"settings": {"worker_timeout_seconds": "abc"}})

        self.assertEqual(response.status, 400)
        self.assertIn("worker_timeout_seconds", response.payload["message"])
        self.assertEqual(cfg["settings"]["worker_timeout_seconds"], 60)

    async def test_bool_is_not_accepted_for_integer_field(self):
        cfg = _server_config()

        response = await self._post(cfg, {"worker_timeout_seconds": True})

        self.assertEqual(response.status, 400)
        self.assertEqual(cfg["settings"]["worker_timeout_seconds"], 60)


class UpdateSettingValidationTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, cfg, key, value):
        with patch.object(config_routes, "load_config", return_value=cfg), patch.object(
            config_routes, "save_config", return_value=True
        ):
            return await config_routes.update_setting_endpoint(
                _FakeRequest({"key": key, "value": value})
            )

    async def test_rejects_wrong_type_for_known_setting(self):
        for key, value in [
            ("worker_timeout_seconds", "abc"),
            ("worker_timeout_seconds", "120"),
            ("worker_timeout_seconds", 1.5),
            ("worker_timeout_seconds", True),
            ("worker_timeout_seconds", 0),
            ("worker_timeout_seconds", -5),
            ("debug", "yes"),
            ("master_delegate_only", 1),
        ]:
            with self.subTest(key=key, value=value):
                cfg = _server_config()
                before = copy.deepcopy(cfg["settings"])
                response = await self._post(cfg, key, value)
                self.assertEqual(response.status, 400)
                self.assertIn(key, response.payload["message"])
                self.assertEqual(cfg["settings"], before)

    async def test_accepts_valid_values(self):
        for key, value in [
            ("worker_timeout_seconds", 120),
            ("debug", True),
            ("debug", False),
            ("has_auto_populated_workers", True),
        ]:
            with self.subTest(key=key, value=value):
                cfg = _server_config()
                response = await self._post(cfg, key, value)
                self.assertEqual(response.status, 200)
                self.assertEqual(cfg["settings"][key], value)


if __name__ == "__main__":
    unittest.main()
