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


if __name__ == "__main__":
    unittest.main()
