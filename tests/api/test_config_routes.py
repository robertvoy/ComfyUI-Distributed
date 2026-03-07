import copy
import importlib.util
import sys
import types
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from tests.api.harness import (
    bootstrap_test_package,
    cleanup_optional_module,
    install_aiohttp_stub,
    install_endpoint_policy_passthrough_stub,
    install_request_guards_stub,
    install_server_stub,
)


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

    bootstrap_test_package(package_name, with_api=True, with_utils=True)

    schemas_module = types.ModuleType(f"{package_name}.api.schemas")
    schemas_module.is_authorized_request = lambda _request, _config: True
    sys.modules[f"{package_name}.api.schemas"] = schemas_module

    install_request_guards_stub(package_name)
    install_endpoint_policy_passthrough_stub(package_name)
    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _FakeResponse(payload, status=status)
    )
    install_server_stub()

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        if isinstance(error, list):
            message = "; ".join(str(item) for item in error)
            error_payload = [str(item) for item in error]
        else:
            message = str(error)
            error_payload = str(error)
        return _FakeResponse(
            {"status": "error", "error": error_payload, "message": message},
            status=status,
        )

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

    @asynccontextmanager
    async def _config_transaction():
        yield config_module.load_config()

    config_module.config_transaction = _config_transaction
    sys.modules[f"{package_name}.utils.config"] = config_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.config_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cleanup_optional_module("aiohttp", created_aiohttp_stub)

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
        with patch.object(config_routes, "load_config", return_value=cfg):
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
