import importlib.util
import sys
import types
import unittest
from pathlib import Path

from tests.api.harness import (
    bootstrap_test_package,
    install_aiohttp_stub,
    install_endpoint_policy_passthrough_stub,
    install_request_guards_stub,
    install_server_stub,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


def _load_tunnel_routes_module():
    module_path = Path(__file__).resolve().parents[2] / "api" / "tunnel_routes.py"
    package_name = "dist_tunnel_routes_testpkg"

    bootstrap_test_package(package_name, with_api=True, with_utils=True)
    install_server_stub()
    install_aiohttp_stub(lambda payload, status=200: _FakeResponse(payload, status=status))

    cloudflare_module = types.ModuleType(f"{package_name}.utils.cloudflare")

    class _TunnelManager:
        def get_status(self):
            return {"status": "running"}

        async def start_tunnel(self):
            return {"status": "running", "public_url": "https://example.trycloudflare.com"}

        async def stop_tunnel(self):
            return {"status": "stopped"}

    cloudflare_module.cloudflare_tunnel_manager = _TunnelManager()
    sys.modules[f"{package_name}.utils.cloudflare"] = cloudflare_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: {"master": {"host": "https://master.example.com"}}
    sys.modules[f"{package_name}.utils.config"] = config_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status=status)

    network_module.handle_api_error = _handle_api_error
    sys.modules[f"{package_name}.utils.network"] = network_module

    install_request_guards_stub(package_name)
    install_endpoint_policy_passthrough_stub(package_name)

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.tunnel_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tunnel_routes = _load_tunnel_routes_module()


class TunnelRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_endpoint_returns_tunnel_and_master_host(self):
        response = await tunnel_routes.tunnel_status_endpoint(request=None)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["status"], "success")
        self.assertEqual(response.payload["tunnel"]["status"], "running")
        self.assertEqual(response.payload["master_host"], "https://master.example.com")

    async def test_start_and_stop_endpoints_return_success_payload(self):
        start_response = await tunnel_routes.tunnel_start_endpoint(request=None)
        stop_response = await tunnel_routes.tunnel_stop_endpoint(request=None)

        self.assertEqual(start_response.status, 200)
        self.assertEqual(start_response.payload["tunnel"]["status"], "running")
        self.assertEqual(stop_response.status, 200)
        self.assertEqual(stop_response.payload["tunnel"]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
