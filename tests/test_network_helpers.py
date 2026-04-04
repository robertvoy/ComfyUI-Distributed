import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_network_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "network.py"

    package_name = "dist_utils_testpkg"
    package_module = types.ModuleType(package_name)
    package_module.__path__ = []  # mark as package
    sys.modules[package_name] = package_module

    logging_module = types.ModuleType(f"{package_name}.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.logging"] = logging_module

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(address="127.0.0.1", port=8188, loop=None)
    )
    sys.modules["server"] = server_module

    if "aiohttp" not in sys.modules:
        aiohttp_module = types.ModuleType("aiohttp")

        class _TCPConnector:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _ClientSession:
            def __init__(self, *args, **kwargs):
                self.closed = False

            async def close(self):
                self.closed = True

        aiohttp_module.TCPConnector = _TCPConnector
        aiohttp_module.ClientSession = _ClientSession
        aiohttp_module.web = types.SimpleNamespace(
            json_response=lambda payload, status=200: {"payload": payload, "status": status}
        )
        sys.modules["aiohttp"] = aiohttp_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.network", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


network = _load_network_module()


class NetworkHelpersTests(unittest.TestCase):
    def test_normalize_host_strips_protocol_and_path(self):
        self.assertEqual(network.normalize_host("  https://example.com/a/b  "), "example.com")

    def test_normalize_host_keeps_none(self):
        self.assertIsNone(network.normalize_host(None))

    def test_build_worker_url_defaults_to_server_address(self):
        worker = {"id": "w1", "port": 8189}
        self.assertEqual(network.build_worker_url(worker, "/prompt"), "http://127.0.0.1:8189/prompt")

    def test_build_worker_url_cloud_defaults_to_https(self):
        worker = {"id": "w2", "host": "foo.proxy.runpod.net", "port": 443}
        self.assertEqual(network.build_worker_url(worker), "https://foo.proxy.runpod.net")

    def test_build_worker_url_keeps_explicit_scheme(self):
        worker = {"id": "w3", "host": "https://worker.example.com", "port": 1234}
        self.assertEqual(network.build_worker_url(worker, "/prompt"), "https://worker.example.com/prompt")

    def test_build_master_url_uses_https_for_cloud_host(self):
        cfg = {"master": {"host": "demo.proxy.runpod.net"}}
        prompt_server = types.SimpleNamespace(address="127.0.0.1", port=8188)
        self.assertEqual(
            network.build_master_url(config=cfg, prompt_server_instance=prompt_server),
            "https://demo.proxy.runpod.net",
        )

    def test_build_master_url_keeps_explicit_scheme(self):
        cfg = {"master": {"host": "https://master.example.com/"}}
        prompt_server = types.SimpleNamespace(address="127.0.0.1", port=8188)
        self.assertEqual(
            network.build_master_url(config=cfg, prompt_server_instance=prompt_server),
            "https://master.example.com",
        )

    def test_build_master_url_ignores_stale_saved_port_and_uses_runtime_port(self):
        cfg = {"master": {"host": "192.168.68.56", "port": 8001}}
        prompt_server = types.SimpleNamespace(address="127.0.0.1", port=8188)
        self.assertEqual(
            network.build_master_url(config=cfg, prompt_server_instance=prompt_server),
            "http://192.168.68.56:8188",
        )

    def test_build_master_url_keeps_explicit_port_in_host(self):
        cfg = {"master": {"host": "192.168.68.56:8001"}}
        prompt_server = types.SimpleNamespace(address="127.0.0.1", port=8188)
        self.assertEqual(
            network.build_master_url(config=cfg, prompt_server_instance=prompt_server),
            "http://192.168.68.56:8001",
        )

    def test_build_master_url_falls_back_to_server_address(self):
        cfg = {"master": {"host": "", "port": 8001}}
        prompt_server = types.SimpleNamespace(address="0.0.0.0", port=8190)
        self.assertEqual(
            network.build_master_url(config=cfg, prompt_server_instance=prompt_server),
            "http://127.0.0.1:8190",
        )


if __name__ == "__main__":
    unittest.main()
