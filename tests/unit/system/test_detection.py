import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_detection_module():
    module_path = Path(__file__).resolve().parents[3] / "workers" / "detection.py"
    package_name = "dist_det_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    workers_pkg = types.ModuleType(f"{package_name}.workers")
    workers_pkg.__path__ = []
    sys.modules[f"{package_name}.workers"] = workers_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.normalize_host = lambda value: value
    network_module.build_worker_url = lambda worker, endpoint="": f"http://{worker.get('host', 'localhost')}{endpoint}"

    async def _fake_session():
        raise RuntimeError("network calls not used in these tests")

    network_module.get_client_session = _fake_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    created_aiohttp_stub = False
    if "aiohttp" not in sys.modules:
        created_aiohttp_stub = True
        aiohttp_module = types.ModuleType("aiohttp")

        class _ClientTimeout:
            def __init__(self, total=None):
                pass

        aiohttp_module.ClientTimeout = _ClientTimeout
        sys.modules["aiohttp"] = aiohttp_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.workers.detection",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    if created_aiohttp_stub:
        sys.modules.pop("aiohttp", None)

    return module


detection = _load_detection_module()


# ---------------------------------------------------------------------------
# is_docker_environment
# ---------------------------------------------------------------------------

class IsDockerEnvironmentTests(unittest.TestCase):
    def test_true_when_dockerenv_file_exists(self):
        with patch.object(detection.os.path, "exists", return_value=True), \
             patch.dict(detection.os.environ, {}, clear=True), \
             patch.object(detection.platform, "node", return_value="my-laptop"):
            self.assertTrue(detection.is_docker_environment())

    def test_true_when_docker_container_env_var_is_set(self):
        with patch.object(detection.os.path, "exists", return_value=False), \
             patch.dict(detection.os.environ, {"DOCKER_CONTAINER": "1"}, clear=True), \
             patch.object(detection.platform, "node", return_value="my-laptop"):
            self.assertTrue(detection.is_docker_environment())

    def test_true_when_platform_node_contains_docker(self):
        with patch.object(detection.os.path, "exists", return_value=False), \
             patch.dict(detection.os.environ, {}, clear=True), \
             patch.object(detection.platform, "node", return_value="my-docker-host"):
            self.assertTrue(detection.is_docker_environment())

    def test_false_when_none_of_the_signals_are_present(self):
        with patch.object(detection.os.path, "exists", return_value=False), \
             patch.dict(detection.os.environ, {}, clear=True), \
             patch.object(detection.platform, "node", return_value="my-laptop"):
            self.assertFalse(detection.is_docker_environment())

    def test_docker_node_name_is_case_insensitive(self):
        with patch.object(detection.os.path, "exists", return_value=False), \
             patch.dict(detection.os.environ, {}, clear=True), \
             patch.object(detection.platform, "node", return_value="My-Docker-Box"):
            self.assertTrue(detection.is_docker_environment())

    def test_docker_env_var_empty_string_is_falsy(self):
        """An empty DOCKER_CONTAINER env var should NOT trigger docker detection."""
        with patch.object(detection.os.path, "exists", return_value=False), \
             patch.dict(detection.os.environ, {"DOCKER_CONTAINER": ""}, clear=True), \
             patch.object(detection.platform, "node", return_value="my-laptop"):
            self.assertFalse(detection.is_docker_environment())


# ---------------------------------------------------------------------------
# is_runpod_environment
# ---------------------------------------------------------------------------

class IsRunpodEnvironmentTests(unittest.TestCase):
    def test_true_when_runpod_pod_id_is_set(self):
        with patch.dict(detection.os.environ, {"RUNPOD_POD_ID": "pod-abc"}, clear=True):
            self.assertTrue(detection.is_runpod_environment())

    def test_true_when_runpod_api_key_is_set(self):
        with patch.dict(detection.os.environ, {"RUNPOD_API_KEY": "key-xyz"}, clear=True):
            self.assertTrue(detection.is_runpod_environment())

    def test_true_when_both_vars_are_set(self):
        with patch.dict(
            detection.os.environ,
            {"RUNPOD_POD_ID": "pod-abc", "RUNPOD_API_KEY": "key-xyz"},
            clear=True,
        ):
            self.assertTrue(detection.is_runpod_environment())

    def test_false_when_neither_var_is_set(self):
        with patch.dict(detection.os.environ, {}, clear=True):
            self.assertFalse(detection.is_runpod_environment())

    def test_true_when_pod_id_is_empty_string(self):
        """is not None check means even empty string counts as detected."""
        with patch.dict(detection.os.environ, {"RUNPOD_POD_ID": ""}, clear=True):
            self.assertTrue(detection.is_runpod_environment())


# ---------------------------------------------------------------------------
# is_local_worker (synchronous paths only)
# ---------------------------------------------------------------------------

class IsLocalWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_true_for_localhost_host(self):
        result = await detection.is_local_worker({"host": "localhost", "port": 8188})
        self.assertTrue(result)

    async def test_true_for_127_0_0_1(self):
        result = await detection.is_local_worker({"host": "127.0.0.1", "port": 8188})
        self.assertTrue(result)

    async def test_true_for_0_0_0_0(self):
        result = await detection.is_local_worker({"host": "0.0.0.0", "port": 8188})  # nosec B104 - explicit wildcard host test case
        self.assertTrue(result)

    async def test_true_when_type_is_local(self):
        result = await detection.is_local_worker({"type": "local", "host": "remote.example.com"})
        self.assertTrue(result)

    async def test_false_for_remote_host(self):
        result = await detection.is_local_worker({"host": "remote.example.com", "port": 8188})
        self.assertFalse(result)

    async def test_true_when_no_host_key(self):
        """Missing host defaults to 'localhost'."""
        result = await detection.is_local_worker({"port": 8188})
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# get_machine_id
# ---------------------------------------------------------------------------

class GetMachineIdTests(unittest.TestCase):
    def test_returns_a_string(self):
        result = detection.get_machine_id()
        self.assertIsInstance(result, str)

    def test_returns_non_empty_string(self):
        result = detection.get_machine_id()
        self.assertTrue(len(result) > 0)

    def test_stable_across_calls(self):
        r1 = detection.get_machine_id()
        r2 = detection.get_machine_id()
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
