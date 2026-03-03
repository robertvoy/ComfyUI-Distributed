import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _bootstrap_package(package_name):
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    workers_pkg = types.ModuleType(f"{package_name}.workers")
    workers_pkg.__path__ = []
    sys.modules[f"{package_name}.workers"] = workers_pkg

    process_pkg = types.ModuleType(f"{package_name}.workers.process")
    process_pkg.__path__ = []
    sys.modules[f"{package_name}.workers.process"] = process_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg


def _load_module(package_name, module_rel_path, module_name):
    module_path = Path(__file__).resolve().parents[3] / module_rel_path
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_lifecycle_module():
    package_name = "dist_lifecycle_testpkg"
    _bootstrap_package(package_name)

    state = {
        "config": {"settings": {"stop_workers_on_master_exit": True}, "managed_processes": {}},
        "saved_configs": [],
        "launch_calls": [],
        "terminated": [],
        "alive_pids": set(),
    }

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: state["config"]
    config_module.save_config = lambda cfg: state["saved_configs"].append(dict(cfg))
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.PROCESS_TERMINATION_TIMEOUT = 5.0
    constants_module.PROCESS_WAIT_TIMEOUT = 1.0
    constants_module.WORKER_CHECK_INTERVAL = 0.01
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    process_module = types.ModuleType(f"{package_name}.utils.process")

    class _FakeProcess:
        def __init__(self, pid=1234, poll_result=None):
            self.pid = pid
            self._poll_result = poll_result
            self.returncode = poll_result

        def poll(self):
            return self._poll_result

    def _launch_process_with_timeout(command, timeout_seconds=10.0, **kwargs):
        state["launch_calls"].append((list(command), float(timeout_seconds), dict(kwargs)))
        return _FakeProcess(pid=4321, poll_result=None)

    def _terminate_process(process, timeout=5):
        state["terminated"].append((process.pid, timeout))
        process._poll_result = 0
        process.returncode = 0

    process_module.launch_process_with_timeout = _launch_process_with_timeout
    process_module.terminate_process = _terminate_process
    process_module.is_process_alive = lambda pid: int(pid) in state["alive_pids"]
    process_module.get_python_executable = lambda: "/usr/bin/python3"
    process_module._FakeProcess = _FakeProcess
    sys.modules[f"{package_name}.utils.process"] = process_module

    lifecycle_module = _load_module(package_name, "workers/process/lifecycle.py", "workers.process.lifecycle")
    return lifecycle_module, state, process_module


lifecycle_module, test_state, process_stub_module = _load_lifecycle_module()


class _FakeManager:
    def __init__(self, comfy_root):
        self._comfy_root = comfy_root
        self.processes = {}
        self.save_calls = 0

    def find_comfy_root(self):
        return self._comfy_root

    def build_launch_command(self, worker_config, _comfy_root):
        return ["python", "main.py", "--port", str(worker_config["port"])]

    def save_processes(self):
        self.save_calls += 1


class ProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        test_state["config"] = {"settings": {"stop_workers_on_master_exit": True}, "managed_processes": {}}
        test_state["saved_configs"].clear()
        test_state["launch_calls"].clear()
        test_state["terminated"].clear()
        test_state["alive_pids"].clear()

    def test_launch_worker_uses_monitor_wrapper_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _FakeManager(tmpdir)
            lifecycle = lifecycle_module.ProcessLifecycle(manager)

            pid = lifecycle.launch_worker(
                {
                    "id": "1",
                    "name": "Worker A",
                    "port": 8189,
                    "cuda_device": 2,
                }
            )

            self.assertEqual(pid, 4321)
            self.assertEqual(manager.save_calls, 1)
            self.assertIn("1", manager.processes)
            self.assertTrue(manager.processes["1"]["is_monitor"])
            self.assertIn("logs/workers", manager.processes["1"]["log_file"])
            self.assertEqual(len(test_state["launch_calls"]), 1)
            launched_command, timeout_seconds, launch_kwargs = test_state["launch_calls"][0]
            self.assertEqual(timeout_seconds, 1.0)
            self.assertEqual(launched_command[0], "/usr/bin/python3")
            self.assertTrue(launched_command[1].endswith("workers/worker_monitor.py"))
            self.assertEqual(launch_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")

    def test_launch_worker_skips_monitor_when_disabled(self):
        test_state["config"] = {"settings": {"stop_workers_on_master_exit": False}, "managed_processes": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _FakeManager(tmpdir)
            lifecycle = lifecycle_module.ProcessLifecycle(manager)

            lifecycle.launch_worker(
                {
                    "id": "2",
                    "name": "Worker-B",
                    "port": 8190,
                    "cuda_device": 0,
                }
            )

            launched_command, _, _ = test_state["launch_calls"][0]
            self.assertEqual(launched_command[0], "python")
            self.assertFalse(manager.processes["2"]["is_monitor"])

    def test_stop_worker_uses_fallback_terminate_when_tree_kill_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _FakeManager(tmpdir)
            lifecycle = lifecycle_module.ProcessLifecycle(manager)
            running_process = process_stub_module._FakeProcess(pid=88, poll_result=None)
            manager.processes["7"] = {
                "pid": 88,
                "process": running_process,
                "started_at": 0.0,
                "config": {},
            }
            lifecycle._kill_process_tree = lambda _pid: False

            ok, message = lifecycle.stop_worker("7")

            self.assertTrue(ok)
            self.assertIn("fallback", message.lower())
            self.assertEqual(test_state["terminated"], [(88, 5.0)])
            self.assertNotIn("7", manager.processes)

    def test_stop_worker_without_process_uses_tree_kill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _FakeManager(tmpdir)
            lifecycle = lifecycle_module.ProcessLifecycle(manager)
            manager.processes["9"] = {
                "pid": 99,
                "process": None,
                "started_at": 0.0,
                "config": {},
            }
            lifecycle._kill_process_tree = lambda _pid: True

            ok, message = lifecycle.stop_worker("9")

            self.assertTrue(ok)
            self.assertEqual(message, "Worker stopped")
            self.assertNotIn("9", manager.processes)

    def test_check_worker_process_falls_back_to_pid_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _FakeManager(tmpdir)
            lifecycle = lifecycle_module.ProcessLifecycle(manager)
            lifecycle._is_process_running = lambda pid: int(pid) == 42

            running, from_subprocess = lifecycle._check_worker_process(
                "any",
                {"pid": 42, "process": None},
            )
            self.assertTrue(running)
            self.assertFalse(from_subprocess)


if __name__ == "__main__":
    unittest.main()
