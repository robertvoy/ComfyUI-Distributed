"""Managed worker PIDs are only adopted or killed when they are provably ours."""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = "dist_pid_identity_testpkg"


def _load_modules():
    for name in list(sys.modules):
        if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
            del sys.modules[name]
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_REPO_ROOT)]
    sys.modules[_PACKAGE] = package
    if "server" not in sys.modules:
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(address="127.0.0.1", port=8188, loop=None)
        )
        sys.modules["server"] = server

    config = __import__(f"{_PACKAGE}.utils.config", fromlist=["*"])
    logging_module = __import__(f"{_PACKAGE}.utils.logging", fromlist=["*"])
    scratch = tempfile.mkdtemp(prefix="pid-identity-import-")
    atexit.register(shutil.rmtree, scratch, True)
    config.CONFIG_FILE = os.path.join(scratch, "gpu_config.json")
    logging_module.CONFIG_FILE = config.CONFIG_FILE
    config.invalidate_config_cache()
    manager = __import__(f"{_PACKAGE}.workers.process_manager", fromlist=["*"])
    lifecycle = __import__(f"{_PACKAGE}.workers.process.lifecycle", fromlist=["*"])
    return config, logging_module, manager, lifecycle


config_module, logging_module, manager_module, lifecycle_module = _load_modules()


class WorkerPidIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = os.path.join(self.tmp.name, "gpu_config.json")
        for module in (config_module, logging_module):
            config_patch = patch.object(module, "CONFIG_FILE", self.config_path)
            config_patch.start()
            self.addCleanup(config_patch.stop)
        config_module.invalidate_config_cache()
        self.addCleanup(config_module.invalidate_config_cache)
        self.children = []
        self.addCleanup(self._kill_children)

    def _kill_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)

    def _spawn(self, *extra_args):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *extra_args])
        self.children.append(child)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                process = psutil.Process(child.pid)
                if not extra_args or extra_args[-1] in process.cmdline():
                    return child, process.create_time()
            except psutil.Error:
                pass
            time.sleep(0.05)
        self.fail("child process did not start")

    def _save_managed(self, managed):
        config = config_module.get_default_config()
        config["managed_processes"] = managed
        self.assertTrue(config_module.save_config(config))

    def _saved_managed(self):
        with open(self.config_path, encoding="utf-8") as handle:
            return json.load(handle).get("managed_processes")

    @staticmethod
    def _entry(pid, create_time, port=8199):
        return {
            "pid": pid,
            "create_time": create_time,
            "started_at": 1.0,
            "config": {"id": "w1", "name": "W1", "port": port},
            "log_file": None,
        }

    def _restored(self, child, create_time, port=8199):
        entry = self._entry(child.pid, create_time, port)
        entry["process"] = None
        return entry

    def test_restore_drops_a_reused_pid_and_persists_the_removal(self):
        child, create_time = self._spawn()
        self._save_managed({"w1": self._entry(child.pid, create_time + 1000.0)})
        manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)
        self.assertEqual(self._saved_managed(), {})
        self.assertIsNone(child.poll())

    def test_restore_keeps_a_pid_whose_creation_time_matches(self):
        child, create_time = self._spawn()
        self._save_managed({"w1": self._entry(child.pid, create_time)})
        manager = manager_module.WorkerProcessManager()
        self.assertEqual(manager.processes["w1"]["pid"], child.pid)

    def test_restore_tolerates_a_creation_time_that_moved_by_a_second(self):
        # Linux derives creation times from a boot time in whole seconds.
        child, create_time = self._spawn()
        self._save_managed({"w1": self._entry(child.pid, create_time + 1.0)})
        manager = manager_module.WorkerProcessManager()
        self.assertEqual(manager.processes["w1"]["pid"], child.pid)

    def test_a_worker_that_exits_before_the_kill_counts_as_stopped(self):
        child, create_time = self._spawn()
        child.kill()
        child.wait(timeout=10)
        manager = manager_module.WorkerProcessManager()
        self.assertTrue(manager._kill_process_tree(child.pid, expected_create_time=create_time))

    def test_legacy_entry_without_creation_time_is_verified_by_port(self):
        # `python -c` ignores the extra arguments; they only shape the command line.
        child, _ = self._spawn(os.path.join("ComfyUI", "main.py"), "--port", "8199")
        for port, expected in ((8199, True), (8200, False)):
            with self.subTest(port=port):
                self._save_managed({"w1": self._entry(child.pid, None, port=port)})
                manager = manager_module.WorkerProcessManager()
                self.assertEqual("w1" in manager.processes, expected)

    def test_legacy_port_match_alone_does_not_prove_a_comfyui_worker(self):
        # A reused PID running some other program that happens to take --port.
        child, _ = self._spawn("--port", "8199")
        self._save_managed({"w1": self._entry(child.pid, None, port=8199)})
        manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)
        self.assertIsNone(child.poll())

    def test_stop_worker_never_kills_a_reused_pid(self):
        child, create_time = self._spawn()
        manager = manager_module.WorkerProcessManager()
        manager.processes["w1"] = self._restored(child, create_time + 1000.0)
        success, _message = manager.stop_worker("w1")
        self.assertFalse(success)
        self.assertNotIn("w1", manager.processes)
        time.sleep(0.2)
        self.assertIsNone(child.poll(), "an unrelated process was killed")

    def test_stop_worker_kills_a_verified_restored_pid(self):
        child, create_time = self._spawn()
        manager = manager_module.WorkerProcessManager()
        manager.processes["w1"] = self._restored(child, create_time)
        success, _message = manager.stop_worker("w1")
        self.assertTrue(success)
        child.wait(timeout=10)

    def test_cleanup_all_skips_unverified_pids(self):
        child, create_time = self._spawn()
        manager = manager_module.WorkerProcessManager()
        manager.processes["w1"] = self._restored(child, create_time + 1000.0)
        manager.cleanup_all()
        time.sleep(0.2)
        self.assertIsNone(child.poll(), "an unrelated process was killed")
        self.assertEqual(manager.processes, {})

    def test_get_managed_workers_drops_and_persists_unverified_pids(self):
        child, create_time = self._spawn()
        self._save_managed({})
        manager = manager_module.WorkerProcessManager()
        manager.processes["w1"] = self._restored(child, create_time + 1000.0)
        self._save_managed({"w1": self._entry(child.pid, create_time + 1000.0)})
        self.assertEqual(manager.get_managed_workers(), {})
        self.assertEqual(self._saved_managed(), {})

    def test_launch_records_creation_time_with_the_pid(self):
        config = config_module.get_default_config()
        config["settings"]["stop_workers_on_master_exit"] = False
        config["workers"] = [{"id": "w1", "name": "W1", "port": 8199}]
        self.assertTrue(config_module.save_config(config))
        manager = manager_module.WorkerProcessManager()
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        with patch.object(manager, "find_comfy_root", return_value=self.tmp.name), \
             patch.object(manager, "build_launch_command", return_value=command), \
             patch.object(lifecycle_module, "validate_worker_port"), \
             patch.object(lifecycle_module, "get_server_port", return_value=8188):
            pid = manager.launch_worker({"id": "w1", "name": "W1", "port": 8199})
            self.assertEqual(manager.get_worker_log_dir(), os.path.join(self.tmp.name, "logs", "workers"))
        process = manager.processes["w1"]["process"]
        self.children.append(process)
        expected = psutil.Process(pid).create_time()
        self.assertAlmostEqual(manager.processes["w1"]["create_time"], expected, places=2)
        self.assertAlmostEqual(self._saved_managed()["w1"]["create_time"], expected, places=2)
        self.assertTrue(manager.stop_worker("w1")[0])
        process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
