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
from unittest.mock import MagicMock, patch

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
process_module = sys.modules[f"{_PACKAGE}.utils.process"]


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

    def test_start_id_proves_identity_when_creation_times_disagree(self):
        # WSL2: a restarted master read a creation time 13 s away from the saved
        # one; the boot id and start ticks (Linux) still match exactly.
        child, create_time = self._spawn()
        entry = self._entry(child.pid, create_time - 13.0)
        entry["start_id"] = "boot-1:4242"
        self._save_managed({"w1": entry})
        with patch.object(lifecycle_module, "get_process_start_id", return_value="boot-1:4242"):
            manager = manager_module.WorkerProcessManager()
            self.assertEqual(manager.processes["w1"]["pid"], child.pid)
            success, _message = manager.stop_worker("w1")
        self.assertTrue(success)
        child.wait(timeout=10)

    def test_a_different_start_id_is_final_even_for_a_comfyui_worker_on_its_port(self):
        child, create_time = self._spawn(os.path.join("ComfyUI", "main.py"), "--port", "8199")
        entry = self._entry(child.pid, create_time)
        entry["start_id"] = "boot-1:4242"
        self._save_managed({"w1": entry})
        with patch.object(lifecycle_module, "get_process_start_id", return_value="boot-1:9999"):
            manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)
        self.assertIsNone(child.poll())

    def test_a_creation_time_mismatch_is_final_even_for_a_comfyui_worker_on_its_port(self):
        # A reused PID that runs main.py --port 8199 must not be adopted (and later killed).
        child, create_time = self._spawn(os.path.join("ComfyUI", "main.py"), "--port", "8199")
        self._save_managed({"w1": self._entry(child.pid, create_time - 13.0)})
        with patch.object(lifecycle_module, "get_process_start_id", return_value=None):
            manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)
        self.assertIsNone(child.poll())

    def test_a_start_id_that_cannot_be_read_here_never_falls_back_to_the_command_line(self):
        child, _ = self._spawn(os.path.join("ComfyUI", "main.py"), "--port", "8199")
        entry = self._entry(child.pid, None)
        entry["start_id"] = "boot-1:4242"
        self._save_managed({"w1": entry})
        with patch.object(lifecycle_module, "get_process_start_id", return_value=None):
            manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)

    def test_a_saved_start_id_that_cannot_be_read_fails_closed_despite_a_matching_creation_time(self):
        child, create_time = self._spawn()
        entry = self._entry(child.pid, create_time)
        entry["start_id"] = "boot-1:4242"
        self._save_managed({"w1": entry})
        with patch.object(lifecycle_module, "get_process_start_id", return_value=None):
            manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)
        self.assertIsNone(child.poll())

    def test_a_start_id_read_after_the_process_was_replaced_is_not_recorded(self):
        replaced = MagicMock()
        replaced.is_running.return_value = False
        self._save_managed({"w1": self._entry(4242, 123.0)})
        with patch.object(lifecycle_module, "open_verified_process", return_value=replaced), \
             patch.object(lifecycle_module, "get_process_start_id", return_value="boot-1:5"):
            manager = manager_module.WorkerProcessManager()
        self.assertNotIn("w1", manager.processes)
        self.assertEqual(self._saved_managed(), {})

    def test_tree_kill_signals_the_verified_handle_without_looking_the_pid_up_again(self):
        handle = MagicMock()
        handle.create_time.return_value = 100.0
        handle.children.return_value = []
        manager = manager_module.WorkerProcessManager()
        with patch.object(lifecycle_module.psutil, "Process", side_effect=AssertionError("PID looked up again")):
            self.assertTrue(manager._kill_process_tree(4242, expected_create_time=100.0, handle=handle))
        handle.terminate.assert_called_once()

    def test_entry_verified_by_creation_time_gains_and_persists_a_start_id(self):
        child, create_time = self._spawn()
        self._save_managed({"w1": self._entry(child.pid, create_time)})
        with patch.object(lifecycle_module, "get_process_start_id", return_value="boot-1:7"):
            manager = manager_module.WorkerProcessManager()
        self.assertEqual(manager.processes["w1"]["start_id"], "boot-1:7")
        # Saved right away, not only on a later save.
        self.assertEqual(self._saved_managed()["w1"]["start_id"], "boot-1:7")

    def test_legacy_entry_verified_by_port_gains_and_persists_a_start_id(self):
        child, _ = self._spawn(os.path.join("ComfyUI", "main.py"), "--port", "8199")
        self._save_managed({"w1": self._entry(child.pid, None)})
        with patch.object(lifecycle_module, "get_process_start_id", return_value="boot-1:8"):
            manager = manager_module.WorkerProcessManager()
        self.assertEqual(manager.processes["w1"]["pid"], child.pid)
        self.assertEqual(self._saved_managed()["w1"]["start_id"], "boot-1:8")

    @unittest.skipUnless(sys.platform.startswith("linux"), "start ids are read from /proc")
    def test_real_start_id_restores_a_worker_whose_creation_time_drifted(self):
        child, create_time = self._spawn()
        start_id = process_module.get_process_start_id(child.pid)
        self.assertRegex(start_id, r"^[0-9a-f-]+:\d+$")
        entry = self._entry(child.pid, create_time - 13.0)
        entry["start_id"] = start_id
        self._save_managed({"w1": entry})
        manager = manager_module.WorkerProcessManager()
        self.assertEqual(manager.processes["w1"]["pid"], child.pid)
        self.assertTrue(manager.stop_worker("w1")[0])
        child.wait(timeout=10)

    def test_start_ticks_are_read_from_field_22_after_a_command_name_with_spaces(self):
        rest = ["S"] + [str(number) for number in range(4, 22)] + ["424242", "99"]
        stat = b"321 (odd ) name) " + " ".join(rest).encode()
        self.assertEqual(process_module._parse_start_ticks(stat), 424242)

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
        expected_start_id = process_module.get_process_start_id(pid)
        self.assertEqual(manager.processes["w1"]["start_id"], expected_start_id)
        self.assertEqual(self._saved_managed()["w1"]["start_id"], expected_start_id)
        self.assertTrue(manager.stop_worker("w1")[0])
        process.wait(timeout=10)

    def test_process_that_exits_after_verification_counts_as_already_stopped(self):
        child, create_time = self._spawn()
        manager = manager_module.WorkerProcessManager()
        manager.processes["w1"] = self._restored(child, create_time)
        verified = MagicMock()
        verified.create_time.side_effect = psutil.NoSuchProcess(child.pid)
        with patch.object(lifecycle_module.ProcessLifecycle, "_open_restored_process", return_value=verified):
            success, message = manager.stop_worker("w1")
        self.assertFalse(success)
        self.assertEqual(message, "Worker already stopped")
        self.assertNotIn("w1", manager.processes)
        self.assertIsNone(child.poll(), "nothing should be killed")

    def test_save_processes_cannot_undo_a_committed_transaction(self):
        import asyncio
        import threading

        child, create_time = self._spawn()
        self._save_managed({})
        manager = manager_module.WorkerProcessManager()
        manager.processes["w1"] = self._restored(child, create_time)
        thread = threading.Thread(target=manager.save_processes)

        async def scenario():
            async with config_module.config_transaction() as cfg:
                thread.start()
                await asyncio.sleep(0.3)
                self.assertEqual(self._saved_managed(), {}, "saved inside the transaction")
                cfg["master"]["host"] = "from-transaction"
            await asyncio.to_thread(thread.join, 10)

        asyncio.run(scenario())
        with open(self.config_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["master"]["host"], "from-transaction")
        self.assertIn("w1", saved["managed_processes"])

    def test_cleanup_all_clears_managed_processes_under_the_lock(self):
        self._save_managed({"w1": self._entry(424242, 1.0)})
        manager = manager_module.WorkerProcessManager()
        manager.processes.clear()
        with patch.object(lifecycle_module, "config_update", wraps=lifecycle_module.config_update) as update:
            manager.cleanup_all()
        update.assert_called_once()
        self.assertEqual(self._saved_managed(), {})


if __name__ == "__main__":
    unittest.main()
