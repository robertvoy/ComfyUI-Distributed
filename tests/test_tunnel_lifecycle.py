"""Cloudflare tunnel lifecycle: no lock re-entry deadlock, no killing of reused PIDs."""
import asyncio
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
_PACKAGE = "dist_tunnel_testpkg"


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
    # Importing the cloudflare package builds a manager that reads/writes the
    # config, so point it at a scratch file first.
    scratch = tempfile.mkdtemp(prefix="tunnel-test-import-")
    atexit.register(shutil.rmtree, scratch, True)
    config.CONFIG_FILE = os.path.join(scratch, "gpu_config.json")
    logging_module.CONFIG_FILE = config.CONFIG_FILE
    config.invalidate_config_cache()
    state = __import__(f"{_PACKAGE}.utils.cloudflare.state", fromlist=["*"])
    tunnel = __import__(f"{_PACKAGE}.utils.cloudflare.tunnel", fromlist=["*"])
    return config, logging_module, state, tunnel


config_module, logging_module, state_module, tunnel_module = _load_modules()
process_module = sys.modules[f"{_PACKAGE}.utils.process"]

_URL_SCRIPT = "import time; print('INF https://abc-def.trycloudflare.com', flush=True); time.sleep(60)"
_SLEEP_SCRIPT = "import time; print('starting', flush=True); time.sleep(60)"
_EXIT_SCRIPT = "print('cloudflared: failed to connect', flush=True)"


class TunnelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = os.path.join(self.tmp.name, "gpu_config.json")
        self.config_patch = patch.object(config_module, "CONFIG_FILE", self.config_path)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        log_patch = patch.object(logging_module, "CONFIG_FILE", self.config_path)
        log_patch.start()
        self.addCleanup(log_patch.stop)
        config_module.invalidate_config_cache()
        self.addCleanup(config_module.invalidate_config_cache)
        self.children = []
        self.addCleanup(self._kill_children)
        # Keep tunnel log files out of the repository.
        base_dir_patch = patch.object(
            tunnel_module.CloudflareTunnelManager, "base_dir", new=property(lambda _self: self.tmp.name)
        )
        base_dir_patch.start()
        self.addCleanup(base_dir_patch.stop)
        port_patch = patch.object(tunnel_module, "get_server_port", return_value=8188)
        port_patch.start()
        self.addCleanup(port_patch.stop)

    def _kill_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)

    def _spawn_sleeper(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.children.append(child)
        # Wait until the OS reports a creation time for it.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                return child, psutil.Process(child.pid).create_time()
            except psutil.Error:
                time.sleep(0.05)
        self.fail("child process did not start")

    def _tunnel_config(self):
        with open(self.config_path, encoding="utf-8") as handle:
            return json.load(handle).get("tunnel", {})

    def _fake_cloudflared(self, script):
        real_popen = subprocess.Popen

        def popen(_cmd, **kwargs):
            process = real_popen([sys.executable, "-c", script], **kwargs)
            self.children.append(process)
            return process

        fake_subprocess = types.SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
        return [
            patch.object(tunnel_module, "subprocess", fake_subprocess),
            patch.object(tunnel_module, "ensure_binary", return_value=sys.executable),
        ]

    def _start_patches(self, patches):
        for item in patches:
            item.start()
            self.addCleanup(item.stop)


class TunnelStartStopTests(TunnelTestCase):
    async def test_start_failure_before_url_raises_instead_of_deadlocking(self):
        self._start_patches(self._fake_cloudflared(_EXIT_SCRIPT))
        manager = tunnel_module.CloudflareTunnelManager()
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(manager.start_tunnel(), timeout=15)
        self.assertFalse(manager._lock.locked())
        self.assertTrue(manager.last_error)
        self.assertEqual(manager.status, "stopped")
        # Later lifecycle calls must not hang on the lock either.
        self.assertEqual((await asyncio.wait_for(manager.stop_tunnel(), timeout=5))["status"], "stopped")
        self.assertIsNone(self._tunnel_config().get("pid"))

    async def test_start_timeout_terminates_the_spawned_cloudflared(self):
        self._start_patches(self._fake_cloudflared(_SLEEP_SCRIPT))
        manager = tunnel_module.CloudflareTunnelManager()
        with patch.object(tunnel_module, "TUNNEL_START_TIMEOUT", 1.0):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                await asyncio.wait_for(manager.start_tunnel(), timeout=15)
        spawned = self.children[-1]
        self.assertIsNotNone(spawned.poll(), "cloudflared from a failed start is still running")
        self.assertIsNone(manager.pid)
        self.assertIsNone(self._tunnel_config().get("pid"))
        self.assertIn("Timed out", manager.get_status()["last_error"])

    async def test_successful_start_persists_identity_and_stop_clears_it(self):
        self._start_patches(self._fake_cloudflared(_URL_SCRIPT))
        manager = tunnel_module.CloudflareTunnelManager()
        result = await asyncio.wait_for(manager.start_tunnel(), timeout=15)
        spawned = self.children[-1]
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["public_url"], "https://abc-def.trycloudflare.com")
        saved = self._tunnel_config()
        self.assertEqual(saved["pid"], spawned.pid)
        self.assertAlmostEqual(saved["pid_create_time"], psutil.Process(spawned.pid).create_time(), places=2)
        self.assertEqual(saved["pid_start_id"], process_module.get_process_start_id(spawned.pid))

        await asyncio.wait_for(manager.stop_tunnel(), timeout=10)
        self.assertIsNotNone(spawned.poll())
        saved = self._tunnel_config()
        self.assertIsNone(saved["pid"])
        self.assertIsNone(saved["pid_create_time"])
        self.assertIsNone(saved["pid_start_id"])


class TunnelPidIdentityTests(TunnelTestCase):
    def test_clear_tunnel_state_forgets_the_pid(self):
        state_module.persist_tunnel_state(status="running", pid=424242, pid_create_time=1.0, log_file="x.log")
        state_module.clear_tunnel_state(log_file="x.log")
        saved = self._tunnel_config()
        self.assertIsNone(saved["pid"])
        self.assertIsNone(saved["pid_create_time"])
        self.assertEqual(saved["status"], "stopped")

    async def test_restore_forgets_a_reused_pid_and_never_kills_it(self):
        child, create_time = self._spawn_sleeper()
        for saved_create_time in (create_time + 1000.0, None):
            with self.subTest(saved_create_time=saved_create_time):
                # A live PID that is not the saved cloudflared (wrong creation
                # time, or a non-cloudflared executable for legacy state).
                state_module.persist_tunnel_state(
                    status="running", pid=child.pid, pid_create_time=saved_create_time
                )
                manager = tunnel_module.CloudflareTunnelManager()
                self.assertIsNone(manager.pid)
                self.assertIsNone(self._tunnel_config()["pid"])
                await asyncio.wait_for(manager.stop_tunnel(), timeout=5)
                self.assertIsNone(child.poll(), "an unrelated process was killed")

    async def test_stop_does_not_kill_a_restored_pid_that_no_longer_matches(self):
        child, create_time = self._spawn_sleeper()
        manager = tunnel_module.CloudflareTunnelManager()
        manager.pid = child.pid
        manager.pid_create_time = create_time + 1000.0
        await asyncio.wait_for(manager.stop_tunnel(), timeout=5)
        self.assertIsNone(child.poll(), "an unrelated process was killed")
        self.assertIsNone(manager.pid)

    async def test_restore_adopts_and_stops_a_verified_cloudflared_pid(self):
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid, pid_create_time=create_time)
        manager = tunnel_module.CloudflareTunnelManager()
        self.assertEqual(manager.pid, child.pid)
        self.assertEqual(manager.get_status()["status"], "running")
        await asyncio.wait_for(manager.stop_tunnel(), timeout=10)
        child.wait(timeout=10)
        self.assertIsNone(self._tunnel_config()["pid"])

    def test_a_local_worker_never_rewrites_the_masters_tunnel_state(self):
        # Local workers share gpu_config.json and build their own manager at import.
        state_module.persist_tunnel_state(status="running", pid=424242, pid_create_time=1.0,
                                          public_url="https://x.trycloudflare.com")
        with patch.dict(os.environ, {"COMFYUI_IS_WORKER": "1"}),                 patch.object(tunnel_module, "_open_tunnel_process", return_value=None):
            manager = tunnel_module.CloudflareTunnelManager()
        self.assertIsNone(manager.pid)
        saved = self._tunnel_config()
        self.assertEqual(saved["pid"], 424242)
        self.assertEqual(saved["public_url"], "https://x.trycloudflare.com")

    def test_restore_tolerates_a_creation_time_that_moved_by_a_second(self):
        # Linux reports creation times from a boot time in whole seconds.
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid, pid_create_time=create_time - 1.0)
        manager = tunnel_module.CloudflareTunnelManager()
        self.assertEqual(manager.pid, child.pid)

    async def test_start_id_proves_identity_when_creation_times_disagree(self):
        # WSL2: a restarted master read a creation time 13 s away from the saved one.
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid,
                                          pid_create_time=create_time - 13.0, pid_start_id="boot-1:4242")
        with patch.object(process_module, "get_process_start_id", return_value="boot-1:4242"):
            manager = tunnel_module.CloudflareTunnelManager()
            self.assertEqual(manager.pid, child.pid)
            await asyncio.wait_for(manager.stop_tunnel(), timeout=10)
        child.wait(timeout=10)
        self.assertIsNone(self._tunnel_config()["pid_start_id"])

    async def test_a_different_start_id_is_final(self):
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid,
                                          pid_create_time=create_time, pid_start_id="boot-1:4242")
        with patch.object(process_module, "get_process_start_id", return_value="boot-1:9999"):
            manager = tunnel_module.CloudflareTunnelManager()
            self.assertIsNone(manager.pid)
            await asyncio.wait_for(manager.stop_tunnel(), timeout=5)
        self.assertIsNone(child.poll(), "an unrelated process was killed")

    def test_a_saved_start_id_that_cannot_be_read_fails_closed_despite_a_matching_creation_time(self):
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid,
                                          pid_create_time=create_time, pid_start_id="boot-1:4242")
        with patch.object(process_module, "get_process_start_id", return_value=None):
            manager = tunnel_module.CloudflareTunnelManager()
        self.assertIsNone(manager.pid)
        self.assertIsNone(child.poll())

    def test_tunnel_verified_by_creation_time_gains_and_persists_a_start_id(self):
        child, create_time = self._spawn_sleeper()
        for is_worker, expected in ((False, "boot-1:9"), (True, None)):
            with self.subTest(is_worker=is_worker):
                state_module.persist_tunnel_state(status="running", pid=child.pid,
                                                  pid_create_time=create_time, pid_start_id=None)
                env = {"COMFYUI_IS_WORKER": "1"} if is_worker else {}
                with patch.dict(os.environ, env), \
                     patch.object(tunnel_module, "get_process_start_id", return_value="boot-1:9"):
                    manager = tunnel_module.CloudflareTunnelManager()
                self.assertEqual(manager.pid, child.pid)
                self.assertEqual(manager.pid_start_id, "boot-1:9")
                # Only the master writes the shared tunnel state.
                self.assertEqual(self._tunnel_config()["pid_start_id"], expected)
                self.assertEqual(self._tunnel_config()["pid"], child.pid)

    def test_a_start_id_read_after_the_process_was_replaced_is_not_recorded(self):
        replaced = MagicMock()
        replaced.is_running.return_value = False
        state_module.persist_tunnel_state(status="running", pid=4242, pid_create_time=123.0)
        with patch.object(tunnel_module, "_open_tunnel_process", return_value=replaced), \
             patch.object(tunnel_module, "get_process_start_id", return_value="boot-1:5"):
            manager = tunnel_module.CloudflareTunnelManager()
        self.assertIsNone(manager.pid)
        self.assertIsNone(self._tunnel_config()["pid"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "start ids are read from /proc")
    def test_real_start_id_restores_a_tunnel_whose_creation_time_drifted(self):
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(
            status="running", pid=child.pid, pid_create_time=create_time - 13.0,
            pid_start_id=process_module.get_process_start_id(child.pid))
        manager = tunnel_module.CloudflareTunnelManager()
        self.assertEqual(manager.pid, child.pid)

    def test_stopping_a_restored_cloudflared_that_ignores_terminate_kills_it(self):
        handle = MagicMock()
        handle.wait.side_effect = [psutil.TimeoutExpired(5), None]
        tunnel_module._terminate_restored(handle, 4242, timeout=0.01)
        handle.terminate.assert_called_once()
        handle.kill.assert_called_once()

    def test_stopping_a_restored_cloudflared_that_already_exited_is_quiet(self):
        handle = MagicMock()
        handle.terminate.side_effect = psutil.NoSuchProcess(4242)
        tunnel_module._terminate_restored(handle, 4242, timeout=0.01)
        handle.kill.assert_not_called()


class TunnelWorkerRefusalTests(TunnelTestCase):
    async def test_a_worker_cannot_stop_the_masters_tunnel(self):
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid, pid_create_time=create_time,
                                          public_url="https://x.trycloudflare.com")
        before = self._tunnel_config()
        with patch.dict(os.environ, {"COMFYUI_IS_WORKER": "1"}):
            manager = tunnel_module.CloudflareTunnelManager()
            self.assertEqual(manager.pid, child.pid)
            with self.assertRaises(tunnel_module.TunnelManagedByMasterError):
                await asyncio.wait_for(manager.stop_tunnel(), timeout=5)
            # Status stays readable for workers.
            self.assertEqual(manager.get_status()["pid"], child.pid)
        self.assertIsNone(child.poll(), "a worker killed the master's cloudflared")
        self.assertEqual(self._tunnel_config(), before)
        self.assertEqual(manager.pid, child.pid)

    async def test_a_worker_cannot_start_a_tunnel(self):
        state_module.persist_tunnel_state(status="stopped", pid=None)
        before = self._tunnel_config()
        spawn = MagicMock()
        fake_subprocess = types.SimpleNamespace(Popen=spawn, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
        with patch.dict(os.environ, {"COMFYUI_IS_WORKER": "1"}), \
             patch.object(tunnel_module, "subprocess", fake_subprocess), \
             patch.object(tunnel_module, "ensure_binary", return_value=sys.executable) as binary:
            manager = tunnel_module.CloudflareTunnelManager()
            with self.assertRaisesRegex(tunnel_module.TunnelManagedByMasterError, "managed by the master"):
                await asyncio.wait_for(manager.start_tunnel(), timeout=5)
        spawn.assert_not_called()
        binary.assert_not_called()
        self.assertEqual(self._tunnel_config(), before)

    async def test_the_master_can_still_stop_its_tunnel(self):
        child, create_time = self._spawn_sleeper()
        state_module.persist_tunnel_state(status="running", pid=child.pid, pid_create_time=create_time)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMFYUI_IS_WORKER", None)
            manager = tunnel_module.CloudflareTunnelManager()
            await asyncio.wait_for(manager.stop_tunnel(), timeout=10)
        child.wait(timeout=10)
        self.assertIsNone(self._tunnel_config()["pid"])


if __name__ == "__main__":
    unittest.main()
