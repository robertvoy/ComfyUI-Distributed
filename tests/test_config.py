import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_config_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "config.py"
    package_name = "dist_cfg_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = [str(module_path.parent)]
    sys.modules[package_name] = root_pkg

    logging_module = types.ModuleType(f"{package_name}.logging")
    logging_module.log = lambda *_args, **_kwargs: None
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.logging"] = logging_module

    constants_module = types.ModuleType(f"{package_name}.constants")
    constants_module.HEARTBEAT_TIMEOUT = 30
    sys.modules[f"{package_name}.constants"] = constants_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


config = _load_config_module()


# ---------------------------------------------------------------------------
# _merge_with_defaults
# ---------------------------------------------------------------------------

class MergeWithDefaultsTests(unittest.TestCase):
    def test_non_dict_input_returns_defaults(self):
        result = config._merge_with_defaults("not a dict", {"key": "default"})
        self.assertEqual(result, {"key": "default"})

    def test_fills_missing_keys_with_defaults(self):
        result = config._merge_with_defaults({}, {"a": 1, "b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_loaded_value_overrides_default(self):
        result = config._merge_with_defaults({"a": 99}, {"a": 1, "b": 2})
        self.assertEqual(result["a"], 99)
        self.assertEqual(result["b"], 2)

    def test_nested_dict_merges_recursively(self):
        defaults = {"settings": {"debug": False, "count": 5}}
        loaded = {"settings": {"debug": True}}
        result = config._merge_with_defaults(loaded, defaults)
        self.assertTrue(result["settings"]["debug"])
        self.assertEqual(result["settings"]["count"], 5)

    def test_preserves_unknown_keys_for_forward_compatibility(self):
        result = config._merge_with_defaults({"extra_key": "extra"}, {"a": 1})
        self.assertEqual(result["extra_key"], "extra")

    def test_none_loaded_value_overrides_default(self):
        """Explicitly set None in config should override non-None default."""
        result = config._merge_with_defaults({"a": None}, {"a": "default"})
        self.assertIsNone(result["a"])

    def test_non_dict_nested_loaded_value_replaces_dict_default(self):
        """If loaded has a scalar where default has a dict, use the scalar."""
        defaults = {"settings": {"debug": False}}
        loaded = {"settings": "flat_string"}
        result = config._merge_with_defaults(loaded, defaults)
        self.assertEqual(result["settings"], "flat_string")


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class LoadConfigTests(unittest.TestCase):
    def setUp(self):
        config.invalidate_config_cache()

    def tearDown(self):
        config.invalidate_config_cache()

    def test_returns_defaults_when_file_missing(self):
        with patch.object(config, "CONFIG_FILE", "/nonexistent/path/config.json"):
            cfg = config.load_config()
        defaults = config.get_default_config()
        self.assertEqual(cfg["settings"]["debug"], defaults["settings"]["debug"])
        self.assertIn("workers", cfg)

    def test_loads_valid_json_file(self):
        data = {
            "workers": [{"id": "w1"}],
            "master": {"host": "test.host"},
            "settings": {},
            "tunnel": {},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            with patch.object(config, "CONFIG_FILE", tmp_path):
                cfg = config.load_config()
            self.assertEqual(cfg["master"]["host"], "test.host")
            self.assertEqual(len(cfg["workers"]), 1)
        finally:
            os.unlink(tmp_path)

    def test_merges_loaded_file_with_defaults(self):
        """Loaded file with partial settings should be filled in from defaults."""
        data = {"master": {"host": "h"}, "workers": [], "settings": {"debug": True}, "tunnel": {}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            with patch.object(config, "CONFIG_FILE", tmp_path):
                cfg = config.load_config()
            # debug was set to True
            self.assertTrue(cfg["settings"]["debug"])
            # auto_launch_workers is a default key and should be present
            self.assertIn("auto_launch_workers", cfg["settings"])
        finally:
            os.unlink(tmp_path)

    def test_falls_back_to_defaults_on_invalid_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{invalid json{{")
            tmp_path = f.name
        try:
            with patch.object(config, "CONFIG_FILE", tmp_path):
                cfg = config.load_config()
            self.assertIn("settings", cfg)
            self.assertIn("workers", cfg)
        finally:
            os.unlink(tmp_path)

    def test_second_call_returns_cached_object(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(config.get_default_config(), f)
            tmp_path = f.name
        try:
            with patch.object(config, "CONFIG_FILE", tmp_path):
                cfg1 = config.load_config()
                cfg2 = config.load_config()
            self.assertIs(cfg1, cfg2)
        finally:
            os.unlink(tmp_path)

    def test_invalidate_cache_forces_reload(self):
        data = config.get_default_config()
        data["master"]["host"] = "first"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            with patch.object(config, "CONFIG_FILE", tmp_path):
                cfg1 = config.load_config()
                config.invalidate_config_cache()
                data["master"]["host"] = "second"
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                cfg2 = config.load_config()
            self.assertEqual(cfg1["master"]["host"], "first")
            self.assertEqual(cfg2["master"]["host"], "second")
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

class SaveConfigTests(unittest.TestCase):
    def setUp(self):
        config.invalidate_config_cache()

    def tearDown(self):
        config.invalidate_config_cache()

    def test_saves_and_reloads_correctly(self):
        data = config.get_default_config()
        data["master"]["host"] = "saved.host"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "config.json")
            with patch.object(config, "CONFIG_FILE", tmp_path):
                result = config.save_config(data)
                self.assertTrue(result)
                loaded = config.load_config()
        self.assertEqual(loaded["master"]["host"], "saved.host")

    def test_returns_false_when_path_unwritable(self):
        with patch.object(config, "CONFIG_FILE", "/nonexistent_dir/config.json"):
            result = config.save_config({})
        self.assertFalse(result)

    def test_save_invalidates_cache(self):
        """After saving, the cache should be cleared so next load re-reads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "config.json")
            with patch.object(config, "CONFIG_FILE", tmp_path):
                data = config.get_default_config()
                config.save_config(data)
                # Cache is now None; load_config should re-read
                self.assertIsNone(config._config_cache)

    def test_written_file_is_valid_json(self):
        data = config.get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "config.json")
            with patch.object(config, "CONFIG_FILE", tmp_path):
                config.save_config(data)
            with open(tmp_path, encoding="utf-8") as fh:
                parsed = json.load(fh)
        self.assertEqual(parsed["master"], data["master"])


class _SharingViolation(PermissionError):
    """PermissionError carrying Windows ERROR_SHARING_VIOLATION on any platform."""

    winerror = 32


class SaveConfigConcurrencyTests(unittest.TestCase):
    """Concurrent saves must never leave a corrupt or partially written config."""

    WRITERS = 4
    SAVES_PER_WRITER = 40

    def setUp(self):
        config.invalidate_config_cache()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = os.path.join(self._tmpdir.name, "config.json")
        patcher = patch.object(config, "CONFIG_FILE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(config.invalidate_config_cache)

    @staticmethod
    def _config_for(writer, index):
        data = config.get_default_config()
        data["master"]["host"] = f"writer-{writer}-save-{index}"
        data["workers"] = [{"id": f"w{writer}-{n}", "name": "x" * 200} for n in range(20)]
        return data

    def test_concurrent_saves_always_leave_parseable_config(self):
        import threading

        initial = self._config_for("initial", 0)
        self.assertTrue(config.save_config(initial))
        written_hosts = {initial["master"]["host"]}
        written = {initial["master"]["host"]: initial}
        for writer in range(self.WRITERS):
            for index in range(self.SAVES_PER_WRITER):
                cfg = self._config_for(writer, index)
                written_hosts.add(cfg["master"]["host"])
                written[cfg["master"]["host"]] = cfg

        stop = threading.Event()
        save_results = []
        corrupt_reads = []
        bad_loads = []
        errors = []

        def _writer(writer):
            try:
                for index in range(self.SAVES_PER_WRITER):
                    save_results.append(config.save_config(self._config_for(writer, index)))
            except Exception as exc:  # pragma: no cover - surfaced by the assertion below
                errors.append(exc)

        def _raw_reader():
            # Simulates another process (e.g. a worker) reading the shared file.
            while not stop.is_set():
                try:
                    with open(self.path, encoding="utf-8") as fh:
                        text = fh.read()
                except OSError:
                    continue
                try:
                    json.loads(text)
                except ValueError:
                    corrupt_reads.append(text[:80])
                time.sleep(0.001)

        def _cached_reader():
            while not stop.is_set():
                host = config.load_config().get("master", {}).get("host")
                if host not in written_hosts:
                    bad_loads.append(host)
                time.sleep(0.001)

        writers = [threading.Thread(target=_writer, args=(w,)) for w in range(self.WRITERS)]
        readers = [threading.Thread(target=_raw_reader), threading.Thread(target=_cached_reader)]
        for thread in readers + writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=120)
        stop.set()
        for thread in readers:
            thread.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(len(save_results), self.WRITERS * self.SAVES_PER_WRITER)
        self.assertTrue(all(save_results), f"{save_results.count(False)} saves failed")
        self.assertEqual(corrupt_reads, [])
        self.assertEqual(bad_loads, [], "load_config fell back to defaults or a foreign config")

        with open(self.path, encoding="utf-8") as fh:
            final = json.load(fh)
        self.assertIn(final["master"]["host"], written)
        self.assertEqual(final, written[final["master"]["host"]])
        self.assertEqual(os.listdir(self._tmpdir.name), ["config.json"], "temp files leaked")

    def test_replace_retries_transient_sharing_violation(self):
        real_replace = os.replace
        attempts = {"count": 0}

        def _flaky_replace(src, dst):
            attempts["count"] += 1
            if attempts["count"] <= 2:
                raise _SharingViolation(13, "file in use by another process")
            return real_replace(src, dst)

        data = self._config_for("retry", 0)
        with patch.object(config.os, "replace", _flaky_replace):
            self.assertTrue(config.save_config(data))

        self.assertEqual(attempts["count"], 3)
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["master"]["host"], "writer-retry-save-0")
        self.assertEqual(os.listdir(self._tmpdir.name), ["config.json"])

    def test_new_config_file_gets_default_permissions_not_owner_only(self):
        modes = []
        real_chmod = os.chmod

        def _recording_chmod(path, mode):
            modes.append(mode)
            return real_chmod(path, mode)

        with patch.object(config.os, "chmod", _recording_chmod):
            self.assertTrue(config.save_config(self._config_for("new", 0)))
        self.assertEqual(modes, [config._get_new_file_mode()])
        self.assertEqual(config._get_new_file_mode(), 0o666 & ~config._read_umask())

    def test_umask_is_read_lazily_and_only_once(self):
        # Reading the umask means briefly setting it for the whole process, so it
        # must not happen at import, nor when saving over an existing file.
        calls = []

        def _fake_read_umask():
            calls.append(1)
            return 0o022

        with patch.object(config, "_new_file_mode", None), \
             patch.object(config, "_read_umask", _fake_read_umask):
            self.assertEqual(calls, [])
            self.assertTrue(config.save_config(self._config_for("lazy", 0)))
            self.assertEqual(calls, [1])
            os.unlink(self.path)
            self.assertTrue(config.save_config(self._config_for("lazy", 1)))
            self.assertEqual(calls, [1])
            self.assertEqual(config._get_new_file_mode(), 0o666 & ~0o022)

    def test_overwriting_an_existing_file_never_reads_the_umask(self):
        self.assertTrue(config.save_config(self._config_for("first", 0)))
        with patch.object(config, "_new_file_mode", None), \
             patch.object(config, "_read_umask", side_effect=AssertionError("umask read")):
            self.assertTrue(config.save_config(self._config_for("second", 0)))

    def test_persistent_sharing_violation_gives_up_quickly(self):
        # Saves run on the event loop thread under the config lock; a file held
        # open by another process must not stall the server for long.
        sleeps = []

        def _always_locked(src, dst):
            raise _SharingViolation(13, "file in use by another process")

        data = self._config_for("locked", 0)
        with patch.object(config.os, "replace", _always_locked),                 patch.object(config.time, "sleep", sleeps.append):
            self.assertFalse(config.save_config(data))

        self.assertTrue(sleeps, "no retry was attempted")
        self.assertLessEqual(sum(sleeps), 0.5)
        self.assertEqual(os.listdir(self._tmpdir.name), [], "temp file leaked")

    def test_non_transient_replace_error_fails_without_leaking_temp_file(self):
        def _denied(_src, _dst):
            raise PermissionError(13, "permission denied")

        with patch.object(config.os, "replace", _denied):
            self.assertFalse(config.save_config(self._config_for("denied", 0)))

        self.assertEqual(os.listdir(self._tmpdir.name), [])


class ConfigTransactionSerialisationTests(unittest.IsolatedAsyncioTestCase):
    """A thread's save must not land between a transaction's load and save."""

    async def asyncSetUp(self):
        config.invalidate_config_cache()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = os.path.join(self._tmpdir.name, "config.json")
        patcher = patch.object(config, "CONFIG_FILE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(config.invalidate_config_cache)

    def _read_host(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)["master"]["host"]

    async def test_thread_save_waits_for_open_transaction(self):
        import asyncio
        import threading

        initial = config.get_default_config()
        initial["master"]["host"] = "initial"
        self.assertTrue(config.save_config(initial))

        thread_cfg = config.get_default_config()
        thread_cfg["master"]["host"] = "from-thread"
        thread_result = []
        thread = threading.Thread(target=lambda: thread_result.append(config.save_config(thread_cfg)))

        async with config.config_transaction() as cfg:
            thread.start()
            # Give the thread ample time to save if nothing stops it.
            await asyncio.sleep(0.3)
            host_during_transaction = self._read_host()
            thread_saved_during_transaction = bool(thread_result)
            cfg["master"]["host"] = "from-transaction"

        await asyncio.to_thread(thread.join, 10)

        self.assertEqual(host_during_transaction, "initial")
        self.assertFalse(thread_saved_during_transaction)
        self.assertEqual(thread_result, [True])
        # The transaction committed first; the waiting thread's save follows it.
        self.assertEqual(self._read_host(), "from-thread")


# ---------------------------------------------------------------------------
# get_worker_timeout_seconds
# ---------------------------------------------------------------------------

class GetWorkerTimeoutSecondsTests(unittest.TestCase):
    def test_returns_configured_value(self):
        cfg = config.get_default_config()
        cfg["settings"]["worker_timeout_seconds"] = 120
        with patch.object(config, "load_config", return_value=cfg):
            self.assertEqual(config.get_worker_timeout_seconds(), 120)

    def test_clamps_zero_to_one(self):
        cfg = config.get_default_config()
        cfg["settings"]["worker_timeout_seconds"] = 0
        with patch.object(config, "load_config", return_value=cfg):
            self.assertEqual(config.get_worker_timeout_seconds(), 1)

    def test_clamps_negative_to_one(self):
        cfg = config.get_default_config()
        cfg["settings"]["worker_timeout_seconds"] = -10
        with patch.object(config, "load_config", return_value=cfg):
            self.assertEqual(config.get_worker_timeout_seconds(), 1)

    def test_falls_back_to_provided_default_when_key_missing(self):
        cfg = config.get_default_config()
        # worker_timeout_seconds is not present in default config
        cfg["settings"].pop("worker_timeout_seconds", None)
        with patch.object(config, "load_config", return_value=cfg):
            result = config.get_worker_timeout_seconds(default=45)
        self.assertEqual(result, 45)

    def test_fallback_also_clamped_to_one(self):
        cfg = config.get_default_config()
        cfg["settings"].pop("worker_timeout_seconds", None)
        with patch.object(config, "load_config", return_value=cfg):
            result = config.get_worker_timeout_seconds(default=0)
        self.assertEqual(result, 1)


# ---------------------------------------------------------------------------
# is_master_delegate_only
# ---------------------------------------------------------------------------

class IsMasterDelegateOnlyTests(unittest.TestCase):
    def test_returns_false_by_default(self):
        cfg = config.get_default_config()
        with patch.object(config, "load_config", return_value=cfg):
            self.assertFalse(config.is_master_delegate_only())

    def test_returns_true_when_enabled(self):
        cfg = config.get_default_config()
        cfg["settings"]["master_delegate_only"] = True
        with patch.object(config, "load_config", return_value=cfg):
            self.assertTrue(config.is_master_delegate_only())

    def test_returns_false_on_exception(self):
        def _raise():
            raise RuntimeError("config exploded")

        with patch.object(config, "load_config", side_effect=RuntimeError("boom")):
            self.assertFalse(config.is_master_delegate_only())


class ConfigUpdateTests(unittest.IsolatedAsyncioTestCase):
    """config_update is the sync transaction: a late save cannot undo a committed one."""

    async def asyncSetUp(self):
        config.invalidate_config_cache()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = os.path.join(self._tmpdir.name, "config.json")
        patcher = patch.object(config, "CONFIG_FILE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(config.invalidate_config_cache)

    def _read(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    async def test_applies_the_change_and_saves_it(self):
        self.assertTrue(config.config_update(lambda cfg: cfg["master"].update(host="h1")))
        self.assertEqual(self._read()["master"]["host"], "h1")

    async def test_unchanged_config_is_not_rewritten(self):
        self.assertTrue(config.config_update(lambda cfg: cfg["master"].update(host="h1")))
        with patch.object(config, "save_config", side_effect=AssertionError("saved")):
            self.assertTrue(config.config_update(lambda cfg: cfg["master"].update(host="h1")))

    async def test_failed_save_restores_the_cached_config(self):
        self.assertTrue(config.config_update(lambda cfg: cfg["master"].update(host="h1")))
        with patch.object(config, "save_config", return_value=False):
            self.assertFalse(config.config_update(lambda cfg: cfg["master"].update(host="lost")))
        self.assertEqual(config.load_config()["master"]["host"], "h1")

    async def test_update_from_a_thread_waits_for_an_open_transaction(self):
        import asyncio
        import threading

        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                config.config_update(lambda cfg: cfg["settings"].update(debug=True))
            )
        )
        async with config.config_transaction() as cfg:
            thread.start()
            await asyncio.sleep(0.3)
            self.assertEqual(results, [], "the update ran inside the transaction")
            cfg["master"]["host"] = "from-transaction"
        await asyncio.to_thread(thread.join, 10)
        self.assertEqual(results, [True])
        saved = self._read()
        # The update was applied on top of the committed transaction, not over it.
        self.assertEqual(saved["master"]["host"], "from-transaction")
        self.assertTrue(saved["settings"]["debug"])


if __name__ == "__main__":
    unittest.main()
