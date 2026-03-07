import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_config_module():
    module_path = Path(__file__).resolve().parents[3] / "utils" / "config.py"
    package_name = "dist_cfg_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    constants_module = types.ModuleType(f"{package_name}.constants")
    constants_module.GPU_CONFIG_FILE = "gpu_config.json"
    constants_module.HEARTBEAT_TIMEOUT = 30
    constants_module.ORCHESTRATION_WORKER_PROBE_CONCURRENCY = 8
    constants_module.ORCHESTRATION_WORKER_PREP_CONCURRENCY = 4
    constants_module.ORCHESTRATION_MEDIA_SYNC_CONCURRENCY = 2
    constants_module.ORCHESTRATION_MEDIA_SYNC_TIMEOUT = 120
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
                self.assertIsNone(config._config_state().cache)

    def test_written_file_is_valid_json(self):
        data = config.get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "config.json")
            with patch.object(config, "CONFIG_FILE", tmp_path):
                config.save_config(data)
            with open(tmp_path, encoding="utf-8") as fh:
                parsed = json.load(fh)
        self.assertEqual(parsed["master"], data["master"])


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


if __name__ == "__main__":
    unittest.main()
