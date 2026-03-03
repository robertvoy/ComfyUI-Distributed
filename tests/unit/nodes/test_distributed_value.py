import json
import unittest


class DistributedValueTests(unittest.TestCase):
    """Unit tests for the DistributedValue node's distribute() method."""

    def _make_node(self):
        # Import inline to avoid plugin-level imports
        import importlib.util
        import sys
        import types
        from pathlib import Path
        from unittest.mock import MagicMock

        module_path = Path(__file__).resolve().parents[3] / "nodes" / "utilities.py"
        pkg_name = "dv_test_pkg"

        for mod_name in list(sys.modules):
            if mod_name == pkg_name or mod_name.startswith(f"{pkg_name}."):
                del sys.modules[mod_name]

        # Mock torch if not available
        if "torch" not in sys.modules:
            sys.modules["torch"] = MagicMock()

        root_pkg = types.ModuleType(pkg_name)
        root_pkg.__path__ = []
        sys.modules[pkg_name] = root_pkg

        utils_pkg = types.ModuleType(f"{pkg_name}.utils")
        utils_pkg.__path__ = []
        sys.modules[f"{pkg_name}.utils"] = utils_pkg

        logging_mod = types.ModuleType(f"{pkg_name}.utils.logging")
        logging_mod.debug_log = lambda *_a, **_k: None
        logging_mod.log = lambda *_a, **_k: None
        sys.modules[f"{pkg_name}.utils.logging"] = logging_mod

        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.nodes.utilities", module_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.DistributedValue()

    def setUp(self):
        self.node = self._make_node()

    def test_master_returns_default(self):
        result = self.node.distribute(
            default_value="model_a",
            worker_values="{}",
            is_worker=False,
            worker_id="",
        )
        self.assertEqual(result, ("model_a",))

    def test_master_coerces_default_int(self):
        values = json.dumps({"_type": "INT"})
        result = self.node.distribute(
            default_value="42",
            worker_values=values,
            is_worker=False,
            worker_id="",
        )
        self.assertEqual(result, (42,))
        self.assertIsInstance(result[0], int)

    def test_master_coerces_default_float(self):
        values = json.dumps({"_type": "FLOAT"})
        result = self.node.distribute(
            default_value="2.5",
            worker_values=values,
            is_worker=False,
            worker_id="",
        )
        self.assertEqual(result, (2.5,))
        self.assertIsInstance(result[0], float)

    def test_worker_returns_specific_value(self):
        values = json.dumps({"1": "model_x", "2": "model_y"})
        result = self.node.distribute(
            default_value="default",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, ("model_x",))

    def test_worker_second_index(self):
        values = json.dumps({"1": "model_x", "2": "model_y"})
        result = self.node.distribute(
            default_value="default",
            worker_values=values,
            is_worker=True,
            worker_id="worker_1",
        )
        self.assertEqual(result, ("model_y",))

    def test_worker_falls_back_to_default_when_key_missing(self):
        values = json.dumps({"_type": "INT", "1": "3"})
        result = self.node.distribute(
            default_value="9",
            worker_values=values,
            is_worker=True,
            worker_id="worker_5",
        )
        self.assertEqual(result, (9,))
        self.assertIsInstance(result[0], int)

    def test_worker_falls_back_to_default_on_empty_value(self):
        values = json.dumps({"1": ""})
        result = self.node.distribute(
            default_value="fallback",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, ("fallback",))

    def test_worker_falls_back_on_invalid_json(self):
        result = self.node.distribute(
            default_value="safe",
            worker_values="not-json",
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, ("safe",))

    def test_worker_falls_back_on_invalid_worker_id(self):
        values = json.dumps({"1": "model_x"})
        result = self.node.distribute(
            default_value="safe",
            worker_values=values,
            is_worker=True,
            worker_id="bad_id",
        )
        self.assertEqual(result, ("safe",))

    def test_worker_id_as_direct_integer(self):
        values = json.dumps({"1": "model_x"})
        result = self.node.distribute(
            default_value="default",
            worker_values=values,
            is_worker=True,
            worker_id="0",
        )
        self.assertEqual(result, ("model_x",))

    def test_type_int_coerces_value(self):
        values = json.dumps({"_type": "INT", "1": "42"})
        result = self.node.distribute(
            default_value="0",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, (42,))
        self.assertIsInstance(result[0], int)

    def test_type_float_coerces_value(self):
        values = json.dumps({"_type": "FLOAT", "1": "3.14"})
        result = self.node.distribute(
            default_value="0",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertAlmostEqual(result[0], 3.14)
        self.assertIsInstance(result[0], float)

    def test_type_combo_stays_string(self):
        values = json.dumps({"_type": "COMBO", "1": "model_v2"})
        result = self.node.distribute(
            default_value="model_v1",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, ("model_v2",))
        self.assertIsInstance(result[0], str)

    def test_type_string_default_stays_string(self):
        values = json.dumps({"1": "hello"})
        result = self.node.distribute(
            default_value="default",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, ("hello",))
        self.assertIsInstance(result[0], str)

    def test_int_coerce_handles_float_string(self):
        """INT coercion of '3.7' should truncate to 3."""
        values = json.dumps({"_type": "INT", "1": "3.7"})
        result = self.node.distribute(
            default_value="0",
            worker_values=values,
            is_worker=True,
            worker_id="worker_0",
        )
        self.assertEqual(result, (3,))


if __name__ == "__main__":
    unittest.main()
