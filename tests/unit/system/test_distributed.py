import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_distributed_module():
    module_path = Path(__file__).resolve().parents[3] / "distributed.py"
    package_name = "dist_distributed_entry_testpkg"
    calls = {
        "build_node_mappings": 0,
        "initialize_runtime": 0,
        "initialize_runtime_args": [],
    }

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    bootstrap_pkg = types.ModuleType(f"{package_name}.bootstrap")
    bootstrap_pkg.__path__ = []
    sys.modules[f"{package_name}.bootstrap"] = bootstrap_pkg

    entrypoint_module = types.ModuleType(f"{package_name}.bootstrap.entrypoint")

    def _build_node_mappings():
        calls["build_node_mappings"] += 1
        return ({"MockNode": object}, {"MockNode": "Mock Node"})

    def _initialize_runtime(prompt_server=None):
        calls["initialize_runtime"] += 1
        calls["initialize_runtime_args"].append(prompt_server)

    entrypoint_module.build_node_mappings = _build_node_mappings
    entrypoint_module.initialize_runtime = _initialize_runtime
    sys.modules[f"{package_name}.bootstrap.entrypoint"] = entrypoint_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.distributed", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, calls


class DistributedEntryTests(unittest.TestCase):
    def test_import_builds_mappings_without_runtime_bootstrap(self):
        module, calls = _load_distributed_module()

        self.assertIn("MockNode", module.NODE_CLASS_MAPPINGS)
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS["MockNode"], "Mock Node")
        self.assertEqual(calls["build_node_mappings"], 1)
        self.assertEqual(calls["initialize_runtime"], 0)

    def test_initialize_runtime_delegates_to_bootstrap_entrypoint(self):
        module, calls = _load_distributed_module()
        sentinel = object()

        module.initialize_runtime(sentinel)

        self.assertEqual(calls["initialize_runtime"], 1)
        self.assertEqual(calls["initialize_runtime_args"], [sentinel])


if __name__ == "__main__":
    unittest.main()
