import importlib.util
import sys
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

    nodes_pkg = types.ModuleType(f"{package_name}.nodes")
    nodes_pkg.__path__ = []
    sys.modules[f"{package_name}.nodes"] = nodes_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module


def _load_module(package_name, module_rel_path, module_name):
    module_path = Path(__file__).resolve().parents[1] / module_rel_path
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_branch_module():
    package_name = "dist_branch_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "nodes/utilities.py", "nodes.utilities")
    return _load_module(package_name, "nodes/branch.py", "nodes.branch")


branch_module = _load_branch_module()


class DistributedBranchTests(unittest.TestCase):
    def test_no_distribution_outputs_all_branches(self):
        node = branch_module.DistributedBranch()
        token = {"k": "v"}

        outputs = node.branch(token, num_branches=2, assigned_branch=-1)

        self.assertEqual(len(outputs), 10)
        self.assertTrue(all(output is token for output in outputs))

    def test_assigned_branch_still_passes_input(self):
        node = branch_module.DistributedBranch()
        token = "payload"

        outputs = node.branch(token, num_branches=3, is_worker=True, worker_id="worker-a", assigned_branch=1)

        self.assertEqual(outputs[0], "payload")
        self.assertEqual(outputs[1], "payload")
        self.assertEqual(outputs[9], "payload")


if __name__ == "__main__":
    unittest.main()
