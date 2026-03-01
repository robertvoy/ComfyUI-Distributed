import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


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


def _load_list_splitter_module():
    package_name = "dist_list_splitter_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "nodes/utilities.py", "nodes.utilities")
    return _load_module(package_name, "nodes/list_splitter.py", "nodes.list_splitter")


splitter_module = _load_list_splitter_module()


class DistributedListSplitterTests(unittest.TestCase):
    def setUp(self):
        self.node = splitter_module.DistributedListSplitter()

    def _images(self, count):
        return [torch.full((1, 4, 4, 3), float(idx), dtype=torch.float32) for idx in range(count)]

    def test_single_participant_returns_full_list(self):
        images = self._images(5)
        out = self.node.split(images, participant_index=0, total_participants=1)[0]
        self.assertEqual(len(out), 5)
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in out], [0, 1, 2, 3, 4])

    def test_two_participants_split_six_images_evenly(self):
        images = self._images(6)
        p0 = self.node.split(images, participant_index=0, total_participants=2)[0]
        p1 = self.node.split(images, participant_index=1, total_participants=2)[0]
        self.assertEqual(len(p0), 3)
        self.assertEqual(len(p1), 3)
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p0], [0, 1, 2])
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p1], [3, 4, 5])

    def test_three_participants_split_seven_images(self):
        images = self._images(7)
        p0 = self.node.split(images, participant_index=0, total_participants=3)[0]
        p1 = self.node.split(images, participant_index=1, total_participants=3)[0]
        p2 = self.node.split(images, participant_index=2, total_participants=3)[0]
        self.assertEqual([len(p0), len(p1), len(p2)], [3, 2, 2])
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p0], [0, 1, 2])
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p1], [3, 4])
        self.assertEqual([int(t[0, 0, 0, 0].item()) for t in p2], [5, 6])

    def test_more_participants_than_images_yields_empty_slices(self):
        images = self._images(2)
        outputs = [self.node.split(images, participant_index=i, total_participants=5)[0] for i in range(5)]
        self.assertEqual([len(chunk) for chunk in outputs], [1, 1, 0, 0, 0])

    def test_mixed_dimensions_are_preserved(self):
        images = [
            torch.zeros((1, 4, 4, 3), dtype=torch.float32),
            torch.zeros((1, 8, 6, 3), dtype=torch.float32),
            torch.zeros((1, 3, 9, 3), dtype=torch.float32),
        ]

        p0 = self.node.split(images, participant_index=0, total_participants=2)[0]
        p1 = self.node.split(images, participant_index=1, total_participants=2)[0]

        self.assertEqual([tuple(t.shape) for t in p0], [(1, 4, 4, 3), (1, 8, 6, 3)])
        self.assertEqual([tuple(t.shape) for t in p1], [(1, 3, 9, 3)])


if __name__ == "__main__":
    unittest.main()
