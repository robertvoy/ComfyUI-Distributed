import importlib.util
import sys
import types
import unittest
from pathlib import Path

from PIL import Image


def _ensure_torch_stubs():
    if "torch" not in sys.modules:
        torch_module = types.ModuleType("torch")

        class _Tensor:
            pass

        torch_module.Tensor = _Tensor
        torch_module.float32 = object()
        torch_module.zeros = lambda *_args, **_kwargs: None
        torch_module.from_numpy = lambda array: array
        torch_module.cat = lambda *_args, **_kwargs: None

        nn_module = types.ModuleType("torch.nn")
        functional_module = types.ModuleType("torch.nn.functional")
        functional_module.interpolate = lambda tensor, **_kwargs: tensor
        nn_module.functional = functional_module

        torch_module.nn = nn_module

        sys.modules["torch"] = torch_module
        sys.modules["torch.nn"] = nn_module
        sys.modules["torch.nn.functional"] = functional_module

    if "torchvision" not in sys.modules:
        torchvision_module = types.ModuleType("torchvision")
        transforms_module = types.ModuleType("torchvision.transforms")

        class _GaussianBlur:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, value):
                return value

        transforms_module.GaussianBlur = _GaussianBlur
        torchvision_module.transforms = transforms_module

        sys.modules["torchvision"] = torchvision_module
        sys.modules["torchvision.transforms"] = transforms_module


def _load_usdu_utils_module():
    _ensure_torch_stubs()
    module_path = Path(__file__).resolve().parents[3] / "utils" / "usdu_utils.py"
    spec = importlib.util.spec_from_file_location("dist_test_usdu_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


usdu_utils = _load_usdu_utils_module()


class UsduUtilsGeometryTests(unittest.TestCase):
    def test_fix_crop_region_decrements_upper_bounds_when_interior(self):
        self.assertEqual(
            usdu_utils.fix_crop_region((2, 3, 9, 9), (10, 10)),
            (2, 3, 8, 8),
        )
        self.assertEqual(
            usdu_utils.fix_crop_region((1, 1, 10, 10), (10, 10)),
            (1, 1, 10, 10),
        )

    def test_get_crop_region_with_padding_and_empty_mask(self):
        mask = Image.new("L", (10, 10), 0)
        for x in range(2, 5):
            for y in range(3, 7):
                mask.putpixel((x, y), 255)

        region = usdu_utils.get_crop_region(mask, pad=1)
        self.assertEqual(region, (1, 2, 5, 7))

        empty_region = usdu_utils.get_crop_region(Image.new("L", (8, 6), 0), pad=2)
        self.assertEqual(empty_region, (6, 4, 1, 1))

    def test_expand_crop_and_resize_region(self):
        expanded_region, expanded_size = usdu_utils.expand_crop(
            region=(2, 2, 6, 6),
            width=12,
            height=12,
            target_width=8,
            target_height=10,
        )
        self.assertEqual(expanded_region, (0, 0, 8, 10))
        self.assertEqual(expanded_size, (8, 10))

        resized = usdu_utils.resize_region(
            region=(2, 2, 6, 10),
            init_size=(8, 16),
            resize_size=(16, 32),
        )
        self.assertEqual(resized, (4, 4, 12, 20))

    def test_region_intersection(self):
        self.assertEqual(
            usdu_utils.region_intersection((0, 0, 5, 5), (3, 2, 8, 7)),
            (3, 2, 5, 5),
        )
        self.assertIsNone(usdu_utils.region_intersection((0, 0, 1, 1), (2, 2, 4, 4)))

    def test_pad_image2_preserves_expected_output_size(self):
        image = Image.new("RGB", (4, 3), "black")
        padded = usdu_utils.pad_image2(
            image,
            left_pad=2,
            right_pad=1,
            top_pad=3,
            bottom_pad=2,
            fill=False,
            blur=False,
        )
        self.assertEqual(padded.size, (7, 8))


if __name__ == "__main__":
    unittest.main()
