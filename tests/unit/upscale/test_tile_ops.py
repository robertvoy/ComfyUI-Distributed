import copy
import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _bootstrap_package(package_name):
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    image_module = types.ModuleType(f"{package_name}.utils.image")

    def tensor_to_pil(image_batch, index):
        tensor = image_batch[index].detach().cpu().clamp(0, 1)
        array = (tensor.numpy() * 255.0).astype("uint8")
        return Image.fromarray(array)

    def pil_to_tensor(image):
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    image_module.tensor_to_pil = tensor_to_pil
    image_module.pil_to_tensor = pil_to_tensor
    sys.modules[f"{package_name}.utils.image"] = image_module

    usdu_utils_module = types.ModuleType(f"{package_name}.utils.usdu_utils")

    def crop_cond(cond, *_args, **_kwargs):
        return cond

    def get_crop_region(mask, padding):
        bbox = mask.getbbox()
        if bbox is None:
            return (0, 0, mask.size[0], mask.size[1])
        x1, y1, x2, y2 = bbox
        return (
            max(0, x1 - int(padding)),
            max(0, y1 - int(padding)),
            min(mask.size[0], x2 + int(padding)),
            min(mask.size[1], y2 + int(padding)),
        )

    def expand_crop(region, width, height, target_w, target_h):
        x1, y1, x2, y2 = region
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w >= target_w and crop_h >= target_h:
            return (region, (crop_w, crop_h))
        new_x2 = min(width, x1 + max(crop_w, target_w))
        new_y2 = min(height, y1 + max(crop_h, target_h))
        return ((x1, y1, new_x2, new_y2), (new_x2 - x1, new_y2 - y1))

    usdu_utils_module.crop_cond = crop_cond
    usdu_utils_module.get_crop_region = get_crop_region
    usdu_utils_module.expand_crop = expand_crop
    sys.modules[f"{package_name}.utils.usdu_utils"] = usdu_utils_module

    crop_model_patch_module = types.ModuleType(f"{package_name}.utils.crop_model_patch")
    crop_model_patch_module.crop_model_cond = lambda model, *_args, **_kwargs: nullcontext(model)
    sys.modules[f"{package_name}.utils.crop_model_patch"] = crop_model_patch_module

    conditioning_module = types.ModuleType(f"{package_name}.upscale.conditioning")
    conditioning_module.clone_conditioning = lambda cond, clone_hints=False: copy.deepcopy(cond)
    sys.modules[f"{package_name}.upscale.conditioning"] = conditioning_module

    comfy_module = types.ModuleType("comfy")
    comfy_samplers = types.ModuleType("comfy.samplers")
    comfy_model_management = types.ModuleType("comfy.model_management")
    comfy_module.samplers = comfy_samplers
    comfy_module.model_management = comfy_model_management
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.samplers"] = comfy_samplers
    sys.modules["comfy.model_management"] = comfy_model_management


def _load_module(package_name, module_rel_path, module_name):
    module_path = Path(__file__).resolve().parents[3] / module_rel_path
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tile_ops_module():
    package_name = "dist_tile_ops_testpkg"
    _bootstrap_package(package_name)
    return _load_module(package_name, "upscale/tile_ops.py", "upscale.tile_ops")


tile_ops_module = _load_tile_ops_module()


class _DummyTileOps(tile_ops_module.TileOpsMixin):
    pass


class _FakeControl:
    def __init__(self, cond_hint_original, previous_controlnet=None):
        self.cond_hint_original = cond_hint_original
        self.previous_controlnet = previous_controlnet


class TileOpsTests(unittest.TestCase):
    def setUp(self):
        self.node = _DummyTileOps()

    def test_round_and_calculate_tiles(self):
        self.assertEqual(self.node.round_to_multiple(13, 8), 16)
        tiles = self.node.calculate_tiles(10, 9, 4, 5, force_uniform_tiles=True)
        self.assertEqual(len(tiles), 6)
        self.assertEqual(tiles[0], (0, 0))
        self.assertEqual(tiles[-1], (8, 5))

    def test_create_tile_mask_and_blend_tile(self):
        base = Image.new("RGB", (8, 8), (0, 0, 0))
        tile = Image.new("RGB", (4, 4), (255, 255, 255))
        mask = self.node.create_tile_mask(8, 8, 2, 2, 4, 4, mask_blur=0)

        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertEqual(mask.getpixel((3, 3)), 255)

        blended = self.node.blend_tile(
            base_image=base,
            tile_image=tile,
            x=2,
            y=2,
            extracted_size=(4, 4),
            mask=mask,
            padding=0,
        )
        self.assertEqual(blended.size, (8, 8))
        self.assertEqual(blended.getpixel((3, 3)), (255, 255, 255))

    def test_slice_conditioning_selects_batch_index(self):
        control = _FakeControl(cond_hint_original=torch.ones((2, 1, 1)))
        positive = [[torch.arange(6, dtype=torch.float32).reshape(2, 3), {"control": control, "mask": torch.ones((2, 1, 1))}]]
        negative = [[torch.arange(6, dtype=torch.float32).reshape(2, 3), {"mask": torch.zeros((2, 1, 1))}]]

        pos_sliced, neg_sliced = self.node._slice_conditioning(positive, negative, batch_idx=1)

        self.assertEqual(tuple(pos_sliced[0][0].shape), (1, 3))
        self.assertEqual(tuple(neg_sliced[0][0].shape), (1, 3))
        self.assertEqual(tuple(pos_sliced[0][1]["mask"].shape), (1, 1, 1))
        self.assertEqual(tuple(neg_sliced[0][1]["mask"].shape), (1, 1, 1))
        self.assertEqual(tuple(pos_sliced[0][1]["control"].cond_hint_original.shape), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
