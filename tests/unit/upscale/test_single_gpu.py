import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency in CI/runtime
    torch = None


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

    modes_pkg = types.ModuleType(f"{package_name}.upscale.modes")
    modes_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale.modes"] = modes_pkg

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
        if torch is None:
            raise RuntimeError("torch is required for this test helper")
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def blend_processed_batch_item(
        result_images,
        processed_batch,
        batch_index,
        blend_fn,
        x1,
        y1,
        ew,
        eh,
        tile_mask,
        padding,
    ):
        tile_pil = tensor_to_pil(processed_batch, batch_index)
        if tile_pil.size != (ew, eh):
            tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
        result_images[batch_index] = blend_fn(
            result_images[batch_index],
            tile_pil,
            x1,
            y1,
            (ew, eh),
            tile_mask,
            padding,
        )

    image_module.tensor_to_pil = tensor_to_pil
    image_module.pil_to_tensor = pil_to_tensor
    image_module.blend_processed_batch_item = blend_processed_batch_item
    sys.modules[f"{package_name}.utils.image"] = image_module

    if torch is None:
        torch_module = types.ModuleType("torch")
        torch_module.cat = lambda *_args, **_kwargs: None
        sys.modules["torch"] = torch_module


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


def _load_single_gpu_mode_module():
    package_name = "dist_single_gpu_mode_testpkg"
    _bootstrap_package(package_name)
    _load_module(package_name, "upscale/processing_args.py", "upscale.processing_args")
    _load_module(package_name, "upscale/tile_processing.py", "upscale.tile_processing")
    return _load_module(package_name, "upscale/modes/single_gpu.py", "upscale.modes.single_gpu")


single_gpu_module = _load_single_gpu_mode_module()


class _DummySingleGpuNode(single_gpu_module.SingleGpuModeMixin):
    def __init__(self):
        self.round_calls = []
        self.process_batch_calls = 0

    def round_to_multiple(self, value):
        self.round_calls.append(int(value))
        return int(value)

    def calculate_tiles(self, _width, _height, _tile_width, _tile_height, _force_uniform_tiles):
        return [(0, 0)]

    def create_tile_mask(self, _width, _height, _tx, _ty, _tile_width, _tile_height, _mask_blur):
        return None

    def extract_batch_tile_with_padding(self, source_batch, _tx, _ty, _tile_width, _tile_height, _padding, _force_uniform_tiles):
        _, h, w, _ = source_batch.shape
        return source_batch, 0, 0, w, h

    def process_tiles_batch(
        self,
        tile_batch,
        _model,
        _positive,
        _negative,
        _vae,
        _seed,
        _steps,
        _cfg,
        _sampler_name,
        _scheduler,
        _denoise,
        _tiled_decode,
        _region,
        _canvas_shape,
    ):
        self.process_batch_calls += 1
        return tile_batch

    def blend_tile(self, _base_image, tile_pil, _x1, _y1, _size, _tile_mask, _padding):
        return tile_pil


@unittest.skipIf(torch is None, "torch is not installed")
class SingleGpuModeTests(unittest.TestCase):
    def setUp(self):
        self.node = _DummySingleGpuNode()
        self.images = torch.rand((2, 8, 8, 3), dtype=torch.float32)
        self.core_args = single_gpu_module.UpscaleCoreArgs(
            model=None,
            positive=None,
            negative=None,
            vae=None,
            seed=1,
            steps=10,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            denoise=0.5,
            tiled_decode=False,
        )

    def test_process_single_gpu_preserves_batch_shape(self):
        result = self.node.process_single_gpu(
            upscaled_image=self.images,
            core_args=self.core_args,
            tile_width=8,
            tile_height=8,
            padding=4,
            mask_blur=2,
            force_uniform_tiles=True,
        )[0]

        self.assertEqual(result.shape, self.images.shape)
        self.assertTrue(torch.allclose(result, self.images, atol=(1.5 / 255.0)))

    def test_process_single_gpu_invokes_rounding_and_batch_processing(self):
        self.node.process_single_gpu(
            upscaled_image=self.images,
            core_args=self.core_args,
            tile_width=16,
            tile_height=24,
            padding=4,
            mask_blur=2,
            force_uniform_tiles=True,
        )

        self.assertEqual(self.node.round_calls[:2], [16, 24])
        self.assertEqual(self.node.process_batch_calls, 1)


if __name__ == "__main__":
    unittest.main()
