import unittest

from upscale.processing_args import UpscaleCoreArgs
from upscale.tile_processing import TileBatchArgs, extract_and_process_tile_batch


class _DummyTileProcessor:
    def __init__(self):
        self.extract_calls = []
        self.process_calls = []

    def extract_batch_tile_with_padding(
        self,
        upscaled_image,
        tx,
        ty,
        tile_width,
        tile_height,
        padding,
        force_uniform_tiles,
    ):
        self.extract_calls.append(
            {
                "upscaled_image": upscaled_image,
                "tx": tx,
                "ty": ty,
                "tile_width": tile_width,
                "tile_height": tile_height,
                "padding": padding,
                "force_uniform_tiles": force_uniform_tiles,
            }
        )
        return "tile-batch", 10, 20, 30, 40

    def process_tiles_batch(
        self,
        tile_batch,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        tiled_decode,
        region,
        canvas_shape,
    ):
        self.process_calls.append(
            {
                "tile_batch": tile_batch,
                "model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "tiled_decode": tiled_decode,
                "region": region,
                "canvas_shape": canvas_shape,
            }
        )
        return "processed-batch"


class TileProcessingTests(unittest.TestCase):
    def test_extract_and_process_tile_batch_forwards_args_and_computes_region(self):
        node = _DummyTileProcessor()
        tile_args = TileBatchArgs(
            core=UpscaleCoreArgs(
                model="model",
                positive="positive",
                negative="negative",
                vae="vae",
                seed=42,
                steps=20,
                cfg=7.5,
                sampler_name="euler",
                scheduler="normal",
                denoise=0.4,
                tiled_decode=False,
            ),
            tile_width=256,
            tile_height=192,
            padding=32,
            force_uniform_tiles=True,
            width=1024,
            height=768,
        )
        result = extract_and_process_tile_batch(
            node=node,
            upscaled_image="upscaled-image",
            tx=3,
            ty=4,
            args=tile_args,
        )

        self.assertEqual(result, ("processed-batch", 10, 20, 30, 40))
        self.assertEqual(len(node.extract_calls), 1)
        self.assertEqual(len(node.process_calls), 1)
        self.assertEqual(node.extract_calls[0]["tx"], 3)
        self.assertEqual(node.extract_calls[0]["ty"], 4)
        self.assertEqual(node.process_calls[0]["region"], (10, 20, 40, 60))
        self.assertEqual(node.process_calls[0]["canvas_shape"], (1024, 768))


if __name__ == "__main__":
    unittest.main()
