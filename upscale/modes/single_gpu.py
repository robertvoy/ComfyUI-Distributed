from __future__ import annotations

import math, torch
from typing import Any, Protocol, cast
from ...utils.logging import log
from ...utils.image import blend_processed_batch_item, pil_to_tensor, tensor_to_pil
from ..processing_args import UpscaleCoreArgs
from ..tile_processing import TileBatchArgs, extract_and_process_tile_batch


class _SingleGpuOps(Protocol):
    def round_to_multiple(self, value: int) -> int: ...
    def calculate_tiles(self, width: int, height: int, tile_width: int, tile_height: int, force_uniform_tiles: bool): ...
    def create_tile_mask(self, width: int, height: int, tx: int, ty: int, tile_width: int, tile_height: int, mask_blur: int): ...
    def blend_tile(self, base_image: Any, tile_pil: Any, x1: int, y1: int, size: tuple[int, int], tile_mask: Any, padding: int): ...
    def extract_batch_tile_with_padding(self, source_batch: Any, tx: int, ty: int, tile_width: int, tile_height: int, padding: int, force_uniform_tiles: bool): ...
    def process_tiles_batch(self, *args: Any, **kwargs: Any): ...


class SingleGpuModeMixin:
    def process_single_gpu(
        self,
        upscaled_image: torch.Tensor,
        core_args: UpscaleCoreArgs,
        tile_width: int,
        tile_height: int,
        padding: int,
        mask_blur: int,
        force_uniform_tiles: bool,
    ) -> tuple[torch.Tensor]:
        """Process all tiles on a single GPU (no distribution), batching per tile like USDU."""
        ops = cast(_SingleGpuOps, self)
        # Round tile dimensions
        tile_width = ops.round_to_multiple(tile_width)
        tile_height = ops.round_to_multiple(tile_height)

        # Get image dimensions and batch size
        batch_size, height, width, _ = upscaled_image.shape

        # Calculate all tiles
        all_tiles = ops.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)

        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        log(
            f"USDU Dist: Single GPU | Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Grid {rows}x{cols} ({len(all_tiles)} tiles/image) | Batch {batch_size}"
        )
        tile_batch_args = TileBatchArgs(
            core=core_args,
            tile_width=tile_width,
            tile_height=tile_height,
            padding=padding,
            force_uniform_tiles=force_uniform_tiles,
            width=width,
            height=height,
        )

        # Prepare result images list
        result_images = []
        for b in range(batch_size):
            image_pil = tensor_to_pil(upscaled_image[b:b+1], 0).convert('RGB')
            result_images.append(image_pil.copy())

        # Precompute tile masks once
        tile_masks = []
        for tx, ty in all_tiles:
            tile_masks.append(ops.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur))

        # Process tiles batched across images
        for tile_idx, (tx, ty) in enumerate(all_tiles):
            # Progressive state parity: extract each tile from the current updated image batch.
            source_batch = torch.cat([pil_to_tensor(img) for img in result_images], dim=0)
            if upscaled_image.is_cuda:
                source_batch = source_batch.cuda()

            processed_batch, x1, y1, ew, eh = extract_and_process_tile_batch(
                node=ops,
                upscaled_image=source_batch,
                tx=tx,
                ty=ty,
                args=tile_batch_args,
            )

            # Blend results back into each image using cached mask
            tile_mask = tile_masks[tile_idx]
            for b in range(batch_size):
                blend_processed_batch_item(
                    result_images,
                    processed_batch,
                    b,
                    ops.blend_tile,
                    x1,
                    y1,
                    ew,
                    eh,
                    tile_mask,
                    padding,
                )

        # Convert back to tensor
        result_tensors = [pil_to_tensor(img) for img in result_images]
        result_tensor = torch.cat(result_tensors, dim=0)
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()

        return (result_tensor,)
