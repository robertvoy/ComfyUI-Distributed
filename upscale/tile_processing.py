from dataclasses import dataclass
from typing import Any

from .processing_args import UpscaleCoreArgs


@dataclass(frozen=True)
class TileBatchArgs:
    """Tile/canvas parameters layered on top of shared core processing args."""

    core: UpscaleCoreArgs
    tile_width: int
    tile_height: int
    padding: int
    force_uniform_tiles: bool
    width: int
    height: int


def extract_and_process_tile_batch(
    *,
    node: Any,
    upscaled_image: Any,
    tx: int,
    ty: int,
    args: TileBatchArgs,
) -> tuple[Any, int, int, int, int]:
    """Extract one tile position for the whole batch and process it."""
    tile_batch, x1, y1, ew, eh = node.extract_batch_tile_with_padding(
        upscaled_image,
        tx,
        ty,
        args.tile_width,
        args.tile_height,
        args.padding,
        args.force_uniform_tiles,
    )
    region = (x1, y1, x1 + ew, y1 + eh)
    core = args.core
    processed_batch = node.process_tiles_batch(
        tile_batch,
        core.model,
        core.positive,
        core.negative,
        core.vae,
        core.seed,
        core.steps,
        core.cfg,
        core.sampler_name,
        core.scheduler,
        core.denoise,
        core.tiled_decode,
        region,
        (args.width, args.height),
    )
    return processed_batch, x1, y1, ew, eh
