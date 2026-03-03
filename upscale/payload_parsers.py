import io
import json
from collections.abc import Mapping
from typing import Any

from PIL import Image


def parse_tiles_from_form(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parse tiles submitted via multipart/form-data into a list of tile dicts."""
    try:
        padding = int(data.get('padding', 0)) if data.get('padding') is not None else 0
    except Exception:
        padding = 0

    meta_raw = data.get('tiles_metadata')
    if meta_raw is None:
        raise ValueError("Missing tiles_metadata")

    try:
        metadata = json.loads(meta_raw)
    except Exception as e:
        raise ValueError(f"Invalid tiles_metadata JSON: {e}")

    if not isinstance(metadata, list):
        raise ValueError("tiles_metadata must be a list")

    tiles = []
    for i, meta in enumerate(metadata):
        file_field = data.get(f'tile_{i}')
        if file_field is None or not hasattr(file_field, 'file'):
            raise ValueError(f"Missing tile data for index {i}")

        raw = file_field.file.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data for tile {i}: {e}")

        try:
            tile_info = {
                'image': img,
                'tile_idx': int(meta.get('tile_idx', i)),
                'x': int(meta.get('x', 0)),
                'y': int(meta.get('y', 0)),
                'extracted_width': int(meta.get('extracted_width', img.width)),
                'extracted_height': int(meta.get('extracted_height', img.height)),
                'padding': int(padding),
            }
        except Exception as e:
            raise ValueError(f"Invalid metadata values for tile {i}: {e}")

        if 'batch_idx' in meta:
            try:
                tile_info['batch_idx'] = int(meta['batch_idx'])
            except Exception as exc:
                raise ValueError(f"Invalid batch_idx for tile {i}: {meta.get('batch_idx')} ({exc})")
        if 'global_idx' in meta:
            try:
                tile_info['global_idx'] = int(meta['global_idx'])
            except Exception as exc:
                raise ValueError(f"Invalid global_idx for tile {i}: {meta.get('global_idx')} ({exc})")

        tiles.append(tile_info)

    return tiles


# Backward compatibility for existing imports.
_parse_tiles_from_form = parse_tiles_from_form
