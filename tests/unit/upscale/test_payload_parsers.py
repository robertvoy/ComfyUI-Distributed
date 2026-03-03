import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _load_payload_parsers_module():
    # payload_parsers.py has no relative imports; only stdlib + PIL
    module_path = Path(__file__).resolve().parents[3] / "upscale" / "payload_parsers.py"
    spec = importlib.util.spec_from_file_location("upscale_payload_parsers", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


if PIL_AVAILABLE:
    pp = _load_payload_parsers_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width=64, height=64, color=(128, 64, 32)):
    """Return raw PNG bytes for a solid-colour image."""
    img = PILImage.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _MockFileField:
    """Minimal multipart file-field stub."""

    class _MockFile:
        def __init__(self, data: bytes):
            self._buf = io.BytesIO(data)

        def read(self) -> bytes:
            return self._buf.read()

    def __init__(self, data: bytes):
        self.file = self._MockFile(data)


def _make_form(n_tiles, *, padding=None, extra_meta=None, image_color=(128, 64, 32)):
    """Build a minimal form-data dict with `n_tiles` tile entries."""
    image_bytes = _make_png_bytes(color=image_color)
    metadata = []
    for i in range(n_tiles):
        entry = {
            "tile_idx": i,
            "x": i * 64,
            "y": 0,
            "extracted_width": 64,
            "extracted_height": 64,
        }
        if extra_meta and i < len(extra_meta):
            entry.update(extra_meta[i])
        metadata.append(entry)

    form = {"tiles_metadata": json.dumps(metadata)}
    if padding is not None:
        form["padding"] = str(padding)
    for i in range(n_tiles):
        form[f"tile_{i}"] = _MockFileField(image_bytes)
    return form


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(PIL_AVAILABLE, "PIL not installed")
class ParseTilesFromFormTests(unittest.TestCase):

    # --- happy paths ---

    def test_single_tile_returns_one_entry(self):
        tiles = pp._parse_tiles_from_form(_make_form(1))
        self.assertEqual(len(tiles), 1)

    def test_multiple_tiles_all_returned(self):
        tiles = pp._parse_tiles_from_form(_make_form(3))
        self.assertEqual(len(tiles), 3)

    def test_tile_image_is_pil_image(self):
        tiles = pp._parse_tiles_from_form(_make_form(1))
        self.assertIsInstance(tiles[0]["image"], PILImage.Image)

    def test_tile_metadata_fields_are_parsed(self):
        tiles = pp._parse_tiles_from_form(_make_form(1))
        tile = tiles[0]
        self.assertEqual(tile["tile_idx"], 0)
        self.assertEqual(tile["x"], 0)
        self.assertEqual(tile["y"], 0)
        self.assertEqual(tile["extracted_width"], 64)
        self.assertEqual(tile["extracted_height"], 64)

    def test_padding_is_parsed_from_form(self):
        tiles = pp._parse_tiles_from_form(_make_form(1, padding=16))
        self.assertEqual(tiles[0]["padding"], 16)

    def test_default_padding_is_zero(self):
        form = _make_form(1)
        form.pop("padding", None)
        tiles = pp._parse_tiles_from_form(form)
        self.assertEqual(tiles[0]["padding"], 0)

    def test_invalid_padding_string_falls_back_to_zero(self):
        form = _make_form(1)
        form["padding"] = "not_a_number"
        tiles = pp._parse_tiles_from_form(form)
        self.assertEqual(tiles[0]["padding"], 0)

    def test_optional_batch_idx_included_when_present(self):
        extra = [{"batch_idx": 2}]
        tiles = pp._parse_tiles_from_form(_make_form(1, extra_meta=extra))
        self.assertEqual(tiles[0]["batch_idx"], 2)

    def test_optional_global_idx_included_when_present(self):
        extra = [{"global_idx": 5}]
        tiles = pp._parse_tiles_from_form(_make_form(1, extra_meta=extra))
        self.assertEqual(tiles[0]["global_idx"], 5)

    def test_batch_idx_and_global_idx_absent_when_not_in_metadata(self):
        tiles = pp._parse_tiles_from_form(_make_form(1))
        self.assertNotIn("batch_idx", tiles[0])
        self.assertNotIn("global_idx", tiles[0])

    def test_tile_indices_match_metadata_order(self):
        tiles = pp._parse_tiles_from_form(_make_form(3))
        for i, tile in enumerate(tiles):
            self.assertEqual(tile["tile_idx"], i)

    def test_x_coordinates_reflect_metadata(self):
        tiles = pp._parse_tiles_from_form(_make_form(3))
        self.assertEqual(tiles[1]["x"], 64)
        self.assertEqual(tiles[2]["x"], 128)

    # --- error cases ---

    def test_missing_tiles_metadata_raises_value_error(self):
        with self.assertRaises(ValueError, msg="Missing tiles_metadata"):
            pp._parse_tiles_from_form({})

    def test_invalid_json_metadata_raises_value_error(self):
        form = {"tiles_metadata": "{not valid json}"}
        with self.assertRaises(ValueError):
            pp._parse_tiles_from_form(form)

    def test_non_list_metadata_raises_value_error(self):
        form = {"tiles_metadata": json.dumps({"not": "a list"})}
        with self.assertRaises(ValueError):
            pp._parse_tiles_from_form(form)

    def test_missing_tile_file_field_raises_value_error(self):
        form = {
            "tiles_metadata": json.dumps([{"tile_idx": 0, "x": 0, "y": 0}]),
            # tile_0 intentionally omitted
        }
        with self.assertRaises(ValueError):
            pp._parse_tiles_from_form(form)

    def test_tile_field_without_file_attr_raises_value_error(self):
        form = {
            "tiles_metadata": json.dumps([{"tile_idx": 0, "x": 0, "y": 0}]),
            "tile_0": "plain string without .file",
        }
        with self.assertRaises(ValueError):
            pp._parse_tiles_from_form(form)

    def test_non_image_bytes_raises_value_error(self):
        class _BadFileField:
            class _BadFile:
                def read(self):
                    return b"this is definitely not image data"
            file = _BadFile()

        form = {
            "tiles_metadata": json.dumps([{"tile_idx": 0, "x": 0, "y": 0}]),
            "tile_0": _BadFileField(),
        }
        with self.assertRaises(ValueError):
            pp._parse_tiles_from_form(form)

    def test_invalid_metadata_value_type_raises_value_error(self):
        """Non-integer metadata fields (x, y, etc.) should raise ValueError."""
        form = {
            "tiles_metadata": json.dumps([{"tile_idx": 0, "x": "not_int", "y": 0}]),
            "tile_0": _MockFileField(_make_png_bytes()),
        }
        with self.assertRaises(ValueError):
            pp._parse_tiles_from_form(form)


if __name__ == "__main__":
    unittest.main()
