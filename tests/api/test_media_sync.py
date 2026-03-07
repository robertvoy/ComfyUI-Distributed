import importlib.util
import sys
import types
import unittest
from pathlib import Path

from tests.api.harness import (
    bootstrap_test_package,
    cleanup_optional_module,
    install_aiohttp_stub,
)


class _AiohttpResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


def _load_media_sync_module():
    module_path = Path(__file__).resolve().parents[2] / "api" / "orchestration" / "media_sync.py"
    package_name = "dist_ms_testpkg"

    bootstrap_test_package(
        package_name,
        with_api=True,
        with_utils=True,
        with_orchestration=True,
    )

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_worker_url = lambda worker, endpoint="": f"http://localhost{endpoint}"

    async def _fake_session():
        raise RuntimeError("network calls not used in pure-function tests")

    network_module.get_client_session = _fake_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    trace_module = types.ModuleType(f"{package_name}.utils.trace_logger")
    trace_module.trace_debug = lambda *_args, **_kwargs: None
    trace_module.trace_info = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.trace_logger"] = trace_module

    auth_module = types.ModuleType(f"{package_name}.utils.auth")
    auth_module.distributed_auth_headers = lambda _config=None: {}
    sys.modules[f"{package_name}.utils.auth"] = auth_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: {"settings": {}}
    sys.modules[f"{package_name}.utils.config"] = config_module

    created_aiohttp_stub = install_aiohttp_stub(
        lambda payload, status=200: _AiohttpResponse(payload, status=status)
    )

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api.orchestration.media_sync",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cleanup_optional_module("aiohttp", created_aiohttp_stub)

    return module


ms = _load_media_sync_module()


# ---------------------------------------------------------------------------
# convert_paths_for_platform
# ---------------------------------------------------------------------------

class ConvertPathsForPlatformTests(unittest.TestCase):
    def test_forward_slash_target_normalises_backslashes(self):
        obj = {"ckpt_name": "C:\\Models\\model.safetensors"}
        result = ms.convert_paths_for_platform(obj, "/")
        self.assertEqual(result["ckpt_name"], "C:/Models/model.safetensors")

    def test_backslash_target_normalises_forward_slashes(self):
        obj = {"ckpt_name": "/models/checkpoints/model.safetensors"}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertIn("\\", result["ckpt_name"])
        self.assertNotIn("/", result["ckpt_name"])

    def test_relative_media_paths_always_stay_forward_slash(self):
        """Relative image/video/audio paths (Comfy annotated style) must not be backslash-ified."""
        obj = {"image": "subfolder/my_photo.png"}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertEqual(result["image"], "subfolder/my_photo.png")

    def test_relative_audio_paths_stay_forward_slash(self):
        obj = {"audio": "subfolder/my_track.wav"}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertEqual(result["audio"], "subfolder/my_track.wav")

    def test_annotated_relative_media_path_stays_forward_slash(self):
        obj = {"image": "input/frame.jpg [abc123]"}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertIn("/", result["image"])
        self.assertNotIn("\\", result["image"].split("[")[0])

    def test_non_filename_strings_are_untouched(self):
        obj = {"prompt": "a beautiful cat", "count": 5}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertEqual(result["prompt"], "a beautiful cat")
        self.assertEqual(result["count"], 5)

    def test_url_strings_are_untouched(self):
        obj = {"url": "https://example.com/model.safetensors"}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertEqual(result["url"], "https://example.com/model.safetensors")

    def test_invalid_separator_returns_obj_unchanged(self):
        obj = {"ckpt_name": "/models/model.safetensors"}
        result = ms.convert_paths_for_platform(obj, "|")
        self.assertEqual(result, obj)

    def test_nested_dict_is_processed_recursively(self):
        obj = {"node": {"ckpt_name": "C:\\Models\\model.safetensors"}}
        result = ms.convert_paths_for_platform(obj, "/")
        self.assertEqual(result["node"]["ckpt_name"], "C:/Models/model.safetensors")

    def test_list_items_are_processed_recursively(self):
        obj = [{"ckpt_name": "C:\\Models\\model.safetensors"}, "plain string"]
        result = ms.convert_paths_for_platform(obj, "/")
        self.assertEqual(result[0]["ckpt_name"], "C:/Models/model.safetensors")
        self.assertEqual(result[1], "plain string")

    def test_non_string_scalar_values_are_untouched(self):
        obj = {"seed": 42, "enabled": True, "ratio": 1.5}
        result = ms.convert_paths_for_platform(obj, "/")
        self.assertEqual(result["seed"], 42)
        self.assertTrue(result["enabled"])

    def test_absolute_unix_path_to_windows(self):
        obj = {"lora": "/home/user/loras/my_lora.safetensors"}
        result = ms.convert_paths_for_platform(obj, "\\")
        self.assertNotIn("/", result["lora"])

    def test_already_normalised_path_is_idempotent(self):
        obj = {"ckpt": "C:/Models/model.safetensors"}
        result = ms.convert_paths_for_platform(obj, "/")
        self.assertEqual(result["ckpt"], "C:/Models/model.safetensors")


# ---------------------------------------------------------------------------
# _find_media_references
# ---------------------------------------------------------------------------

class FindMediaReferencesTests(unittest.TestCase):
    def test_finds_image_input(self):
        prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("photo.png", refs)

    def test_finds_video_input(self):
        prompt = {"1": {"class_type": "LoadVideo", "inputs": {"video": "clip.mp4"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("clip.mp4", refs)

    def test_finds_file_input_for_load_video(self):
        prompt = {"1": {"class_type": "LoadVideo", "inputs": {"file": "1 - Copy.mp4"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("1 - Copy.mp4", refs)

    def test_finds_audio_input(self):
        prompt = {"1": {"class_type": "LoadAudio", "inputs": {"audio": "track.wav"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("track.wav", refs)

    def test_strips_annotation_suffix(self):
        prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "photo.jpg [abc123]"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("photo.jpg", refs)
        self.assertFalse(any("[" in r for r in refs))

    def test_normalises_backslashes_in_path(self):
        prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "sub\\img.png"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("sub/img.png", refs)

    def test_ignores_non_media_text_inputs(self):
        prompt = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}}}
        refs = ms._find_media_references(prompt)
        self.assertEqual(refs, [])

    def test_ignores_node_link_values(self):
        """Inputs that are [node_id, slot] lists should be ignored."""
        prompt = {"1": {"class_type": "Anything", "inputs": {"image": ["2", 0]}}}
        refs = ms._find_media_references(prompt)
        self.assertEqual(refs, [])

    def test_deduplicates_same_file_across_nodes(self):
        prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "cat.png"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "cat.png"}},
        }
        refs = ms._find_media_references(prompt)
        self.assertEqual(len(refs), 1)

    def test_returns_sorted_list(self):
        prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "z_image.png"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "a_image.jpg"}},
        }
        refs = ms._find_media_references(prompt)
        self.assertEqual(refs, sorted(refs))

    def test_ignores_non_dict_nodes(self):
        prompt = {"1": "not a node dict", "2": {"class_type": "LoadImage", "inputs": {"image": "img.png"}}}
        refs = ms._find_media_references(prompt)
        self.assertIn("img.png", refs)

    def test_empty_prompt_returns_empty_list(self):
        self.assertEqual(ms._find_media_references({}), [])

    def test_multiple_media_types_all_found(self):
        prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "frame.png"}},
            "2": {"class_type": "LoadVideo", "inputs": {"video": "clip.mp4"}},
            "3": {"class_type": "LoadAudio", "inputs": {"audio": "track.wav"}},
        }
        refs = ms._find_media_references(prompt)
        self.assertIn("frame.png", refs)
        self.assertIn("clip.mp4", refs)
        self.assertIn("track.wav", refs)


class RewritePromptMediaInputsTests(unittest.TestCase):
    def test_rewrites_video_file_input_to_worker_path(self):
        prompt = {
            "79": {"class_type": "LoadVideo", "inputs": {"file": "1 - Copy.mp4"}},
        }
        ms._rewrite_prompt_media_inputs(prompt, {"1 - Copy.mp4": "videos/1 - Copy.mp4"})
        self.assertEqual(prompt["79"]["inputs"]["file"], "videos/1 - Copy.mp4")

    def test_rewrites_audio_input_and_strips_annotation_when_matching(self):
        prompt = {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": "song.wav [input]"}},
        }
        ms._rewrite_prompt_media_inputs(prompt, {"song.wav": "song.wav"})
        self.assertEqual(prompt["1"]["inputs"]["audio"], "song.wav")


if __name__ == "__main__":
    unittest.main()
    class _AiohttpResponse:
        def __init__(self, payload, status=200):
            self.payload = payload
            self.status = status
