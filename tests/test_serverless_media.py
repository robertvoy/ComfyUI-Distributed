import base64
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "api" / "orchestration" / "serverless_media.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dist_serverless_media_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


media = load_module()


def png_bytes(mode="RGB", size=(2, 1), color=None):
    if color is None:
        color = (10, 20, 30, 40) if mode == "RGBA" else (10, 20, 30)
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class FolderPathsFixture:
    def __init__(self, root):
        self.input = root / "input"
        self.output = root / "output"
        self.temp = root / "temp"
        for directory in (self.input, self.output, self.temp):
            directory.mkdir(parents=True, exist_ok=True)

    def get_input_directory(self):
        return str(self.input)

    def get_output_directory(self):
        return str(self.output)

    def get_temp_directory(self):
        return str(self.temp)

    def get_annotated_filepath(self, filename):
        value = filename.replace("\\", "/")
        # Match real ComfyUI folder_paths: root annotations are suffixes,
        # not directory prefixes (output/foo.png is an input subdirectory).
        for marker, root in (("input", self.input), ("output", self.output), ("temp", self.temp)):
            suffix = f" [{marker}]"
            if value.endswith(suffix):
                return str(root / value[:-len(suffix)])
        return str(self.input / value)


class ServerlessMediaInputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.paths = FolderPathsFixture(Path(self.tempdir.name))
        self.folder_paths = types.ModuleType("folder_paths")
        self.folder_paths.get_input_directory = self.paths.get_input_directory
        self.folder_paths.get_output_directory = self.paths.get_output_directory
        self.folder_paths.get_temp_directory = self.paths.get_temp_directory
        self.folder_paths.get_annotated_filepath = self.paths.get_annotated_filepath
        self.folder_patch = patch.dict(sys.modules, {"folder_paths": self.folder_paths})
        self.folder_patch.start()
        self.addCleanup(self.folder_patch.stop)
        self.addCleanup(self.tempdir.cleanup)

    async def test_packs_image_and_rewrites_duplicate_loadimage_inputs(self):
        payload = png_bytes()
        (self.paths.input / "cat.png").write_bytes(payload)
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "cat.png [input]", "other": "unchanged"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "cat.png [input]"}},
            "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "stable"}},
        }

        packed = await media.prepare_images(workflow)

        self.assertEqual(len(packed), 1)
        self.assertEqual(set(packed[0]), {"name", "image"})
        self.assertRegex(packed[0]["name"], r"^[A-Za-z0-9_.-]+\.png$")
        self.assertEqual(base64.b64decode(packed[0]["image"], validate=True), payload)
        self.assertEqual(workflow["1"]["inputs"]["image"], packed[0]["name"])
        self.assertEqual(workflow["2"]["inputs"]["image"], packed[0]["name"])
        self.assertEqual(workflow["1"]["inputs"]["other"], "unchanged")
        self.assertEqual(workflow["3"]["inputs"]["filename_prefix"], "stable")

    async def test_resolves_annotated_input_output_and_temp_sources(self):
        sources = ((self.paths.input, "one.png"), (self.paths.output, "two.jpg"), (self.paths.temp, "three.webp"))
        workflow = {}
        for index, (directory, filename) in enumerate(sources, 1):
            (directory / filename).write_bytes(png_bytes())
            root = directory.name
            workflow[str(index)] = {"class_type": "LoadImage", "inputs": {"image": f"{filename} [{root}]"}}

        packed = await media.prepare_images(workflow)

        self.assertEqual(len(packed), 3)
        self.assertEqual({base64.b64decode(item["image"]) for item in packed}, {png_bytes()})
        self.assertEqual({node["inputs"]["image"] for node in workflow.values()}, {item["name"] for item in packed})

    async def test_directory_prefix_is_not_a_root_annotation(self):
        (self.paths.input / "output").mkdir()
        payload = png_bytes(color=(100, 0, 0))
        (self.paths.input / "output" / "same.png").write_bytes(payload)
        (self.paths.output / "same.png").write_bytes(png_bytes())
        workflow = {"1": {"class_type": "LoadImage", "inputs": {"image": "output/same.png"}}}
        packed = await media.prepare_images(workflow)
        self.assertEqual(base64.b64decode(packed[0]["image"]), payload)

    async def test_rejects_unknown_root_annotation(self):
        (self.paths.input / "cat.png").write_bytes(png_bytes())
        workflow = {"1": {"class_type": "LoadImage", "inputs": {"image": "cat.png [abc123]"}}}
        with self.assertRaisesRegex(media.ServerlessMediaError, "annotation"):
            await media.prepare_images(workflow)

    async def test_recognizes_custom_file_image_reference(self):
        (self.paths.input / "source.bmp").write_bytes(png_bytes())
        workflow = {"1": {"class_type": "CustomImageLoader", "inputs": {"file": "source.bmp"}}}

        packed = await media.prepare_images(workflow)

        self.assertEqual(len(packed), 1)
        self.assertEqual(workflow["1"]["inputs"]["file"], packed[0]["name"])

    async def test_rejects_missing_and_dynamic_loadimage_values(self):
        cases = (
            {"1": {"class_type": "LoadImage", "inputs": {}}},
            {"1": {"class_type": "LoadImage"}},
            {"1": {"class_type": "LoadImage", "inputs": None}},
            {"1": {"class_type": "LoadImage", "inputs": {"image": ["2", 0]}}},
        )
        for workflow in cases:
            with self.subTest(workflow=workflow):
                with self.assertRaisesRegex(media.ServerlessMediaError, "LoadImage"):
                    await media.prepare_images(workflow)

    async def test_rejects_missing_source_and_unapproved_extension(self):
        for value in ("absent.png", "animation.gif", "clip.mp4"):
            with self.subTest(value=value):
                workflow = {"1": {"class_type": "LoadImage", "inputs": {"image": value}}}
                with self.assertRaises(media.ServerlessMediaError):
                    await media.prepare_images(workflow)

    async def test_rejects_unsupported_audio_or_video_references_without_reading(self):
        for key, value in (("audio", "private.wav"), ("video", "private.mp4"), ("file", "private.wav")):
            with self.subTest(key=key):
                workflow = {"1": {"class_type": "CustomLoader", "inputs": {key: value}}}
                with self.assertRaisesRegex(media.ServerlessMediaError, "audio|video|media"):
                    await media.prepare_images(workflow)

    async def test_rejects_parent_traversal_and_absolute_outside_roots(self):
        outside = Path(self.tempdir.name).parent / "outside.png"
        outside.write_bytes(png_bytes())
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        for value in ("../outside.png", str(outside)):
            with self.subTest(value=value):
                workflow = {"1": {"class_type": "LoadImage", "inputs": {"image": value}}}
                with self.assertRaises(media.ServerlessMediaError):
                    await media.prepare_images(workflow)

    async def test_rejects_symlink_escape_from_allowed_root(self):
        outside = Path(self.tempdir.name).parent / "linked-outside.png"
        outside.write_bytes(png_bytes())
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.paths.input / "link.png").symlink_to(outside)
        workflow = {"1": {"class_type": "LoadImage", "inputs": {"image": "link.png"}}}

        with self.assertRaises(media.ServerlessMediaError):
            await media.prepare_images(workflow)

    async def test_rejects_individual_and_aggregate_input_size_overages(self):
        huge = self.paths.input / "large.jpg"
        with huge.open("wb") as handle:
            handle.truncate(media.MAX_INPUT_IMAGE_BYTES + 1)
        workflow = {"1": {"class_type": "LoadImage", "inputs": {"image": "large.jpg"}}}
        with self.assertRaisesRegex(media.ServerlessMediaError, "size"):
            await media.prepare_images(workflow)

        huge.unlink()
        first = self.paths.input / "first.png"
        second = self.paths.input / "second.png"
        first.write_bytes(b"a" * (media.MAX_INPUT_TOTAL_BYTES // 2 + 1))
        second.write_bytes(b"b" * (media.MAX_INPUT_TOTAL_BYTES // 2 + 1))
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "first.png"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "second.png"}},
        }
        with self.assertRaisesRegex(media.ServerlessMediaError, "size"):
            await media.prepare_images(workflow)


class ServerlessMediaOutputTests(unittest.IsolatedAsyncioTestCase):
    def manifest(self, prefix="render"):
        return [{"collector_id": "collector-1", "job_id": "job-1", "node_id": "12", "kind": "image", "prefix": prefix}]

    def artifact(self, filename, payload=None, **extra):
        entry = {
            "filename": filename,
            "artifact_type": "output",
            "artifact_path": f"/safe/output/{filename}",
            "subfolder": "",
            "type": "base64",
            "data": base64.b64encode(payload if payload is not None else png_bytes()).decode("ascii"),
        }
        entry.update(extra)
        return entry

    async def test_decodes_rgb_to_normalized_float32_and_orders_by_counter(self):
        output = {"images": [self.artifact("render_00009_.png"), self.artifact("render_00002_.png")]}

        result = await media.collect_images(output, self.manifest())

        self.assertEqual(list(result), ["job-1"])
        self.assertEqual(len(result["job-1"]), 2)
        self.assertTrue(all(image.dtype.name == "float32" for image in result["job-1"]))
        self.assertEqual(result["job-1"][0].shape, (1, 2, 3))
        self.assertAlmostEqual(float(result["job-1"][0][0, 0, 0]), 10 / 255)

    async def test_groups_images_under_manifest_job_ids(self):
        manifest = [
            self.manifest("first")[0],
            {"collector_id": "collector-2", "job_id": "job-2", "node_id": "14", "kind": "image", "prefix": "second"},
        ]
        output = {"images": [self.artifact("first_00001_.png"), self.artifact("second_00001_.png")]}

        result = await media.collect_images(output, manifest)

        self.assertEqual(set(result), {"job-1", "job-2"})
        self.assertEqual(len(result["job-1"]), 1)
        self.assertEqual(len(result["job-2"]), 1)

    async def test_preserves_rgba_alpha(self):
        output = {"images": [self.artifact("render_00001_.png", png_bytes(mode="RGBA"))]}

        result = await media.collect_images(output, self.manifest())

        self.assertEqual(result["job-1"][0].shape, (1, 2, 4))
        self.assertAlmostEqual(float(result["job-1"][0][0, 0, 3]), 40 / 255)

    async def test_signed_url_is_downloaded_only_through_injected_downloader(self):
        output = {"images": [{
            "filename": "render_00001_.png", "artifact_type": "output", "type": "gcs_url",
            "url": "https://storage.example.test/signed",
        }]}
        fetch = AsyncMock(return_value=png_bytes())

        result = await media.collect_images(output, self.manifest(), download=fetch)

        self.assertEqual(result["job-1"][0].shape, (1, 2, 3))
        fetch.assert_awaited_once_with("https://storage.example.test/signed")

    async def test_ignores_unrelated_artifacts_but_requires_every_manifest_collector(self):
        output = {"images": [
            self.artifact("preview_00001_.png"),
            self.artifact("render_00001_.png"),
            {"filename": "mesh.obj", "type": "url", "url": "not a fetch target"},
        ]}

        result = await media.collect_images(output, self.manifest())

        self.assertEqual(len(result["job-1"]), 1)
        with self.assertRaisesRegex(media.ServerlessMediaError, "expected|missing"):
            await media.collect_images({"images": [self.artifact("preview_00001_.png")]}, self.manifest())

    async def test_accepts_compiler_length_flat_safe_prefix(self):
        prefix = "serverless_" + "j" * 24 + "_" + "w" * 24 + "_" + "c" * 24 + "_image_" + "a" * 64
        manifest = self.manifest(prefix)
        output = {"images": [self.artifact(f"{prefix}_00001_.png")]}

        result = await media.collect_images(output, manifest)

        self.assertEqual(len(result["job-1"]), 1)

    async def test_rejects_duplicate_manifest_prefix_and_unsafe_prefix(self):
        duplicate = self.manifest() * 2
        with self.assertRaises(media.ServerlessMediaError):
            await media.collect_images({"images": []}, duplicate)
        unsafe = self.manifest("../render")
        with self.assertRaises(media.ServerlessMediaError):
            await media.collect_images({"images": []}, unsafe)

    async def test_rejects_duplicate_canonical_artifact_identity(self):
        item = self.artifact("render_00001_.png")
        with self.assertRaisesRegex(media.ServerlessMediaError, "[Dd]uplicate"):
            await media.collect_images({"images": [item, dict(item)]}, self.manifest())

    async def test_rejects_matching_non_output_artifact_and_non_numeric_counter(self):
        wrong_type = self.artifact("render_00001_.png", artifact_type="temp")
        with self.assertRaises(media.ServerlessMediaError):
            await media.collect_images({"images": [wrong_type]}, self.manifest())
        with self.assertRaises(media.ServerlessMediaError):
            await media.collect_images({"images": [self.artifact("render_final.png")]}, self.manifest())

    async def test_rejects_provider_errors_even_when_job_completed(self):
        for output in (
            {"error": "raw provider secret", "images": []},
            {"errors": ["raw provider secret"], "images": []},
            {"status": "COMPLETED", "output": {"error": "raw provider secret", "images": []}},
        ):
            with self.subTest(output=output):
                with self.assertRaises(media.ServerlessMediaError) as ctx:
                    await media.collect_images(output, self.manifest())
                self.assertNotIn("raw provider secret", str(ctx.exception))

    async def test_rejects_invalid_base64_and_malformed_image(self):
        for item in (
            self.artifact("render_00001_.png", data="!not base64!"),
            self.artifact("render_00001_.png", data=base64.b64encode(b"not an image").decode()),
        ):
            with self.subTest(item=item):
                with self.assertRaises(media.ServerlessMediaError):
                    await media.collect_images({"images": [item]}, self.manifest())

    async def test_validates_shape_consistency_within_collector(self):
        output = {"images": [
            self.artifact("render_00001_.png", png_bytes(size=(2, 1))),
            self.artifact("render_00002_.png", png_bytes(size=(3, 1))),
        ]}
        with self.assertRaisesRegex(media.ServerlessMediaError, "shape|dimensions"):
            await media.collect_images(output, self.manifest())

    async def test_bounds_output_file_count_and_bytes(self):
        too_many = {"images": [{}] * (media.MAX_OUTPUT_FILES + 1)}
        with self.assertRaisesRegex(media.ServerlessMediaError, "files"):
            await media.collect_images(too_many, self.manifest())

        with patch.object(media, "MAX_OUTPUT_IMAGE_BYTES", 4):
            output = {"images": [self.artifact("render_00001_.png", data=base64.b64encode(b"12345").decode())]}
            with self.assertRaisesRegex(media.ServerlessMediaError, "size"):
                await media.collect_images(output, self.manifest())

    async def test_aggregate_payload_limit_is_checked_as_artifacts_are_read(self):
        output = {"images": [
            self.artifact("render_00001_.png", data=base64.b64encode(b"12").decode()),
            self.artifact("render_00002_.png", data=base64.b64encode(b"34").decode()),
        ]}
        with patch.object(media, "MAX_OUTPUT_TOTAL_BYTES", 3):
            with self.assertRaisesRegex(media.ServerlessMediaError, "size"):
                await media.collect_images(output, self.manifest())

    async def test_rejects_non_public_or_non_https_artifact_urls_before_downloading(self):
        for url in ("http://example.test/a", "https://127.0.0.1/a", "https://user:pass@example.test/a"):
            with self.subTest(url=url):
                fetch = AsyncMock(return_value=png_bytes())
                output = {"images": [{"filename": "render_00001_.png", "type": "url", "url": url}]}
                with self.assertRaises(media.ServerlessMediaError):
                    await media.collect_images(output, self.manifest(), download=fetch)
                fetch.assert_not_awaited()

    async def test_public_resolver_rejects_non_global_dns_answers(self):
        resolver = media._PublicResolver()
        resolver._getaddrinfo = AsyncMock(return_value=[(2, 1, 6, "", ("8.8.8.8", 443))])
        self.assertEqual(len(await resolver.resolve("cdn.example.test", 443)), 1)
        resolver._getaddrinfo = AsyncMock(return_value=[(2, 1, 6, "", ("10.0.0.5", 443))])
        with self.assertRaises(OSError):
            await resolver.resolve("cdn.example.test", 443)

    async def test_default_download_revalidates_redirects_and_disables_environment_proxies(self):
        class Content:
            def __init__(self, chunks):
                self.chunks = iter(chunks)
            async def iter_chunked(self, _size):
                for chunk in self.chunks:
                    yield chunk

        class Response:
            def __init__(self, status, headers=None, chunks=()):
                self.status, self.headers = status, headers or {}
                self.content = Content(chunks)
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_args):
                return False

        class Session:
            def __init__(self, responses):
                self.responses, self.calls = list(responses), []
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_args):
                return False
            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return self.responses.pop(0)

        session = Session([Response(302, {"Location": "https://cdn.example.test/final"}), Response(200, chunks=[b"abc", b"def"])])
        with patch.object(media.aiohttp, "ClientSession", return_value=session) as create_session:
            payload = await media.download_public_artifact("https://storage.example.test/signed")

        self.assertEqual(payload, b"abcdef")
        self.assertEqual([call[0] for call in session.calls], ["https://storage.example.test/signed", "https://cdn.example.test/final"])
        self.assertTrue(all(call[1]["allow_redirects"] is False for call in session.calls))
        self.assertFalse(create_session.call_args.kwargs["trust_env"])
        self.assertIsInstance(create_session.call_args.kwargs["connector"]._resolver, media._PublicResolver)
        self.assertNotIn("Authorization", session.calls[0][1].get("headers", {}))
        await create_session.call_args.kwargs["connector"].close()

        blocked = Session([Response(302, {"Location": "https://127.0.0.1/metadata"})])
        with patch.object(media.aiohttp, "ClientSession", return_value=blocked) as blocked_session:
            with self.assertRaises(media.ServerlessMediaError):
                await media.download_public_artifact("https://storage.example.test/signed")
        self.assertEqual(len(blocked.calls), 1)
        await blocked_session.call_args.kwargs["connector"].close()


if __name__ == "__main__":
    unittest.main()
