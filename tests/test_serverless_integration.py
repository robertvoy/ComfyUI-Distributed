"""Offline integration: real graph preparation + orchestration, fake provider."""
import asyncio
import aiohttp as real_aiohttp
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PKG = "dist_serverless_integration"


def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    return value


def load_modules():
    modules = {}
    for part, directory in [("", ROOT), (".api", ROOT / "api"), (".api.orchestration", ROOT / "api/orchestration"), (".utils", ROOT / "utils")]:
        modules[PKG + part] = module(PKG + part, __path__=[str(directory)])
    ps = types.SimpleNamespace(routes=types.SimpleNamespace(get=lambda _: lambda fn: fn, post=lambda _: lambda fn: fn))
    modules["server"] = module("server", PromptServer=types.SimpleNamespace(instance=ps))
    modules[PKG + ".utils.logging"] = module("logging", debug_log=lambda *a: None, log=lambda *a: None)
    modules[PKG + ".utils.trace_logger"] = module("trace_logger", trace_debug=lambda *a: None)
    modules[PKG + ".utils.config"] = module("config", load_config=lambda: {})
    modules[PKG + ".utils.async_helpers"] = module("async_helpers", queue_prompt_payload=AsyncMock())
    modules[PKG + ".utils.network"] = module("network", build_master_url=lambda **k: "http://master", build_master_callback_url=lambda *a, **k: "http://master")
    modules[PKG + ".api.orchestration.dispatch"] = module("dispatch", dispatch_worker_prompt=AsyncMock(), select_active_workers=AsyncMock(), select_least_busy_worker=AsyncMock())
    modules[PKG + ".api.orchestration.media_sync"] = module("media_sync", convert_paths_for_platform=lambda p, _: p, fetch_worker_path_separator=AsyncMock(), fetch_worker_has_node=AsyncMock(return_value=False), sync_worker_media=AsyncMock())
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(PKG + ".api.queue_orchestration", ROOT / "api/queue_orchestration.py")
        q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(q)
        # Keep imported package namespace only; restore shared Comfy runtime stub.
        modules.update({k: v for k, v in sys.modules.items() if k.startswith(PKG)})
    return q, modules, ps


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.q, modules, self.ps = load_modules()
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.pool = {"id": "pool", "name": "Runpod", "type": "serverless", "provider": "runpod", "endpoint_id": "endpoint", "enabled": True, "jobs_per_run": 2, "concurrency": 1, "timeout_seconds": 30}
        self.q.load_config = lambda: {"workers": [self.pool], "settings": {"master_delegate_only": True}}
        self.q.select_active_workers = AsyncMock(side_effect=lambda workers, _ws, delegate, **kw: (workers, delegate))
        self.q.queue_prompt_payload = AsyncMock(return_value={"prompt_id": "master-prompt", "number": 1})
        self.manager = types.SimpleNamespace(start=AsyncMock(), cancel=AsyncMock(), ensure_capacity=lambda _: None)
        sys.modules[PKG + ".api.orchestration.serverless_runtime"] = module("runtime", get_manager=lambda: self.manager)
        sys.modules[PKG + ".api.orchestration.serverless_media"] = module("media", prepare_images=AsyncMock(return_value=[]))
        self.prompt = {
            "1": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0}},
            "2": {"class_type": "DistributedCollector", "inputs": {"images": ["1", 0], "load_balance": False}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "local"}},
        }

    async def test_real_compiler_slots_to_master_collector_without_callbacks(self):
        result = await self.q.orchestrate_distributed_execution(self.prompt, None, "client")
        self.assertEqual(result[2], 2)
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.manager.start.assert_awaited_once()
        jobs = self.manager.start.call_args.args[2]
        self.assertEqual([job["worker"]["id"] for job in jobs], ["serverless:pool:1", "serverless:pool:2"])
        self.assertEqual(len({job["manifest"][0]["prefix"] for job in jobs}), 2)
        for job in jobs:
            graph = job["workflow"]
            self.assertTrue(any(n["class_type"] == "SaveImage" for n in graph.values()))
            self.assertFalse(any("Distributed" in n["class_type"] for n in graph.values()))
            self.assertNotIn("http://master", json.dumps(graph))
        master_graph = self.q.queue_prompt_payload.call_args.args[0]
        self.assertEqual(json.loads(master_graph["2"]["inputs"]["enabled_worker_ids"]), ["serverless:pool:1", "serverless:pool:2"])

    async def test_compile_pack_provider_decode_and_collect_real_components(self):
        """Exercise the whole image path, mocking only the provider and Comfy queue."""
        import base64
        import io
        import tempfile
        import torch
        from PIL import Image
        from test_collector_list_inputs import _load_collector_module

        sys.modules["aiohttp"] = real_aiohttp
        for leaf in ("serverless_media", "serverless_runtime"):
            name = PKG + ".api.orchestration." + leaf
            spec = importlib.util.spec_from_file_location(name, ROOT / "api/orchestration" / (leaf + ".py"))
            real_module = importlib.util.module_from_spec(spec)
            sys.modules[name] = real_module
            spec.loader.exec_module(real_module)
        runtime = sys.modules[PKG + ".api.orchestration.serverless_runtime"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_buffer = io.BytesIO()
            Image.new("RGB", (2, 3), color=(255, 0, 128)).save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()
            (root / "source.png").write_bytes(image_bytes)
            paths = module("folder_paths", get_input_directory=lambda: directory,
                           get_output_directory=lambda: directory, get_temp_directory=lambda: directory,
                           get_annotated_filepath=lambda name: str(root / name))
            submissions = []

            class FakeProvider:
                async def submit(self, graph, images, timeout_seconds):
                    self.prefix = next(n["inputs"]["filename_prefix"] for n in graph.values() if n["class_type"] == "SaveImage")
                    submissions.append((graph, images))
                    self_image = next(n["inputs"]["image"] for n in graph.values() if n["class_type"] == "LoadImage")
                    assert self_image == images[0]["name"]
                    assert base64.b64decode(images[0]["image"]) == image_bytes
                    return "provider-job-" + str(len(submissions))

                async def status(self, _job_id):
                    filename = self.prefix + "_00001_.png"
                    return {"status": "COMPLETED", "output": {"images": [{
                        "filename": filename, "artifact_path": filename, "artifact_type": "output",
                        "type": "base64", "data": base64.b64encode(image_bytes).decode(),
                    }]}}

            manager = runtime.ServerlessManager(self.ps, None, root / "state" / "jobs.json", None)
            manager._client = AsyncMock(side_effect=lambda _: FakeProvider())
            self.ps.distributed_serverless = manager
            self.prompt["1"] = {"class_type": "LoadImage", "inputs": {"image": "source.png"}}
            collector_module = _load_collector_module()
            collector_module.prompt_server = self.ps
            sys.modules["aiohttp"] = real_aiohttp
            with patch.object(runtime, "get_manager", return_value=manager), patch.dict(sys.modules, {"folder_paths": paths}):
                result = await self.q.orchestrate_distributed_execution(self.prompt, None, "client")
                master_graph = self.q.queue_prompt_payload.call_args.args[0]
                inputs = master_graph["2"]["inputs"]
                images, _audio = await asyncio.wait_for(collector_module.DistributedCollectorNode().execute(
                    images=None, audio=None, multi_job_id=inputs["multi_job_id"],
                    enabled_worker_ids=inputs["enabled_worker_ids"], delegate_only=True,
                ), timeout=3)
                await asyncio.gather(*manager.tasks.values())
            self.assertEqual(result[2], 2)
            self.assertEqual(len(submissions), 2)
            self.assertEqual(tuple(images.shape), (2, 3, 2, 3))
            self.assertEqual(images.dtype, torch.float32)
            self.assertTrue(torch.equal(images[0], images[1]))
            self.assertEqual(float(images[0, 0, 0, 0]), 1.0)
            self.assertEqual([record["status"] for record in manager.ledger.all()], ["COMPLETED", "COMPLETED"])
            self.assertEqual(self.ps.distributed_pending_jobs, {})
            self.q.dispatch_worker_prompt.assert_not_awaited()

    async def test_audio_rejected_before_any_dispatch_or_master_queue(self):
        self.prompt["2"]["inputs"]["audio"] = ["9", 0]
        self.prompt["9"] = {"class_type": "LoadAudio", "inputs": {"audio": "test.wav"}}
        with self.assertRaises(ValueError):
            await self.q.orchestrate_distributed_execution(self.prompt, None, "client")
        self.manager.start.assert_not_awaited()
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_disabled_pool_excluded(self):
        self.pool["enabled"] = False
        self.assertEqual(self.q._resolve_enabled_workers({"workers": [self.pool]}), [])

    async def test_master_validation_failure_submits_no_paid_job(self):
        self.q.queue_prompt_payload.side_effect = ValueError("invalid local graph")
        with self.assertRaisesRegex(ValueError, "invalid local graph"):
            await self.q.orchestrate_distributed_execution(self.prompt, None, "client")
        self.manager.start.assert_not_awaited()

    async def test_oversized_serverless_request_fails_before_dispatch_or_master_queue(self):
        media_module = sys.modules[PKG + ".api.orchestration.serverless_media"]
        media_module.prepare_images.return_value = [{
            "name": "private-image-name.png",
            "image": "x" * 10_000_001,
        }]

        with self.assertRaisesRegex(ValueError, "10,000,") as ctx:
            await self.q.orchestrate_distributed_execution(self.prompt, None, "client")

        self.assertNotIn("private-image-name.png", str(ctx.exception))
        self.assertNotIn("x" * 100, str(ctx.exception))
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.q.queue_prompt_payload.assert_not_awaited()
        self.manager.start.assert_not_awaited()
        self.assertEqual(self.ps.distributed_pending_jobs, {})
