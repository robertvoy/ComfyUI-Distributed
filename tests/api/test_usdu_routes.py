import asyncio
import importlib.util
import io
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeRequest:
    def __init__(self, json_payload=None, post_payload=None, headers=None, query=None):
        self._json_payload = json_payload
        self._post_payload = post_payload or {}
        self.headers = headers or {}
        self.query = query or {}

    async def json(self):
        return self._json_payload

    async def post(self):
        return self._post_payload


class _Routes:
    def post(self, _path):
        def _decorator(fn):
            return fn

        return _decorator

    def get(self, _path):
        def _decorator(fn):
            return fn

        return _decorator


def _load_usdu_routes_module():
    module_path = Path(__file__).resolve().parents[2] / "api" / "usdu_routes.py"
    package_name = "dist_api_usdu_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    api_pkg = types.ModuleType(f"{package_name}.api")
    api_pkg.__path__ = []
    sys.modules[f"{package_name}.api"] = api_pkg

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    prompt_server_holder = {
        "value": types.SimpleNamespace(
            distributed_tile_jobs_lock=asyncio.Lock(),
            distributed_pending_tile_jobs={},
        )
    }

    created_aiohttp_stub = False
    if "aiohttp" not in sys.modules:
        created_aiohttp_stub = True
        aiohttp_module = types.ModuleType("aiohttp")
        aiohttp_module.web = types.SimpleNamespace(
            json_response=lambda payload, status=200: _FakeResponse(payload, status=status)
        )
        sys.modules["aiohttp"] = aiohttp_module

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    sys.modules["server"] = server_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _handle_api_error(_request, error, status=500):
        return _FakeResponse({"status": "error", "message": str(error)}, status=status)

    network_module.handle_api_error = _handle_api_error
    sys.modules[f"{package_name}.utils.network"] = network_module

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.MAX_PAYLOAD_SIZE = 1024
    job_store_module.ensure_tile_jobs_initialized = lambda: prompt_server_holder["value"]

    async def _init_dynamic_job(multi_job_id, batch_size, enabled_workers, all_indices=None):
        prompt_server = prompt_server_holder["value"]
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                return
            job_data = ImageJobState(multi_job_id=multi_job_id)
            job_data.worker_status = {str(worker_id): 0 for worker_id in (enabled_workers or [])}
            job_data.assigned_to_workers = {str(worker_id): [] for worker_id in (enabled_workers or [])}
            indices = all_indices if all_indices is not None else list(range(int(batch_size or 0)))
            for idx in indices:
                await job_data.pending_images.put(int(idx))
            prompt_server.distributed_pending_tile_jobs[multi_job_id] = job_data

    job_store_module.init_dynamic_job = _init_dynamic_job
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    job_models_module = types.ModuleType(f"{package_name}.upscale.job_models")

    class BaseJobState:
        pass

    @dataclass
    class TileJobState(BaseJobState):
        multi_job_id: str
        mode: str = field(default="static", init=False)
        queue: asyncio.Queue = field(default_factory=asyncio.Queue)
        pending_tasks: asyncio.Queue = field(default_factory=asyncio.Queue)
        completed_tasks: dict = field(default_factory=dict)
        worker_status: dict = field(default_factory=dict)
        assigned_to_workers: dict = field(default_factory=dict)
        batch_size: int = 0
        num_tiles_per_image: int = 0
        batched_static: bool = False

    @dataclass
    class ImageJobState(BaseJobState):
        multi_job_id: str
        mode: str = field(default="dynamic", init=False)
        queue: asyncio.Queue = field(default_factory=asyncio.Queue)
        pending_images: asyncio.Queue = field(default_factory=asyncio.Queue)
        completed_images: dict = field(default_factory=dict)
        worker_status: dict = field(default_factory=dict)
        assigned_to_workers: dict = field(default_factory=dict)
        batch_size: int = 0
        num_tiles_per_image: int = 0
        batched_static: bool = False

        @property
        def pending_tasks(self):
            return self.pending_images

        @property
        def completed_tasks(self):
            return self.completed_images

    job_models_module.BaseJobState = BaseJobState
    job_models_module.TileJobState = TileJobState
    job_models_module.ImageJobState = ImageJobState
    sys.modules[f"{package_name}.upscale.job_models"] = job_models_module

    parsers_module = types.ModuleType(f"{package_name}.upscale.payload_parsers")
    parsers_module.parse_tiles_from_form = lambda _data: []
    parsers_module._parse_tiles_from_form = lambda _data: []
    sys.modules[f"{package_name}.upscale.payload_parsers"] = parsers_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.api.usdu_routes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    if created_aiohttp_stub:
        sys.modules.pop("aiohttp", None)

    module.web = types.SimpleNamespace(
        json_response=lambda payload, status=200: _FakeResponse(payload, status=status)
    )

    module._prompt_server_holder = prompt_server_holder
    module._TileJobState = TileJobState
    module._ImageJobState = ImageJobState
    return module


usdu_routes = _load_usdu_routes_module()


class _UploadField:
    def __init__(self, data):
        self.file = io.BytesIO(data)


def _tiny_png_bytes():
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class USDURoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        usdu_routes._prompt_server_holder["value"] = types.SimpleNamespace(
            distributed_tile_jobs_lock=asyncio.Lock(),
            distributed_pending_tile_jobs={},
        )

    async def test_heartbeat_updates_worker_status(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        job_data = usdu_routes._TileJobState("job-1")
        prompt_server.distributed_pending_tile_jobs["job-1"] = job_data

        request = _FakeRequest(json_payload={"worker_id": "worker-a", "multi_job_id": "job-1"})
        response = await usdu_routes.heartbeat_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "success")
        self.assertIn("worker-a", job_data.worker_status)

    async def test_heartbeat_missing_fields_returns_400(self):
        request = _FakeRequest(json_payload={"worker_id": "worker-a"})
        response = await usdu_routes.heartbeat_endpoint(request)

        self.assertEqual(response.status, 400)
        self.assertIn("missing", response.payload.get("message", "").lower())

    async def test_request_image_dynamic_assigns_next_image(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        job_data = usdu_routes._ImageJobState("job-2")
        await job_data.pending_images.put(7)
        prompt_server.distributed_pending_tile_jobs["job-2"] = job_data

        request = _FakeRequest(json_payload={"worker_id": "worker-a", "multi_job_id": "job-2"})
        response = await usdu_routes.request_image_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("image_idx"), 7)
        self.assertEqual(response.payload.get("estimated_remaining"), 0)
        self.assertEqual(job_data.assigned_to_workers["worker-a"], [7])
        self.assertIn("worker-a", job_data.worker_status)

    async def test_request_image_static_assigns_tile_and_batched_flag(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        job_data = usdu_routes._TileJobState("job-3")
        job_data.batched_static = True
        await job_data.pending_tasks.put(4)
        prompt_server.distributed_pending_tile_jobs["job-3"] = job_data

        request = _FakeRequest(json_payload={"worker_id": "worker-a", "multi_job_id": "job-3"})
        response = await usdu_routes.request_image_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("tile_idx"), 4)
        self.assertTrue(response.payload.get("batched_static"))
        self.assertEqual(job_data.assigned_to_workers["worker-a"], [4])

    async def test_init_list_queue_initializes_pending_items(self):
        request = _FakeRequest(
            json_payload={
                "multi_job_id": "list-job-1",
                "list_size": 3,
                "enabled_workers": ["worker-a", "worker-b"],
            }
        )
        response = await usdu_routes.init_list_queue_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "success")
        self.assertEqual(response.payload.get("remaining"), 3)

    async def test_request_list_item_assigns_next_index(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        job_data = usdu_routes._ImageJobState("list-job-2")
        await job_data.pending_images.put(11)
        prompt_server.distributed_pending_tile_jobs["list-job-2"] = job_data

        request = _FakeRequest(json_payload={"worker_id": "worker-a", "multi_job_id": "list-job-2"})
        response = await usdu_routes.request_list_item_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("item_idx"), 11)
        self.assertEqual(response.payload.get("estimated_remaining"), 0)
        self.assertEqual(job_data.assigned_to_workers["worker-a"], [11])

    async def test_submit_tiles_completion_signal_enqueues_last_marker(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        job_data = usdu_routes._TileJobState("job-4")
        prompt_server.distributed_pending_tile_jobs["job-4"] = job_data

        request = _FakeRequest(
            post_payload={
                "multi_job_id": "job-4",
                "worker_id": "worker-a",
                "batch_size": "0",
                "is_last": "true",
            },
            headers={"content-length": "128"},
        )

        response = await usdu_routes.submit_tiles_endpoint(request)

        self.assertEqual(response.status, 200)
        queued = await job_data.queue.get()
        self.assertEqual(queued["worker_id"], "worker-a")
        self.assertTrue(queued["is_last"])
        self.assertEqual(queued["tiles"], [])

    async def test_submit_image_enqueues_processed_image_payload(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        job_data = usdu_routes._ImageJobState("job-5")
        prompt_server.distributed_pending_tile_jobs["job-5"] = job_data

        request = _FakeRequest(
            post_payload={
                "multi_job_id": "job-5",
                "worker_id": "worker-a",
                "image_idx": "2",
                "full_image": _UploadField(_tiny_png_bytes()),
                "is_last": "false",
            },
            headers={"content-length": "256"},
        )

        response = await usdu_routes.submit_image_endpoint(request)

        self.assertEqual(response.status, 200)
        queued = await job_data.queue.get()
        self.assertEqual(queued["worker_id"], "worker-a")
        self.assertEqual(queued["image_idx"], 2)
        self.assertIn("image", queued)

    async def test_job_status_endpoint_reports_ready(self):
        prompt_server = usdu_routes._prompt_server_holder["value"]
        prompt_server.distributed_pending_tile_jobs["job-6"] = usdu_routes._TileJobState("job-6")

        request = _FakeRequest(query={"multi_job_id": "job-6"})
        response = await usdu_routes.job_status_endpoint(request)

        self.assertEqual(response.status, 200)
        self.assertTrue(response.payload.get("ready"))


if __name__ == "__main__":
    unittest.main()
