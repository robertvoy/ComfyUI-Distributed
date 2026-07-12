import importlib.util
import sys
import types
import asyncio
from pathlib import Path

import torch


def _load_collector_module():
    module_path = Path(__file__).resolve().parents[1] / "nodes" / "collector.py"
    package_name = "dist_collector_list_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    nodes_pkg = types.ModuleType(f"{package_name}.nodes")
    nodes_pkg.__path__ = []
    sys.modules[f"{package_name}.nodes"] = nodes_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    class _Routes:
        def post(self, _path):
            return lambda fn: fn

        def get(self, _path):
            return lambda fn: fn

    prompt_server = types.SimpleNamespace(
        routes=_Routes(),
        distributed_jobs_lock=None,
        distributed_pending_jobs={},
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=prompt_server)
    sys.modules["server"] = server_module

    comfy_module = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")

    class InterruptProcessingException(Exception):
        pass

    model_management.InterruptProcessingException = InterruptProcessingException
    model_management.throw_exception_if_processing_interrupted = lambda: None
    comfy_module.model_management = model_management

    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, _total):
            self.total = _total
            self.updates = []

        def update(self, value):
            self.updates.append(value)

    comfy_utils.ProgressBar = ProgressBar
    comfy_module.utils = comfy_utils
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = comfy_utils

    aiohttp_module = types.ModuleType("aiohttp")
    aiohttp_module.ClientTimeout = lambda total: types.SimpleNamespace(total=total)
    sys.modules["aiohttp"] = aiohttp_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.get_worker_timeout_seconds = lambda: 0.1
    config_module.load_config = lambda: {"workers": []}
    config_module.is_master_delegate_only = lambda: False
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.HEARTBEAT_INTERVAL = 1.0
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    image_module = types.ModuleType(f"{package_name}.utils.image")
    def _ensure_contiguous(tensor):
        return tensor.contiguous() if hasattr(tensor, "contiguous") else tensor

    image_module.ensure_contiguous = _ensure_contiguous
    image_module.tensor_to_pil = lambda *_args, **_kwargs: None
    image_module.pil_to_tensor = lambda value: value
    sys.modules[f"{package_name}.utils.image"] = image_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_worker_url = lambda worker: "http://worker"
    network_module.get_client_session = lambda: None
    network_module.probe_worker = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.network"] = network_module

    audio_payload_module = types.ModuleType(f"{package_name}.utils.audio_payload")
    audio_payload_module.encode_audio_payload = lambda audio: audio
    sys.modules[f"{package_name}.utils.audio_payload"] = audio_payload_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.run_async_in_server_loop = lambda coro: asyncio.run(coro)
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.nodes.collector", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collector_opts_into_comfyui_list_inputs():
    collector = _load_collector_module().DistributedCollectorNode

    assert collector.INPUT_IS_LIST is True


def test_collector_exposes_images_as_optional_input():
    input_types = _load_collector_module().DistributedCollectorNode.INPUT_TYPES()

    assert "images" not in input_types["required"]
    assert input_types["optional"]["images"] == ("IMAGE",)


def test_audio_only_pass_through_returns_no_images_and_preserves_audio():
    collector = _load_collector_module().DistributedCollectorNode()
    audio = {"waveform": torch.ones(1, 2, 4), "sample_rate": 48000}

    images, returned_audio = collector.run(images=None, audio=[audio])

    assert images is None
    assert returned_audio is audio


def test_collector_rejects_missing_images_and_audio():
    collector = _load_collector_module().DistributedCollectorNode()

    try:
        collector.run(images=None, audio=None)
    except ValueError as exc:
        assert "image or audio" in str(exc).lower()
    else:
        raise AssertionError("Expected collector to reject a run with no media input")


def test_delegate_only_master_allows_no_local_media_input():
    collector = _load_collector_module().DistributedCollectorNode()

    images, audio = collector.run(
        images=None,
        audio=None,
        multi_job_id=["delegate-audio-job"],
        delegate_only=[True],
        enabled_worker_ids=["[]"],
    )

    assert images is None
    assert tuple(audio["waveform"].shape) == (1, 2, 1)


def test_pass_through_collapses_comfyui_image_list_to_batch_and_unwraps_hidden_inputs():
    collector = _load_collector_module().DistributedCollectorNode()
    first = torch.zeros(1, 2, 2, 3)
    second = torch.ones(1, 2, 2, 3)

    images, audio = collector.run(
        images=[first, second],
        load_balance=[False],
        audio=[None],
        multi_job_id=[""],
        is_worker=[False],
        master_url=[""],
        enabled_worker_ids=["[]"],
        worker_batch_size=[1],
        worker_id=[""],
        pass_through=[False],
        delegate_only=[False],
    )

    assert tuple(images.shape) == (2, 2, 2, 3)
    assert torch.equal(images[0:1], first)
    assert torch.equal(images[1:2], second)
    assert tuple(audio["waveform"].shape) == (1, 2, 1)


def test_worker_list_input_sends_one_completion_sequence_with_last_only_on_final_item():
    module = _load_collector_module()
    collector = module.DistributedCollectorNode()
    first = torch.zeros(1, 2, 2, 3)
    second = torch.ones(1, 2, 2, 3)
    posted_payloads = []

    class _FakeImage:
        def save(self, fp, format=None, compress_level=None):
            fp.write(b"png-bytes")

    class _FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

    class _FakeSession:
        def post(self, url, json, timeout):
            posted_payloads.append(json)
            return _FakeResponse()

    async def _fake_get_client_session():
        return _FakeSession()

    module.tensor_to_pil = lambda *_args, **_kwargs: _FakeImage()
    module.get_client_session = _fake_get_client_session
    module.encode_audio_payload = lambda _audio: None

    images, audio = collector.run(
        images=[first, second],
        load_balance=[False],
        audio=[None],
        multi_job_id=["job-list-1"],
        is_worker=[True],
        master_url=["http://master"],
        enabled_worker_ids=["[]"],
        worker_batch_size=[1],
        worker_id=["worker-a"],
        pass_through=[False],
        delegate_only=[False],
    )

    assert tuple(images.shape) == (2, 2, 2, 3)
    assert tuple(audio["waveform"].shape) == (1, 2, 1)
    assert len(posted_payloads) == 2
    assert [payload["batch_idx"] for payload in posted_payloads] == [0, 1]
    assert [payload["is_last"] for payload in posted_payloads] == [False, True]
    assert {payload["job_id"] for payload in posted_payloads} == {"job-list-1"}
    assert {payload["worker_id"] for payload in posted_payloads} == {"worker-a"}


def test_audio_only_worker_sends_one_completion_without_image():
    module = _load_collector_module()
    collector = module.DistributedCollectorNode()
    audio = {"waveform": torch.ones(1, 2, 4), "sample_rate": 48000}
    posted = []

    class _FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

    class _FakeSession:
        def post(self, url, json, timeout):
            posted.append((url, json, timeout.total))
            return _FakeResponse()

    async def _fake_get_client_session():
        return _FakeSession()

    module.get_client_session = _fake_get_client_session
    module.encode_audio_payload = lambda value: {"encoded": value is audio}

    images, returned_audio = collector.run(
        images=None,
        audio=[audio],
        multi_job_id=["audio-job"],
        is_worker=[True],
        master_url=["http://master"],
        worker_id=["worker-a"],
    )

    assert images is None
    assert returned_audio is audio
    assert len(posted) == 1
    assert posted[0][0] == "http://master/distributed/job_complete"
    assert posted[0][2] == 600
    assert posted[0][1] == {
        "job_id": "audio-job",
        "worker_id": "worker-a",
        "batch_idx": 0,
        "audio": {"encoded": True},
        "is_last": True,
    }


def test_audio_only_master_combines_local_and_worker_audio():
    module = _load_collector_module()
    collector = module.DistributedCollectorNode()
    master_audio = {"waveform": torch.ones(1, 2, 2), "sample_rate": 48000}
    worker_audio = {"waveform": torch.full((1, 2, 3), 2.0), "sample_rate": 48000}
    module.prompt_server.distributed_jobs_lock = asyncio.Lock()
    queue = asyncio.Queue()
    queue.put_nowait(
        {
            "worker_id": "worker-a",
            "image_index": 0,
            "tensor": None,
            "audio": worker_audio,
            "is_last": True,
        }
    )
    module.prompt_server.distributed_pending_jobs = {"audio-job": queue}

    images, combined_audio = asyncio.run(
        collector.execute(
            images=None,
            audio=master_audio,
            multi_job_id="audio-job",
            enabled_worker_ids='["worker-a"]',
        )
    )

    assert images is None
    assert combined_audio["sample_rate"] == 48000
    assert tuple(combined_audio["waveform"].shape) == (1, 2, 5)
    assert torch.equal(combined_audio["waveform"][..., :2], master_audio["waveform"])
    assert torch.equal(combined_audio["waveform"][..., 2:], worker_audio["waveform"])


def test_delegate_only_audio_collects_worker_audio_without_placeholder_image():
    module = _load_collector_module()
    collector = module.DistributedCollectorNode()
    worker_audio = {"waveform": torch.full((1, 2, 3), 2.0), "sample_rate": 48000}
    module.prompt_server.distributed_jobs_lock = asyncio.Lock()
    queue = asyncio.Queue()
    queue.put_nowait(
        {
            "worker_id": "worker-a",
            "image_index": 0,
            "tensor": None,
            "audio": worker_audio,
            "is_last": True,
        }
    )
    module.prompt_server.distributed_pending_jobs = {"delegate-audio-job": queue}

    images, combined_audio = asyncio.run(
        collector.execute(
            images=None,
            audio=None,
            multi_job_id="delegate-audio-job",
            enabled_worker_ids='["worker-a"]',
            delegate_only=True,
        )
    )

    assert images is None
    assert combined_audio["sample_rate"] == 48000
    assert torch.equal(combined_audio["waveform"], worker_audio["waveform"])
