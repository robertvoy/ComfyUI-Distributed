"""Real PNG/tensor regressions; only server/config/HTTP plumbing is stubbed."""
import asyncio
import importlib.util
import io
import base64
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from aiohttp import web
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def media_modules():
    # Existing harnesses provide ComfyUI/config doubles. Restore their module
    # replacements so this file can coexist with the rest of the test suite.
    with patch.dict(sys.modules):
        image_utils = _load_file("real_image_utils", ROOT / "utils/image.py")
        collector_tests = _load_file("collector_harness", ROOT / "tests/test_collector_list_inputs.py")
        collector = collector_tests._load_collector_module()
        sys.modules["aiohttp"].web = web
        route_tests = _load_file("route_harness", ROOT / "tests/api/test_distributed_queue.py")
        routes = route_tests._load_job_routes_module()
        for module in (collector, routes):
            module.pil_to_tensor = image_utils.pil_to_tensor
            module.ensure_contiguous = image_utils.ensure_contiguous
        collector.tensor_to_pil = image_utils.tensor_to_pil
        routes.prompt_server = collector.prompt_server
        yield collector, routes


def _png_payload(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "palette", "palette_alpha"])
def test_png_decoder_preserves_alpha_and_normalizes_opaque_modes(media_modules, mode):
    _, routes = media_modules
    if mode.startswith("palette"):
        image = Image.new("P", (12, 8), 0)
        image.putpalette([16, 32, 64] + [0] * 765)
        if mode == "palette_alpha":
            image.info["transparency"] = 0
    else:
        color = {"RGB": (16, 32, 64), "RGBA": (16, 32, 64, 96), "L": 32, "LA": (32, 96)}[mode]
        image = Image.new(mode, (12, 8), color)
    alpha = mode in {"RGBA", "LA", "palette_alpha"}
    expected = np.array(image.convert("RGBA" if alpha else "RGB"), dtype=np.float32) / 255
    decoded = routes._decode_canonical_png_tensor(_png_payload(image))
    assert decoded.shape == (1, 8, 12, 4 if alpha else 3)
    assert decoded.is_contiguous()
    np.testing.assert_array_equal(decoded[0].numpy(), expected)


async def _master_collect(module, local, worker_images, delegate=False):
    module.prompt_server.distributed_jobs_lock = asyncio.Lock()
    queue = asyncio.Queue()
    for index, tensor in enumerate(worker_images):
        queue.put_nowait({"worker_id": "worker-a", "image_index": index,
                          "tensor": tensor, "is_last": index == len(worker_images) - 1})
    module.prompt_server.distributed_pending_jobs = {"image-test": queue}
    return await module.DistributedCollectorNode().execute(
        local, None, multi_job_id="image-test", enabled_worker_ids='["worker-a"]',
        delegate_only=delegate,
    )


@pytest.mark.parametrize("local_channels,worker_channels", [(3, 3), (4, 4), (3, 4), (4, 3), (None, 4)])
def test_worker_png_callback_and_collector_preserve_pixels_and_alpha(
    media_modules, local_channels, worker_channels
):
    module, routes = media_modules
    local = None if local_channels is None else torch.full((1, 8, 12, local_channels), 0.5)
    worker = torch.full((2, 8, 12, worker_channels), 0.25)
    if worker_channels == 4:
        worker[..., 3] = torch.linspace(0, 1, 12).reshape(1, 1, 12)
    responses = []

    async def scenario():
        module.prompt_server.distributed_jobs_lock = asyncio.Lock()
        module.prompt_server.distributed_pending_jobs = {"image-test": asyncio.Queue()}

        class Response:
            def __init__(self, payload):
                self.payload = payload

            async def __aenter__(self):
                async def request_json():
                    return self.payload
                response = await routes.job_complete_endpoint(types.SimpleNamespace(json=request_json))
                responses.append(response.status)
                assert response.status == 200, response.text
                return self

            async def __aexit__(self, *_args):
                return False

            def raise_for_status(self):
                pass

        class Session:
            def post(self, _url, json, timeout):
                return Response(json)

        async def get_session():
            return Session()

        module.get_client_session = get_session
        module.encode_audio_payload = lambda _audio: None
        collector = module.DistributedCollectorNode()
        await collector.send_batch_to_master(worker, None, "image-test", "http://master", "worker-a")
        result, _ = await collector.execute(
            local, None, multi_job_id="image-test", enabled_worker_ids='["worker-a"]',
            delegate_only=local is None,
        )
        assert "image-test" not in module.prompt_server.distributed_pending_jobs
        return result

    result = asyncio.run(scenario())
    channels = max(local_channels or 3, worker_channels)
    offset = 0 if local is None else 1
    assert responses == [200, 200]
    assert result.shape == (offset + 2, 8, 12, channels)
    assert result.is_contiguous()
    if local is not None:
        assert torch.equal(result[:1, ..., :local_channels], local)
        if local_channels == 3 and channels == 4:
            assert torch.all(result[:1, ..., 3] == 1)
    assert torch.allclose(result[offset:, ..., :worker_channels], worker, atol=1 / 255)
    if worker_channels == 3 and channels == 4:
        assert torch.all(result[offset:, ..., 3] == 1)


def test_collector_mixed_rgb_rgba_list_promotes_rgb_to_opaque_rgba(media_modules):
    module, _ = media_modules
    rgb = torch.zeros(1, 8, 12, 3)
    rgba = torch.full((1, 8, 12, 4), 0.5)
    result, _ = module.DistributedCollectorNode().run(images=[rgb, rgba])
    assert result.shape == (2, 8, 12, 4)
    assert torch.equal(result[:1, ..., :3], rgb)
    assert torch.all(result[:1, ..., 3] == 1)
    assert torch.equal(result[1:], rgba)


@pytest.mark.parametrize("other_channels", [1, 3, 4])
def test_collector_preserves_grayscale_or_promotes_it_for_color_batches(media_modules, other_channels):
    module, _ = media_modules
    gray = torch.full((1, 8, 12, 1), 0.25)
    other = torch.full((1, 8, 12, other_channels), 0.5)
    result, _ = module.DistributedCollectorNode().run(images=[gray, other])
    assert result.shape == (2, 8, 12, other_channels)
    assert torch.all(result[:1, ..., :min(other_channels, 3)] == 0.25)
    if other_channels == 4:
        assert torch.all(result[:1, ..., 3] == 1)
    assert torch.equal(result[1:], other)


def test_mixed_channel_merge_keeps_configured_worker_and_image_order(media_modules):
    module, _ = media_modules
    collector = module.DistributedCollectorNode()
    master = torch.zeros(1, 8, 12, 3)
    worker_a = torch.full((1, 8, 12, 4), 0.25)
    worker_b = torch.full((1, 8, 12, 3), 0.5)
    unexpected = torch.full((1, 8, 12, 3), 0.75)
    result = collector._reorder_and_combine_tensors(
        {"worker-a": {0: worker_a}, "worker-b": {2: worker_b, 0: master},
         "unexpected": {0: unexpected}},
        ["worker-b", "worker-a"], 1, master, False, master,
    )
    assert result.shape == (5, 8, 12, 4)
    assert result[:, 0, 0, 0].tolist() == [0, 0, 0.5, 0.25, 0.75]
    assert result[:, 0, 0, 3].tolist() == [1, 1, 1, 0.25, 1]


def test_collector_spatial_mismatch_raises_with_shapes_not_successful_fallback(media_modules):
    module, _ = media_modules
    local = torch.zeros(1, 8, 12, 3)
    worker = torch.zeros(1, 8, 13, 3)
    with pytest.raises(ValueError, match="spatial dimensions") as error:
        asyncio.run(_master_collect(module, local, [worker]))
    message = str(error.value)
    assert "master image 0" in message
    assert "worker 'worker-a' image 0" in message
    assert "(1, 8, 12, 3)" in message and "(1, 8, 13, 3)" in message
    assert "image-test" not in module.prompt_server.distributed_pending_jobs


@pytest.mark.parametrize("shape", [(1, 8, 12), (1, 8, 12, 5)])
def test_collector_rejects_invalid_worker_image_shape(media_modules, shape):
    module, _ = media_modules
    with pytest.raises(ValueError, match="worker 'worker-a' image 0"):
        asyncio.run(_master_collect(module, None, [torch.zeros(shape)], delegate=True))


def test_delegate_master_ignores_empty_placeholder_shape(media_modules):
    module, _ = media_modules
    placeholder = torch.empty(0, 64, 64, 3)
    worker = torch.full((1, 8, 12, 4), 0.5)
    result, _ = asyncio.run(_master_collect(module, placeholder, [worker], delegate=True))
    assert torch.equal(result, worker)


def test_empty_image_results_raise_instead_of_reaching_save_image(media_modules):
    module, _ = media_modules
    with pytest.raises(ValueError, match="empty image batch"):
        asyncio.run(_master_collect(module, torch.empty(0, 64, 64, 3),
                                    [torch.empty(0, 8, 12, 3)], delegate=True))
