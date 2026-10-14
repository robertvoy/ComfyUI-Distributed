"""Real Collector aggregation of typed bridge deliveries and direct results.

Only Comfy's host services are stubbed by the shared Collector loader. Tensor
normalization, participant ordering, queue consumption and cleanup are real.
"""
import asyncio
import json
import types
from unittest.mock import AsyncMock

import pytest
import torch

from test_collector_list_inputs import _load_collector_module


@pytest.fixture
def collector_module():
    module = _load_collector_module()
    module.prompt_server.distributed_jobs_lock = asyncio.Lock()
    module.prompt_server.distributed_serverless = types.SimpleNamespace(cancel=AsyncMock())
    return module


def native_media(value=0):
    image = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4) / 997 + value
    image[0, 0, 0, 0] = -0.0
    audio = {"waveform": torch.arange(6, dtype=torch.float64).reshape(1, 2, 3) / 997 + value,
             "sample_rate": 24000}
    return {"images": image, "audio": audio}


def assert_exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def collect_queue(module, messages, participants, *, local=None, delegate=True):
    queue = asyncio.Queue()
    for message in messages:
        queue.put_nowait(message)
    module.prompt_server.distributed_pending_jobs["typed-job"] = queue
    values = local or {"images": None, "audio": None}
    return asyncio.run(asyncio.wait_for(module.DistributedCollectorNode().execute(
        **values, multi_job_id="typed-job", enabled_worker_ids=json.dumps(participants),
        delegate_only=delegate), timeout=2))


@pytest.mark.parametrize("delegate", [False, True])
@pytest.mark.parametrize("socket_set", [("images",), ("audio",), ("images", "audio")])
def test_typed_and_direct_results_preserve_batch_and_audio_order(collector_module, delegate, socket_set):
    module = collector_module
    local, typed_a, typed_b, direct = [native_media(n) for n in range(4)]
    typed = {socket: [typed_a[socket], typed_b[socket]] for socket in socket_set}
    direct_message = {"worker_id": "direct", "is_last": True}
    if "images" in socket_set:
        direct_message.update(tensor=direct["images"], image_index=0)
    if "audio" in socket_set:
        direct_message["audio"] = direct["audio"]
    messages = [direct_message, {"worker_id": "serverless:pool:1", "media": typed, "is_last": True}]
    # Delivery order differs from participant order. Duplicate configured IDs
    # must not duplicate a participant's data or leave collection waiting.
    participants = ["serverless:pool:1", "direct", "serverless:pool:1"]
    local_inputs = {socket: local[socket] if socket in socket_set else None for socket in ("images", "audio")}
    images, audio = collect_queue(module, messages, participants, local=local_inputs, delegate=delegate)
    expected = ([] if delegate else [local]) + [typed_a, typed_b, direct]
    if "images" in socket_set:
        assert_exact(images, torch.cat([row["images"] for row in expected]))
    else:
        assert images is None
    if "audio" in socket_set:
        assert audio["sample_rate"] == 24000
        assert_exact(audio["waveform"], torch.cat([row["audio"]["waveform"] for row in expected], dim=-1))
    else:
        assert torch.equal(audio["waveform"], module.DistributedCollectorNode.EMPTY_AUDIO["waveform"])
    assert module.prompt_server.distributed_pending_jobs == {}
    module.prompt_server.distributed_serverless.cancel.assert_not_awaited()


@pytest.mark.parametrize("media", [{}, {"texts": ["not a Collector socket"]}, {"images": []},
                                      {"audio": [None]}, {"images": "not a list"},
                                      {"images": [torch.zeros(2, 2, 3)]}])
def test_invalid_typed_delivery_cancels_and_removes_queue(collector_module, media):
    module = collector_module
    with pytest.raises(RuntimeError, match="media|items|IMAGE"):
        collect_queue(module, [{"worker_id": "serverless:pool:1", "media": media, "is_last": True}],
                      ["serverless:pool:1"])
    assert module.prompt_server.distributed_pending_jobs == {}
    module.prompt_server.distributed_serverless.cancel.assert_awaited_once_with(collector_job_id="typed-job")


@pytest.mark.parametrize("failure", ["nonfinal", "unexpected", "duplicate", "provider-error", "shape"])
def test_typed_delivery_protocol_failures_are_not_partial_success(collector_module, failure):
    media = {"images": [native_media()["images"]]}
    first = {"worker_id": "serverless:pool:1", "media": media, "is_last": True}
    messages = [first]
    participants = ["serverless:pool:1", "serverless:pool:2"]
    if failure == "nonfinal":
        first["is_last"] = False
    elif failure == "unexpected":
        first["worker_id"] = "other"
    elif failure == "duplicate":
        messages.append(dict(first))
    elif failure == "provider-error":
        messages.append({"worker_id": "serverless:pool:2", "serverless_error": "offline failure", "is_last": True})
    else:
        messages.append({"worker_id": "serverless:pool:2", "media": {"images": [torch.zeros(1, 4, 5, 3)]},
                         "is_last": True})
    with pytest.raises(RuntimeError, match="atomic|Unexpected|Duplicate|failed|shape mismatch"):
        collect_queue(collector_module, messages, participants)
    assert collector_module.prompt_server.distributed_pending_jobs == {}
    collector_module.prompt_server.distributed_serverless.cancel.assert_awaited_once_with(collector_job_id="typed-job")
