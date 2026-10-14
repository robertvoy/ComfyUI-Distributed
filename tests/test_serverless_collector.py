import asyncio

import pytest
import torch

from test_collector_list_inputs import _load_collector_module


class _FakeManager:
    def __init__(self, *, pending=False, owned=True):
        self.pending = pending
        self.owned = owned
        self.pending_calls = []
        self.owned_calls = []
        self.cancel_calls = []

    def is_pending(self, collector_job_id, participant_id):
        self.pending_calls.append((collector_job_id, participant_id))
        return self.pending

    def owns(self, collector_job_id, participant_id):
        self.owned_calls.append((collector_job_id, participant_id))
        return self.owned

    async def cancel(self, worker_id=None, execution_id=None, collector_job_id=None):
        self.cancel_calls.append({
            "worker_id": worker_id,
            "execution_id": execution_id,
            "collector_job_id": collector_job_id,
        })
        return 1


def _set_up_collector(module, job_id, queue, manager=None):
    module.prompt_server.distributed_jobs_lock = asyncio.Lock()
    module.prompt_server.distributed_pending_jobs = {job_id: queue}
    if manager is not None:
        module.prompt_server.distributed_serverless = manager
    return module.DistributedCollectorNode()


def test_serverless_pending_waits_past_normal_worker_timeout():
    module = _load_collector_module()
    job_id = "collector-pending"
    participant = "serverless:pool:1"
    queue = asyncio.Queue()
    manager = _FakeManager(pending=True)
    collector = _set_up_collector(module, job_id, queue, manager)
    module.get_worker_timeout_seconds = lambda: 0.01
    module.HEARTBEAT_INTERVAL = 0.1
    probe_calls = []
    module.probe_worker = lambda *args, **kwargs: probe_calls.append(args)

    async def deliver_after_provider_work():
        await asyncio.sleep(0.14)
        queue.put_nowait({
            "worker_id": participant,
            "image_index": 0,
            "tensor": torch.ones(1, 2, 2, 3),
            "is_last": True,
        })

    async def run_collection():
        producer = asyncio.create_task(deliver_after_provider_work())
        result = await collector.execute(
            images=None,
            audio=None,
            multi_job_id=job_id,
            enabled_worker_ids=f'["{participant}"]',
            delegate_only=True,
        )
        await producer
        return result

    images, _audio = asyncio.run(run_collection())

    assert tuple(images.shape) == (1, 2, 2, 3)
    assert manager.pending_calls
    assert probe_calls == []
    assert job_id not in module.prompt_server.distributed_pending_jobs


def test_serverless_queue_error_raises_and_cancels_only_collector_binding():
    module = _load_collector_module()
    job_id = "collector-error"
    queue = asyncio.Queue()
    queue.put_nowait({
        "worker_id": "serverless:pool:1",
        "serverless_error": "Provider execution failed safely",
        "is_last": True,
    })
    manager = _FakeManager()
    collector = _set_up_collector(module, job_id, queue, manager)

    with pytest.raises(RuntimeError, match="serverless.*Provider execution failed safely"):
        asyncio.run(collector.execute(
            images=None,
            audio=None,
            multi_job_id=job_id,
            enabled_worker_ids='["serverless:pool:1"]',
            delegate_only=True,
        ))

    assert manager.cancel_calls == [{
        "worker_id": None,
        "execution_id": None,
        "collector_job_id": job_id,
    }]
    assert job_id not in module.prompt_server.distributed_pending_jobs


def test_owned_serverless_slot_without_pending_result_fails_instead_of_partial_success():
    module = _load_collector_module()
    job_id = "collector-stopped"
    queue = asyncio.Queue()
    manager = _FakeManager(pending=False, owned=True)
    collector = _set_up_collector(module, job_id, queue, manager)
    module.get_worker_timeout_seconds = lambda: 0.01
    module.HEARTBEAT_INTERVAL = 0.1
    module.probe_worker = lambda *_args, **_kwargs: pytest.fail("serverless slots are not worker URLs")

    with pytest.raises(RuntimeError, match="serverless.*stopped.*result|serverless.*result.*stopped"):
        asyncio.run(collector.execute(
            images=None,
            audio=None,
            multi_job_id=job_id,
            enabled_worker_ids='["serverless:pool:1"]',
            delegate_only=True,
        ))

    assert manager.owned_calls
    assert manager.cancel_calls[0]["collector_job_id"] == job_id
    assert job_id not in module.prompt_server.distributed_pending_jobs


def test_interruption_cancels_serverless_job_and_removes_collector_queue():
    module = _load_collector_module()
    job_id = "collector-interrupted"
    queue = asyncio.Queue()
    manager = _FakeManager(pending=True)
    collector = _set_up_collector(module, job_id, queue, manager)
    interrupted = module.comfy.model_management.InterruptProcessingException("stop")
    module.comfy.model_management.throw_exception_if_processing_interrupted = lambda: (_ for _ in ()).throw(interrupted)

    with pytest.raises(module.comfy.model_management.InterruptProcessingException, match="stop"):
        asyncio.run(collector.execute(
            images=None,
            audio=None,
            multi_job_id=job_id,
            enabled_worker_ids='["serverless:pool:1"]',
            delegate_only=True,
        ))

    assert manager.cancel_calls[0]["collector_job_id"] == job_id
    assert job_id not in module.prompt_server.distributed_pending_jobs


def test_serverless_tensor_shape_mismatch_raises_instead_of_silent_fallback():
    module = _load_collector_module()
    job_id = "collector-shape-mismatch"
    queue = asyncio.Queue()
    queue.put_nowait({
        "worker_id": "serverless:pool:1",
        "image_index": 0,
        "tensor": torch.ones(1, 3, 2, 3),
        "is_last": True,
    })
    manager = _FakeManager()
    collector = _set_up_collector(module, job_id, queue, manager)
    local_images = torch.zeros(1, 2, 2, 3)

    with pytest.raises(RuntimeError, match="serverless.*shape|shape.*serverless"):
        asyncio.run(collector.execute(
            images=local_images,
            audio=None,
            multi_job_id=job_id,
            enabled_worker_ids='["serverless:pool:1"]',
        ))

    assert manager.cancel_calls[0]["collector_job_id"] == job_id
    assert job_id not in module.prompt_server.distributed_pending_jobs


def test_regular_worker_timeout_keeps_partial_behavior_without_creating_manager():
    module = _load_collector_module()
    job_id = "collector-regular-timeout"
    queue = asyncio.Queue()
    collector = _set_up_collector(module, job_id, queue)
    module.get_worker_timeout_seconds = lambda: 0.01
    module.HEARTBEAT_INTERVAL = 0.1
    local_images = torch.zeros(1, 2, 2, 3)
    collector._get_serverless_manager = lambda *_args, **_kwargs: pytest.fail("regular collection must not create manager")

    images, _audio = asyncio.run(collector.execute(
        images=local_images,
        audio=None,
        multi_job_id=job_id,
        enabled_worker_ids='["worker-a"]',
    ))

    assert torch.equal(images, local_images)
    assert job_id not in module.prompt_server.distributed_pending_jobs
