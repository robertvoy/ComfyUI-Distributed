"""Collector result queues are freed when their master prompt never collects them."""
import asyncio
import importlib.util
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from test_serverless_integration import load_modules, PKG, ROOT


def _load_collector_queues():
    spec = importlib.util.spec_from_file_location(
        "dist_collector_queues_test", ROOT / "api" / "orchestration" / "collector_queues.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cq = _load_collector_queues()


def _server(queues=None):
    return types.SimpleNamespace(
        distributed_pending_jobs=dict(queues or {}),
        distributed_jobs_lock=asyncio.Lock(),
    )


class SweepTests(unittest.IsolatedAsyncioTestCase):
    async def test_sweep_frees_queues_of_prompts_that_left_the_queue(self):
        leaked = asyncio.Queue()
        leaked.put_nowait({"worker_id": "w1", "image": "decoded tensor"})
        ps = _server({"job-gone": leaked, "job-live": asyncio.Queue()})
        cq.register_queue_owner(ps, ["job-gone"], "prompt-gone")
        cq.register_queue_owner(ps, ["job-live"], "prompt-live")
        # Owned, but its Collector already removed the queue on success.
        cq.register_queue_owner(ps, ["job-done"], "prompt-gone")

        with patch.object(cq, "_live_prompt_ids", return_value={"prompt-live"}):
            removed = await cq.sweep_orphan_queues(ps)

        self.assertEqual(removed, 1)
        self.assertEqual(set(ps.distributed_pending_jobs), {"job-live"})
        self.assertEqual(ps.distributed_pending_job_owners, {"job-live": "prompt-live"})

    async def test_run_registered_while_the_sweep_waits_for_the_lock_is_kept(self):
        """A stale queue snapshot must never be applied to a run queued after it."""
        ps = _server({"old-job": asyncio.Queue()})
        cq.register_queue_owner(ps, ["old-job"], "old-prompt")
        await ps.distributed_jobs_lock.acquire()  # e.g. job_complete is enqueuing
        with patch.object(cq, "_live_prompt_ids", return_value=set()):  # snapshot: nothing live
            sweep = asyncio.create_task(cq.sweep_orphan_queues(ps))
            await asyncio.sleep(0)  # the sweep took its snapshot and now waits for the lock
            # Meanwhile a new run is queued and registers its queue.
            ps.distributed_pending_jobs["new-job"] = asyncio.Queue()
            cq.register_queue_owner(ps, ["new-job"], "new-prompt")
            ps.distributed_jobs_lock.release()
            self.assertEqual(await sweep, 1)
        self.assertEqual(set(ps.distributed_pending_jobs), {"new-job"})
        self.assertEqual(ps.distributed_pending_job_owners, {"new-job": "new-prompt"})

    async def test_unreadable_prompt_queue_never_frees_anything(self):
        ps = _server({"job": asyncio.Queue()})
        cq.register_queue_owner(ps, ["job"], "prompt")
        with patch.object(cq, "_live_prompt_ids", side_effect=RuntimeError("no reader")):
            self.assertEqual(await cq.sweep_orphan_queues(ps), 0)
        self.assertIn("job", ps.distributed_pending_jobs)
        self.assertEqual(ps.distributed_pending_job_owners, {"job": "prompt"})

    async def test_unowned_queues_are_left_alone(self):
        # Queues of load-balanced runs (or a run still starting) have no owner.
        ps = _server({"unowned": asyncio.Queue()})
        with patch.object(cq, "_live_prompt_ids", return_value=set()):
            self.assertEqual(await cq.sweep_orphan_queues(ps), 0)
        self.assertIn("unowned", ps.distributed_pending_jobs)

    async def test_release_queues_drops_queue_and_owner(self):
        ps = _server({"a": asyncio.Queue(), "b": asyncio.Queue()})
        cq.register_queue_owner(ps, ["a"], "prompt")
        await cq.release_queues(ps, ["a"])
        self.assertEqual(set(ps.distributed_pending_jobs), {"b"})
        self.assertEqual(ps.distributed_pending_job_owners, {})

    async def test_sweeper_installs_once_and_stops_with_the_app(self):
        app = types.SimpleNamespace(cleanup_ctx=[])
        ps = types.SimpleNamespace(app=app)
        cq.install_collector_queue_sweeper(ps, interval_seconds=60)
        cq.install_collector_queue_sweeper(ps, interval_seconds=60)
        self.assertEqual(len(app.cleanup_ctx), 1)

        context = app.cleanup_ctx[0](app)
        await context.__anext__()  # app startup: the sweep task starts
        tasks = [t for t in asyncio.all_tasks() if t.get_name() == "distributed-collector-queue-sweeper"]
        self.assertEqual(len(tasks), 1)
        with self.assertRaises(StopAsyncIteration):
            await context.__anext__()  # app cleanup: the task is cancelled
        self.assertTrue(tasks[0].cancelled())


class OrchestrationQueueCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.q, modules, self.ps = load_modules()
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.worker = {"id": "w1", "name": "Worker", "type": "local", "host": "localhost",
                       "port": 8189, "enabled": True}
        self.q.load_config = lambda: {"workers": [self.worker], "settings": {"master_delegate_only": False}}
        self.q.select_active_workers = AsyncMock(side_effect=lambda workers, _ws, delegate, **kw: (workers, delegate))
        self.q.queue_prompt_payload = AsyncMock(return_value={"prompt_id": "master-prompt", "number": 1})
        self.q.dispatch_worker_prompt = AsyncMock()
        self.prompt = {
            "1": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0}},
            "2": {"class_type": "DistributedCollector", "inputs": {"images": ["1", 0], "load_balance": False}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "local"}},
        }

    async def _orchestrate(self):
        return await self.q.orchestrate_distributed_execution(self.prompt, None, "client")

    async def test_failed_worker_dispatch_releases_the_collector_queues(self):
        self.q.dispatch_worker_prompt.side_effect = RuntimeError("worker rejected the prompt")
        with self.assertRaises(RuntimeError):
            await self._orchestrate()
        self.q.queue_prompt_payload.assert_not_awaited()
        self.assertEqual(self.ps.distributed_pending_jobs, {})

    async def test_failed_master_queueing_releases_the_collector_queues(self):
        self.q.queue_prompt_payload.side_effect = RuntimeError("prompt failed validation")
        with self.assertRaises(RuntimeError):
            await self._orchestrate()
        self.q.dispatch_worker_prompt.assert_awaited()
        self.assertEqual(self.ps.distributed_pending_jobs, {})

    async def test_queued_run_owns_its_queues_and_the_sweep_frees_them_after_it_ends(self):
        prompt_id, *_ = await self._orchestrate()
        job_ids = set(self.ps.distributed_pending_jobs)
        self.assertTrue(job_ids)
        self.assertEqual(self.ps.distributed_pending_job_owners, {job_id: prompt_id for job_id in job_ids})

        sweeper = sys.modules[PKG + ".api.orchestration.collector_queues"]
        with patch.object(sweeper, "_live_prompt_ids", return_value={prompt_id}):
            self.assertEqual(await sweeper.sweep_orphan_queues(self.ps), 0)
        self.assertEqual(set(self.ps.distributed_pending_jobs), job_ids)

        # The master prompt was interrupted before its Collector ran.
        with patch.object(sweeper, "_live_prompt_ids", return_value=set()):
            self.assertEqual(await sweeper.sweep_orphan_queues(self.ps), len(job_ids))
        self.assertEqual(self.ps.distributed_pending_jobs, {})


if __name__ == "__main__":
    unittest.main()
