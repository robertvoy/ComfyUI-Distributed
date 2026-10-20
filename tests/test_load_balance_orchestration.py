"""Queue orchestration keeps a load-balanced run waiting when every GPU is busy."""
import asyncio
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from test_serverless_integration import load_modules, module, PKG


class FakeLoadBalanceQueue:
    def __init__(self):
        self.records = []

    def register(self, record):
        self.records.append(record)
        return record

    def _remove(self, record):
        self.records.remove(record)

    @property
    def _records(self):
        return {record.prompt_id: record for record in self.records}


class FakePendingRun:
    def __init__(self, prompt_id, job_ids, candidates, master_may_run, dispatch_to_worker, requeue_as_delegate,
                 trace_execution_id=None):
        self.prompt_id = prompt_id
        self.job_ids = set(job_ids)
        self.candidates = list(candidates)
        self.master_may_run = master_may_run
        self.dispatch_to_worker = dispatch_to_worker
        self.requeue_as_delegate = requeue_as_delegate


class DeferredLoadBalanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.q, modules, self.ps = load_modules()
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.worker = {"id": "w1", "name": "Worker", "type": "local", "port": 8189, "enabled": True}
        self.settings = {"master_delegate_only": False}
        self.q.load_config = lambda: {"workers": [self.worker], "settings": self.settings}
        self.q.select_active_workers = AsyncMock(side_effect=lambda workers, _ws, delegate, **kw: (workers, delegate))
        self.q.select_least_busy_worker = AsyncMock()
        # The real helper queues under the id it is handed; echo it like it does.
        self.q.queue_prompt_payload = AsyncMock(side_effect=lambda *a, **kw: {
            "prompt_id": kw.get("prompt_id", "master-prompt"), "number": 3, "node_errors": {}})
        self.lb_queue = FakeLoadBalanceQueue()
        self.select_idle_worker = AsyncMock(return_value=(None, 1))
        sys.modules[PKG + ".api.orchestration.load_balance_queue"] = module(
            "load_balance_queue",
            select_idle_worker=self.select_idle_worker,
            PendingRun=FakePendingRun,
            get_load_balance_queue=lambda *a: self.lb_queue,
        )
        self.prompt = {
            "1": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0}},
            "2": {"class_type": "DistributedCollector", "inputs": {"images": ["1", 0], "load_balance": True}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "local"}},
        }

    async def _run(self):
        return await self.q.orchestrate_distributed_execution(self.prompt, None, "client")

    async def test_all_busy_keeps_full_prompt_waiting_for_a_participating_master(self):
        prompt_id, number, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 0)
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.q.select_least_busy_worker.assert_not_awaited()
        candidates = self.select_idle_worker.call_args.args[0]
        self.assertEqual([c["id"] for c in candidates], ["w1", "master"])

        self.assertEqual(len(self.lb_queue.records), 1)
        record = self.lb_queue.records[0]
        self.assertEqual(record.prompt_id, prompt_id)
        self.assertTrue(record.master_may_run)
        self.assertEqual([c["id"] for c in record.candidates], ["w1"])

        master_prompt = self.q.queue_prompt_payload.call_args.args[0]
        self.assertEqual(self.q.queue_prompt_payload.call_args.kwargs["prompt_id"], prompt_id)
        self.assertIn("1", master_prompt)  # full prompt: the master may run it
        collector = master_prompt["2"]["inputs"]
        self.assertEqual(json.loads(collector["enabled_worker_ids"]), [])
        self.assertFalse(collector["delegate_only"])
        self.assertEqual(record.job_ids, {collector["multi_job_id"]})

    async def test_all_busy_queues_delegate_prompt_for_orchestrator_only_master(self):
        self.settings["master_delegate_only"] = True
        prompt_id, *_ = await self._run()
        record = self.lb_queue.records[0]
        self.assertFalse(record.master_may_run)
        candidates = self.select_idle_worker.call_args.args[0]
        self.assertEqual([c["id"] for c in candidates], ["w1"])
        master_prompt = self.q.queue_prompt_payload.call_args.args[0]
        self.assertNotIn("1", master_prompt)  # pruned: nothing runs on the master
        self.assertTrue(master_prompt["2"]["inputs"]["delegate_only"])
        self.assertEqual(json.loads(master_prompt["2"]["inputs"]["enabled_worker_ids"]), [])

    async def test_waiting_run_dispatches_and_requeues_when_a_worker_frees_up(self):
        prompt_id, *_ = await self._run()
        record = self.lb_queue.records[0]
        self.q.queue_prompt_payload.reset_mock()

        await record.dispatch_to_worker(self.worker)
        self.q.dispatch_worker_prompt.assert_awaited_once()
        worker, worker_prompt = self.q.dispatch_worker_prompt.call_args.args[:2]
        self.assertEqual(worker["id"], "w1")
        self.assertTrue(worker_prompt["2"]["inputs"]["is_worker"])
        self.assertEqual(json.loads(worker_prompt["2"]["inputs"]["enabled_worker_ids"]), ["w1"])

        await record.requeue_as_delegate(self.worker)
        self.q.queue_prompt_payload.assert_awaited_once()
        delegate_prompt = self.q.queue_prompt_payload.call_args.args[0]
        self.assertEqual(self.q.queue_prompt_payload.call_args.kwargs["prompt_id"], prompt_id)
        self.assertNotIn("1", delegate_prompt)
        self.assertEqual(json.loads(delegate_prompt["2"]["inputs"]["enabled_worker_ids"]), ["w1"])
        self.assertTrue(delegate_prompt["2"]["inputs"]["delegate_only"])

    async def test_idle_worker_is_still_dispatched_right_away(self):
        self.select_idle_worker.return_value = (self.worker, 2)
        _, _, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 1)
        self.q.dispatch_worker_prompt.assert_awaited_once()
        self.assertEqual(self.lb_queue.records, [])

    async def test_unreachable_probe_falls_back_to_first_candidate(self):
        self.select_idle_worker.return_value = (None, 0)
        _, _, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 1)
        self.assertEqual(self.lb_queue.records, [])

    async def test_master_queue_failure_discards_the_waiting_run(self):
        self.q.queue_prompt_payload.side_effect = RuntimeError("validation failed")
        with self.assertRaises(RuntimeError):
            await self._run()
        self.assertEqual(self.lb_queue.records, [])

    async def test_upscale_prompts_keep_queue_depth_selection(self):
        self.prompt["4"] = {"class_type": "UltimateSDUpscaleDistributed", "inputs": {"upscaled_image": ["1", 0]}}
        self.q.select_least_busy_worker.return_value = self.worker
        await self._run()
        self.select_idle_worker.assert_not_awaited()
        self.q.select_least_busy_worker.assert_awaited_once()
        self.assertEqual(self.lb_queue.records, [])


if __name__ == "__main__":
    unittest.main()
