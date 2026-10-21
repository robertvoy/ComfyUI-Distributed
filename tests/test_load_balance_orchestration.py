"""Queue orchestration keeps a load-balanced run waiting when every GPU is busy."""
import asyncio
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from test_serverless_integration import load_modules, module, PKG, ROOT


class FakeLoadBalanceQueue:
    def __init__(self):
        self.records = []
        self.queued = []

    def register(self, record):
        self.records.append(record)
        return record

    def mark_queued(self, record):
        assert record in self.records
        self.queued.append(record)

    def register_dispatched(self, record, worker):
        self.records.append(record)
        record.dispatched_worker = worker

    def discard_run(self, prompt_id):
        self.records = [record for record in self.records if record.prompt_id != prompt_id]

    def mark_unresolved(self, record, worker, item=None):
        record.unresolved = worker

    def _remove(self, record):
        self.records.remove(record)

    @property
    def _records(self):
        return {record.prompt_id: record for record in self.records}


class FakePendingRun:
    def __init__(self, prompt_id, job_ids, candidates, master_may_run, prepare_for_worker, send_to_worker,
                 requeue_as_delegate, trace_execution_id=None):
        self.prompt_id = prompt_id
        self.job_ids = set(job_ids)
        self.candidates = list(candidates)
        self.master_may_run = master_may_run
        self.prepare_for_worker = prepare_for_worker
        self.send_to_worker = send_to_worker
        self.requeue_as_delegate = requeue_as_delegate
        self.unresolved = None
        self.cancelled = False


class AmbiguousDispatchError(RuntimeError):
    pass


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
        sys.modules[PKG + ".api.orchestration.dispatch"].AmbiguousDispatchError = AmbiguousDispatchError
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
        self.assertEqual(self.lb_queue.queued, [record])  # released only after the master prompt is queued
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

        payload = await record.prepare_for_worker(self.worker)
        self.q.dispatch_worker_prompt.assert_not_awaited()  # preparation submits nothing
        await record.send_to_worker(self.worker, payload)
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
        # The insertion guard follows the run's cancellation flag.
        insert_if = self.q.queue_prompt_payload.call_args.kwargs["insert_if"]
        self.assertTrue(insert_if())
        record.cancelled = True
        self.assertFalse(insert_if())

    async def test_idle_worker_is_dispatched_right_away_and_master_stays_free(self):
        self.select_idle_worker.return_value = (self.worker, 2)
        prompt_id, _, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 1)
        self.q.dispatch_worker_prompt.assert_awaited_once()
        # Participating master: no collect-only prompt parked on its executor.
        self.q.queue_prompt_payload.assert_not_awaited()
        self.assertEqual(len(self.lb_queue.records), 1)
        record = self.lb_queue.records[0]
        self.assertEqual(record.prompt_id, prompt_id)
        self.assertEqual(record.dispatched_worker["id"], "w1")
        self.assertFalse(record.master_may_run)
        # Delivered later: the delegate prompt is queued under the same id.
        await record.requeue_as_delegate(self.worker)
        delegate_prompt = self.q.queue_prompt_payload.call_args.args[0]
        self.assertEqual(self.q.queue_prompt_payload.call_args.kwargs["prompt_id"], prompt_id)
        self.assertEqual(json.loads(delegate_prompt["2"]["inputs"]["enabled_worker_ids"]), ["w1"])

    async def test_failed_dispatch_of_held_run_discards_it(self):
        self.select_idle_worker.return_value = (self.worker, 2)
        self.q.dispatch_worker_prompt.side_effect = RuntimeError("worker refused")
        with self.assertRaises(RuntimeError):
            await self._run()
        self.assertEqual(self.lb_queue.records, [])

    async def test_ambiguous_eager_dispatch_keeps_the_run_on_hold(self):
        """Lost acknowledgement on the immediate dispatch: report it, but never resend or forget it."""
        self.select_idle_worker.return_value = (self.worker, 2)
        self.q.dispatch_worker_prompt.side_effect = AmbiguousDispatchError("ack lost")
        with self.assertRaises(AmbiguousDispatchError):
            await self._run()
        self.assertEqual(len(self.lb_queue.records), 1)
        self.assertEqual(self.lb_queue.records[0].unresolved["id"], "w1")
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_master_selected_alone_creates_no_result_queues(self):
        self.select_idle_worker.return_value = ({"id": "master", "name": "Master", "type": "local"}, 2)
        _, _, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 0)
        self.q.queue_prompt_payload.assert_awaited_once()
        self.assertEqual(self.ps.distributed_pending_jobs, {})
        self.assertEqual(self.lb_queue.records, [])

    async def test_orchestrator_only_master_queues_delegate_immediately(self):
        self.settings["master_delegate_only"] = True
        self.select_idle_worker.return_value = (self.worker, 1)
        _, _, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 1)
        self.q.dispatch_worker_prompt.assert_awaited_once()
        self.q.queue_prompt_payload.assert_awaited_once()
        self.assertEqual(self.lb_queue.records, [])

    async def test_unreachable_probe_falls_back_to_first_candidate(self):
        self.select_idle_worker.return_value = (None, 0)
        _, _, worker_count, _ = await self._run()
        self.assertEqual(worker_count, 1)
        self.q.dispatch_worker_prompt.assert_awaited_once()
        # Sent to the first worker; the participating master still holds the delegate back.
        self.assertEqual(len(self.lb_queue.records), 1)
        self.assertEqual(self.lb_queue.records[0].dispatched_worker["id"], "w1")
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_master_queue_failure_discards_the_waiting_run(self):
        self.q.queue_prompt_payload.side_effect = RuntimeError("validation failed")
        with self.assertRaises(RuntimeError):
            await self._run()
        self.assertEqual(self.lb_queue.records, [])
        self.assertEqual(self.lb_queue.queued, [])

    async def test_dispatcher_sweep_during_prompt_validation_keeps_the_run(self):
        """Regression: validation yields to the loop before the prompt is queued."""
        from test_load_balance_queue import FakePromptQueue
        prompt_queue = FakePromptQueue()
        self.ps.prompt_queue = prompt_queue
        network = sys.modules[PKG + ".utils.network"]
        network.build_worker_url = lambda worker, *a: f"http://{worker['id']}"
        network.probe_worker = AsyncMock(return_value=None)
        dispatch = sys.modules[PKG + ".api.orchestration.dispatch"]

        async def always_busy_probe(worker, semaphore, probe_timeout):
            return {"worker": worker, "queue_remaining": 1}

        dispatch._probe_worker_queue = always_busy_probe
        dispatch._select_idle_round_robin = lambda statuses: statuses[0]
        spec = importlib.util.spec_from_file_location(
            PKG + ".api.orchestration.load_balance_queue", ROOT / "api" / "orchestration" / "load_balance_queue.py")
        lb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lb)
        lb.select_idle_worker = AsyncMock(return_value=(None, 1))
        sys.modules[PKG + ".api.orchestration.load_balance_queue"] = lb
        queue = lb.get_load_balance_queue(self.ps)
        queue.poll_interval = 0.01
        sweeps = []

        async def slow_validation(prompt, workflow_meta, client_id, include_queue_metadata=False, *, prompt_id=None, **kw):
            # Yield long enough for the dispatcher task to run its sweep.
            for _ in range(5):
                await asyncio.sleep(0.02)
                queue.sweep()
                sweeps.append(len(queue))
            prompt_queue.put((0, prompt_id, prompt, {}, [], {}))
            return {"prompt_id": prompt_id, "number": 0, "node_errors": {}}

        self.q.queue_prompt_payload = AsyncMock(side_effect=slow_validation)
        try:
            prompt_id, *_ = await self._run()
            self.assertTrue(sweeps and all(count == 1 for count in sweeps))
            record = queue._records.get(prompt_id)
            self.assertIsNotNone(record)
            self.assertTrue(record.queued)
            self.assertEqual(prompt_queue.prompt_ids(), [prompt_id])
            queue.sweep()
            self.assertIs(queue._records.get(prompt_id), record)
        finally:
            queue._records.clear()
            await asyncio.sleep(0.05)

    async def test_upscale_prompts_keep_queue_depth_selection(self):
        self.prompt["4"] = {"class_type": "UltimateSDUpscaleDistributed", "inputs": {"upscaled_image": ["1", 0]}}
        self.q.select_least_busy_worker.return_value = self.worker
        await self._run()
        self.select_idle_worker.assert_not_awaited()
        self.q.select_least_busy_worker.assert_awaited_once()
        self.assertEqual(self.lb_queue.records, [])


if __name__ == "__main__":
    unittest.main()
