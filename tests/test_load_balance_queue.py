"""Deferred load-balance dispatch: runs wait in the master queue for a free GPU."""
import asyncio
import heapq
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PKG = "dist_lb_queue_testpkg"


def _module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    return value


class FakePromptQueue:
    """Just enough of ComfyUI's PromptQueue: a heap, a mutex and currently_running."""

    def __init__(self):
        self.mutex = threading.RLock()
        self.queue = []
        self.currently_running = {}
        self._task = 0

    def put(self, item):
        with self.mutex:
            heapq.heappush(self.queue, item)

    def start_next(self):
        with self.mutex:
            item = heapq.heappop(self.queue)
            self._task += 1
            self.currently_running[self._task] = item
            return item

    def finish_running(self, prompt_id):
        with self.mutex:
            for task_id, item in list(self.currently_running.items()):
                if item[1] == prompt_id:
                    self.currently_running.pop(task_id)

    def delete_queue_item(self, function):
        with self.mutex:
            for index in range(len(self.queue)):
                if function(self.queue[index]):
                    self.queue.pop(index)
                    heapq.heapify(self.queue)
                    return True
        return False

    def prompt_ids(self):
        return [item[1] for item in sorted(self.queue)]


def _load_module(prompt_queue):
    for suffix in ("", ".api", ".api.orchestration", ".utils"):
        package = types.ModuleType(PKG + suffix)
        package.__path__ = []
        sys.modules[PKG + suffix] = package
    ps = types.SimpleNamespace(prompt_queue=prompt_queue)
    sys.modules["server"] = _module("server", PromptServer=types.SimpleNamespace(instance=ps))
    sys.modules[PKG + ".utils.logging"] = _module("logging", debug_log=lambda *a, **k: None, log=lambda *a, **k: None)
    sys.modules[PKG + ".utils.network"] = _module(
        "network", build_worker_url=lambda worker, *a: f"http://{worker['id']}", probe_worker=AsyncMock(return_value=None)
    )
    sys.modules[PKG + ".api.orchestration.dispatch"] = _module(
        "dispatch",
        _extract_queue_remaining=lambda payload: int(payload.get("exec_info", {}).get("queue_remaining", 0)),
        _select_idle_round_robin=lambda statuses: statuses[0],
    )
    spec = importlib.util.spec_from_file_location(
        PKG + ".api.orchestration.load_balance_queue", ROOT / "api" / "orchestration" / "load_balance_queue.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, ps


class LoadBalanceQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.prompt_queue = FakePromptQueue()
        self.lb, self.ps = _load_module(self.prompt_queue)
        self.queue = self.lb.LoadBalanceQueue(lambda: self.prompt_queue)
        # Drive cycles explicitly instead of through the background task.
        self.queue._ensure_task = lambda: None
        self.workers = [{"id": "w1", "name": "Slow"}, {"id": "w2", "name": "Fast"}]
        self.remaining = {"w1": 1, "w2": 1}
        self.dispatched = []
        self.requeued = []

        async def fake_probe(url, timeout=3.0):
            worker_id = url.rsplit("/", 1)[-1]
            value = self.remaining.get(worker_id)
            return None if value is None else {"exec_info": {"queue_remaining": value}}

        self.lb.probe_worker = fake_probe

    def _queue_run(self, prompt_id, master_may_run=True, dispatch_error=None):
        number = len(self.prompt_queue.queue) + len(self.prompt_queue.currently_running)
        self.prompt_queue.put((number, prompt_id, {"full": prompt_id}, {}, [], {}))

        async def dispatch_to_worker(worker):
            if dispatch_error is not None:
                raise dispatch_error
            self.dispatched.append((prompt_id, worker["id"]))

        async def requeue_as_delegate(worker):
            self.requeued.append((prompt_id, worker["id"]))
            self.prompt_queue.put((100 + len(self.requeued), prompt_id, {"delegate": worker["id"]}, {}, [], {}))

        record = self.lb.PendingRun(prompt_id, [f"job-{prompt_id}"], self.workers, master_may_run,
                                    dispatch_to_worker, requeue_as_delegate)
        self.queue.register(record)
        return record

    async def test_busy_workers_leave_runs_waiting(self):
        self._queue_run("p1")
        claims = await self.queue.run_cycle()
        self.assertEqual(claims, [])
        self.assertEqual(self.dispatched, [])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])

    async def test_first_free_worker_takes_oldest_waiting_run(self):
        self._queue_run("p1")
        self._queue_run("p2")
        self.remaining["w2"] = 0
        await self.queue.run_cycle()
        self.assertEqual(self.dispatched, [("p1", "w2")])
        self.assertEqual(self.requeued, [("p1", "w2")])
        # p1 moved behind p2 as a delegate prompt under the same id
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p2", "p1"])
        self.assertEqual(self.queue.lookup_job("job-p1").assigned_id, "w2")
        self.assertIsNone(self.queue.lookup_job("job-p2").assigned)
        # the worker that just got a run is not probed again right away
        self.remaining["w2"] = 0
        await self.queue.run_cycle()
        self.assertEqual(len(self.dispatched), 1)

    async def test_faster_worker_keeps_pulling_until_queue_drains(self):
        for index in range(4):
            self._queue_run(f"p{index}")
        self.remaining["w2"] = 0
        order = []
        for _ in range(4):
            await self.queue.run_cycle()
            order.extend(self.dispatched[len(order):])
            self.queue._recent_dispatch.clear()
        self.assertEqual(order, [(f"p{i}", "w2") for i in range(4)])
        self.assertEqual(self.queue.pending_runs(), [])

    async def test_master_executor_claims_run_it_reaches_first(self):
        record = self._queue_run("p1")
        self.prompt_queue.start_next()  # master popped it; it is no longer claimable
        self.remaining["w2"] = 0
        await self.queue.run_cycle()
        self.assertEqual(self.dispatched, [])
        self.assertEqual(record.assigned, self.lb.MASTER)
        assignment = await self.queue.wait_for_assignment(record)
        self.assertEqual(assignment, self.lb.MASTER)

    async def test_collector_claims_master_when_dispatcher_never_looked(self):
        record = self._queue_run("p1")
        self.prompt_queue.start_next()
        self.assertEqual(await self.queue.wait_for_assignment(record), self.lb.MASTER)
        self.queue.finish_job(record, "job-p1")
        self.assertIsNone(self.queue.lookup_job("job-p1"))

    async def test_orchestrator_only_head_run_is_assigned_while_collector_waits(self):
        record = self._queue_run("p1", master_may_run=False)
        self.prompt_queue.start_next()  # delegate prompt is executing, its Collector waits
        waiter = asyncio.ensure_future(self.queue.wait_for_assignment(record))
        await asyncio.sleep(0)
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        assignment = await asyncio.wait_for(waiter, timeout=2)
        self.assertEqual(assignment["id"], "w1")
        self.assertEqual(self.dispatched, [("p1", "w1")])
        self.assertEqual(self.requeued, [])  # already executing: nothing to re-queue

    async def test_failed_dispatch_restores_run_and_keeps_waiting(self):
        record = self._queue_run("p1", dispatch_error=RuntimeError("worker offline"))
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        self.assertIsNone(record.assigned)
        self.assertFalse(record.claiming)
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])
        self.assertEqual(self.requeued, [])

    async def test_cancelled_runs_are_dropped_by_sweep(self):
        record = self._queue_run("p1")
        self.prompt_queue.delete_queue_item(lambda item: item[1] == "p1")
        self.queue.sweep()
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertEqual(len(self.queue), 0)
        self.assertIsNone(record.assigned)

    async def test_interrupt_check_breaks_the_wait(self):
        record = self._queue_run("p1", master_may_run=False)

        def interrupted():
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            await self.queue.wait_for_assignment(record, interrupt_check=interrupted)

    async def test_select_idle_worker_reports_busy_versus_unreachable(self):
        worker, probed = await self.lb.select_idle_worker(self.workers)
        self.assertIsNone(worker)
        self.assertEqual(probed, 2)
        self.remaining = {}
        worker, probed = await self.lb.select_idle_worker(self.workers)
        self.assertIsNone(worker)
        self.assertEqual(probed, 0)
        self.remaining = {"w1": 3, "w2": 0}
        worker, probed = await self.lb.select_idle_worker(self.workers)
        self.assertEqual(worker["id"], "w2")

    async def test_concurrent_cycles_never_double_book_a_worker(self):
        self._queue_run("p1")
        self._queue_run("p2")
        self.remaining["w2"] = 0
        await asyncio.gather(self.queue.run_cycle(), self.queue.run_cycle())
        self.assertEqual(self.dispatched, [("p1", "w2")])

    async def test_background_task_dispatches_on_wake(self):
        queue = self.lb.LoadBalanceQueue(lambda: self.prompt_queue)
        queue.poll_interval = 0.05
        saved, self.queue = self.queue, queue
        try:
            record = self._queue_run("p1")
            self.remaining["w1"] = 0
            for _ in range(50):
                if record.assigned is not None:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(record.assigned_id, "w1")
            queue.finish_job(record, "job-p1")
            await asyncio.sleep(0.1)
            self.assertIsNone(queue._task)  # loop exits once nothing is registered
        finally:
            self.queue = saved

    async def test_registry_is_shared_on_the_prompt_server(self):
        first = self.lb.get_load_balance_queue(self.ps)
        self.assertIs(first, self.lb.get_load_balance_queue(self.ps))
        self.assertIs(first._prompt_queue(), self.prompt_queue)


if __name__ == "__main__":
    unittest.main()
