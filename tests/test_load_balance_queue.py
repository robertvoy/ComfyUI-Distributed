"""Deferred load-balance dispatch: runs wait in the master queue for a free GPU."""
import asyncio
import heapq
import importlib.util
import sys
import threading
import time
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


class InsertionRefused(RuntimeError):
    """What queue_prompt_payload raises when insert_if refuses the insertion."""

    cancelled_insertion = True


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

    def wipe_queue(self):
        with self.mutex:
            self.queue = []

    def delete_queue_item(self, function):
        # Mirrors ComfyUI: deleting the last item goes through wipe_queue().
        with self.mutex:
            for index in range(len(self.queue)):
                if function(self.queue[index]):
                    if len(self.queue) == 1:
                        self.wipe_queue()
                    else:
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
        self.pending_jobs = {}  # job id -> result queue, like prompt_server.distributed_pending_jobs
        self.queue = self.lb.LoadBalanceQueue(lambda: self.prompt_queue, lambda: self.pending_jobs)
        self.queue.install_cancellation_hooks()
        # Drive cycles explicitly instead of through the background task.
        self.queue._ensure_task = lambda: None
        self.workers = [{"id": "w1", "name": "Slow"}, {"id": "w2", "name": "Fast"}]
        self.remaining = {"w1": 1, "w2": 1}
        self.dispatched = []
        self.prepared = []
        self.requeued = []

        async def fake_probe(url, timeout=3.0):
            worker_id = url.rsplit("/", 1)[-1]
            value = self.remaining.get(worker_id)
            return None if value is None else {"exec_info": {"queue_remaining": value}}

        self.lb.probe_worker = fake_probe
        # What a worker answers when asked about a specific run (its job ids):
        # "running" | "finished" | "absent" | None (could not be asked).
        self.job_state = {}
        self.job_state_queries = []

        async def fake_job_state(worker, job_ids, timeout=5.0):
            self.job_state_queries.append((worker["id"], set(job_ids)))
            return self.job_state.get(worker["id"], "absent")

        self.fetch_worker_job_state = self.lb.fetch_worker_job_state
        self.lb.fetch_worker_job_state = fake_job_state

    def _queue_run(self, prompt_id, master_may_run=True, dispatch_error=None, requeue_errors=0, queued=True,
                   prepare_error=None, prepare_gate=None, ambiguous=False, requeue_gate=None):
        """Register a run. ``prepare_gate`` (an Event) delays preparation until set;
        ``ambiguous`` makes the send raise AmbiguousDispatchError after the worker
        has in fact accepted the job (its acknowledgement was lost)."""
        if queued:
            number = len(self.prompt_queue.queue) + len(self.prompt_queue.currently_running)
            self.prompt_queue.put((number, prompt_id, {"full": prompt_id}, {}, [], {}))
        self.pending_jobs[f"job-{prompt_id}"] = asyncio.Queue()
        remaining_requeue_errors = [requeue_errors]

        async def prepare_for_worker(worker):
            if prepare_gate is not None:
                await prepare_gate.wait()
            if prepare_error is not None:
                raise prepare_error
            self.prepared.append((prompt_id, worker["id"]))
            return {"payload": prompt_id}

        async def send_to_worker(worker, payload):
            assert payload == {"payload": prompt_id}
            if dispatch_error is not None:
                raise dispatch_error
            self.dispatched.append((prompt_id, worker["id"]))
            if ambiguous:
                raise self.lb.AmbiguousDispatchError("acknowledgement lost")

        async def requeue_as_delegate(worker):
            if requeue_gate is not None:
                await requeue_gate.wait()  # prompt validation yielding to the loop
            if remaining_requeue_errors[0] > 0:
                remaining_requeue_errors[0] -= 1
                raise RuntimeError("delegate prompt failed validation")
            # Mirrors queue_prompt_payload(insert_if=lambda: not record.cancelled).
            if record.cancelled:
                raise InsertionRefused("cancelled before insertion")
            self.requeued.append((prompt_id, worker["id"]))
            self.prompt_queue.put((100 + len(self.requeued), prompt_id, {"delegate": worker["id"]}, {}, [], {}))

        record = self.lb.PendingRun(prompt_id, [f"job-{prompt_id}"], self.workers, master_may_run,
                                    prepare_for_worker, send_to_worker, requeue_as_delegate)
        self.queue.register(record)
        if queued:
            self.queue.mark_queued(record)
        return record

    async def _until_in_flight(self, record):
        """Wait until the dispatcher has taken ``record`` out of the queue for a hand-off."""
        for _ in range(200):
            if record.claiming and record.prompt_id not in self.prompt_queue.prompt_ids():
                return
            await asyncio.sleep(0.005)
        self.fail("hand-off did not start")

    # ----- ambiguous acceptance ------------------------------------------------
    async def _ambiguous_run(self, prompt_id="p1", **kwargs):
        record = self._queue_run(prompt_id, ambiguous=True, **kwargs)
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        self.assertEqual(self.dispatched, [(prompt_id, "w1")])
        self.assertEqual(record.unresolved["id"], "w1")
        self.assertIsNone(record.assigned)
        self.assertEqual(self.prompt_queue.prompt_ids(), [])  # not restored: the master cannot pick it up
        record.unresolved_since = 0  # past the grace period
        self.queue._recent_dispatch.clear()
        return record

    async def test_lost_acknowledgement_never_replays_the_run(self):
        """The worker took the job, lost the acknowledgement, ran it and lost the
        delivery too: it now looks idle. The original prompt must not run again."""
        record = await self._ambiguous_run()
        self.remaining["w1"] = 0  # idle again; nothing delivered
        self.job_state["w1"] = "finished"  # but the worker's history has the run
        await self.queue.run_cycle()
        self.assertEqual(len(self.dispatched), 1)  # no second submission
        self.assertEqual(self.prompt_queue.queue, [])  # original prompt never restored
        self.assertIsNone(record.unresolved)
        self.assertEqual(record.assigned_id, "w1")
        self.assertEqual(self.job_state_queries, [("w1", {"job-p1"})])
        # Nothing else may claim it either.
        self.remaining["w2"] = 0
        await self.queue.run_cycle()
        self.assertEqual(len(self.dispatched), 1)
        # The lost result is reported through the collector, never by re-running.
        record.dispatched_at = 0
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w1")])
        self.assertEqual([item[2] for item in self.prompt_queue.queue], [{"delegate": "w1"}])
        self.assertEqual(len(self.dispatched), 1)

    async def test_lost_acknowledgement_with_job_still_running_is_collected(self):
        record = await self._ambiguous_run()
        self.remaining["w1"] = 0  # aggregate probe says idle (e.g. a race); the job itself is there
        self.job_state["w1"] = "running"
        await self.queue.run_cycle()
        self.assertEqual(record.assigned_id, "w1")
        self.assertTrue(record.requeue_pending and record.wait_for_result)
        self.assertEqual(len(self.dispatched), 1)
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w1")])
        self.queue.finish_job(record, "job-p1")
        self.assertIsNone(self.queue.lookup_job("job-p1"))

    async def test_lost_acknowledgement_delivery_resolves_it_without_asking(self):
        record = await self._ambiguous_run()
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(record.assigned_id, "w1")
        self.assertEqual(self.requeued, [("p1", "w1")])
        self.assertEqual(self.job_state_queries, [])
        self.assertEqual(len(self.dispatched), 1)

    async def _assert_missing_history_never_replays(self, history):
        """Exercise the real queue/history lookup, not a mocked 'absent' answer."""
        record = await self._ambiguous_run()
        self.remaining = {"w1": 0, "w2": 0}
        requests = []

        class Response:
            status = 200

            def __init__(self, payload):
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def json(self):
                return self.payload

        class Session:
            def get(self, url, **kwargs):
                requests.append(url)
                if url.endswith("/queue"):
                    return Response({"queue_running": [], "queue_pending": []})
                if url.endswith("/history"):
                    limit = int(kwargs["params"]["max_items"])
                    return Response(dict(list(history.items())[-limit:]))
                raise AssertionError(f"Unexpected request: {url}")

        network = sys.modules[PKG + ".utils.network"]
        self.lb.fetch_worker_job_state = self.fetch_worker_job_state
        with patch.object(network, "get_client_session", AsyncMock(return_value=Session()), create=True):
            for _ in range(5):
                self.queue._recent_dispatch.clear()
                await self.queue.run_cycle()
                self.assertEqual(self.dispatched, [("p1", "w1")])
                self.assertEqual(self.prompt_queue.prompt_ids(), [])
                self.assertEqual(self.requeued, [])
                self.assertIsNone(record.assigned)
                self.assertEqual(record.unresolved["id"], "w1")
                self.assertIs(self.queue.lookup_job("job-p1"), record)
                self.assertIn("job-p1", self.pending_jobs)
        self.assertEqual(requests.count("http://w1/history"), 5)
        # Missing history must not discard ownership: a late result still collects.
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w1")])
        self.assertEqual(self.dispatched, [("p1", "w1")])

    async def test_lost_acknowledgement_outside_history_window_never_replays(self):
        history = {"accepted": {"prompt": [0, "accepted", {
            "collector": {"inputs": {"multi_job_id": "job-p1"}}
        }]}}
        for index in range(self.lb.HISTORY_LOOKBACK_ITEMS):
            history[f"newer-{index}"] = {"prompt": [index + 1, f"newer-{index}", {}]}
        await self._assert_missing_history_never_replays(history)

    async def test_lost_acknowledgement_cleared_history_never_replays(self):
        await self._assert_missing_history_never_replays({})

    async def test_busy_worker_on_unrelated_work_is_not_taken_as_acceptance(self):
        record = await self._ambiguous_run(master_may_run=False)
        self.remaining["w1"] = 3  # busy, but with other prompts
        self.job_state["w1"] = "absent"
        await self.queue.run_cycle()
        self.assertEqual(record.unresolved["id"], "w1")
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        for _ in range(5):
            await self.queue.run_cycle()
        self.assertEqual(record.unresolved["id"], "w1")  # repeated absence still proves nothing
        self.assertIsNone(record.assigned)
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        self.assertEqual(self.dispatched, [("p1", "w1")])
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        self.assertEqual(self.requeued, [])

    async def test_lost_acknowledgement_stays_on_hold_while_worker_cannot_be_asked(self):
        record = await self._ambiguous_run()
        self.job_state["w1"] = None
        for _ in range(self.lb.WORKER_UNREACHABLE_PROBES + 2):
            await self.queue.run_cycle()
        self.assertEqual(record.unresolved["id"], "w1")  # never replayed, never dropped
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        self.assertEqual(len(self.dispatched), 1)
        # Neither missing history nor failed probes establishes rejection.
        self.job_state["w1"] = "absent"
        await self.queue.run_cycle()
        self.job_state["w1"] = None
        await self.queue.run_cycle()
        self.assertEqual(record.absent_checks, 0)
        self.assertEqual(record.unresolved["id"], "w1")

    async def test_eager_run_with_lost_acknowledgement_is_collected_after_delivery(self):
        """Sent straight to an idle worker by a participating master, no master prompt exists yet."""
        record = self._queue_run("p1", master_may_run=False, queued=False)
        self.queue.register_dispatched(record, self.workers[0])
        self.queue.mark_unresolved(record, self.workers[0])  # the eager dispatch raised ambiguity
        self.assertIsNone(record.assigned)
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [])
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(record.assigned_id, "w1")
        self.assertEqual(self.requeued, [("p1", "w1")])  # a collector is queued for the result
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])
        self.assertFalse(record.requeue_pending)
        self.queue.finish_job(record, "job-p1")
        self.assertIsNone(self.queue.lookup_job("job-p1"))

    async def test_eager_run_with_lost_acknowledgement_confirmed_running_is_collected(self):
        record = self._queue_run("p1", master_may_run=False, queued=False)
        self.queue.register_dispatched(record, self.workers[0])
        self.queue.mark_unresolved(record, self.workers[0])
        record.unresolved_since = 0
        self.job_state["w1"] = "running"
        await self.queue.run_cycle()
        self.assertEqual(record.assigned_id, "w1")
        self.assertTrue(record.requeue_pending)
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w1")])

    async def test_eager_run_with_lost_acknowledgement_missing_history_stays_tracked(self):
        record = self._queue_run("p1", master_may_run=False, queued=False)
        self.queue.register_dispatched(record, self.workers[0])
        self.queue.mark_unresolved(record, self.workers[0])
        record.unresolved_since = 0
        for _ in range(5):
            await self.queue.run_cycle()
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        self.assertIn("job-p1", self.pending_jobs)
        self.assertEqual(record.unresolved["id"], "w1")
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w1")])

    async def test_definite_failure_before_submission_still_restores_the_run(self):
        record = self._queue_run("p1", dispatch_error=RuntimeError("connection refused"))
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        self.assertIsNone(record.unresolved)
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])
        record2 = self._queue_run("p2", prepare_error=RuntimeError("media sync failed"))
        self.remaining["w2"] = 0
        self.queue._recent_dispatch.clear()
        await self.queue.run_cycle()
        self.assertIsNone(record2.assigned)
        self.assertIn("p2", self.prompt_queue.prompt_ids())

    # ----- cancellation during the hand-off ------------------------------------
    async def test_clear_queue_during_delayed_preparation_drops_run_without_sending(self):
        gate = asyncio.Event()
        record = self._queue_run("p1", prepare_gate=gate)
        self.remaining["w1"] = 0
        cycle = asyncio.ensure_future(self.queue.run_cycle())
        await self._until_in_flight(record)
        self.assertEqual(self.prompt_queue.prompt_ids(), [])  # taken out for the hand-off
        self.prompt_queue.wipe_queue()  # the user clears the queue
        self.assertTrue(record.cancelled)
        gate.set()
        await cycle
        self.assertEqual(self.dispatched, [])  # never submitted
        self.assertEqual(self.prompt_queue.prompt_ids(), [])  # and it does not reappear
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)  # result queue gone
        self.assertTrue(self.queue.is_cancelled_job("job-p1"))
        self.queue.sweep()
        await self.queue.run_cycle()
        self.assertEqual(self.prompt_queue.prompt_ids(), [])

    async def test_delete_of_one_run_during_hand_off_cancels_only_that_run(self):
        gate = asyncio.Event()
        record1 = self._queue_run("p1", prepare_gate=gate)
        record2 = self._queue_run("p2", prepare_gate=gate)
        self.remaining = {"w1": 0, "w2": 0}
        cycle = asyncio.ensure_future(self.queue.run_cycle())
        await self._until_in_flight(record1)
        await self._until_in_flight(record2)
        self.prompt_queue.delete_queue_item(lambda item: item[1] == "p2")  # /queue delete of p2
        self.assertFalse(record1.cancelled)
        self.assertTrue(record2.cancelled)
        gate.set()
        await cycle
        self.assertEqual(self.dispatched, [("p1", "w1")])
        self.assertIsNone(self.queue.lookup_job("job-p2"))
        self.assertIs(self.queue.lookup_job("job-p1"), record1)

    async def test_deleting_the_last_queued_item_does_not_cancel_in_flight_runs(self):
        gate = asyncio.Event()
        record = self._queue_run("p1", prepare_gate=gate)
        self._queue_run("p9", queued=True)  # stays queued; deleting it wipes the (one-item) queue internally
        self.remaining = {"w1": 0}
        cycle = asyncio.ensure_future(self.queue.run_cycle())
        await self._until_in_flight(record)
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p9"])
        self.prompt_queue.delete_queue_item(lambda item: item[1] == "p9")
        self.assertFalse(record.cancelled)
        gate.set()
        await cycle
        self.assertEqual(self.dispatched, [("p1", "w1")])

    async def test_accepted_then_cleared_run_is_discarded_when_the_worker_delivers(self):
        record = self._queue_run("p1")
        self.remaining["w1"] = 0
        await self.queue.run_cycle()  # accepted; delegate held until delivery
        self.prompt_queue.wipe_queue()
        self.assertTrue(record.cancelled)
        self.assertIs(self.queue.lookup_job("job-p1"), record)  # accepted work stays owned
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [])  # no collector queued for cancelled work
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)

    async def test_clear_queue_during_pending_requeue_retry_does_not_requeue(self):
        record = self._queue_run("p1", master_may_run=False, requeue_errors=1)
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        self.assertTrue(record.requeue_pending)
        self.prompt_queue.wipe_queue()
        self.assertTrue(record.cancelled)
        record.requeue_retry_at = 0
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [])
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        record.dispatched_at = 0
        self.remaining["w1"] = 0  # worker finished without delivering
        await self.queue.run_cycle()
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)

    async def test_ambiguous_then_cleared_run_stays_tracked_until_delivery_never_restored(self):
        record = self._queue_run("p1", ambiguous=True)
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        self.prompt_queue.wipe_queue()
        self.assertTrue(record.cancelled)
        record.unresolved_since = 0
        for _ in range(5):
            await self.queue.run_cycle()
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        self.assertEqual(self.dispatched, [("p1", "w1")])
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        self.assertIn("job-p1", self.pending_jobs)
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()  # late result is discarded, never collected or replayed
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)
        self.assertEqual(self.requeued, [])
        self.assertEqual(self.prompt_queue.prompt_ids(), [])

    async def test_clear_queue_during_delegate_validation_does_not_insert_it(self):
        """Cancellation that lands while the delegate prompt is being validated."""
        gate = asyncio.Event()
        record = self._queue_run("p1", master_may_run=False, requeue_gate=gate)
        self.remaining["w1"] = 0
        cycle = asyncio.ensure_future(self.queue.run_cycle())
        for _ in range(200):
            if self.dispatched:
                break
            await asyncio.sleep(0.005)
        self.assertEqual(self.dispatched, [("p1", "w1")])  # accepted; delegate now validating
        self.prompt_queue.wipe_queue()
        self.assertTrue(record.cancelled)
        gate.set()
        await cycle
        self.assertEqual(self.requeued, [])
        self.assertEqual(self.prompt_queue.prompt_ids(), [])  # never inserted
        self.assertEqual(record.requeue_failures, 0)  # refused insertion is not a retryable failure
        self.assertFalse(record.requeue_pending)
        self.assertIs(self.queue.lookup_job("job-p1"), record)  # accepted work stays owned
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)
        self.assertEqual(self.prompt_queue.prompt_ids(), [])

    # ----- interrupted assignment on an orchestrator-only master ---------------
    async def test_interrupt_during_assignment_cleans_result_queue_and_stops_dispatch(self):
        gate = asyncio.Event()
        record = self._queue_run("p1", master_may_run=False, prepare_gate=gate)
        self.prompt_queue.start_next()  # delegate prompt executing; its Collector waits
        self.remaining["w1"] = 0
        cycle = asyncio.ensure_future(self.queue.run_cycle())
        for _ in range(200):
            if record.claiming:
                break
            await asyncio.sleep(0.005)
        self.assertTrue(record.claiming)
        # The user interrupts the master prompt: the Collector gives up on the run.
        self.queue.finish_job(record, "job-p1", failed=True)
        self.prompt_queue.finish_running("p1")
        gate.set()
        await cycle
        self.assertEqual(self.dispatched, [])
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)
        # A late worker cannot populate or recreate the abandoned result queue.
        self.assertTrue(self.queue.is_cancelled_job("job-p1"))
        self.assertFalse(self.queue.result_delivered("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)

    async def test_interrupt_after_acceptance_discards_late_result(self):
        record = self._queue_run("p1", master_may_run=False)
        self.prompt_queue.start_next()
        self.remaining["w1"] = 0
        await self.queue.run_cycle()  # accepted by w1; Collector would now wait for it
        self.queue.finish_job(record, "job-p1", failed=True)  # interrupted instead
        self.prompt_queue.finish_running("p1")
        self.assertIs(self.queue.lookup_job("job-p1"), record)  # still owned while the worker runs
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)
        self.assertEqual(self.requeued, [])

    async def test_cancelling_a_queued_run_drops_its_result_queue(self):
        self._queue_run("p1")
        self.prompt_queue.wipe_queue()
        self.queue.sweep()
        self.assertIsNone(self.queue.lookup_job("job-p1"))
        self.assertNotIn("job-p1", self.pending_jobs)

    async def test_run_is_safe_from_sweep_until_its_master_prompt_is_queued(self):
        # Registered, but the orchestrator is still validating the master prompt.
        record = self._queue_run("p1", master_may_run=False, queued=False)
        self.remaining["w1"] = 0
        self.queue.sweep()
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        await self.queue.run_cycle()
        self.assertEqual(self.dispatched, [])  # not claimable before it is queued
        self.assertIsNone(record.assigned)
        # Now the prompt lands in the queue and the dispatcher may take it.
        self.prompt_queue.put((0, "p1", {"full": "p1"}, {}, [], {}))
        self.queue.mark_queued(record)
        self.queue._recent_dispatch.clear()
        await self.queue.run_cycle()
        self.assertEqual(self.dispatched, [("p1", "w1")])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])

    async def test_unqueued_run_is_dropped_when_orchestrator_discards_it(self):
        record = self._queue_run("p1", queued=False)
        self.queue._remove(record)
        self.assertEqual(len(self.queue), 0)
        self.queue.sweep()
        self.assertEqual(len(self.queue), 0)

    async def test_requeue_failure_keeps_tracking_the_remote_job_and_retries(self):
        record = self._queue_run("p1", master_may_run=False, requeue_errors=2)
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        # Dispatched, master delegate prompt missing, run still tracked.
        self.assertEqual(self.dispatched, [("p1", "w1")])
        self.assertEqual(self.requeued, [])
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        self.assertEqual(record.assigned_id, "w1")
        self.assertTrue(record.requeue_pending)
        self.assertEqual(self.prompt_queue.prompt_ids(), [])
        # Sweeps never drop it while the requeue is outstanding.
        self.queue.sweep()
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        # Backoff holds the retry, then it is attempted again until it lands.
        await self.queue.run_cycle()
        self.assertTrue(record.requeue_pending)
        record.requeue_retry_at = 0
        await self.queue.run_cycle()
        self.assertTrue(record.requeue_pending)  # second failure
        self.assertEqual(record.requeue_failures, 2)
        record.requeue_retry_at = 0
        await self.queue.run_cycle()
        self.assertFalse(record.requeue_pending)
        self.assertEqual(self.requeued, [("p1", "w1")])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])
        self.assertEqual(len(self.dispatched), 1)  # the worker was never sent the job twice
        # Collected on the master: tracking ends only now.
        self.queue.finish_job(record, "job-p1")
        self.assertIsNone(self.queue.lookup_job("job-p1"))

    async def test_requeue_backoff_grows_and_caps(self):
        record = self._queue_run("p1", master_may_run=False, requeue_errors=10)
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        delays = []
        for _ in range(7):
            record.requeue_retry_at = 0
            before = time.monotonic()
            await self.queue.run_cycle()
            delays.append(round(record.requeue_retry_at - before))
        self.assertEqual(delays, [2, 4, 8, 16, 30, 30, 30])
        self.assertIs(self.queue.lookup_job("job-p1"), record)

    async def test_busy_workers_leave_runs_waiting(self):
        self._queue_run("p1")
        claims = await self.queue.run_cycle()
        self.assertEqual(claims, [])
        self.assertEqual(self.dispatched, [])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])

    async def test_first_free_worker_takes_oldest_waiting_run(self):
        self._queue_run("p1", master_may_run=False)
        self._queue_run("p2", master_may_run=False)
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

    async def test_participating_master_gets_delegate_only_after_worker_delivers(self):
        """The master executor must not be parked on a collector waiting for a worker."""
        record = self._queue_run("p1")
        self._queue_run("p2")
        self.remaining["w2"] = 0
        await self.queue.run_cycle()
        self.assertEqual(self.dispatched, [("p1", "w2")])
        self.assertEqual(record.assigned_id, "w2")
        # Nothing collect-only on the master yet: p2 is next for the master executor.
        self.assertEqual(self.requeued, [])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p2"])
        self.assertTrue(record.requeue_pending and record.wait_for_result)
        self.queue.sweep()
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        # Still nothing while the worker is running it, even after the dispatch grace.
        record.dispatched_at = 0
        self.job_state["w2"] = "running"
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [])
        # The worker delivers: the collector is queued behind the real work.
        self.assertTrue(self.queue.result_delivered("job-p1"))
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w2")])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p2", "p1"])
        self.assertFalse(record.requeue_pending)
        self.queue.finish_job(record, "job-p1")
        self.assertIsNone(self.queue.lookup_job("job-p1"))

    async def test_delegate_is_queued_when_worker_finished_without_delivering(self):
        record = self._queue_run("p1")
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [])
        record.dispatched_at = 0  # past the dispatch grace
        self.job_state["w1"] = "running"
        await self.queue.run_cycle()  # still running there: keep waiting
        self.assertEqual(self.requeued, [])
        self.job_state["w1"] = "finished"
        await self.queue.run_cycle()  # ran it, nothing delivered: let the collector report it
        self.assertEqual(self.requeued, [("p1", "w1")])
        self.assertFalse(record.wait_for_result)

    async def test_delegate_is_queued_when_worker_becomes_unreachable(self):
        record = self._queue_run("p1")
        self.remaining["w1"] = 0
        await self.queue.run_cycle()
        record.dispatched_at = 0
        self.job_state["w1"] = None  # the worker cannot be asked from now on
        for _ in range(self.lb.WORKER_UNREACHABLE_PROBES - 1):
            await self.queue.run_cycle()
            self.assertEqual(self.requeued, [])
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w1")])

    async def test_eagerly_dispatched_run_is_tracked_until_its_result_is_collected(self):
        record = self._queue_run("p1", queued=False)
        self.queue.register_dispatched(record, self.workers[1])
        self.assertEqual(record.assigned_id, "w2")
        self.assertTrue(record.requeue_pending)
        self.queue.sweep()
        self.assertIs(self.queue.lookup_job("job-p1"), record)
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [])
        self.queue.result_delivered("job-p1")
        await self.queue.run_cycle()
        self.assertEqual(self.requeued, [("p1", "w2")])
        self.assertEqual(self.prompt_queue.prompt_ids(), ["p1"])
        self.assertFalse(self.queue.result_delivered("job-unknown"))

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
        queue = self.lb.LoadBalanceQueue(lambda: self.prompt_queue, lambda: self.pending_jobs)
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
