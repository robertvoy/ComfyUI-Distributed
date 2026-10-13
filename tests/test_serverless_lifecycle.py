"""Billing-safe lifecycle regressions: local completion is not remote completion."""
import asyncio
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_serverless_runtime import runtime, Secrets


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.server = types.SimpleNamespace(distributed_pending_jobs={"collector": asyncio.Queue()}, distributed_jobs_lock=asyncio.Lock())
        self.path = Path(self.temp.name) / "jobs.json"
        self.manager = runtime.ServerlessManager(self.server, Secrets(), self.path, AsyncMock(return_value=object()))
        self.worker = {"id": "serverless:pool:1", "pool_id": "pool", "endpoint_id": "ep", "timeout_seconds": 30, "type": "serverless"}
        self.prepared = {"worker": self.worker, "workflow": {}, "images": [], "manifest": [{"job_id": "collector", "prefix": "x", "kind": "image"}]}
        self.api = types.SimpleNamespace(submit=AsyncMock(return_value="job"), status=AsyncMock(), cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        self.client_patch = patch.object(runtime, "RunpodClient", return_value=self.api)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.decode_patch = patch.object(runtime, "decode_results", AsyncMock(return_value={"collector": ["tensor"]}))
        self.decode_patch.start()
        self.addCleanup(self.decode_patch.stop)
        # These regressions exercise capacity holding, so use a limit of one.
        self.limit_patch = patch.object(runtime, "PER_WORKER_JOB_LIMIT", 1)
        self.limit_patch.start()
        self.addCleanup(self.limit_patch.stop)

    async def asyncTearDown(self):
        await self.manager.cancel()

    def second(self, **changes):
        return dict(self.prepared, worker=dict(self.worker, id="serverless:pool:2", **changes))

    def retained(self, status="CANCEL_REQUESTED", job_id: str | None = "old-job"):
        self.manager.ledger.update("old", worker_id="pool", endpoint_id="ep", job_id=job_id,
                                   status=status, created_at=1, updated_at=1)

    async def test_capacity_is_held_until_slow_cancellation_is_confirmed(self):
        cancel_started, release_cancel = asyncio.Event(), asyncio.Event()
        self.api.submit.side_effect = ["job-1", "job-2"]
        self.api.status.side_effect = [runtime.RunpodError("poll failed"), {"status": "COMPLETED", "output": {}}]
        async def cancel(_job_id):
            cancel_started.set()
            await release_cancel.wait()
            return {"status": "CANCELLED"}
        self.api.cancel.side_effect = cancel
        await self.manager.start("exec", "prompt", [self.prepared, self.second()])
        try:
            await asyncio.wait_for(cancel_started.wait(), 1)
            await asyncio.sleep(0.01)
            self.assertEqual(self.api.submit.await_count, 1)
        finally:
            release_cancel.set()
            await self.manager.wait_for_execution("exec")
        self.assertEqual(self.api.submit.await_count, 2)

    async def test_unconfirmed_cancel_holds_capacity_until_reconciled(self):
        self.api.status.side_effect = runtime.RunpodError("poll failed")
        self.api.cancel.return_value = {"status": "IN_PROGRESS"}
        await self.manager.start("first", "prompt", [self.prepared])
        await self.manager.wait_for_execution("first")
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCEL_REQUESTED")
        self.assertTrue(self.manager.pool_is_active("pool"))
        await self.manager.start("second", "prompt", [self.second()])
        try:
            await asyncio.sleep(0.01)
            self.assertEqual(self.api.submit.await_count, 1)
            self.api.status.side_effect = None
            self.api.status.return_value = {"status": "COMPLETED", "output": {}}
            await self.manager.reconcile(worker_id="pool")
            await asyncio.wait_for(self.manager.wait_for_execution("second"), 1)
            self.assertEqual(self.api.submit.await_count, 2)
        finally:
            self.api.cancel.return_value = {"status": "CANCELLED"}
            await self.manager.cancel()

    async def test_restarted_manager_reserves_known_unresolved_jobs(self):
        self.retained("IN_PROGRESS")
        self.manager = runtime.ServerlessManager(self.server, Secrets(), self.path, AsyncMock(return_value=object()))
        self.assertTrue(self.manager.pool_is_active("pool"))
        self.assertFalse(self.manager.pool_is_active("other"))
        await self.manager.start("exec", "prompt", [self.second(timeout_seconds=0.02)])
        await self.manager.wait_for_execution("exec")
        self.api.submit.assert_not_awaited()
        self.assertEqual(self.manager.ledger.all()[-1]["status"], "TIMED_OUT")

    async def test_recovery_closes_never_submitted_waiting_records(self):
        self.retained("WAITING", job_id=None)
        self.manager = runtime.ServerlessManager(self.server, Secrets(), self.path, AsyncMock(return_value=object()))
        self.assertFalse(self.manager.pool_is_active("pool"))
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCELLED")
        self.api.submit.assert_not_awaited()

    async def test_recovery_keeps_interrupted_submission_unresolved(self):
        self.retained("SUBMITTING", job_id=None)
        self.manager = runtime.ServerlessManager(self.server, Secrets(), self.path, AsyncMock(return_value=object()))
        self.assertTrue(self.manager.pool_is_active("pool"))
        self.assertEqual(self.manager.ledger.all()[0]["status"], "UNKNOWN")
        await self.manager.reconcile(worker_id="pool")
        self.api.status.assert_not_awaited()
        self.api.submit.assert_not_awaited()

    async def test_ambiguous_submission_requires_explicit_recovery_acknowledgment(self):
        self.api.submit.side_effect = runtime.AmbiguousSubmission("acceptance unknown")
        await self.manager.start("exec", "prompt", [self.prepared])
        await self.manager.wait_for_execution("exec")
        record = self.manager.ledger.all()[0]
        self.assertTrue(self.manager.pool_is_active("pool"))
        with self.assertRaises(ValueError):
            await self.manager.acknowledge_recovery("pool", record["record_id"], provider_checked=False)
        with self.assertRaises(ValueError):
            await self.manager.acknowledge_recovery("other", record["record_id"], provider_checked=True)
        await self.manager.acknowledge_recovery("pool", record["record_id"], provider_checked=True)
        self.assertFalse(self.manager.pool_is_active("pool"))
        self.assertEqual(self.manager.ledger.all()[0]["status"], "RESOLVED")
        self.api.submit.assert_awaited_once()

    async def test_cancel_already_completed_provider_job_is_terminal(self):
        self.retained()
        self.api.cancel.return_value = {"status": "COMPLETED"}
        await self.manager.cancel(worker_id="pool")
        self.assertEqual(self.manager.ledger.all()[0]["status"], "COMPLETED")
        self.assertFalse(self.manager.pool_is_active("pool"))

    async def test_reconciliation_failure_never_releases_capacity(self):
        self.retained()
        self.api.status.side_effect = runtime.RunpodError("not reachable")
        await self.manager.reconcile(worker_id="pool")
        self.assertTrue(self.manager.pool_is_active("pool"))
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCEL_REQUESTED")

    async def test_cancel_before_first_task_step_finalizes_local_record(self):
        await self.manager.start("exec", "prompt", [self.prepared])
        await self.manager.cancel(worker_id="pool")
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCELLED")
        self.assertFalse(self.server.distributed_pending_jobs["collector"].empty())
        self.assertFalse(self.manager.pool_is_active("pool"))
        self.api.submit.assert_not_awaited()

    async def test_live_record_cannot_be_acknowledged_or_reconciled(self):
        polling = asyncio.Event()
        async def status(_job_id):
            polling.set()
            await asyncio.Event().wait()
        self.api.status.side_effect = status
        await self.manager.start("exec", "prompt", [self.prepared])
        await asyncio.wait_for(polling.wait(), 1)
        record = self.manager.ledger.all()[0]
        with self.assertRaises(ValueError):
            await self.manager.acknowledge_recovery("pool", record["record_id"], provider_checked=True)
        await self.manager.reconcile(worker_id="pool")
        self.assertEqual(self.api.status.await_count, 1)
        self.assertTrue(self.manager.pool_is_active("pool"))

    async def test_repeated_cancel_does_not_interrupt_provider_cleanup(self):
        polling, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def status(_job_id):
            polling.set()
            await asyncio.Event().wait()
        async def cancel(_job_id):
            cancelling.set()
            await release.wait()
            return {"status": "CANCELLED"}
        self.api.status.side_effect = status
        self.api.cancel.side_effect = cancel
        await self.manager.start("exec", "prompt", [self.prepared])
        await polling.wait()
        first = asyncio.create_task(self.manager.cancel(worker_id="pool"))
        await cancelling.wait()
        second = asyncio.create_task(self.manager.cancel(worker_id="pool"))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)
        self.api.cancel.assert_awaited_once_with("job")
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCELLED")

    async def test_partial_start_cleanup_does_not_leave_waiting_protection(self):
        broken = self.second()
        del broken["worker"]["endpoint_id"]
        with self.assertRaises(KeyError):
            await self.manager.start("exec", "prompt", [self.prepared, broken])
        await self.manager.cancel(execution_id="exec")
        self.assertFalse(self.manager.pool_is_active("pool"))
        self.api.submit.assert_not_awaited()

    async def test_public_jobs_never_hides_unresolved_recovery_ids(self):
        for index in range(22):
            self.manager.ledger.update(str(index), worker_id="pool", endpoint_id="ep", status="UNKNOWN", updated_at=index)
        for index in range(5):
            self.manager.ledger.update("done-" + str(index), worker_id="pool", status="COMPLETED", updated_at=30 + index)
        records = self.manager.public_jobs("pool")
        self.assertEqual({record["record_id"] for record in records}, {str(index) for index in range(22)})
        self.assertTrue(all(record["status"] == "UNKNOWN" for record in records))
