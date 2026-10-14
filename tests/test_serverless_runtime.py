import asyncio
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "dist_serverless_runtime_tests"
for name, path in [(PACKAGE, ROOT), (PACKAGE + ".api", ROOT / "api"), (PACKAGE + ".api.orchestration", ROOT / "api/orchestration")]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + ".api.orchestration." + name, ROOT / "api/orchestration" / (name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

runtime = load("serverless_runtime")

class Secrets:
    def get(self, worker_id):
        return "private-test-key"

class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = types.SimpleNamespace(distributed_pending_jobs={"collector-job": asyncio.Queue()}, distributed_jobs_lock=asyncio.Lock())
        self.manager = runtime.ServerlessManager(self.server, Secrets(), Path(self.temp.name) / "jobs.json", AsyncMock(return_value=object()))
        self.manager.poll_seconds = 0
        self.worker = {"id": "pool:1", "pool_id": "pool", "endpoint_id": "ep1", "timeout_seconds": 30, "type": "serverless"}
        self.prepared = {"worker": self.worker, "workflow": {"1": {}}, "images": [], "manifest": [{"job_id": "collector-job", "prefix": "dist_x", "kind": "image"}]}

    async def asyncTearDown(self):
        await self.manager.cancel()
        self.temp.cleanup()

    async def test_job_id_persisted_before_status_and_results_injected(self):
        api = types.SimpleNamespace(submit=AsyncMock(return_value="provider-job"), cancel=AsyncMock())
        async def status(job_id):
            self.assertEqual(self.manager.ledger.all()[0]["job_id"], "provider-job")
            return {"status": "COMPLETED", "output": {"images": []}}
        api.status = status
        with patch.object(runtime, "RunpodClient", return_value=api), patch.object(runtime, "decode_results", AsyncMock(return_value={"collector-job": ["tensor"]})):
            await self.manager.start("execution", "prompt", [self.prepared])
            await self.manager.wait_for_execution("execution")
        self.assertEqual(api.submit.await_count, 1)
        item = self.server.distributed_pending_jobs["collector-job"].get_nowait()
        self.assertEqual(item["worker_id"], "pool:1")
        self.assertEqual(item["tensor"], "tensor")
        self.assertTrue(item["is_last"])
        self.assertEqual(self.manager.ledger.all()[0]["status"], "COMPLETED")

    async def test_output_error_marks_failed_and_delivers_explicit_error(self):
        api = types.SimpleNamespace(submit=AsyncMock(return_value="provider-job"), status=AsyncMock(return_value={"status": "COMPLETED", "output": {"error": "private-test-key"}}), cancel=AsyncMock())
        with patch.object(runtime, "RunpodClient", return_value=api):
            await self.manager.start("execution", "prompt", [self.prepared])
            await self.manager.wait_for_execution("execution")
        record = self.manager.ledger.all()[0]
        self.assertEqual(record["status"], "FAILED")
        item = self.server.distributed_pending_jobs["collector-job"].get_nowait()
        self.assertIn("serverless_error", item)
        self.assertNotIn("private-test-key", str(record) + str(item))

    async def test_ambiguous_submit_never_retries(self):
        api = types.SimpleNamespace(submit=AsyncMock(side_effect=runtime.AmbiguousSubmission("Unknown submission")), cancel=AsyncMock())
        with patch.object(runtime, "RunpodClient", return_value=api):
            await self.manager.start("execution", "prompt", [self.prepared])
            await self.manager.wait_for_execution("execution")
        self.assertEqual(api.submit.await_count, 1)
        self.assertEqual(self.manager.ledger.all()[0]["status"], "UNKNOWN")

    async def test_cancel_targets_owned_job_only(self):
        polled = asyncio.Event()
        async def status(_job):
            polled.set()
            await asyncio.Event().wait()
        api = types.SimpleNamespace(submit=AsyncMock(return_value="owned-job"), status=status, cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        with patch.object(runtime, "RunpodClient", return_value=api):
            await self.manager.start("execution", "prompt", [self.prepared])
            await polled.wait()
            await self.manager.cancel(worker_id="other-pool")
            api.cancel.assert_not_awaited()
            await self.manager.cancel(worker_id="pool")
        api.cancel.assert_awaited_once_with("owned-job")

    async def test_all_images_of_a_run_submit_at_once_ignoring_legacy_concurrency(self):
        self.assertEqual(runtime.PER_WORKER_JOB_LIMIT, 16)
        gate = asyncio.Event()
        submitted = []
        async def submit(*args):
            submitted.append(args)
            return f"provider-{len(submitted)}"
        async def status(_job):
            await gate.wait()
            return {"status": "COMPLETED", "output": {"images": []}}
        api = types.SimpleNamespace(submit=submit, status=status, cancel=AsyncMock())
        # A config saved before the setting was removed still carries concurrency=1.
        legacy = dict(self.worker, concurrency=1)
        jobs = [dict(self.prepared, worker=dict(legacy, id=f"pool:{i}")) for i in range(1, 4)]
        with patch.object(runtime, "RunpodClient", return_value=api), patch.object(runtime, "decode_results", AsyncMock(return_value={"collector-job": ["tensor"]})):
            await self.manager.start("execution", "prompt", jobs)
            for _ in range(50):
                if len(submitted) == 3:
                    break
                await asyncio.sleep(0)
            self.assertEqual(len(submitted), 3)
            gate.set()
            await self.manager.wait_for_execution("execution")

    async def test_per_pool_job_limit_bounds_submission(self):
        patcher = patch.object(runtime, "PER_WORKER_JOB_LIMIT", 1)
        patcher.start()
        self.addCleanup(patcher.stop)
        gate = asyncio.Event()
        reached = asyncio.Event()
        count = 0
        async def submit(*args):
            nonlocal count
            count += 1
            reached.set()
            return f"provider-{count}"
        async def status(_job):
            await gate.wait()
            return {"status": "COMPLETED", "output": {"images": []}}
        api = types.SimpleNamespace(submit=submit, status=status, cancel=AsyncMock())
        second = dict(self.prepared, worker=dict(self.worker, id="pool:2"))
        with patch.object(runtime, "RunpodClient", return_value=api), patch.object(runtime, "decode_results", AsyncMock(return_value={"collector-job": ["tensor"]})):
            await self.manager.start("execution", "prompt", [self.prepared, second])
            await reached.wait()
            await asyncio.sleep(0)
            self.assertEqual(count, 1)
            gate.set()
            await self.manager.wait_for_execution("execution")
        self.assertEqual(count, 2)

    async def test_cancel_persisted_job_after_master_restart(self):
        self.manager.ledger.update("old", worker_id="pool", endpoint_id="old-endpoint", job_id="old-job", status="IN_PROGRESS", created_at=1, updated_at=1)
        api = types.SimpleNamespace(cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        with patch.object(runtime, "RunpodClient", return_value=api) as client:
            await self.manager.cancel(worker_id="pool")
        api.cancel.assert_awaited_once_with("old-job")
        self.assertEqual(client.call_args.args[1], "old-endpoint")
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCELLED")

    async def test_master_failure_cancels_job_without_partial_success(self):
        self.server.prompt_queue = types.SimpleNamespace(get_history=lambda _id: {"prompt": {"status": {"status_str": "error"}}})
        api = types.SimpleNamespace(submit=AsyncMock(return_value="job"), cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        with patch.object(runtime, "RunpodClient", return_value=api):
            await self.manager.start("execution", "prompt", [self.prepared])
            await self.manager.wait_for_execution("execution")
        api.submit.assert_not_awaited()
        self.assertEqual(self.manager.ledger.all()[0]["status"], "FAILED")

    async def test_deadline_includes_local_slot_wait(self):
        self.prepared["worker"] = dict(self.worker, timeout_seconds=0.01)
        api = types.SimpleNamespace(submit=AsyncMock(return_value="job"), status=AsyncMock(side_effect=lambda _: asyncio.sleep(5)), cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        # A blocked slot expires without a billable submission.
        self.manager.pool_limits["pool"] = asyncio.Semaphore(0)
        with patch.object(runtime, "RunpodClient", return_value=api):
            await self.manager.start("execution", "prompt", [self.prepared])
            await self.manager.wait_for_execution("execution")
        api.submit.assert_not_awaited()
        self.assertEqual(self.manager.ledger.all()[0]["status"], "TIMED_OUT")
