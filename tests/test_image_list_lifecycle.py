"""Tests for periodic image-list server lifecycle cleanup."""
import asyncio
import time
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from aiohttp import web

from image_list import lifecycle, runtime


class ExecutionTargetProofTests(unittest.TestCase):
    def setUp(self):
        from test_image_list_master_runtime import graph
        self.prompt = graph()
        self.registry = SimpleNamespace(NODE_CLASS_MAPPINGS={
            'Operation': type('Flagged', (), {'OUTPUT_NODE': True})})
        self.modules = patch.dict('sys.modules', {'nodes': self.registry})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def server(self, roots, *, prompt=None, running=True):
        item = (0, 'owned', self.prompt if prompt is None else prompt, {}, roots, {})
        return SimpleNamespace(prompt_queue=SimpleNamespace(
            get_current_queue_volatile=lambda: ([item], []) if running else ([], [item])))

    def test_outer_roots_in_running_queue_are_authoritative(self):
        self.assertTrue(lifecycle.has_safe_execution_targets(self.server(['end']), 'owned', self.prompt))
        for roots in (['op', 'end'], [], ['end', 'end'], None, [True]):
            with self.subTest(roots=roots):
                self.assertFalse(lifecycle.has_safe_execution_targets(self.server(roots), 'owned', self.prompt))
        self.assertFalse(lifecycle.has_safe_execution_targets(self.server(['end']), 'other', self.prompt))
        self.assertFalse(lifecycle.has_safe_execution_targets(self.server(['end'], running=False), 'owned', self.prompt))
        self.assertFalse(lifecycle.has_safe_execution_targets(self.server(['end'], prompt={}), 'owned', self.prompt))

    def test_prompt_marker_is_not_an_execution_guarantee(self):
        self.prompt['start']['inputs']['image_list_explicit_targets'] = True
        self.assertFalse(lifecycle.has_safe_execution_targets(self.server(['op', 'end']), 'owned', self.prompt))
        with self.assertRaisesRegex(ValueError, 'must be placed after'):
            runtime.create_master_job('start', 'end', self.prompt, [], [], '', master_participates=True)
        self.assertEqual(runtime._MASTER_JOBS, {})

    def test_start_node_uses_running_queue_proof(self):
        import importlib.util
        import sys
        from pathlib import Path
        from types import ModuleType
        import torch

        root = Path(__file__).resolve().parents[1]
        package = '_image_list_start_targets'
        modules = {}
        for suffix in ('', '.nodes', '.image_list', '.utils'):
            module = ModuleType(package + suffix)
            module.__path__ = [str(root / suffix.lstrip('.'))]
            modules[package + suffix] = module
        modules[package + '.image_list.runtime'] = runtime
        modules[package + '.image_list.lifecycle'] = lifecycle
        modules[package + '.utils.async_helpers'] = SimpleNamespace(run_async_in_server_loop=None)
        io = SimpleNamespace(ComfyNode=type('ComfyNode', (), {}), NodeOutput=lambda *args: args,
                             **{name: object() for name in ('Image', 'Mask', 'String', 'Int', 'Float', 'Boolean')})
        modules['comfy_api.v0_0_2'] = SimpleNamespace(io=io)
        modules['comfy_execution.utils'] = SimpleNamespace(
            get_executing_context=lambda: SimpleNamespace(prompt_id='owned'))
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=self.server(['end'])))
        modules['server'] = server
        spec = importlib.util.spec_from_file_location(package + '.nodes.image_list', root / 'nodes/image_list.py')
        node_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(node_module)
            start = node_module.DistributedImageListStart
            start.hidden = SimpleNamespace(prompt=self.prompt, unique_id='start')
            result = start.execute(images=torch.zeros((1, 2, 2, 3)), image_list_master_participates=True)
            context = result[3][0]
            self.assertIsNotNone(runtime.get_master_job(context))
            runtime.remove_master_job(context)
            server.PromptServer.instance = self.server(['op', 'end'])
            with self.assertRaisesRegex(ValueError, 'must be placed after'):
                start.execute(images=torch.zeros((1, 2, 2, 3)), image_list_master_participates=True)
        self.assertEqual(runtime._MASTER_JOBS, {})

    def test_owned_target_controlled_start_can_create_job(self):
        context = runtime.create_master_job('start', 'end', self.prompt, [], [], '',
                                            master_participates=True, explicit_targets=True)
        self.addCleanup(runtime.remove_master_job, context)
        self.assertIsNotNone(runtime.get_master_job(context))


class ImageListLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with runtime._LOCK:
            runtime._MASTER_JOBS.clear()
            runtime._WORKER_ITEMS.clear()

    def tearDown(self):
        with runtime._LOCK:
            runtime._MASTER_JOBS.clear()
            runtime._WORKER_ITEMS.clear()

    @staticmethod
    def _state(prompt_id=None, *, active=False, age=10_000.0):
        return SimpleNamespace(
            prompt_id=prompt_id,
            active=active,
            created_at=time.monotonic() - age,
        )

    @staticmethod
    def _server(queue):
        return SimpleNamespace(app=web.Application(), prompt_queue=queue)

    async def _start(self, server, *, interval=0.01, grace=0.01):
        lifecycle.install_lifecycle(
            server,
            runtime,
            interval_seconds=interval,
            orphan_grace_seconds=grace,
        )
        runner = web.AppRunner(server.app)
        await runner.setup()
        return runner

    async def test_unattended_sweep_removes_orphan_and_expired_staging(self):
        context_id = "orphan"
        with runtime._LOCK:
            cast(Any, runtime._MASTER_JOBS)[context_id] = self._state()
            runtime._WORKER_ITEMS["expired-token"] = (time.monotonic() - 1, {"image": "stale"})

        queue = SimpleNamespace(
            get_current_queue_volatile=lambda: ([], []),
        )
        server = self._server(queue)
        swept = asyncio.Event()
        sweep_worker_items = runtime._sweep_worker_items

        def track_worker_sweep():
            swept.set()
            sweep_worker_items()

        with patch.object(runtime, "_sweep_worker_items", side_effect=track_worker_sweep):
            runner = await self._start(server)
            try:
                await asyncio.wait_for(swept.wait(), timeout=0.5)
                self.assertNotIn(context_id, runtime._MASTER_JOBS)
                self.assertNotIn("expired-token", runtime._WORKER_ITEMS)
                with self.assertRaisesRegex(ValueError, "Unknown or expired"):
                    runtime.set_master_result(context_id, 0, "late-token", {})
            finally:
                await runner.cleanup()

    async def test_old_active_and_queued_contexts_survive_but_orphans_expire(self):
        states = {
            "active": self._state("active-prompt", active=True),
            "queued": self._state("queued-prompt"),
            "orphan": self._state("missing-prompt"),
            "unbound": self._state(None),
        }
        with runtime._LOCK:
            cast(Any, runtime._MASTER_JOBS).update(states)
        queue = SimpleNamespace(
            get_current_queue_volatile=lambda: (
                [[1, "active-prompt", {}]],
                [(2, "queued-prompt", {}, {}, None, None)],
            ),
        )
        runner = await self._start(self._server(queue))
        try:
            await asyncio.sleep(0.04)
            self.assertIs(runtime._MASTER_JOBS.get("active"), states["active"])
            self.assertIs(runtime._MASTER_JOBS.get("queued"), states["queued"])
            self.assertNotIn("orphan", runtime._MASTER_JOBS)
            self.assertNotIn("unbound", runtime._MASTER_JOBS)
        finally:
            await runner.cleanup()

    async def test_falls_back_to_nonvolatile_queue_reader(self):
        state = self._state("queued-prompt")
        with runtime._LOCK:
            cast(Any, runtime._MASTER_JOBS)["queued"] = state
        fallback_reads = 0

        def volatile_failure():
            raise RuntimeError("volatile view unavailable")

        def current_queue():
            nonlocal fallback_reads
            fallback_reads += 1
            return ([], [[1, "queued-prompt", {}]])

        queue = SimpleNamespace(
            get_current_queue_volatile=volatile_failure,
            get_current_queue=current_queue,
        )
        runner = await self._start(self._server(queue))
        try:
            await asyncio.sleep(0.04)
            self.assertGreaterEqual(fallback_reads, 1)
            self.assertIs(runtime._MASTER_JOBS.get("queued"), state)
        finally:
            await runner.cleanup()

    async def test_queue_read_error_protects_bound_contexts(self):
        with runtime._LOCK:
            cast(Any, runtime._MASTER_JOBS)["bound"] = self._state("bound-prompt")
            cast(Any, runtime._MASTER_JOBS)["unbound"] = self._state(None)

        def fail_read():
            raise RuntimeError("queue unavailable")

        queue = SimpleNamespace(
            get_current_queue_volatile=fail_read,
            get_current_queue=fail_read,
        )
        runner = await self._start(self._server(queue))
        try:
            await asyncio.sleep(0.04)
            self.assertIn("bound", runtime._MASTER_JOBS)
            self.assertNotIn("unbound", runtime._MASTER_JOBS)
        finally:
            await runner.cleanup()

    async def test_install_and_removal_are_idempotent_and_task_stops_on_cleanup(self):
        queue_reads = 0

        def read_queue():
            nonlocal queue_reads
            queue_reads += 1
            return ([], [])

        server = self._server(SimpleNamespace(get_current_queue_volatile=read_queue))
        lifecycle.install_lifecycle(server, runtime, interval_seconds=0.01, orphan_grace_seconds=0.01)
        lifecycle.install_lifecycle(server, runtime, interval_seconds=0.01, orphan_grace_seconds=0.01)
        self.assertEqual(len(server.app.cleanup_ctx), 1)

        with runtime._LOCK:
            cast(Any, runtime._MASTER_JOBS)["removed"] = self._state()
        runtime.remove_master_job("removed")
        runtime.remove_master_job("removed")
        self.assertNotIn("removed", runtime._MASTER_JOBS)

        runner = web.AppRunner(server.app)
        await runner.setup()
        await asyncio.sleep(0.035)
        self.assertGreaterEqual(queue_reads, 1)
        await runner.cleanup()
        reads_after_cleanup = queue_reads
        await asyncio.sleep(0.035)
        self.assertEqual(queue_reads, reads_after_cleanup)


if __name__ == "__main__":
    unittest.main()
