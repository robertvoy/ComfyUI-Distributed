"""Offline image-list queue/scheduler integration; no provider or GPU calls."""
import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import torch

import test_image_list_master_runtime as direct_fixture
from test_serverless_integration import PKG, load_modules


def pool(**updates):
    return dict({"id": "pool", "name": "Cloud", "type": "serverless", "enabled": True,
                 "provider": "runpod", "endpoint_id": "endpoint", "jobs_per_run": 16,
                 "timeout_seconds": 30, "media_profile": "comfy-media-v1"}, **updates)


class MixedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reuse only fixture setup, not inherited tests. The transport/slot doubles
        # execute real runtime staging, assignment and result ordering code.
        self.fixture = direct_fixture.MasterRuntimeTests()
        self.fixture.setUp()
        # This reused TestCase has no asyncio runner: invoke its patch cleanup
        # directly, not doCleanups(), which would silently fail and leak aiohttp.
        self.addCleanup(self.fixture.modules.stop)
        self.f = f = self.fixture
        self.rt = f.rt
        self.configured_pool = pool()
        f.config["workers"] = [self.configured_pool]
        self.cloud_calls = []
        self.completed = []
        self.closed = 0
        self.hold = False
        self.missing = False
        self.dispatch_error = False
        self.poll_entered = asyncio.Event()
        self.dispatch_entered = asyncio.Event()
        self.hold_dispatch = False
        self.slot_count = 2
        fixture = self

        class Controller:
            def __init__(self, state, event):
                self.state, self.event = state, event
                self.slots = [("serverless", "pool", index) for index in range(fixture.slot_count)]
                self.tasks = []
                self.closed = False

            async def dispatch(self, slot, index):
                assert not self.closed
                fixture.cloud_calls.append((slot, index))
                fixture.dispatch_entered.set()
                if fixture.dispatch_error:
                    raise RuntimeError("serverless dispatch failed")

                async def execute():
                    if fixture.hold:
                        await asyncio.Event().wait()
                    # Make a later source item complete before the first local one.
                    await asyncio.sleep(0.001)
                    fixture.completed.append(index)
                    if fixture.missing:
                        raise RuntimeError("serverless completed without an image result")
                    item = state.items[index]
                    return {"image": item["image"], "mask": item["mask"],
                            "text": item["values"]["value_1"]}

                state = self.state
                task = asyncio.create_task(execute())
                self.tasks.append(task)
                task.add_done_callback(lambda _: self.event.set())
                # Own the task before any await that might be cancelled.
                if fixture.hold_dispatch:
                    await asyncio.Event().wait()
                return {"index": index, "task": task}

            def poll(self, inflight):
                fixture.poll_entered.set()
                for slot, handle in inflight.items():
                    if slot in self.slots and handle["task"].done():
                        return slot, handle["index"], handle["task"].result()
                return None

            async def close(self):
                self.closed = True
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self.tasks, return_exceptions=True)
                fixture.closed += 1

        async def create(state, region, config, event):
            self.assertEqual(region["end_id"], "end")
            self.assertEqual(config["workers"][-1]["id"], "pool")
            self.controller = Controller(state, event)
            return self.controller

        self.create_controller = AsyncMock(side_effect=create)
        sys.modules['_master_slot_tests.image_list.serverless'] = types.SimpleNamespace(
            create_serverless_slots=self.create_controller)
        self.direct_dispatch = sys.modules['_master_slot_tests.api.orchestration.dispatch'].dispatch_worker_prompt

    def create(self, *, master=True, direct=True, count=9):
        f = self.f
        if direct:
            f.workers.append({"id": "peer", "type": "local"})
        f.config["workers"] = f.workers + [self.configured_pool]
        f.local_delay = 0.04
        context = f.create(participates=master, count=count) if (master or direct) else None
        if context is None:
            # Fixture.create assumes only direct peers; cloud-only uses a pool ID.
            f.items = self.rt.normalize_image_list_inputs(torch.zeros((count, 2, 2, 3)),
                mask=torch.zeros((count, 2, 2)), values={"value_1": [str(i) for i in range(count)]})
            context = self.rt.create_master_job('start', 'end', direct_fixture.graph(), f.items, ['pool'], '')
            f.context = context
            f.state = self.rt.get_master_job(context)
        else:
            f.state.worker_ids.append('pool')
        return context

    async def test_mixed_roles_share_one_queue_and_preserve_pairing(self):
        context = self.create()
        original = copy.deepcopy(self.f.state.original_prompt)
        result = await self.rt.run_image_list_job(context)
        indices = self.f.local_calls + self.f.dispatched + [i for _, i in self.cloud_calls]
        self.assertEqual(sorted(indices), list(range(9)))
        self.assertTrue(self.f.local_calls and self.f.dispatched and self.cloud_calls)
        self.assertNotEqual(self.completed[0], 0)
        self.assertEqual(result[2], [str(i) for i in range(9)])
        for index, (image, mask) in enumerate(zip(*result[:2])):
            self.assertTrue(torch.equal(image, self.f.items[index]['image']))
            self.assertTrue(torch.equal(mask, self.f.items[index]['mask']))
        self.assertEqual(self.f.state.original_prompt, original)
        self.f.preflight.assert_awaited_once_with(self.f.workers, require_idle=False)
        for call in self.direct_dispatch.await_args_list:
            self.assertEqual(call.kwargs['partial_execution_targets'], ['end'])
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.rt._MASTER_JOBS)

    async def test_cloud_only_one_job_per_item_ignores_images_per_run(self):
        for jobs_per_run in (1, 16):
            with self.subTest(jobs_per_run=jobs_per_run):
                self.configured_pool['jobs_per_run'] = jobs_per_run
                self.cloud_calls.clear()
                result = await self.rt.run_image_list_job(self.create(master=False, direct=False, count=7))
                self.assertEqual([i for _, i in self.cloud_calls], list(range(7)))
                self.assertEqual(result[2], [str(i) for i in range(7)])
                self.assertEqual(len({slot for slot, _ in self.cloud_calls}), self.slot_count)
        self.f.preflight.assert_not_awaited()
        self.f.network.get_client_session.assert_not_awaited()
        self.assertFalse(self.f.slot_instances)

    async def test_missing_cloud_result_fails_without_refill(self):
        self.missing = True
        with self.assertRaisesRegex(RuntimeError, 'without an image result'):
            await self.rt.run_image_list_job(self.create(master=False, direct=False))
        self.assertEqual(len(self.cloud_calls), self.slot_count)
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.rt._MASTER_JOBS)

    async def test_controller_closed_when_direct_preflight_fails(self):
        self.f.preflight.side_effect = RuntimeError('direct preflight failed')
        with self.assertRaisesRegex(RuntimeError, 'direct preflight failed'):
            await self.rt.run_image_list_job(self.create())
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.cloud_calls)

    async def test_controller_closed_when_local_setup_fails(self):
        slot_module = sys.modules['_master_slot_tests.image_list.local_executor']
        with patch.object(slot_module, 'LocalExecutionSlot', side_effect=RuntimeError('local setup failed')):
            with self.assertRaisesRegex(RuntimeError, 'local setup failed'):
                await self.rt.run_image_list_job(self.create())
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.cloud_calls)

    async def test_controller_closed_on_dispatch_failure(self):
        self.dispatch_error = True
        with self.assertRaisesRegex(RuntimeError, 'serverless dispatch failed'):
            await self.rt.run_image_list_job(self.create(master=False, direct=False))
        self.assertEqual(len(self.cloud_calls), 1)
        self.assertEqual(self.closed, 1)

    async def test_cancelled_dispatch_without_returned_handle_is_owned(self):
        self.hold = self.hold_dispatch = True
        with patch.object(self.rt, 'SUBMISSION_CANCEL_GRACE_SECONDS', 0.01):
            task = asyncio.create_task(self.rt.run_image_list_job(self.create(master=False, direct=False)))
            await asyncio.wait_for(self.dispatch_entered.wait(), 1)
            task.cancel()
            with self.assertRaises(self.rt.ImageListCancelledError):
                await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.cloud_calls), 1)
        self.assertEqual(self.closed, 1)
        self.assertTrue(all(t.done() for t in self.controller.tasks))

    async def interrupt_probe(self, models):
        self.hold = True
        task = asyncio.create_task(self.rt.run_image_list_job(self.create(master=False, direct=False)))
        try:
            await asyncio.wait_for(self.poll_entered.wait(), 1)
            assigned = list(self.cloud_calls)
            models.interrupt_current_processing(True)
            with self.assertRaises(self.rt.ImageListCancelledError):
                await asyncio.wait_for(task, 1)
            self.assertEqual(self.cloud_calls, assigned)
            self.assertEqual(self.closed, 1)
            self.assertTrue(all(t.done() for t in self.controller.tasks))
            self.assertFalse(self.rt._MASTER_JOBS)
        finally:
            models.interrupt_current_processing(False)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


class QueueSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.q, modules, self.ps = load_modules()
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.pool = pool()
        self.peer = {'id': 'peer', 'name': 'Peer', 'type': 'local', 'enabled': True}
        self.config = {'workers': [self.pool], 'settings': {'master_delegate_only': True}}
        self.q.load_config = lambda: self.config
        self.q.select_active_workers = AsyncMock(side_effect=lambda workers, ws, delegate, **kw: (workers, delegate))
        self.preflight = AsyncMock(side_effect=lambda workers, **kw: {w['id']: True for w in workers})
        sys.modules[PKG + '.api.orchestration.dispatch'].preflight_image_list_workers = self.preflight
        self.q.queue_prompt_payload = AsyncMock(return_value={'prompt_id': 'outer', 'number': 1})
        self.prompt = direct_fixture.graph()

    async def queue(self, **kwargs):
        return await self.q.orchestrate_distributed_execution(self.prompt, None, 'client', **kwargs)

    async def test_delegate_pool_selected_once_no_direct_probe_or_source_mutation(self):
        original = copy.deepcopy(self.prompt)
        result = await self.queue()
        selected = self.q.select_active_workers.await_args.args[0]
        self.assertEqual([w['id'] for w in selected], ['pool'])
        prepared = self.q.queue_prompt_payload.await_args.args[0]
        inputs = prepared['start']['inputs']
        self.assertEqual(json.loads(inputs['image_list_enabled_worker_ids']), ['pool'])
        self.assertIs(inputs['image_list_master_participates'], False)
        self.assertEqual(self.q.queue_prompt_payload.await_args.kwargs['partial_execution_targets'], ['end'])
        self.assertEqual(result[2], 1)
        self.assertEqual(self.prompt, original)
        self.preflight.assert_not_awaited()
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.assertEqual(self.ps.distributed_pending_jobs, {})

    async def test_mixed_selection_preflights_only_direct_peers(self):
        self.config['workers'].insert(0, self.peer)
        result = await self.queue(delegate_master=False)
        self.assertEqual([w['id'] for w in self.preflight.await_args.args[0]], ['peer'])
        self.assertFalse(self.preflight.await_args.kwargs['require_idle'])
        inputs = self.q.queue_prompt_payload.await_args.args[0]['start']['inputs']
        self.assertEqual(json.loads(inputs['image_list_enabled_worker_ids']), ['peer', 'pool'])
        self.assertEqual(result[2], 2)

    async def test_legacy_pool_rejected_before_probe_or_queue(self):
        self.pool['media_profile'] = 'legacy-image-v1'
        with self.assertRaisesRegex((ValueError, RuntimeError), 'comfy-media-v1'):
            await self.queue()
        self.q.select_active_workers.assert_not_awaited()
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_unknown_selection_rejected_before_queue(self):
        with self.assertRaisesRegex((ValueError, RuntimeError), 'Unknown|unknown'):
            await self.queue(enabled_worker_ids=['missing'])
        self.q.select_active_workers.assert_not_awaited()
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_disabled_pool_can_be_explicitly_selected(self):
        self.pool['enabled'] = False
        await self.queue(enabled_worker_ids=['pool'])
        self.assertEqual([w['id'] for w in self.q.select_active_workers.await_args.args[0]], ['pool'])

    async def test_explicit_empty_selection_does_not_import_cloud_adapter(self):
        sys.modules[PKG + '.image_list.serverless'] = None
        result = await self.queue(enabled_worker_ids=[], delegate_master=False)
        self.assertEqual(result[2], 0)
        self.preflight.assert_not_awaited()

    async def test_cloud_auth_or_capacity_failure_never_queues_master(self):
        for error in ('authentication failed', 'capacity exhausted'):
            with self.subTest(error=error):
                self.q.select_active_workers.side_effect = ValueError(error)
                with self.assertRaisesRegex(ValueError, error):
                    await self.queue()
                self.q.queue_prompt_payload.assert_not_awaited()

    async def test_real_selection_uses_manager_health_not_direct_http(self):
        from test_dispatch_selection import _load_dispatch_module
        dispatch = _load_dispatch_module()
        health = AsyncMock(return_value={'ok': True, 'queueRemaining': 3})
        sys.modules['dist_dispatch_testpkg.api.orchestration.serverless_runtime'] = types.SimpleNamespace(
            get_manager=lambda: types.SimpleNamespace(health=health))
        self.q.select_active_workers = dispatch.select_active_workers
        with patch.object(dispatch, 'get_client_session', AsyncMock(side_effect=AssertionError('no HTTP'))), \
                patch.object(dispatch, 'probe_worker', AsyncMock(side_effect=AssertionError('no direct probe'))):
            await self.queue()
            health.assert_awaited_once()
            self.assertEqual(health.await_args.args[0]['id'], 'pool')
            self.preflight.assert_not_awaited()
            self.q.queue_prompt_payload.reset_mock()
            health.side_effect = ValueError('provider authentication failed')
            with self.assertRaisesRegex(ValueError, 'authentication failed'):
                await self.queue()
            self.q.queue_prompt_payload.assert_not_awaited()

    async def test_unreachable_pool_never_enables_master_fallback(self):
        self.q.select_active_workers.side_effect = None
        self.q.select_active_workers.return_value = ([], False)
        with self.assertRaisesRegex(RuntimeError, 'orchestrator|Orchestrator'):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()


class RealInterruptTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('COMFYUI_SOURCE_ROOT'), 'set COMFYUI_SOURCE_ROOT for CPU interrupt probe')
    def test_processing_interrupted_signal_during_cloud_result_wait(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--interrupt-probe',
                                 os.environ['COMFYUI_SOURCE_ROOT']], text=True, capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('SERVERLESS_INTERRUPT_OK', result.stdout)


def real_interrupt_probe(comfy_root):
    sys.path.insert(0, str(Path(comfy_root).resolve()))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.disable_assets = True
    import comfy.model_management as models
    real_comfy = sys.modules['comfy']
    case = MixedRuntimeTests()
    case.setUp()
    try:
        # Keep actual thread-safe ComfyUI interruption state, not a task.cancel
        # surrogate or a fake processing_interrupted callback.
        sys.modules['comfy'] = real_comfy
        sys.modules['comfy.model_management'] = models
        asyncio.run(case.interrupt_probe(models))
    finally:
        case.f.modules.stop()
    print('SERVERLESS_INTERRUPT_OK no refill, owned close, actual ComfyUI flag')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--interrupt-probe':
        real_interrupt_probe(sys.argv[2])
    else:
        unittest.main()
