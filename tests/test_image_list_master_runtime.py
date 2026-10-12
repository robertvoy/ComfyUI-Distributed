"""Master slot regressions, plus an opt-in real CPU ComfyUI subprocess probe."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def graph():
    return {
        'source': {'class_type': 'Source', 'inputs': {}},
        'start': {'class_type': 'DistributedImageListStart', 'inputs': {'images': ['source', 0]}},
        'op': {'class_type': 'Operation', 'inputs': {'image': ['start', 0]}},
        'end': {'class_type': 'DistributedImageListEnd', 'inputs': {
            'images': ['op', 0], 'context': ['start', 7]}},
    }


class Response:
    status = 200

    def __init__(self, data=None):
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def json(self):
        return self.data

    def raise_for_status(self):
        pass


class MasterRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        name = '_master_slot_tests'
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for suffix in ('', '.image_list', '.api', '.api.orchestration', '.utils'):
            module = types.ModuleType(name + suffix)
            module.__path__ = [str(ROOT / suffix.lstrip('.').replace('.', '/'))]
            sys.modules[name + suffix] = module
        self.rt = importlib.import_module(name + '.image_list.runtime')
        self.workers = []
        self.preflight = AsyncMock(side_effect=lambda workers, **kw: {w['id']: True for w in workers})
        self.dispatched = []
        self.local_calls = []
        self.tokens = []
        self.closed = 0
        self.local_error = None
        self.local_delay = 0.005
        self.interrupted = False
        self.local_started = asyncio.Event()
        self.local_finished = asyncio.Event()
        self.deadline = 0
        self.slot_instances = []
        self.config = {'workers': self.workers, 'settings': {}}
        fixture = self

        class FakeSlot:
            def __init__(self, model_callback=None):
                fixture.slot_instances.append(self)
                self.tasks = []

            def submit(self, prompt, prompt_id, end_id, workflow_meta=None):
                async def execute():
                    inputs = prompt[end_id]['inputs']
                    index = inputs['worker_item_index']
                    token = inputs['worker_result_token']
                    fixture.local_calls.append(index)
                    fixture.tokens.append(token)
                    fixture.local_started.set()
                    try:
                        await asyncio.sleep(fixture.local_delay)
                        if fixture.local_error:
                            raise fixture.local_error
                        item = fixture.rt.consume_worker_item(token)
                        fixture.rt.set_master_result(fixture.context, index, token, {
                            'image': item['image'], 'mask': item.get('mask'), 'text': item.get('text')})
                    finally:
                        fixture.local_finished.set()
                task = asyncio.create_task(execute())
                self.tasks.append(task)
                return task

            def request_stop(self):
                pass

            async def close(self):
                # Model a synchronous node that must finish before its thread can join.
                await asyncio.gather(*self.tasks, return_exceptions=True)
                fixture.closed += 1

        dependencies = {
            '.image_list.local_executor': types.SimpleNamespace(LocalExecutionSlot=FakeSlot),
            '.api.orchestration.dispatch': types.SimpleNamespace(
                preflight_image_list_workers=self.preflight,
                dispatch_worker_prompt=AsyncMock(side_effect=self.dispatch)),
            '.api.orchestration.prompt_transform': types.SimpleNamespace(
                _resolve_direct_workers_for_image_list=lambda config, ids: self.workers),
            '.api.orchestration.media_sync': types.SimpleNamespace(
                sync_worker_media=AsyncMock(), fetch_worker_path_separator=AsyncMock(return_value=None)),
            '.utils.config': types.SimpleNamespace(load_config=lambda: self.config,
                get_worker_timeout_seconds=lambda: 60,
                get_image_list_job_timeout_seconds=lambda: self.deadline),
            '.utils.network': types.SimpleNamespace(
                build_master_callback_url=lambda *a, **kw: 'http://master',
                build_worker_url=lambda worker, path: 'http://remote' + path,
                get_client_session=AsyncMock(return_value=self)),
            '.utils.logging': types.SimpleNamespace(log=lambda *args: None),
        }
        for suffix, module in dependencies.items():
            sys.modules[name + suffix] = module
        self.network = dependencies['.utils.network']
        sys.modules['aiohttp'] = types.SimpleNamespace(ClientTimeout=lambda **kw: kw, ClientError=ConnectionError)
        comfy = types.ModuleType('comfy')
        comfy.model_management = types.SimpleNamespace(processing_interrupted=lambda: self.interrupted)
        sys.modules['comfy'] = comfy
        sys.modules['comfy.model_management'] = comfy.model_management

    def create(self, participates=True, count=5):
        self.items = [{'image': torch.full((1, i + 1, 2, 3), float(i)),
                       'mask': torch.full((1, i + 1, 2), float(i)), 'text': str(i), 'index': i}
                      for i in range(count)]
        self.context = self.rt.create_master_job('start', 'end', graph(), self.items,
            [w['id'] for w in self.workers], '', master_participates=participates)
        self.state = self.rt.get_master_job(self.context)
        return self.context

    def post(self, url, *, data=None, **kwargs):
        if url.endswith('/distributed/image-list/item'):
            payload = json.loads(data)
            self.rt.store_worker_item(payload['token'], payload['item'])
        return Response({})

    def delete(self, url, **kwargs):
        self.rt.delete_worker_item(url.rsplit('/', 1)[-1])
        return Response({})

    def get(self, url, **kwargs):
        pid = url.rsplit('/', 1)[-1]
        return Response({pid: {'status': {'completed': True, 'status_str': 'success'}}})

    async def dispatch(self, worker, prompt, meta, **kwargs):
        inputs = prompt['end']['inputs']
        index, token = inputs['worker_item_index'], inputs['worker_result_token']
        self.dispatched.append(index)
        # A remote callback is ready immediately; the local slot is slower.
        item = self.rt.consume_worker_item(token)
        self.rt.set_master_result(self.context, index, token, {
            'image': item['image'], 'mask': item['mask'], 'text': item['text']})
        return f'remote-{index}'

    async def test_local_only_preserves_order_and_uses_no_http(self):
        result = await self.rt.run_image_list_job(self.create())
        self.assertEqual(result[2], [str(i) for i in range(5)])
        self.assertEqual(self.local_calls, list(range(5)))
        self.assertEqual(len(self.slot_instances), 1)
        self.assertEqual(len(set(self.tokens)), 5)
        self.network.get_client_session.assert_not_awaited()
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.rt._WORKER_ITEMS)
        self.assertFalse(self.rt._MASTER_JOBS)
        for i, (image, mask) in enumerate(zip(*result[:2])):
            self.assertTrue(torch.equal(image, self.items[i]['image']))
            self.assertTrue(torch.equal(mask, self.items[i]['mask']))

    async def test_mixed_pool_claims_from_one_queue_and_relaxes_remote_idle(self):
        self.workers.append({'id': 'master', 'type': 'local'})  # Not the in-process master slot.
        self.local_delay = 0.04
        result = await self.rt.run_image_list_job(self.create(count=9))
        self.assertTrue(self.local_calls)
        self.assertTrue(self.dispatched)
        self.assertEqual(sorted(self.local_calls + self.dispatched), list(range(9)))
        self.assertEqual(result[2], [str(i) for i in range(9)])
        self.preflight.assert_awaited_once_with(self.workers, require_idle=False)

    async def test_orchestrator_mode_never_instantiates_local_slot(self):
        self.workers.append({'id': 'worker', 'type': 'remote'})
        await self.rt.run_image_list_job(self.create(participates=False))
        self.assertFalse(self.slot_instances)
        self.preflight.assert_awaited_once_with(self.workers, require_idle=True)
        with self.assertRaisesRegex(RuntimeError, 'worker|fallback'):
            self.rt.create_master_job('start', 'end', graph(), self.items, [], '')

    async def test_local_failure_surfaces_and_reclaims_staged_token(self):
        self.local_error = RuntimeError('local branch failed')
        with self.assertRaisesRegex(RuntimeError, 'local branch failed'):
            await self.rt.run_image_list_job(self.create())
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.rt._WORKER_ITEMS)
        self.assertFalse(self.rt._MASTER_JOBS)

    async def test_deadline_joins_local_execution_before_removing_context(self):
        self.deadline, self.local_delay = 0.03, 0.1
        with self.assertRaisesRegex(TimeoutError, 'total deadline'):
            await self.rt.run_image_list_job(self.create())
        self.assertTrue(self.local_finished.is_set())
        self.assertEqual(self.closed, 1)
        self.assertFalse(self.rt._WORKER_ITEMS)
        self.assertFalse(self.rt._MASTER_JOBS)


    async def test_cancelled_scheduler_joins_owned_execution_even_if_cancelled_again(self):
        self.local_delay = 0.12
        task = asyncio.create_task(self.rt.run_image_list_job(self.create()))
        await self.local_started.wait()
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(self.rt.ImageListCancelledError):
            await task
        self.assertTrue(self.local_finished.is_set())
        self.assertFalse(self.rt._WORKER_ITEMS)
        self.assertFalse(self.rt._MASTER_JOBS)


class RealFrameworkTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('COMFYUI_SOURCE_ROOT'), 'set COMFYUI_SOURCE_ROOT for the real CPU framework probe')
    def test_real_framework_local_slot(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--real-probe',
                                 os.environ['COMFYUI_SOURCE_ROOT']], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('MASTER_SLOT_REAL_OK', result.stdout)


def real_framework_probe(comfy_root):
    import threading
    import time
    sys.path.insert(0, str(Path(comfy_root).resolve()))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.disable_assets = True
    import execution
    import nodes
    import comfy.memory_management as memory
    import comfy.model_management as models
    import comfy.utils as utils
    from comfy_execution import progress
    from comfy_execution.graph_utils import GraphBuilder
    from comfy_api.v0_0_2 import io

    name = '_real_master_slot'
    for suffix in ('', '.image_list', '.utils', '.nodes'):
        module = types.ModuleType(name + suffix)
        module.__path__ = [str(ROOT / suffix.lstrip('.').replace('.', '/'))]
        sys.modules[name + suffix] = module
    # No HTTP/service/extension bootstrap is needed by the local slot. Fail if
    # either boundary tries to use a server-loop bridge instead of direct memory.
    def forbidden(*args, **kwargs):
        raise AssertionError('local boundary used the server loop or /prompt')
    sys.modules[name + '.utils.async_helpers'] = types.SimpleNamespace(run_async_in_server_loop=forbidden)
    config = {'settings': {}}
    sys.modules[name + '.utils.config'] = types.SimpleNamespace(load_config=lambda: config)
    rt = importlib.import_module(name + '.image_list.runtime')
    local = importlib.import_module(name + '.image_list.local_executor')
    boundary = importlib.import_module(name + '.nodes.image_list')
    nodes.NODE_CLASS_MAPPINGS.update(DistributedImageListStart=boundary.DistributedImageListStart,
                                   DistributedImageListEnd=boundary.DistributedImageListEnd)
    loads, calls = [], []
    entered, stopped = threading.Event(), threading.Event()
    mode = ['normal']
    model = object()

    class Loader:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {}}
        RETURN_TYPES = ('MODEL',)
        FUNCTION = 'run'
        def run(self):
            loads.append(True)
            return (model,)

    class Operation:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'image': ('IMAGE',), 'model': ('MODEL',),
                                 'mask': ('MASK',), 'text': ('STRING', {'forceInput': True})}}
        RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
        FUNCTION = 'run'
        async def run(self, image, model, mask, text):
            calls.append(text)
            entered.set()
            if mode[0] == 'error':
                raise ValueError('probe branch failure')
            if mode[0] in ('cancel', 'interrupt'):
                try:
                    while True:
                        utils.ProgressBar(1).update(1)
                        await asyncio.sleep(0.01)
                finally:
                    await asyncio.sleep(0.02)
                    stopped.set()
            if mode[0] == 'sync':
                # No progress/interrupt checks: cleanup still must join, not
                # release the parent just because an asyncio Future is cancelled.
                time.sleep(0.08)
                stopped.set()
            await asyncio.sleep(0)
            return image + 1, mask + 1, text + ':done'

    class Expanded:
        @classmethod
        def INPUT_TYPES(cls):
            return Operation.INPUT_TYPES()
        RETURN_TYPES = Operation.RETURN_TYPES
        FUNCTION = 'run'
        def run(self, image, model, mask, text):
            builder = GraphBuilder()
            node = builder.node('Operation', image=image, model=model, mask=mask, text=text)
            return {'result': (node.out(0), node.out(1), node.out(2)), 'expand': builder.finalize()}

    nodes.NODE_CLASS_MAPPINGS.update(Loader=Loader, Operation=Operation, Expanded=Expanded)
    prompt = graph()
    prompt['loader'] = {'class_type': 'Loader', 'inputs': {}}
    prompt['op']['inputs'].update(model=['loader', 0], mask=['start', 2], text=['start', 3])
    prompt['end']['inputs'].update(masks=['op', 1], texts=['op', 2])
    items = [{'image': torch.full((1, i + 1, 2, 3), float(i)),
              'mask': torch.full((1, i + 1, 2), float(i)), 'text': str(i), 'index': i} for i in range(3)]
    region = rt.validate_image_list_regions(prompt)[0]
    parent_progress = progress.ProgressRegistry('outer', execution.DynamicPrompt({}))
    parent_progress.nodes['outer'] = {'state': progress.NodeState.Running, 'value': 3, 'max': 7}
    progress.global_progress_registry = parent_progress
    parent_hook = lambda *a, **kw: None
    utils.PROGRESS_BAR_HOOK = parent_hook
    ram_callback = lambda *a, **kw: None
    memory.set_ram_cache_release_state(ram_callback, 123)
    GraphBuilder.set_default_prefix('outer-prefix', 7, 9)

    def restored():
        assert progress.global_progress_registry is parent_progress
        assert parent_progress.nodes == {'outer': {'state': progress.NodeState.Running, 'value': 3, 'max': 7}}
        assert utils.PROGRESS_BAR_HOOK is parent_hook
        assert memory.extra_ram_release_callback is ram_callback and memory.RAM_CACHE_HEADROOM == 123
        assert (GraphBuilder._default_prefix_root, GraphBuilder._default_prefix_call_index,
                GraphBuilder._default_prefix_graph_index) == ('outer-prefix', 7, 9)
        assert not any(t.name.startswith('image-list-master') for t in threading.enumerate())

    async def probe():
        # Verify actual V3 input parsing and list mapping, not only direct execute.
        for supplied, delegated, expected in [(None, False, True), (None, True, False),
                                               (False, False, False), (True, True, True)]:
            config['settings']['master_delegate_only'] = delegated
            inputs = {'image_list_enabled_worker_ids': '["remote"]'}
            if supplied is not None:
                inputs['image_list_master_participates'] = supplied
            cls = boundary.DistributedImageListStart
            data, _, v3 = execution.get_input_data(inputs, cls, 'start', dynprompt=execution.DynamicPrompt(prompt))
            data['images'] = [torch.zeros((3, 2, 2, 3))]
            result = await execution._async_map_node_over_list('outer', 'start', cls, data, cls.FUNCTION, v3_data=v3)
            result = await execution.resolve_map_node_over_list_results(result)
            context = result[0].result[7][0]
            assert rt.get_master_job(context).master_participates is expected
            rt.remove_master_job(context)

        for behavior in ('normal', 'expand', 'error', 'cancel', 'interrupt', 'sync', 'already-interrupted'):
            mode[0] = behavior
            prompt['op']['class_type'] = 'Expanded' if behavior == 'expand' else 'Operation'
            loads.clear()
            calls.clear()
            entered.clear()
            stopped.clear()
            tracked = []
            slot = local.LocalExecutionSlot(model_callback=tracked.append)
            context = rt.create_master_job('start', 'end', prompt, items, [], '', master_participates=True)
            try:
                for index in range(3 if behavior in ('normal', 'expand') else 1):
                    token = rt.create_worker_task_token()
                    rt.register_assignment(context, index, 'local', token)
                    rt._WORKER_ITEMS[token] = (time.monotonic() + 60, items[index])
                    worker_prompt = rt.build_worker_prompt(prompt, region, token, context, index, '')
                    for key in ('start', 'end'):
                        worker_prompt[key]['inputs']['image_list_local_execution'] = True
                    if behavior == 'already-interrupted':
                        models.interrupt_current_processing(True)
                    future = slot.submit(worker_prompt, f'local-{index}', 'end')
                    if behavior in ('cancel', 'interrupt', 'sync'):
                        assert await asyncio.to_thread(entered.wait, 5)
                        if behavior == 'interrupt':
                            models.interrupt_current_processing(True)
                        else:
                            slot.request_stop()
                    try:
                        await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), 5)
                    except rt.ImageListCancelledError:
                        assert behavior in ('cancel', 'interrupt', 'sync', 'already-interrupted'), behavior
                    except RuntimeError as exc:
                        assert behavior == 'error' and 'probe branch failure' in str(exc), str(exc)
                    else:
                        assert behavior in ('normal', 'expand'), behavior
                        result = rt.get_master_job(context).results[index]
                        assert torch.equal(result['image'], items[index]['image'] + 1)
                        assert torch.equal(result['mask'], items[index]['mask'] + 1)
                        assert result['text'] == f'{index}:done'
                if behavior in ('normal', 'expand'):
                    assert len(loads) == 1 and calls == ['0', '1', '2'], (loads, calls)
                    assert any(model in outputs for outputs in tracked), tracked
                if behavior in ('cancel', 'interrupt', 'sync'):
                    assert stopped.is_set(), behavior
                if behavior == 'already-interrupted':
                    assert models.processing_interrupted() and not calls
            finally:
                await slot.close()
                for token in rt.get_master_job(context).expected_tokens.values():
                    rt.delete_worker_item(token)
                rt.remove_master_job(context)
                models.interrupt_current_processing(False)  # Only test setup/teardown.
            restored()
        assert not rt._MASTER_JOBS and not rt._WORKER_ITEMS
    asyncio.run(probe())
    print('MASTER_SLOT_REAL_OK V3 metadata; cache reuse; async/expanded graph; errors; cancellation; state restoration')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--real-probe':
        real_framework_probe(sys.argv[2])
    else:
        unittest.main()
