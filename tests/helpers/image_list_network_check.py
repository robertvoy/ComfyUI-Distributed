"""Model-free multi-process acceptance using real ComfyUI executors and loopback HTTP.

Only the transport test facade (/prompt, /queue, /history, /interrupt) is synthetic.
Start/End, graph extraction, scheduling, payload codecs and callbacks are production code.
No managed ComfyUI service, user workflow, model, or remote worker is touched.
"""
import asyncio
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = Path(sys.argv[1]).resolve()
WORKER = '--worker' in sys.argv
DELAY = float(sys.argv[3]) if WORKER else 0
sys.path.insert(0, str(COMFY_ROOT))
os.environ['COMFYUI_IS_WORKER'] = '1'
import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.disable_assets = True
from app.assets.manager import default_asset_manager
import comfy.model_management
import execution
import nodes
import server
import torch
from aiohttp import web
import aiohttp

torch.set_num_threads(1)
SOURCE_CALLS, BRANCH_CALLS, CAPTURED, MODEL_CALLS = [], [], [], []
ALLOW_MASTER = False
COUNT = 7


def source_rows():
    return [{'image': torch.full((1, 5 + i, 7, 3 if i % 2 == 0 else 4), i / 16, dtype=torch.float32),
             'reference_image': torch.full((1, 2, 3, 3), (i + 1) / 16),
             'mask': torch.full((1, 5 + i, 7), (i + 1) / 16),
             'text': f'item-{i}', 'strength': (i + 1) / 16, 'seed': 100 + i, 'index': i}
            for i in range(COUNT)]


class TestImageSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'STRING', 'FLOAT', 'INT')
    OUTPUT_IS_LIST = (True,) * 6
    FUNCTION = 'run'
    def run(self):
        assert not WORKER, 'upstream image source leaked into worker prompt'
        SOURCE_CALLS.append(True)
        rows = source_rows()
        return tuple([row[key] for row in rows] for key in
                     ('image', 'reference_image', 'mask', 'text', 'strength', 'seed'))


class TestSharedFlag:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    RETURN_TYPES = ('BOOLEAN',)
    FUNCTION = 'run'
    def run(self):
        assert not WORKER, 'shared value source leaked into worker prompt'
        return (True,)


class TestCachedModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'run'
    def run(self):
        MODEL_CALLS.append(True)
        return (object(),)


class TestImageOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'reference_image': ('IMAGE',), 'mask': ('MASK',),
                             'text': ('STRING', {'forceInput': True}),
                             'strength': ('FLOAT', {'forceInput': True}),
                             'seed': ('INT', {'forceInput': True}), 'index': ('INT', {'forceInput': True}),
                             'flag': ('BOOLEAN', {'forceInput': True}), 'model': ('MODEL',)}}
    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    FUNCTION = 'run'
    def run(self, image, reference_image, mask, text, strength, seed, index, flag, model):
        assert WORKER or ALLOW_MASTER, 'per-image branch executed in orchestrator-only mode'
        BRANCH_CALLS.append(index)
        # Sleep in short chunks so the real Comfy interrupt signal can be tested.
        until = time.monotonic() + DELAY
        while time.monotonic() < until:
            comfy.model_management.throw_exception_if_processing_interrupted()
            time.sleep(0.01)
        assert seed == 100 + index
        assert flag is True
        assert text == f'item-{index}'
        assert strength == (index + 1) / 16
        assert torch.all(reference_image == strength)
        return image + strength, mask * 0.5, f'{text}:seed-{seed}'


class TestCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'masks': ('MASK',), 'texts': ('STRING', {'forceInput': True})}}
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = 'run'
    def run(self, images, masks, texts):
        CAPTURED.append((images, masks, texts))
        return ()


async def setup():
    assets = default_asset_manager()
    ps = server.PromptServer(asyncio.get_running_loop(), assets)
    assert await nodes.load_custom_node(str(ROOT))
    module = sys.modules[str(ROOT).replace('.', '_x_')]
    nodes.NODE_CLASS_MAPPINGS.update(TestImageSource=TestImageSource,
                                   TestImageOperation=TestImageOperation, TestCapture=TestCapture,
                                   TestCachedModel=TestCachedModel, TestSharedFlag=TestSharedFlag)
    # Match the CORS-enabled direct-worker launch contract for canvas probes.
    app = web.Application(middlewares=[server.create_cors_middleware('*')])
    # Register the actual feature routes, not test replacements.
    app.add_routes([route for route in ps.routes if route.path.startswith('/distributed/image-list/')])
    return ps, assets, module, app


async def listen(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def worker_main():
    ps, assets, module, app = await setup()
    pending, running, history = [], {}, {}
    wake = asyncio.Event()
    interrupted_ids = []
    async def prompt(request):
        data = await request.json()
        prompt_id = str(uuid.uuid4())
        graph = data['prompt']
        valid = await execution.validate_prompt(prompt_id, graph, None)
        if not valid[0]:
            return web.json_response({'error': valid[1], 'node_errors': valid[3]}, status=400)
        pending.append((prompt_id, graph, valid[2]))
        wake.set()
        return web.json_response({'prompt_id': prompt_id})
    async def queue(request):
        if request.method == 'POST':
            deleted = (await request.json()).get('delete', [])
            pending[:] = [entry for entry in pending if entry[0] not in deleted]
            return web.json_response({})
        return web.json_response({'queue_running': [[0, key] for key in running],
                                  'queue_pending': [[0, entry[0]] for entry in pending]})
    async def get_history(request):
        pid = request.match_info['prompt_id']
        return web.json_response({pid: history[pid]} if pid in history else {})
    async def interrupt(request):
        pid = (await request.json())['prompt_id']
        interrupted_ids.append(pid)
        if pid in running:
            comfy.model_management.interrupt_current_processing(True)
        return web.json_response({})
    async def stats(request):
        return web.json_response({'calls': BRANCH_CALLS, 'running': list(running),
                                  'pending': [e[0] for e in pending], 'interrupted': interrupted_ids,
                                  'history': history})
    async def consume():
        executor = execution.PromptExecutor(ps, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=assets)
        while True:
            await wake.wait()
            while pending:
                pid, graph, outputs = pending.pop(0)
                running[pid] = True
                await asyncio.to_thread(executor.execute, graph, pid, {}, outputs)
                history[pid] = {'status': {'completed': True,
                    'status_str': 'success' if executor.success else 'error',
                    'messages': str(executor.status_messages)}}
                running.pop(pid, None)
            wake.clear()
    async def probe(_request):
        return web.json_response({'exec_info': {'queue_remaining': len(pending) + len(running)}})
    app.router.add_get('/prompt', probe)
    app.router.add_post('/prompt', prompt)
    app.router.add_get('/queue', queue)
    app.router.add_post('/queue', queue)
    app.router.add_get('/history/{prompt_id}', get_history)
    app.router.add_post('/interrupt', interrupt)
    app.router.add_get('/test-stats', stats)
    runner, port = await listen(app)
    consumer = asyncio.create_task(consume())
    print(f'READY:{port}', flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await runner.cleanup()


def graph(callback, worker_ids):
    return {
        '1': {'class_type': 'TestImageSource', 'inputs': {}},
        # Generic Autogrow value slots carry per-item reference images, text,
        # strength and seed (one value per image), plus a shared boolean.
        '2': {'class_type': 'DistributedImageListStart', 'inputs': {
            'images': ['1', 0], 'mask': ['1', 2],
            'values.value_1': ['1', 1], 'values.value_2': ['1', 3],
            'values.value_3': ['1', 4], 'values.value_4': ['1', 5], 'values.value_5': ['7', 0],
            'image_list_enabled_worker_ids': json.dumps(worker_ids), 'image_list_master_callback': callback}},
        '3': {'class_type': 'TestImageOperation', 'inputs': {
            key: ['2', index] for key, index in [('image', 0), ('mask', 1), ('index', 2),
                ('reference_image', 4), ('text', 5), ('strength', 6), ('seed', 7), ('flag', 8)]}
              | {'model': ['6', 0]}},
        '4': {'class_type': 'DistributedImageListEnd', 'inputs': {
            'images': ['3', 0], 'masks': ['3', 1], 'texts': ['3', 2], 'context': ['2', 3]}},
        '5': {'class_type': 'TestCapture', 'inputs': {'images': ['4', 0], 'masks': ['4', 1], 'texts': ['4', 2]}},
        '6': {'class_type': 'TestCachedModel', 'inputs': {}},
        '7': {'class_type': 'TestSharedFlag', 'inputs': {}},
    }


async def coordinator():
    global ALLOW_MASTER
    ps, assets, module, app = await setup()
    runtime = importlib.import_module(module.__name__ + '.image_list.runtime')
    config_mod = importlib.import_module(module.__name__ + '.utils.config')
    orchestration = importlib.import_module(module.__name__ + '.api.queue_orchestration')
    runner, port = await listen(app)
    ps.port = port  # Match the listener used by local-worker callback resolution.
    processes, stderr_files = [], []
    # Old worker capability refusal is read-only and precedes input evaluation.
    legacy_app = web.Application()
    legacy_requests = []
    async def legacy(request):
        legacy_requests.append(request.method)
        return web.json_response({'error': 'not found'}, status=404)
    legacy_app.router.add_route('*', '/{path:.*}', legacy)
    legacy_runner, legacy_port = await listen(legacy_app)
    dispatch_module = importlib.import_module(module.__name__ + '.api.orchestration.dispatch')
    try:
        try:
            await dispatch_module.preflight_image_list_workers([
                {'id': 'legacy', 'type': 'local', 'host': '127.0.0.1', 'port': legacy_port}])
            raise AssertionError('legacy worker accepted')
        except RuntimeError as exc:
            assert 'upgrade' in str(exc), str(exc)
        assert legacy_requests == ['GET'] and not SOURCE_CALLS
    finally:
        await legacy_runner.cleanup()
    print('IMAGE_LIST_PREFLIGHT_OK legacy worker rejected before materialization or writes')
    scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='image-list-network-', dir=scratch) as tmp:
            workers = []
            for name, delay in [('fast', 0.02), ('slow', 0.5)]:
                err = open(Path(tmp) / f'{name}.log', 'w+')
                stderr_files.append(err)
                proc = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()),
                    str(COMFY_ROOT), '--worker', str(delay), stdout=asyncio.subprocess.PIPE, stderr=err,
                    env={**os.environ, 'TMPDIR': tmp})
                processes.append(proc)
                async def ready():
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            err.seek(0)
                            raise AssertionError(f'Worker exited: {err.read()}')
                        if line.startswith(b'READY:'):
                            return int(line.split(b':')[1])
                worker_port = await asyncio.wait_for(ready(), 60)
                workers.append({'id': name, 'type': 'local', 'host': '127.0.0.1',
                                'port': worker_port, 'enabled': True})
            config = {'workers': workers, 'settings': {'websocket_orchestration': False, 'master_delegate_only': True},
                      'master': {'host': f'http://127.0.0.1:{port}'}}
            prompt = graph(f'http://127.0.0.1:{port}', [w['id'] for w in workers])
            valid = await execution.validate_prompt('network-acceptance', prompt, None)
            assert valid[0], valid
            executor = execution.PromptExecutor(ps, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=assets)
            # Exercise production queue preparation; only master enqueue is redirected
            # to this isolated executor, not the managed server's live queue.
            prompt['2']['inputs'].pop('image_list_enabled_worker_ids')
            prompt['2']['inputs'].pop('image_list_master_callback')
            async def enqueue(prepared, workflow, client_id, **kwargs):
                checked = await execution.validate_prompt('network-acceptance', prepared, None)
                assert checked[0], checked
                await asyncio.to_thread(executor.execute, prepared, 'network-acceptance', {}, checked[2])
                return {'prompt_id': 'network-acceptance', 'number': 1, 'node_errors': {}}
            with patch.object(config_mod, 'load_config', return_value=config), \
                 patch.object(orchestration, 'load_config', return_value=config), \
                 patch.object(orchestration, 'queue_prompt_payload', enqueue), \
                 patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                queued = await asyncio.wait_for(orchestration.orchestrate_distributed_execution(
                    prompt, None, 'test', [w['id'] for w in workers]), 45)
                assert queued[0] == 'network-acceptance' and queued[2] == 2, queued
                assert not ps.distributed_pending_jobs, 'image-list created unused Collector queues'
            if not executor.success:
                for err in stderr_files:
                    err.flush(); err.seek(0); print(err.read(), file=sys.stderr)
            assert executor.success, executor.status_messages
            assert len(SOURCE_CALLS) == 1 and not BRANCH_CALLS
            assert len(CAPTURED) == 1
            images, masks, texts = CAPTURED[0]
            assert len(images) == len(masks) == len(texts) == COUNT
            for i, row in enumerate(source_rows()):
                assert torch.equal(images[i], row['image'] + row['strength']), (i, images[i])
                assert torch.equal(masks[i], row['mask'] * 0.5), i
                assert texts[i] == f'item-{i}:seed-{100 + i}', texts
            async with aiohttp.ClientSession() as session:
                stats = []
                for worker in workers:
                    async with session.get(f"http://127.0.0.1:{worker['port']}/test-stats") as response:
                        stats.append(await response.json())
                counts = [len(row['calls']) for row in stats]
                assert counts[0] > counts[1] > 0, counts
                assert Counter(i for row in stats for i in row['calls']) == Counter(range(COUNT))
            assert not runtime._MASTER_JOBS
            # Three sequential regions on one worker, including optional outputs
            # left unconnected. Region N+1 must not race the prior worker End.
            chained = {'1': {'class_type': 'EmptyImage', 'inputs': {
                'height': 8, 'width': 8, 'batch_size': 2, 'color': 0}}}
            previous = '1'
            for start, end in [('2', '3'), ('4', '5'), ('6', '7')]:
                chained[start] = {'class_type': 'DistributedImageListStart', 'inputs': {
                    'images': [previous, 0]}}
                chained[end] = {'class_type': 'DistributedImageListEnd', 'inputs': {
                    'images': [start, 0], 'context': [start, 3]}}
                previous = end
            chained['8'] = {'class_type': 'TestCapture', 'inputs': {
                'images': ['7', 0], 'masks': ['7', 1], 'texts': ['7', 2]}}
            with patch.object(config_mod, 'load_config', return_value=config), \
                 patch.object(orchestration, 'load_config', return_value=config), \
                 patch.object(orchestration, 'queue_prompt_payload', enqueue), \
                 patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                await asyncio.wait_for(orchestration.orchestrate_distributed_execution(
                    chained, None, 'test', ['slow']), 45)
            assert executor.success, executor.status_messages
            assert len(CAPTURED) == 2 and len(CAPTURED[1][0]) == 2
            assert all(torch.equal(image, torch.zeros((1, 8, 8, 3))) for image in CAPTURED[1][0])
            assert CAPTURED[1][1:] == ([None, None], [None, None])
            assert not runtime._MASTER_JOBS
            print('IMAGE_LIST_SEQUENTIAL_OK three regions; one worker; optional outputs omitted')
            # Same public queue path, now with master participation enabled. The
            # parent queue stays occupied while its local slot processes items.
            ALLOW_MASTER = True
            config['settings']['master_delegate_only'] = False
            participation_counts = {}
            for mode, selected in [('mixed', [w['id'] for w in workers]), ('master-only', [])]:
                executor = execution.PromptExecutor(ps, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=assets)
                before_sources, before_local, before_models = len(SOURCE_CALLS), len(BRANCH_CALLS), len(MODEL_CALLS)
                async with aiohttp.ClientSession() as session:
                    before_worker_calls = []
                    for worker in workers:
                        async with session.get(f"http://127.0.0.1:{worker['port']}/test-stats") as response:
                            before_worker_calls.append(len((await response.json())['calls']))
                with patch.object(config_mod, 'load_config', return_value=config), \
                     patch.object(orchestration, 'load_config', return_value=config), \
                     patch.object(orchestration, 'queue_prompt_payload', enqueue), \
                     patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                    queued = await asyncio.wait_for(orchestration.orchestrate_distributed_execution(
                        prompt, None, 'test', selected), 45)
                assert executor.success, (mode, executor.status_messages)
                assert queued[2] == len(selected)
                assert len(SOURCE_CALLS) == before_sources + 1, 'source was reevaluated per local item'
                local_calls = BRANCH_CALLS[before_local:]
                assert local_calls, f'{mode}: master did not process any items'
                assert len(MODEL_CALLS) == before_models + 1, 'local model dependency did not stay cached'
                remote_calls = []
                async with aiohttp.ClientSession() as session:
                    for worker, offset in zip(workers, before_worker_calls):
                        async with session.get(f"http://127.0.0.1:{worker['port']}/test-stats") as response:
                            current = await response.json()
                        remote_calls.extend(current['calls'][offset:])
                        assert not current['running'] and not current['pending'], current
                if mode == 'mixed':
                    assert remote_calls, 'mixed execution failed to use direct workers'
                else:
                    assert not remote_calls and len(local_calls) == COUNT
                assert Counter(local_calls + remote_calls) == Counter(range(COUNT)), (local_calls, remote_calls)
                images, masks, texts = CAPTURED[-1]
                assert len(images) == len(masks) == len(texts) == COUNT
                for i, row in enumerate(source_rows()):
                    assert torch.equal(images[i], row['image'] + row['strength']), (mode, i)
                    assert torch.equal(masks[i], row['mask'] * 0.5), (mode, i)
                    assert texts[i] == f'item-{i}:seed-{100 + i}', (mode, texts)
                assert not runtime._MASTER_JOBS and not runtime._WORKER_ITEMS
                participation_counts[mode] = {'master': len(local_calls), 'workers': len(remote_calls)}
            print('IMAGE_LIST_MASTER_OK ' + json.dumps(participation_counts))
            ALLOW_MASTER = False
            config['settings']['master_delegate_only'] = True
            # Real Comfy user-interrupt flag while a real worker executor is active.
            slow = workers[1]
            async with aiohttp.ClientSession() as session:
                async def slow_stats():
                    async with session.get(f"http://127.0.0.1:{slow['port']}/test-stats") as response:
                        return await response.json()
                async def until(predicate):
                    for _ in range(150):
                        current = await slow_stats()
                        if predicate(current):
                            return current
                        await asyncio.sleep(0.02)
                    raise AssertionError('worker state transition timed out')
                before = await until(lambda state: not state['running'] and not state['pending'])
                context = runtime.create_master_job('2', '4', prompt, source_rows(), ['slow'],
                                                  f'http://127.0.0.1:{port}')
                with patch.object(config_mod, 'load_config', return_value=config), \
                     patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                    task = asyncio.create_task(runtime.run_image_list_job(context))
                    active = await until(lambda state: bool(state['running']))
                    owned_id = active['running'][0]
                    comfy.model_management.interrupt_current_processing(True)
                    try:
                        await asyncio.wait_for(task, 5)
                        raise AssertionError('interrupt was ignored')
                    except runtime.ImageListCancelledError:
                        pass
                    finally:
                        comfy.model_management.interrupt_current_processing(False)
                after = await until(lambda state: not state['running'] and not state['pending'])
                assert after['interrupted'] == [owned_id], after
                assert len(after['calls']) <= len(before['calls']) + 1
                assert set(before['history']) <= set(after['history'])
                assert not runtime._MASTER_JOBS
            print('IMAGE_LIST_CANCEL_OK owned prompt stopped; no further dispatch; context removed')
            # A worker exception before End must fail, not hang or return a partial list.
            bad_rows = source_rows()
            bad_rows[0]['seed'] = -1
            context = runtime.create_master_job('2', '4', prompt, bad_rows, ['fast'],
                                              f'http://127.0.0.1:{port}')
            with patch.object(config_mod, 'load_config', return_value=config), \
                 patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                try:
                    await asyncio.wait_for(runtime.run_image_list_job(context), 5)
                    raise AssertionError('worker exception was ignored')
                except RuntimeError as exc:
                    assert 'failed image-list item' in str(exc), str(exc)
            assert not runtime._MASTER_JOBS
            print('IMAGE_LIST_FAILURE_OK worker exception surfaced; context removed')
            print('IMAGE_LIST_NETWORK_OK ' + json.dumps({'workers': 2, 'items': COUNT,
                'assignments': dict(zip(['fast', 'slow'], counts)), 'ordered_exact_outputs': True,
                'master_branch_calls': len(BRANCH_CALLS), 'source_calls': len(SOURCE_CALLS)}))
    finally:
        for proc in processes:
            if proc.returncode is None:
                proc.terminate()
        await asyncio.gather(*(proc.wait() for proc in processes))
        for err in stderr_files:
            err.close()
        await runner.cleanup()
        network = importlib.import_module(module.__name__ + '.utils.network')
        session = await network.get_client_session()
        await session.close()


if __name__ == '__main__':
    asyncio.run(worker_main() if WORKER else coordinator())
