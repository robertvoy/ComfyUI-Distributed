"""Model-free multi-process acceptance using real ComfyUI executors and loopback HTTP.

Isolated listeners use real core /prompt, /queue, /history and /interrupt handlers.
Start/End, graph extraction, scheduling, payload codecs and callbacks are production code.
No managed ComfyUI service, user workflow, model, or remote worker is touched.
"""
import asyncio
from collections import Counter
import copy
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
UTILITY_CALLS, ITEM_EXECUTIONS, VALUE_CAPTURES = [], [], []
ALLOW_MASTER = False
COUNT = 7


def source_rows():
    return [{'image': torch.full((1, 5 + i, 7, 3 if i % 2 == 0 else 4), i / 16, dtype=torch.float32),
             'reference_image': torch.full((2, 2, 3, 3), (i + 1) / 16),
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


def assert_item_execution(prompt, index):
    """Prove the operation belongs to an item, not merely a participating host."""
    from comfy_execution.utils import get_executing_context
    context = get_executing_context()
    assert context is not None
    assert prompt['2']['inputs'].get('worker_item_token'), 'per-item node ran in outer graph'
    running, _ = server.PromptServer.instance.prompt_queue.get_current_queue_volatile()
    assert len(running) == 1, running
    entry = running[0]
    if WORKER:
        assert entry[1] == context.prompt_id and entry[2] == prompt
        assert entry[4] == ['4'], entry[4]
    else:
        assert ALLOW_MASTER, 'per-item node ran on delegate-only master'
        assert entry[1] != context.prompt_id, 'per-item node ran in outer executor'
        assert set(entry[4]) == {'4', '5'}, entry[4]
    return {'index': index, 'prompt_id': context.prompt_id,
            'role': 'worker' if WORKER else 'master-item', 'queue_prompt_id': entry[1]}


class TestRequiredOutputUtility:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'index': ('INT', {'forceInput': True})},
                'hidden': {'prompt': 'PROMPT'}}
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    def run(self, image, index, prompt):
        assert_item_execution(prompt, index)
        UTILITY_CALLS.append(index)
        return (image,)


class TestDiscardedPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',)}}
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = 'run'
    def run(self, image):
        raise AssertionError('discarded per-item output preview executed')


class TestImageOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'reference_image': ('IMAGE',), 'mask': ('MASK',),
                             'text': ('STRING', {'forceInput': True}),
                             'strength': ('FLOAT', {'forceInput': True}),
                             'seed': ('INT', {'forceInput': True}), 'index': ('INT', {'forceInput': True}),
                             'flag': ('BOOLEAN', {'forceInput': True}), 'model': ('MODEL',)},
                'hidden': {'prompt': 'PROMPT'}}
    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    FUNCTION = 'run'
    def run(self, image, reference_image, mask, text, strength, seed, index, flag, model, prompt):
        assert WORKER or ALLOW_MASTER, 'per-image branch executed in orchestrator-only mode'
        ITEM_EXECUTIONS.append(assert_item_execution(prompt, index))
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


class TestValueOverride:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput': True}),
                             'index': ('INT', {'forceInput': True})}}
    RETURN_TYPES = ('STRING', 'INT')
    FUNCTION = 'run'
    def run(self, text, index):
        assert WORKER or ALLOW_MASTER, 'override branch executed on delegate-only master'
        return (text + ':override', 200 + index)


class TestCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'masks': ('MASK',), 'texts': ('STRING', {'forceInput': True})},
                'optional': {'values': ('DISTRIBUTED_IMAGE_LIST_VALUES',),
                             'picked': ('STRING', {'forceInput': True})}}
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = 'run'
    def run(self, images, masks, texts, values=None, picked=None):
        assert not WORKER, 'post-End capture leaked into a worker prompt'
        CAPTURED.append((images, masks, texts))
        VALUE_CAPTURES.append((values, picked))
        return ()


async def setup():
    assets = default_asset_manager()
    ps = server.PromptServer(asyncio.get_running_loop(), assets)
    assert await nodes.load_custom_node(str(ROOT))
    module = sys.modules[str(ROOT).replace('.', '_x_')]
    boundary = importlib.import_module(module.__name__ + '.nodes.image_list')
    encode_result = boundary.encode_result
    def checked_encode(result):
        encoded = encode_result(result)
        if 'values' in encoded:
            assert set(encoded['values']) == {f'value_{n}' for n in (2, 3, 4, 5, 7, 8, 9, 10)}
        return encoded
    boundary.encode_result = checked_encode
    nodes.NODE_CLASS_MAPPINGS.update(TestImageSource=TestImageSource,
                                   TestImageOperation=TestImageOperation, TestCapture=TestCapture,
                                   TestCachedModel=TestCachedModel, TestSharedFlag=TestSharedFlag,
                                   TestRequiredOutputUtility=TestRequiredOutputUtility,
                                   TestDiscardedPreview=TestDiscardedPreview, TestValueOverride=TestValueOverride)
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


def finish_queue_item(ps, item_id, executor):
    ps.prompt_queue.task_done(item_id, executor.history_result,
        status=execution.PromptQueue.ExecutionStatus(
            status_str='success' if executor.success else 'error',
            completed=executor.success, messages=executor.status_messages),
        process_item=lambda item: item[:5] + item[6:])


async def worker_main():
    ps, assets, module, app = await setup()
    wake = asyncio.Event()
    interrupted_ids, submissions = [], []
    core_handlers = {(route.method, route.path): route.handler for route in ps.routes}
    async def prompt(request):
        data = await request.json()
        # Use core's actual validation/enqueue/response schema, including
        # partial_execution_targets; only observe it and wake our CPU consumer.
        response = await core_handlers['POST', '/prompt'](request)
        if response.status == 200:
            accepted = json.loads(response.body)
            _, pending = ps.prompt_queue.get_current_queue_volatile()
            entry = next(item for item in pending if item[1] == accepted['prompt_id'])
            submissions.append({'prompt_id': entry[1], 'requested': data.get('partial_execution_targets'),
                                'queued': entry[4], 'node_ids': sorted(entry[2]),
                                'end_ids': [key for key, node in entry[2].items()
                                            if node['class_type'] == 'DistributedImageListEnd']})
            wake.set()
        return response
    async def interrupt(request):
        interrupted_ids.append((await request.json())['prompt_id'])
        return await core_handlers['POST', '/interrupt'](request)
    async def stats(request):
        running, pending = ps.prompt_queue.get_current_queue_volatile()
        return web.json_response({'calls': BRANCH_CALLS, 'utility_calls': UTILITY_CALLS,
                                  'item_executions': ITEM_EXECUTIONS, 'submissions': submissions,
                                  'running': [e[1] for e in running], 'pending': [e[1] for e in pending],
                                  'interrupted': interrupted_ids, 'history': ps.prompt_queue.get_history()})
    async def consume():
        executor = execution.PromptExecutor(ps, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=assets)
        while True:
            await wake.wait()
            while (queued := ps.prompt_queue.get(timeout=0)) is not None:
                entry, item_id = queued
                await asyncio.to_thread(executor.execute, entry[2], entry[1], entry[3], entry[4])
                finish_queue_item(ps, item_id, executor)
            wake.clear()
    app.add_routes([route for route in ps.routes if
                    (route.path in ('/queue', '/history/{prompt_id}') or
                     (route.path == '/prompt' and route.method == 'GET'))])
    app.router.add_post('/prompt', prompt)
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
            'values.value_6': ['1', 2],
            'image_list_enabled_worker_ids': json.dumps(worker_ids), 'image_list_master_callback': callback}},
        '3': {'class_type': 'TestImageOperation', 'inputs': {
            key: ['2', index] for key, index in [('image', 0), ('mask', 1), ('index', 2),
                ('reference_image', 4), ('text', 5), ('strength', 6), ('seed', 7), ('flag', 8)]}
              | {'image': ['8', 0], 'model': ['6', 0]}},
        '4': {'class_type': 'DistributedImageListEnd', 'inputs': {
            'images': ['3', 0], 'masks': ['3', 1], 'texts': ['3', 2], 'context': ['2', 3],
            'value_2': ['10', 0], 'value_3': ['2', 6], 'value_4': ['2', 7], 'value_5': ['2', 8],
            'value_7': ['3', 1], 'value_8': ['2', 4], 'value_9': ['2', 9], 'value_10': ['10', 1]}},
        '5': {'class_type': 'TestCapture', 'inputs': {'images': ['4', 0], 'masks': ['4', 1], 'texts': ['4', 2],
                                                     'values': ['4', 3], 'picked': ['get', 0]}},
        '6': {'class_type': 'TestCachedModel', 'inputs': {}},
        '7': {'class_type': 'TestSharedFlag', 'inputs': {}},
        '8': {'class_type': 'TestRequiredOutputUtility', 'inputs': {'image': ['2', 0], 'index': ['2', 2]}},
        '9': {'class_type': 'TestDiscardedPreview', 'inputs': {'image': ['8', 0]}},
        '10': {'class_type': 'TestValueOverride', 'inputs': {'text': ['2', 5], 'index': ['2', 2]}},
        'get': {'class_type': 'DistributedImageListGetValue', 'inputs': {'values': ['4', 3], 'value_name': 'value_2'}},
    }


def check_carried_values():
    values, picked = VALUE_CAPTURES[-1]
    assert len(values) == len(picked) == COUNT
    for index, (actual, source) in enumerate(zip(values, source_rows())):
        assert set(actual) == {f'value_{n}' for n in range(1, 11)}, actual.keys()
        assert torch.equal(actual['value_1'], source['reference_image'])
        assert actual['value_2'] == picked[index] == f'item-{index}:override'
        assert actual['value_3'] == source['strength']
        assert actual['value_4'] == source['seed']
        assert actual['value_5'] is True
        assert torch.equal(actual['value_6'], source['mask'])
        assert torch.equal(actual['value_7'], source['mask'] * 0.5)
        assert torch.equal(actual['value_8'], source['reference_image'])
        assert actual['value_8'].shape[0] == 2, 'value tensor batch was split'
        assert torch.equal(actual['value_9'], source['mask'])
        assert actual['value_10'] == 200 + index


async def coordinator():
    global ALLOW_MASTER
    ps, assets, module, app = await setup()
    runtime = importlib.import_module(module.__name__ + '.image_list.runtime')
    execution_plan = importlib.import_module(module.__name__ + '.image_list.execution_plan')
    lifecycle = importlib.import_module(module.__name__ + '.image_list.lifecycle')
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
            assert {'4', '5', '8', '9'} == set(valid[2]), valid
            executor = execution.PromptExecutor(ps, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=assets)
            # Exercise production queue preparation; only master enqueue is redirected
            # to this isolated executor, not the managed server's live queue.
            prompt['2']['inputs'].pop('image_list_enabled_worker_ids')
            prompt['2']['inputs'].pop('image_list_master_callback')
            saved_prompt = copy.deepcopy(prompt)
            outer_executions = []
            async def enqueue(prepared, workflow, client_id, **kwargs):
                targets = kwargs['partial_execution_targets']
                planned, planned_targets = execution_plan.prepare_image_list_execution(prepared)
                assert planned == prepared and targets == planned_targets
                if '9' in saved_prompt and 'TestImageOperation' in {
                        node['class_type'] for node in prepared.values()}:
                    assert set(prepared) == set(saved_prompt) - {'9'}, prepared
                    assert prepared['8'] == saved_prompt['8'], 'required utility changed'
                    assert prepared['5'] == saved_prompt['5'], 'post-End output changed'
                    assert set(targets) == {'4', '5'}, targets
                    assert nodes.NODE_CLASS_MAPPINGS['TestRequiredOutputUtility'].OUTPUT_NODE is True
                    assert nodes.NODE_CLASS_MAPPINGS['TestDiscardedPreview'].OUTPUT_NODE is True
                pid = 'network-acceptance' if not outer_executions else str(uuid.uuid4())
                checked = await execution.validate_prompt(pid, prepared, targets)
                assert checked[0], checked
                assert not checked[3] and Counter(checked[2]) == Counter(targets), checked
                assert not lifecycle.has_safe_execution_targets(ps, pid, prepared)
                # Exactly the core queue tuple/put/get/task_done lifecycle. Start
                # must prove its roots from this real running record, not a mock.
                number = len(outer_executions)
                ps.prompt_queue.put((number, pid, prepared, {'client_id': client_id}, checked[2], {}))
                entry, item_id = ps.prompt_queue.get(timeout=0)
                assert lifecycle.has_safe_execution_targets(ps, pid, prepared)
                before_captures = len(CAPTURED)
                await asyncio.to_thread(executor.execute, entry[2], entry[1], entry[3], entry[4])
                finish_queue_item(ps, item_id, executor)
                assert ps.prompt_queue.get_current_queue_volatile() == ([], [])
                history = ps.prompt_queue.get_history(pid)[pid]
                assert Counter(history['prompt'][4]) == Counter(targets)
                assert not lifecycle.has_safe_execution_targets(ps, pid, prepared)
                outer_executions.append({'prompt_id': pid, 'targets': targets,
                                         'captures': len(CAPTURED) - before_captures})
                if executor.success:
                    assert len(CAPTURED) == before_captures + 1, 'post-End Capture must run exactly once'
                return {'prompt_id': pid, 'number': number, 'node_errors': checked[3]}
            with patch.object(config_mod, 'load_config', return_value=config), \
                 patch.object(orchestration, 'load_config', return_value=config), \
                 patch.object(orchestration, 'queue_prompt_payload', enqueue), \
                 patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                queued = await asyncio.wait_for(orchestration.orchestrate_distributed_execution(
                    prompt, None, 'test', [w['id'] for w in workers]), 45)
                assert queued[0] == 'network-acceptance' and queued[2] == 2, queued
                assert not ps.distributed_pending_jobs, 'image-list created unused Collector queues'
            assert prompt == saved_prompt, 'queue preparation changed the saved/source graph'
            if not executor.success:
                for err in stderr_files:
                    err.flush(); err.seek(0); print(err.read(), file=sys.stderr)
            assert executor.success, executor.status_messages
            assert len(SOURCE_CALLS) == 1 and not BRANCH_CALLS and not UTILITY_CALLS
            assert len(CAPTURED) == 1
            check_carried_values()
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
                for row in stats:
                    assert Counter(row['utility_calls']) == Counter(row['calls']), row
                    assert len(row['submissions']) == len(row['calls'])
                    assert len({event['prompt_id'] for event in row['item_executions']}) == len(row['calls'])
                    for submission in row['submissions']:
                        assert submission['requested'] == submission['queued'] == ['4'], submission
                        assert '8' in submission['node_ids'] and '9' not in submission['node_ids']
                        assert '5' not in submission['node_ids']
            assert not runtime._MASTER_JOBS
            # Three sequential regions on one worker, including optional outputs
            # left unconnected. Region N+1 must not race the prior worker End.
            chained = {'1': {'class_type': 'EmptyImage', 'inputs': {
                'height': 8, 'width': 8, 'batch_size': 2, 'color': 0}}}
            chained['constant'] = {'class_type': 'TestSharedFlag', 'inputs': {}}
            previous = '1'
            value_source = 'constant'
            for start, end in [('2', '3'), ('4', '5'), ('6', '7')]:
                chained[start] = {'class_type': 'DistributedImageListStart', 'inputs': {
                    'images': [previous, 0], 'values.value_1': [value_source, 0]}}
                chained[end] = {'class_type': 'DistributedImageListEnd', 'inputs': {
                    'images': [start, 0], 'context': [start, 3]}}
                value_source = f'get-{end}'
                chained[value_source] = {'class_type': 'DistributedImageListGetValue', 'inputs': {
                    'values': [end, 3], 'value_name': 'value_1'}}
                previous = end
            chained['8'] = {'class_type': 'TestCapture', 'inputs': {
                'images': ['7', 0], 'masks': ['7', 1], 'texts': ['7', 2], 'values': ['7', 3]}}
            saved_chained = copy.deepcopy(chained)
            with patch.object(config_mod, 'load_config', return_value=config), \
                 patch.object(orchestration, 'load_config', return_value=config), \
                 patch.object(orchestration, 'queue_prompt_payload', enqueue), \
                 patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                await asyncio.wait_for(orchestration.orchestrate_distributed_execution(
                    chained, None, 'test', ['slow']), 45)
            assert chained == saved_chained
            assert executor.success, executor.status_messages
            assert len(CAPTURED) == 2 and len(CAPTURED[1][0]) == 2
            assert all(torch.equal(image, torch.zeros((1, 8, 8, 3))) for image in CAPTURED[1][0])
            assert CAPTURED[1][1:] == ([None, None], [None, None])
            assert VALUE_CAPTURES[1][0] == [{'value_1': True}, {'value_1': True}]
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
                before_utilities, before_events = len(UTILITY_CALLS), len(ITEM_EXECUTIONS)
                async with aiohttp.ClientSession() as session:
                    before_worker_stats = []
                    for worker in workers:
                        async with session.get(f"http://127.0.0.1:{worker['port']}/test-stats") as response:
                            before_worker_stats.append(await response.json())
                with patch.object(config_mod, 'load_config', return_value=config), \
                     patch.object(orchestration, 'load_config', return_value=config), \
                     patch.object(orchestration, 'queue_prompt_payload', enqueue), \
                     patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                    queued = await asyncio.wait_for(orchestration.orchestrate_distributed_execution(
                        prompt, None, 'test', selected), 45)
                assert prompt == saved_prompt, (mode, 'saved/source graph changed')
                assert executor.success, (mode, executor.status_messages)
                assert queued[2] == len(selected)
                assert len(SOURCE_CALLS) == before_sources + 1, 'source was reevaluated per local item'
                local_calls = BRANCH_CALLS[before_local:]
                assert Counter(UTILITY_CALLS[before_utilities:]) == Counter(local_calls), mode
                assert len({event['prompt_id'] for event in ITEM_EXECUTIONS[before_events:]}) == len(local_calls)
                assert local_calls, f'{mode}: master did not process any items'
                assert len(MODEL_CALLS) == before_models + 1, 'local model dependency did not stay cached'
                remote_calls = []
                async with aiohttp.ClientSession() as session:
                    for worker, before_stats in zip(workers, before_worker_stats):
                        async with session.get(f"http://127.0.0.1:{worker['port']}/test-stats") as response:
                            current = await response.json()
                        calls = current['calls'][len(before_stats['calls']):]
                        utilities = current['utility_calls'][len(before_stats['utility_calls']):]
                        submissions = current['submissions'][len(before_stats['submissions']):]
                        events = current['item_executions'][len(before_stats['item_executions']):]
                        assert Counter(utilities) == Counter(calls), (mode, current)
                        assert len(submissions) == len(calls) == len({event['prompt_id'] for event in events})
                        for submission in submissions:
                            assert submission['requested'] == submission['queued'] == ['4'], submission
                            assert '8' in submission['node_ids'] and '9' not in submission['node_ids']
                            assert '5' not in submission['node_ids']
                        remote_calls.extend(calls)
                        assert not current['running'] and not current['pending'], current
                if mode == 'mixed':
                    assert remote_calls, 'mixed execution failed to use direct workers'
                else:
                    assert not remote_calls and len(local_calls) == COUNT
                assert Counter(local_calls + remote_calls) == Counter(range(COUNT)), (local_calls, remote_calls)
                check_carried_values()
                images, masks, texts = CAPTURED[-1]
                assert len(images) == len(masks) == len(texts) == COUNT
                for i, row in enumerate(source_rows()):
                    assert torch.equal(images[i], row['image'] + row['strength']), (mode, i)
                    assert torch.equal(masks[i], row['mask'] * 0.5), (mode, i)
                    assert texts[i] == f'item-{i}:seed-{100 + i}', (mode, texts)
                assert not runtime._MASTER_JOBS and not runtime._WORKER_ITEMS
                participation_counts[mode] = {'master': len(local_calls), 'workers': len(remote_calls)}
            print('IMAGE_LIST_MASTER_OK ' + json.dumps(participation_counts))
            print('IMAGE_LIST_VALUES_OK passthrough all six types; sparse override/add; Get; sequential Get-to-Start; direct/mixed/master-only')
            assert len(outer_executions) == 4 and all(row['captures'] == 1 for row in outer_executions)
            print('IMAGE_LIST_TARGETS_OK ' + json.dumps({
                'modes': ['delegate-only', 'mixed', 'master-only'],
                'outer_targets': outer_executions[0]['targets'], 'worker_targets': ['4'],
                'required_utility_once_per_item': True, 'preview_discarded': True,
                'saved_prompt_unchanged': prompt == saved_prompt, 'capture_once_per_outer_prompt': True}))
            ALLOW_MASTER = False
            config['settings']['master_delegate_only'] = True
            # Preserve the original direct-runtime cancellation/failure fixture:
            # it has no outer queue and must not claim target-control evidence.
            direct_prompt = copy.deepcopy(prompt)
            del direct_prompt['8'], direct_prompt['9']
            direct_prompt['3']['inputs']['image'] = ['2', 0]
            # Real Comfy user-interrupt flag while a real worker executor is active.
            rows = source_rows()
            items = runtime.normalize_image_list_inputs(
                [row['image'] for row in rows], [row['mask'] for row in rows],
                {f'value_{n}': [row[key] for row in rows]
                 for n, key in ((1, 'reference_image'), (2, 'text'), (3, 'strength'), (4, 'seed'), (6, 'mask'))}
                | {'value_5': True})
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
                context = runtime.create_master_job('2', '4', direct_prompt, items, ['slow'],
                                                  f'http://127.0.0.1:{port}')
                with patch.object(config_mod, 'load_config', return_value=config), \
                     patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                    task = asyncio.create_task(runtime.run_image_list_job(context, include_values=True))
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
            bad_rows = copy.deepcopy(items)
            bad_rows[0]['values']['value_4'] = -1
            context = runtime.create_master_job('2', '4', direct_prompt, bad_rows, ['fast'],
                                              f'http://127.0.0.1:{port}')
            with patch.object(config_mod, 'load_config', return_value=config), \
                 patch.object(config_mod, 'get_worker_timeout_seconds', return_value=15):
                try:
                    await asyncio.wait_for(runtime.run_image_list_job(context, include_values=True), 5)
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
