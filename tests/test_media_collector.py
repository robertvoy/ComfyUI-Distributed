"""CPU-only typed collection and optional native V3 mapping acceptance."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOCKETS = ('images', 'audio', 'masks', 'texts', 'video')
WORKERS = ['serverless:pool:1', 'serverless:pool:2']


class Manager:
    def __init__(self, *, pending=False, owned=True):
        self.pending, self.owned = pending, owned
        self.cancelled = []

    def owns(self, job, worker):
        return self.owned

    def is_pending(self, job, worker):
        return self.pending

    async def cancel(self, **kwargs):
        self.cancelled.append(kwargs)


@pytest.fixture
def module(monkeypatch):
    package = '_media_collector_test'
    for name, path in ((package, ROOT), (package + '.nodes', ROOT / 'nodes'),
                       (package + '.utils', ROOT / 'utils'),
                       (package + '.serverless_bridge', ROOT / 'serverless_bridge')):
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, stub)
    io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *items: types.SimpleNamespace(result=items))
    api = types.ModuleType('comfy_api.v0_0_2')
    api.io = io
    monkeypatch.setitem(sys.modules, 'comfy_api', types.ModuleType('comfy_api'))
    monkeypatch.setitem(sys.modules, 'comfy_api.v0_0_2', api)
    management = types.ModuleType('comfy.model_management')
    management.throw_exception_if_processing_interrupted = lambda: None
    comfy = types.ModuleType('comfy')
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.model_management', management)
    ps = types.SimpleNamespace(distributed_pending_jobs={}, distributed_jobs_lock=asyncio.Lock())
    server = types.ModuleType('server')
    server.PromptServer = types.SimpleNamespace(instance=ps)
    monkeypatch.setitem(sys.modules, 'server', server)
    helper = types.ModuleType(package + '.utils.async_helpers')
    helper.run_async_in_server_loop = asyncio.run
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    spec = importlib.util.spec_from_file_location(package + '.nodes.media_collector', ROOT / 'nodes/media_collector.py')
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, loaded)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(loaded, 'POLL_SECONDS', 0.005)
    monkeypatch.setattr(loaded, 'START_GRACE_SECONDS', 0.025)
    loaded.test_server = ps
    loaded.test_management = management
    return loaded


def native_media(offset=0):
    return {
        'images': [torch.full((1, 2 + offset, 3, 4), offset, dtype=torch.float64),
                   torch.zeros((2, 1, 1, 3), dtype=torch.float16)],
        'audio': [{'waveform': torch.zeros((1, 1 + offset, 8), dtype=torch.float64),
                   'sample_rate': 44100 + offset}],
        'masks': [torch.zeros((1, 2 + offset, 3), dtype=torch.float64)],
        'texts': ['', '  café\n🎥\r\n'],
        'video': [object()],
    }


def setup(module, messages=(), *, manager=None):
    queue = asyncio.Queue()
    for message in messages:
        queue.put_nowait(message)
    ps = module.test_server
    ps.distributed_pending_jobs = {'job': queue, 'unrelated': asyncio.Queue()}
    ps.distributed_serverless = manager if manager is not None else Manager()
    return ps, queue


def message(worker, media):
    return {'worker_id': worker, 'media': media, 'is_last': True}


def collect(module, **kwargs):
    return module.DistributedMediaCollector.execute(**kwargs).result


def remote(module, **kwargs):
    controls = dict(multi_job_id=['job'], enabled_worker_ids=[json.dumps(WORKERS)],
                    expected_media=[json.dumps(['texts'])], delegate_only=[True])
    controls.update(kwargs)
    return collect(module, **controls)


@pytest.mark.parametrize('controls', [{}, {'pass_through': [True]}, {'enabled_worker_ids': ['[]']}])
def test_all_five_passthrough_preserves_lists_and_native_values(module, controls):
    values = native_media()
    result = collect(module, **values, **controls)
    assert len(result) == 5
    for index, socket in enumerate(SOCKETS):
        assert result[index] is values[socket]


def test_reversed_arrivals_master_then_workers_then_items(module):
    master, first, second = native_media(), native_media(1), native_media(2)
    ps, _ = setup(module, [message(WORKERS[1], second), message(WORKERS[0], first)])
    result = remote(module, **master, delegate_only=[False], expected_media=[json.dumps(list(SOCKETS))])
    for index, socket in enumerate(SOCKETS):
        expected = master[socket] + first[socket] + second[socket]
        assert len(result[index]) == len(expected)
        assert all(actual is original for actual, original in zip(result[index], expected))
    assert list(ps.distributed_pending_jobs) == ['unrelated']
    assert ps.distributed_serverless.cancelled == []


def test_delegate_only_and_empty_string(module):
    ps, _ = setup(module, [message(WORKERS[1], {'texts': ['b']}), message(WORKERS[0], {'texts': ['']})])
    result = remote(module)
    assert result == ([], [], [], ['', 'b'], [])
    assert 'job' not in ps.distributed_pending_jobs


@pytest.mark.parametrize('kwargs', [{}, {'texts': []}, {'texts': ''}, {'texts': ['x'], 'is_worker': [True]},
    {'texts': ['x'], 'enabled_worker_ids': ['["direct-worker"]']},
    {'texts': ['x'], 'expected_media': ['["images"]']},
    {'texts': ['x'], 'pass_through': [True, False]}, {'delegate_only': True},
    {'texts': ['x'], 'enabled_worker_ids': ['["serverless:pool:1","serverless:pool:1"]']},
    {'texts': ['x'], 'load_balance': ['false']}])
def test_rejects_missing_media_or_invalid_controls(module, kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        collect(module, **kwargs)


@pytest.mark.parametrize('bad', [
    {'worker_id': 'serverless:other:1', 'media': {'texts': ['x']}, 'is_last': True},
    {'worker_id': WORKERS[0], 'media': {'texts': ['x']}, 'is_last': False},
    {'worker_id': WORKERS[0], 'media': {'images': [object()]}, 'is_last': True},
    {'worker_id': WORKERS[0], 'media': {'texts': []}, 'is_last': True},
    {'worker_id': WORKERS[0], 'media': {'texts': 'x'}, 'is_last': True},
    {'worker_id': WORKERS[0], 'media': {'texts': [None]}, 'is_last': True},
    {'worker_id': WORKERS[0], 'media': {'texts': ['x'] * 65}, 'is_last': True},
    {'worker_id': WORKERS[0], 'media': {'texts': ['x'], 'unknown': ['x']}, 'is_last': True},
    {'worker_id': WORKERS[0], 'serverless_error': 'sanitized failure', 'is_last': True},
])
def test_rejects_bad_delivery_without_partial_output_and_cleans(module, bad):
    ps, _ = setup(module, [bad])
    with pytest.raises((ValueError, RuntimeError)):
        remote(module)
    assert list(ps.distributed_pending_jobs) == ['unrelated']
    assert ps.distributed_serverless.cancelled == [{'collector_job_id': 'job'}]


@pytest.mark.parametrize('after_complete', [False, True])
def test_duplicate_delivery_rejected_including_final_queue_drain(module, after_complete):
    first = message(WORKERS[0], {'texts': ['a']})
    second = message(WORKERS[1], {'texts': ['b']})
    messages = [first, second, first] if after_complete else [first, first, second]
    ps, _ = setup(module, messages)
    with pytest.raises(RuntimeError, match='[Dd]uplicate'):
        remote(module)
    assert 'job' not in ps.distributed_pending_jobs
    assert ps.distributed_serverless.cancelled == [{'collector_job_id': 'job'}]


@pytest.mark.parametrize('manager', [None, Manager(owned=False), Manager(pending=False)])
def test_missing_manager_or_finished_participant_fails_bounded(module, manager):
    ps, _ = setup(module)
    ps.distributed_serverless = manager
    with pytest.raises(RuntimeError, match='[Ss]erverless'):
        remote(module)
    assert 'job' not in ps.distributed_pending_jobs


def test_control_error_also_cleans_exact_job(module):
    ps, _ = setup(module)
    with pytest.raises(ValueError):
        remote(module, expected_media='["unknown"]')
    assert list(ps.distributed_pending_jobs) == ['unrelated']
    assert ps.distributed_serverless.cancelled == [{'collector_job_id': 'job'}]


def test_launch_race_and_pending_deadline_owned_by_manager(module, monkeypatch):
    ps, queue = setup(module)
    ps.distributed_serverless = None
    async def runner(coro):
        async def produce():
            await asyncio.sleep(0.01)
            ps.distributed_serverless = Manager(pending=True)
            await asyncio.sleep(0.035)
            queue.put_nowait(message(WORKERS[1], {'texts': ['b']}))
            queue.put_nowait(message(WORKERS[0], {'texts': ['a']}))
        task = asyncio.create_task(produce())
        result = await coro
        await task
        return result
    monkeypatch.setattr(module, 'run_async_in_server_loop', lambda coro: asyncio.run(runner(coro)))
    assert remote(module)[3] == ['a', 'b']
    assert ps.distributed_serverless.cancelled == []


@pytest.mark.parametrize('cancel_task', [False, True])
def test_cancellation_while_blocked_cleans_and_propagates(module, monkeypatch, cancel_task):
    ps, _ = setup(module, manager=Manager(pending=True))
    class Interrupted(Exception):
        pass
    async def runner(coro):
        task = asyncio.create_task(coro)
        await asyncio.sleep(0.012)
        if cancel_task:
            task.cancel()
        else:
            def interrupt():
                raise Interrupted('stop')
            module.test_management.throw_exception_if_processing_interrupted = interrupt
        return await task
    monkeypatch.setattr(module, 'run_async_in_server_loop', lambda coro: asyncio.run(runner(coro)))
    with pytest.raises(asyncio.CancelledError if cancel_task else Interrupted):
        remote(module)
    assert list(ps.distributed_pending_jobs) == ['unrelated']
    assert ps.distributed_serverless.cancelled == [{'collector_job_id': 'job'}]


def test_replaced_queue_not_removed(module, monkeypatch):
    ps, _ = setup(module, manager=Manager(pending=True))
    replacement = asyncio.Queue()
    async def runner(coro):
        task = asyncio.create_task(coro)
        await asyncio.sleep(0.012)
        ps.distributed_pending_jobs['job'] = replacement
        return await task
    monkeypatch.setattr(module, 'run_async_in_server_loop', lambda coro: asyncio.run(runner(coro)))
    with pytest.raises(RuntimeError, match='queue'):
        remote(module)
    assert ps.distributed_pending_jobs['job'] is replacement


def test_native_v3_schema_and_list_mapping_cpu():
    core = os.environ.get('COMFYUI_SOURCE_ROOT')
    if not core:
        pytest.skip('Set COMFYUI_SOURCE_ROOT for CPU-only native V3 mapping')
    probe = r'''
import asyncio, importlib.util, sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.disable_assets = True
import execution
import torch
import server
# No HTTP listener or PromptServer bootstrap is needed for schema/list mapping.
server.PromptServer.instance = types.SimpleNamespace()
root = Path(sys.argv[2])
for name, path in [('typed_collector', root), ('typed_collector.nodes', root/'nodes'),
                   ('typed_collector.utils', root/'utils'),
                   ('typed_collector.serverless_bridge', root/'serverless_bridge')]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
spec = importlib.util.spec_from_file_location('typed_collector.nodes.media_collector', root/'nodes/media_collector.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
cls = module.DistributedMediaCollector
info = cls.GET_NODE_INFO_V1()
assert info['output'] == ['IMAGE','AUDIO','MASK','STRING','VIDEO'], info
assert info['output_is_list'] == [True]*5, info
assert cls.define_schema().is_input_list and cls.define_schema().accept_all_inputs
for name, kind in zip(['images','audio','masks','texts','video'], info['output']):
    field = info['input']['optional'][name]
    assert field[0] == kind and field[1]['forceInput'] is True, field
assert info['input']['required']['load_balance'][1]['default'] is False
async def check():
    values = {'load_balance':False, 'pass_through':True}
    prepared, missing, hidden = execution.get_input_data(values, cls, unique_id='probe')
    assert not missing, missing
    image = torch.zeros((1,2,3,4), dtype=torch.float64)
    audio = {'waveform':torch.zeros((1,2,7)), 'sample_rate':48000}
    mask = torch.ones((1,2,3), dtype=torch.float64)
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
    from fractions import Fraction
    video = VideoFromComponents(VideoComponents(images=image[:,:,:,:3].float(), frame_rate=Fraction(24,1)))
    native = {'images':[image], 'audio':[audio], 'masks':[mask], 'texts':[''], 'video':[video]}
    prepared.update(native)
    results = await execution._async_map_node_over_list('cpu','probe',cls,prepared,cls.FUNCTION,v3_data=hidden)
    results = await execution.resolve_map_node_over_list_results(results)
    assert len(results) == 1
    for index, name in enumerate(native):
        assert results[0].result[index] is native[name]
    assert results[0].result[4][0] is video
    print('MEDIA_COLLECTOR_CPU_OK')
asyncio.run(check())
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', probe, core, str(ROOT)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'MEDIA_COLLECTOR_CPU_OK' in result.stdout
