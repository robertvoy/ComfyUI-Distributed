"""Stable content signatures with fresh, expiring delivery capabilities."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import AsyncMock

import pytest
import torch

from image_list import runtime as rt

ROOT = Path(__file__).resolve().parents[1]


def item():
    return rt.normalize_image_list_inputs(
        torch.arange(18, dtype=torch.float64).reshape(1, 2, 3, 3),
        torch.zeros(1, 2, 3, dtype=torch.bfloat16),
        {'value_1': 'original', 'value_2': True, 'value_3': 7, 'value_4': 0.25,
         'value_10': torch.ones(2, 3, dtype=torch.float16)})[0]


def graph():
    return {
        'source': {'class_type': 'Source', 'inputs': {}},
        'start': {'class_type': rt.START_CLASS, 'inputs': {
            'images': ['source', 0], 'image_list_execution_id': 'transient',
            'image_list_workflow_meta': {'job': 'metadata'},
            'image_list_enabled_worker_ids': '["worker"]',
            'image_list_master_callback': 'http://old-master',
            'values.value_1': 'inherited', 'worker_context_id': 'old-context'}},
        'op': {'class_type': 'Operation', 'inputs': {'image': ['start', 0]}},
        'end': {'class_type': rt.END_CLASS, 'inputs': {
            'images': ['op', 0], 'context': ['start', 3], 'value_3': ['start', 2]}},
    }


def worker_prompt(token, context, data, *, local=True):
    prompt = graph()
    result = rt.build_worker_prompt(prompt, rt.validate_image_list_regions(prompt)[0],
                                    token, context, data['index'], 'http://master',
                                    item_identity=rt.item_content_identity(data))
    if local:
        result['start']['inputs']['image_list_local_execution'] = True
        result['end']['inputs']['image_list_local_execution'] = True
    return result


@pytest.fixture(autouse=True)
def clean_state():
    yield
    rt._WORKER_ITEMS.clear()
    rt._MASTER_JOBS.clear()


@pytest.fixture
def boundaries(monkeypatch):
    """Load only the boundary module; restore every replaced module on exit."""
    name = '_native_cache_identity_tests'
    for suffix in ('', '.nodes', '.utils', '.image_list', '.api'):
        module = types.ModuleType(name + suffix)
        module.__path__ = [str(ROOT / suffix.lstrip('.').replace('.', '/'))]
        monkeypatch.setitem(sys.modules, name + suffix, module)
    monkeypatch.setitem(sys.modules, name + '.image_list.runtime', rt)
    io = types.SimpleNamespace(ComfyNode=type('ComfyNode', (), {}),
                              NodeOutput=lambda *args: args)
    for cls in ('Image', 'Mask', 'String', 'Int', 'Float', 'Boolean'):
        setattr(io, cls, object())
    monkeypatch.setitem(sys.modules, 'comfy_api', types.ModuleType('comfy_api'))
    monkeypatch.setitem(sys.modules, 'comfy_api.v0_0_2', types.SimpleNamespace(io=io))
    monkeypatch.setitem(sys.modules, name + '.utils.async_helpers',
                        types.SimpleNamespace(run_async_in_server_loop=lambda coro, **kw: asyncio.run(coro)))
    spec = importlib.util.spec_from_file_location(name + '.nodes.image_list', ROOT / 'nodes/image_list.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def start(boundaries, prompt):
    boundaries.DistributedImageListStart.hidden = types.SimpleNamespace(prompt=prompt, unique_id='start')
    return boundaries.DistributedImageListStart.execute(**prompt['start']['inputs'])


def end(boundaries, prompt, cached):
    boundaries.DistributedImageListEnd.hidden = types.SimpleNamespace(prompt=prompt, unique_id='end')
    inputs = {k: v for k, v in prompt['end']['inputs'].items()
              if k not in ('images', 'context', 'value_3')}
    return boundaries.DistributedImageListEnd.execute(
        images=cached[0], context=cached[3], value_3=cached[2], **inputs)


def stage(data):
    token = rt.create_worker_task_token()
    rt.store_worker_item(token, rt.encode_item(data))
    return token


def master(data, token):
    context = rt.create_master_job('start', 'end', graph(), [data], ['worker'], '')
    rt.register_assignment(context, data['index'], 'worker', token)
    return context


def test_identity_is_exact_typed_order_independent_and_not_transport_bounded(monkeypatch):
    original = item()
    identity = rt.item_content_identity(original)
    assert rt.is_valid_item_identity(identity)
    reordered = copy.deepcopy(original)
    reordered['values'] = dict(reversed(list(reordered['values'].items())))
    assert rt.item_content_identity(reordered) == identity
    strided = copy.deepcopy(original)
    strided['image'] = original['image'].transpose(1, 2).contiguous().transpose(1, 2)
    assert not strided['image'].is_contiguous()
    assert rt.item_content_identity(strided) == identity
    monkeypatch.setattr(rt, 'MAX_PAYLOAD_BYTES', 1)
    with pytest.raises(ValueError, match='transport limit'):
        rt.encode_item(original)
    assert rt.item_content_identity(original) == identity
    scalar_identities = set()
    for value in (True, 1, 1.0, '1', 0.0, -0.0):
        changed = copy.deepcopy(original)
        changed['values']['value_2'] = value
        scalar_identities.add(rt.item_content_identity(changed))
    assert len(scalar_identities) == 6


@pytest.mark.parametrize('change', ['image', 'mask', 'value', 'index', 'dtype', 'shape', 'slot', 'missing-mask'])
def test_every_item_component_invalidates_identity(change):
    original, changed = item(), item()
    if change in ('image', 'mask'):
        changed[change].reshape(-1)[0] = 1
    elif change == 'value':
        changed['values']['value_10'][0, 0] = 2
    elif change == 'index':
        changed['index'] = 1
    elif change == 'dtype':
        changed['image'] = changed['image'].float()
    elif change == 'shape':
        changed['values']['value_10'] = changed['values']['value_10'].reshape(3, 2)
    elif change == 'slot':
        changed['values']['value_9'] = changed['values'].pop('value_10')
    else:
        changed['mask'] = None
    assert rt.item_content_identity(original) != rt.item_content_identity(changed)


def test_float64_low_bits_and_bfloat16_bytes_are_not_quantized():
    data, changed = item(), item()
    changed['image'].reshape(-1)[0] = torch.nextafter(torch.tensor(0., dtype=torch.float64),
                                                  torch.tensor(1., dtype=torch.float64))
    assert rt.item_content_identity(data) != rt.item_content_identity(changed)
    changed = item()
    changed['mask'].view(torch.int16).reshape(-1)[0] = 1
    assert rt.item_content_identity(data) != rt.item_content_identity(changed)


def test_stable_start_replaced_wholesale_and_legacy_graph_preserved():
    prompt = graph()
    frozen = copy.deepcopy(prompt)
    region = rt.validate_image_list_regions(prompt)[0]
    legacy = rt.build_worker_prompt(prompt, region, 'a', 'ctx-a', 0, 'http://a')
    assert legacy['start']['inputs']['worker_item_token'] == 'a'
    assert legacy['start']['inputs']['image_list_execution_id'] == 'transient'
    identity = rt.item_content_identity(item())
    a = rt.build_worker_prompt(prompt, region, 'a', 'ctx-a', 0, 'http://a', item_identity=identity)
    b = rt.build_worker_prompt(prompt, region, 'b', 'ctx-b', 0, 'http://b', item_identity=identity)
    assert a['start']['inputs'] == b['start']['inputs'] == {'worker_item_identity': identity}
    assert a['end']['inputs']['worker_context_id'] == 'ctx-a'
    assert b['end']['inputs']['worker_result_token'] == 'b'
    assert b['end']['inputs']['worker_item_identity'] == identity
    assert b['end']['inputs']['value_3'] == ['start', 2]
    assert prompt == frozen


@pytest.mark.parametrize('bad', ['', 'sha256:' + 'z' * 64, 'a' * 64, False, ['bad']])
def test_malformed_identity_rejected(bad):
    assert not rt.is_valid_item_identity(bad)
    prompt = graph()
    with pytest.raises((ValueError, TypeError), match='identity'):
        rt.build_worker_prompt(prompt, rt.validate_image_list_regions(prompt)[0],
                               'token', 'context', 0, '', item_identity=bad)


def test_cache_miss_then_hit_delivers_current_context_token_and_values(boundaries):
    data = item()
    a = stage(data)
    context_a = master(data, a)
    prompt_a = worker_prompt(a, context_a, data)
    cached = start(boundaries, prompt_a)
    assert cached[3] == [rt.item_content_identity(data)]
    assert 'image' not in rt._WORKER_ITEMS[a][1]  # only a small, expiring receipt remains
    end(boundaries, prompt_a, cached)
    assert a not in rt._WORKER_ITEMS
    assert rt.get_master_job(context_a).results[0]['values'] == {'value_3': 0}
    b = stage(data)
    context_b = master(data, b)
    prompt_b = worker_prompt(b, context_b, data)
    assert prompt_a['start'] == prompt_b['start']
    # A native hit does NOT execute Start; End must consume this current upload.
    end(boundaries, prompt_b, cached)
    assert b not in rt._WORKER_ITEMS
    state = rt.get_master_job(context_b)
    assert state.expected_tokens[0] == b
    assert torch.equal(state.results[0]['image'], data['image'])
    assert rt._collect_output_lists([state.results[0]], state=state, include_values=True)[3][0]['value_3'] == 0
    with pytest.raises(RuntimeError, match='missing|expired|consumed'):
        end(boundaries, prompt_b, cached)


@pytest.mark.parametrize('miss', [False, True])
def test_expired_item_cannot_deliver_even_with_cached_start(boundaries, miss):
    data = item()
    token = stage(data)
    context = master(data, token)
    prompt = worker_prompt(token, context, data)
    cached = start(boundaries, prompt) if miss else ([data['image']], [data['mask']], [0], [rt.item_content_identity(data)])
    _, staged = rt._WORKER_ITEMS[token]
    rt._WORKER_ITEMS[token] = (0, staged)
    with pytest.raises(RuntimeError, match='missing|expired|consumed'):
        end(boundaries, prompt, cached)
    assert not rt.get_master_job(context).received_indexes
    assert token not in rt._WORKER_ITEMS


def test_identity_mismatch_on_start_and_on_cache_hit_fails_closed(boundaries):
    data, different = item(), item()
    different['image'][0, 0, 0, 0] = 3
    for hit in (False, True):
        token = stage(different)
        context = master(different, token)
        prompt = worker_prompt(token, context, data)
        cached = ([data['image']], [data['mask']], [0], [rt.item_content_identity(data)])
        with pytest.raises((ValueError, RuntimeError), match='identity'):
            end(boundaries, prompt, cached) if hit else start(boundaries, prompt)
        assert not rt.get_master_job(context).received_indexes
        rt.delete_worker_item(token)


@pytest.mark.parametrize('change', ['ambiguous', 'wrong-link', 'wrong-start-id', 'end-identity', 'end-index'])
def test_start_requires_exact_paired_end_binding(boundaries, change):
    data = item()
    token = stage(data)
    prompt = worker_prompt(token, 'context', data)
    if change == 'ambiguous':
        prompt['second-end'] = copy.deepcopy(prompt['end'])
    elif change == 'wrong-link':
        prompt['end']['inputs']['context'] = ['start', 2]
    elif change == 'wrong-start-id':
        prompt['renamed'] = prompt.pop('start')
    elif change == 'end-identity':
        prompt['end']['inputs']['worker_item_identity'] = 'sha256:' + '0' * 64
    else:
        prompt['end']['inputs']['worker_item_index'] = 1
    boundaries.DistributedImageListStart.hidden = types.SimpleNamespace(prompt=prompt, unique_id='start')
    with pytest.raises((RuntimeError, ValueError), match='paired|identity|index'):
        boundaries.DistributedImageListStart.execute(worker_item_identity=rt.item_content_identity(data),
                                                     image_list_local_execution=True)


def test_end_checks_stable_context_before_delivering(boundaries):
    data = item()
    token = stage(data)
    context = master(data, token)
    prompt = worker_prompt(token, context, data)
    cached = list(start(boundaries, prompt))
    cached[3] = ['sha256:' + '0' * 64]
    with pytest.raises((RuntimeError, ValueError), match='identity|context'):
        end(boundaries, prompt, cached)
    assert not rt.get_master_job(context).received_indexes


def test_legacy_worker_path_still_consumes_and_delivers_job_context(boundaries):
    data = item()
    token = stage(data)
    context = master(data, token)
    prompt = graph()
    prompt = rt.build_worker_prompt(prompt, rt.validate_image_list_regions(prompt)[0], token, context, 0, '')
    prompt['start']['inputs']['image_list_local_execution'] = True
    prompt['end']['inputs']['image_list_local_execution'] = True
    cached = start(boundaries, prompt)
    assert cached[3] == [context]
    assert token not in rt._WORKER_ITEMS
    end(boundaries, prompt, cached)
    assert rt.get_master_job(context).received_indexes == {0}


@pytest.mark.parametrize('fmt', ['exact', 'png'])
def test_remote_cache_hit_submits_fresh_callback_metadata(boundaries, monkeypatch, fmt):
    data = item()
    token = stage(data)
    prompt = worker_prompt(token, 'fresh-context', data, local=False)
    prompt['end']['inputs'].update(transfer_format=fmt, transfer_quality=73)
    submit = AsyncMock()
    monkeypatch.setattr(boundaries, 'submit_worker_result', submit)
    cached = ([data['image']], [data['mask']], [0], [rt.item_content_identity(data)])
    result = end(boundaries, prompt, cached)
    assert len(result) == 4
    assert submit.await_args.args[:4] == ('http://master', token, 'fresh-context', 0)
    assert submit.await_args.args[4]['values']['value_3'] == {'type': 'INT', 'value': 0}
    payload = submit.await_args.args[4]
    if fmt == 'png':
        assert payload['image']['descriptor']['format'] == fmt
        assert payload['image']['descriptor']['quality'] == 73
    else:
        assert 'encoding' not in payload['image']
    assert token not in rt._WORKER_ITEMS


def test_item_route_returns_server_computed_identity(boundaries, monkeypatch):
    from aiohttp import web
    name = '_native_cache_identity_tests'
    server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=web.RouteTableDef())))
    monkeypatch.setitem(sys.modules, 'server', server)
    spec = importlib.util.spec_from_file_location(name + '.api.image_list_routes', ROOT / 'api/image_list_routes.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    data = item()
    token = rt.create_worker_task_token()
    monkeypatch.setattr(module, '_json_body', AsyncMock(return_value={
        'token': token, 'context_id': 'context', 'item': rt.encode_item(data),
        'item_identity': 'sha256:' + '0' * 64}))
    response = asyncio.run(module.image_list_item_endpoint(object()))
    assert response.status == 200
    assert json.loads(response.body)['item_identity'] == rt.item_content_identity(data)
    assert len(rt._WORKER_ITEMS[token]) == 2


@pytest.mark.parametrize('responses', [
    ('valid', 'valid'), ('valid', 'absent'), ('invalid', 'valid'),
    ('bad-json', 'valid'), ('non-object', 'valid'),
])
def test_runtime_opts_in_per_upload_and_can_downgrade(monkeypatch, responses):
    name = '_native_cache_identity_runtime'
    for suffix in ('', '.image_list', '.api', '.api.orchestration', '.utils'):
        module = types.ModuleType(name + suffix)
        module.__path__ = [str(ROOT / suffix.lstrip('.').replace('.', '/'))]
        monkeypatch.setitem(sys.modules, name + suffix, module)
    spec = importlib.util.spec_from_file_location(name + '.image_list.runtime', ROOT / 'image_list/runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runtime)
    spec.loader.exec_module(runtime)
    workers = [{'id': 'worker', 'type': 'local'}]
    dispatched, uploads = [], []
    preflight = AsyncMock(return_value={'worker': True})
    sync_media = AsyncMock()

    class Response:
        status = 200

        def __init__(self, data):
            self.data = data

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            if self.data == 'bad-json':
                raise ValueError('old response is not JSON')
            return self.data

        def raise_for_status(self):
            pass

    class Session:
        def post(self, url, *, data, **kw):
            assert url.endswith('/distributed/image-list/item')
            body = json.loads(data)
            identity = runtime.store_worker_item(body['token'], body['item'])
            mode = responses[len(uploads)]
            uploads.append(body)
            response = {'valid': {'item_identity': identity}, 'absent': {'status': 'stored'},
                        'invalid': {'item_identity': 'bad'}, 'bad-json': 'bad-json', 'non-object': []}[mode]
            return Response(response)

        def get(self, url, **kw):
            pid = url.rsplit('/', 1)[-1]
            return Response({pid: {'status': {'completed': True, 'status_str': 'success'}}})

        def delete(self, url, **kw):
            runtime.delete_worker_item(url.rsplit('/', 1)[-1])
            return Response({})

    async def dispatch(worker, prompt, meta, **kwargs):
        dispatched.append(copy.deepcopy(prompt))
        delivery = prompt['end']['inputs']
        token, index = delivery['worker_result_token'], delivery['worker_item_index']
        identity = prompt['start']['inputs'].get('worker_item_identity')
        options = {'item_identity': identity, 'item_index': index} if identity else {}
        data = runtime.consume_worker_item(token, **options)
        if identity:
            runtime.finish_worker_item(token, identity, index)
        runtime.set_master_result(context, index, token, {
            'image': data['image'], 'text': str(index), 'values': {'value_3': index}})
        assert kwargs['partial_execution_targets'] == ['end']
        return f'worker-{index}'

    dependencies = {
        '.api.orchestration.dispatch': types.SimpleNamespace(
            preflight_image_list_workers=preflight, dispatch_worker_prompt=dispatch),
        '.api.orchestration.prompt_transform': types.SimpleNamespace(
            _resolve_direct_workers_for_image_list=lambda *args: workers),
        '.api.orchestration.media_sync': types.SimpleNamespace(sync_worker_media=sync_media),
        '.utils.config': types.SimpleNamespace(load_config=lambda: {'settings': {}},
                                              get_worker_timeout_seconds=lambda: 60),
        '.utils.network': types.SimpleNamespace(build_master_callback_url=lambda *a, **kw: 'http://master',
            build_worker_url=lambda worker, path: 'http://worker' + path,
            get_client_session=AsyncMock(return_value=Session())),
        '.utils.logging': types.SimpleNamespace(log=lambda *args: None),
    }
    for suffix, module in dependencies.items():
        monkeypatch.setitem(sys.modules, name + suffix, module)
    monkeypatch.setitem(sys.modules, 'aiohttp', types.SimpleNamespace(
        ClientTimeout=lambda **kwargs: kwargs, ClientError=ConnectionError))
    comfy = types.ModuleType('comfy')
    comfy.model_management = types.SimpleNamespace(processing_interrupted=lambda: False)
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.model_management', comfy.model_management)
    items = [item(), item()]
    items[1]['index'] = 1
    context = runtime.create_master_job('start', 'end', graph(), items, ['worker'], '')
    state = runtime.get_master_job(context)
    try:
        result = asyncio.run(runtime._run_image_list_job(state, include_values=True))
        assert result[2] == ['0', '1']
        assert [row['value_3'] for row in result[3]] == [0, 1]
        assert len(uploads) == len(dispatched) == 2
        assert sync_media.await_count == 1
        for index, prompt in enumerate(dispatched):
            start_inputs = prompt['start']['inputs']
            if responses[index] == 'valid':
                assert start_inputs == {'worker_item_identity': runtime.item_content_identity(items[index])}
                assert prompt['end']['inputs']['worker_context_id'] == context
            else:
                assert 'worker_item_identity' not in start_inputs
                assert start_inputs['worker_item_token'] == uploads[index]['token']
                assert start_inputs['worker_context_id'] == context
                assert 'worker_item_identity' not in prompt['end']['inputs']
            assert prompt['end']['inputs']['worker_result_token'] == uploads[index]['token']
        assert not runtime._WORKER_ITEMS
    finally:
        runtime.remove_master_job(context)
        runtime._WORKER_ITEMS.clear()


def test_runtime_local_path_hashes_without_transport(monkeypatch):
    name = '_native_cache_identity_local_runtime'
    for suffix in ('', '.image_list', '.api', '.api.orchestration', '.utils'):
        module = types.ModuleType(name + suffix)
        module.__path__ = [str(ROOT / suffix.lstrip('.').replace('.', '/'))]
        monkeypatch.setitem(sys.modules, name + suffix, module)
    spec = importlib.util.spec_from_file_location(name + '.image_list.runtime', ROOT / 'image_list/runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runtime)
    spec.loader.exec_module(runtime)
    dispatched = []

    class Slot:
        def __init__(self, **kw):
            pass

        def submit(self, prompt, prompt_id, end_id, workflow_meta):
            dispatched.append(prompt)
            delivery = prompt[end_id]['inputs']
            identity = prompt['start']['inputs']['worker_item_identity']
            data = runtime.consume_worker_item(delivery['worker_result_token'], item_identity=identity,
                                                item_index=delivery['worker_item_index'])
            runtime.finish_worker_item(delivery['worker_result_token'], identity, data['index'])
            runtime.set_master_result(delivery['worker_context_id'], data['index'], delivery['worker_result_token'],
                                      {'image': data['image'], 'values': {'value_3': data['index']}})
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

        def request_stop(self):
            pass

        async def close(self):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError('local path attempted HTTP/transport')

    dependencies = {
        '.image_list.local_executor': types.SimpleNamespace(LocalExecutionSlot=Slot),
        '.api.orchestration.dispatch': types.SimpleNamespace(
            preflight_image_list_workers=forbidden, dispatch_worker_prompt=forbidden),
        '.api.orchestration.prompt_transform': types.SimpleNamespace(
            _resolve_direct_workers_for_image_list=lambda *args: []),
        '.utils.config': types.SimpleNamespace(load_config=lambda: {'settings': {}},
                                              get_worker_timeout_seconds=lambda: 60),
        '.utils.network': types.SimpleNamespace(build_master_callback_url=forbidden,
                                                build_worker_url=forbidden, get_client_session=forbidden),
        '.utils.logging': types.SimpleNamespace(log=lambda *args: None),
    }
    for suffix, module in dependencies.items():
        monkeypatch.setitem(sys.modules, name + suffix, module)
    monkeypatch.setitem(sys.modules, 'aiohttp', types.SimpleNamespace(
        ClientTimeout=lambda **kwargs: kwargs, ClientError=ConnectionError))
    comfy = types.ModuleType('comfy')
    comfy.model_management = types.SimpleNamespace(processing_interrupted=lambda: False)
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.model_management', comfy.model_management)
    monkeypatch.setattr(runtime, 'MAX_PAYLOAD_BYTES', 1)
    monkeypatch.setattr(runtime, 'encode_item', forbidden)
    data = item()
    context = runtime.create_master_job('start', 'end', graph(), [data], [], '', master_participates=True)
    try:
        result = asyncio.run(runtime._run_image_list_job(runtime.get_master_job(context), include_values=True))
        assert dispatched[0]['start']['inputs'] == {'worker_item_identity': runtime.item_content_identity(data),
                                                   'image_list_local_execution': True}
        assert result[3][0]['value_3'] == 0
        assert not runtime._WORKER_ITEMS
    finally:
        runtime.remove_master_job(context)
        runtime._WORKER_ITEMS.clear()


def test_real_native_signatures_and_cached_delivery():
    core = Path(os.environ.get('COMFYUI_SOURCE_ROOT', str(Path(sys.executable).parents[2])))
    if not (core / 'comfy_execution/caching.py').exists():
        pytest.skip('real ComfyUI source not available beside this interpreter')
    result = subprocess.run([sys.executable, str(ROOT / 'tests/helpers/image_list_native_cache_identity.py'), str(core)],
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'NATIVE_CACHE_IDENTITY_OK' in result.stdout
