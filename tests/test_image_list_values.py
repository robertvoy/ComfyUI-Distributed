"""Context-carried values and sparse, per-item End overrides."""
import copy
from unittest.mock import patch

import pytest
import torch

from image_list import runtime as rt
from image_list.execution_plan import prepare_image_list_execution


def value_graph():
    # Use a minimal, closed region with an override-only ancestor.
    return {
        'source': {'class_type': 'Source', 'inputs': {}},
        'start': {'class_type': rt.START_CLASS, 'inputs': {'images': ['source', 0]}},
        'override': {'class_type': 'Transform', 'inputs': {'value': ['start', 4]}},
        'end': {'class_type': rt.END_CLASS, 'inputs': {
            'images': ['start', 0], 'context': ['start', 3], 'value_1': ['override', 0],
            'value_10': ['start', 5]}},
    }


def values():
    return {'value_1': torch.arange(72, dtype=torch.float64).reshape(2, 3, 4, 3),
            'value_2': torch.ones(3, 2, 2, dtype=torch.bfloat16), 'value_3': 'name',
            'value_4': 42, 'value_5': 0.125, 'value_6': True}


def assert_values(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert type(actual[key]) is type(value)
        if isinstance(value, torch.Tensor):
            assert actual[key].dtype == value.dtype
            assert torch.equal(actual[key], value)
        else:
            assert actual[key] == value


def test_normalize_and_roundtrip_all_override_kinds_keeps_batches_whole():
    expected = values()
    result = rt.normalize_worker_result(torch.zeros(1, 2, 2, 3),
                                       values={key: [value] for key, value in expected.items()})
    assert_values(result['values'], expected)
    wire = rt.encode_result(result)
    assert_values(rt.decode_result(wire)['values'], expected)
    assert wire['values']['value_1']['shape'][0] == 2
    assert wire['values']['value_2']['shape'][0] == 3


@pytest.mark.parametrize('bad', [None, [], [None], [1, 2], [[1]], object(),
                                     float('nan'), float('inf'), torch.tensor([float('nan')]),
                                     torch.tensor([float('inf')]), torch.ones(1, dtype=torch.int32)])
def test_invalid_override_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        rt.normalize_worker_result(torch.zeros(1, 2, 2, 3), values={'value_1': bad})


def test_unknown_override_names_and_wire_types_rejected():
    for name in ('value_0', 'value_11', 'values.value_1'):
        with pytest.raises(ValueError, match='slot'):
            rt.normalize_worker_result(torch.zeros(1, 2, 2, 3), values={name: 1})
    base = rt.encode_result({'image': torch.zeros(1, 2, 2, 3)})
    for payload in (None, [], {'value_0': {'type': 'INT', 'value': 1}},
                    {'value_1': {'type': 'LATENT', 'value': {}}},
                    {'value_1': {'type': 'FLOAT', 'value': float('nan')}},
                    {'value_1': {'type': 'STRING', 'value': None}},
                    {'value_1': rt.encode_value(torch.tensor([float('inf')]), 'TENSOR')}):
        with pytest.raises((ValueError, TypeError)):
            rt.decode_result({**base, 'values': payload})


def test_no_overrides_keeps_legacy_wire_and_does_not_encode_original_tensors():
    image = torch.zeros(1, 2, 2, 3)
    result = rt.normalize_worker_result(image)
    assert set(result) == {'image', 'mask', 'text'}
    with patch.object(rt, 'encode_value', wraps=rt.encode_value) as encode:
        wire = rt.encode_result(result)
    assert set(wire) == {'image', 'mask', 'text'}
    assert [call.args[1] for call in encode.call_args_list] == ['IMAGE', 'MASK', 'STRING']
    assert set(rt.decode_result(wire)) == {'image', 'mask', 'text'}


def test_override_ancestors_survive_plan_and_worker_prompt():
    prompt = value_graph()
    frozen = copy.deepcopy(prompt)
    prepared, targets = prepare_image_list_execution(prompt)
    assert prepared == frozen and prompt == frozen
    assert targets == ['end']
    region = rt.validate_image_list_regions(prepared)[0]
    assert 'override' in region['branch_ids']
    worker = rt.build_worker_prompt(prepared, region, 'token', 'context', 0, 'http://master')
    assert worker['end']['inputs']['value_1'] == ['override', 0]
    assert worker['end']['inputs']['value_10'] == ['start', 5]
    assert 'source' not in worker
    for bad in ('value_0', 'value_11', 'values.value_1'):
        invalid = copy.deepcopy(prompt)
        invalid['end']['inputs'][bad] = ['start', 4]
        with pytest.raises(ValueError, match='slot'):
            rt.validate_image_list_regions(invalid)


def test_missing_or_unexpected_result_override_fails_before_recording():
    prompt = value_graph()
    items = rt.normalize_image_list_inputs(torch.zeros(1, 2, 2, 3), values={'value_1': 'original'})
    context = rt.create_master_job('start', 'end', prompt, items, ['worker'], '')
    try:
        rt.register_assignment(context, 0, 'worker', 'token')
        for result_values in ({}, {'value_1': 'updated'},
                              {'value_1': 'updated', 'value_10': 10, 'value_9': False}):
            with pytest.raises(ValueError, match='override'):
                rt.set_master_result(context, 0, 'token', {'image': items[0]['image'], 'values': result_values})
            assert rt.get_master_job(context).received_indexes == set()
        rt.set_master_result(context, 0, 'token', {'image': items[0]['image'],
                                                  'values': {'value_1': 'updated', 'value_10': 10}})
        assert rt.get_master_job(context).received_indexes == {0}
    finally:
        rt.remove_master_job(context)


def test_override_return_request_budget_is_aggregate():
    result = rt.normalize_worker_result(torch.zeros(1, 2, 2, 3), values={'value_1': ['x' * 500]})
    with patch.object(rt, 'MAX_REQUEST_BYTES', 400):
        with pytest.raises(ValueError, match='request'):
            rt.encode_request_payload({'result': rt.encode_result(result)})


class TestResultHTTP:
    def test_missing_and_invalid_overrides_fail_through_real_http(self):
        import asyncio
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from test_image_list_integration_gaps import load_routes
        async def check():
            _module, routes = load_routes()
            app = web.Application()
            app.add_routes(routes)
            prompt = value_graph()
            items = rt.normalize_image_list_inputs(torch.zeros(1, 2, 2, 3), values={'value_1': 'original'})
            context = rt.create_master_job('start', 'end', prompt, items, ['worker'], '')
            token = rt.create_worker_task_token()
            rt.register_assignment(context, 0, 'worker', token)
            try:
                async with TestClient(TestServer(app)) as client:
                    caps = await (await client.get('/distributed/image-list/capabilities')).json()
                    assert caps['value_results'] is True and caps['protocol_version'] == 2
                    result = rt.encode_result({'image': items[0]['image']})
                    envelope = {'context_id': context, 'token': token, 'item_index': 0, 'result': result}
                    response = await client.post('/distributed/image-list/result', json=envelope)
                    assert response.status == 400
                    assert 'missing' in (await response.json())['error']
                    assert not rt.get_master_job(context).received_indexes
                    result['values'] = {'value_1': {'type': 'INT', 'value': [1, 2]},
                                        'value_10': {'type': 'INT', 'value': 10}}
                    response = await client.post('/distributed/image-list/result', json=envelope)
                    assert response.status == 400
                    assert not rt.get_master_job(context).received_indexes
                    result['values']['value_1'] = {'type': 'STRING', 'value': 'new'}
                    response = await client.post('/distributed/image-list/result', json=envelope)
                    assert response.status == 200
                    assert rt.get_master_job(context).results[0]['values'] == {'value_1': 'new', 'value_10': 10}
            finally:
                rt.remove_master_job(context)
        asyncio.run(check())
