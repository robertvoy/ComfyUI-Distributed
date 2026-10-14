"""Sparse End overrides through the standalone serverless bridge, CPU only."""
import asyncio
import copy
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch

from unittest.mock import patch

from test_image_list_serverless import adapter, codec, contract, handler, item, prepare, source


@pytest.mark.skipif(not os.environ.get('COMFYUI_SOURCE_ROOT'), reason='set COMFYUI_SOURCE_ROOT for real CPU executor')
def test_real_cpu_standalone_bridge_calculated_overrides(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    helper = Path(__file__).parent / 'helpers' / 'image_list_serverless_values_check.py'
    result = subprocess.run([sys.executable, str(helper), os.environ['COMFYUI_SOURCE_ROOT'], str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'IMAGE_LIST_SERVERLESS_VALUES_CPU_OK' in result.stdout


def assert_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    elif isinstance(expected, float):
        assert struct.pack('!d', actual) == struct.pack('!d', expected)
    else:
        assert actual == expected


@pytest.mark.parametrize('value', [False, True, 2**63 + 123, -(2**100), -0.0, 1.0000000000000002,
                                    '', ' \r\n雪 ', torch.ones(2, 2, 3, 4, dtype=torch.float64),
                                    torch.ones(2, 2, 3, dtype=torch.bfloat16)])
def test_value_codec_exact_types(value):
    descriptor, payload = codec.encode_payload('VALUE', value)
    assert_exact(codec.decode_payload('VALUE', descriptor, payload), value)


@pytest.mark.parametrize('value', [None, {}, [], (1,), object(), float('inf'), float('nan'),
                                    torch.ones(2), torch.ones(1, 2, 3, dtype=torch.int64),
                                    torch.full((1, 2, 3), float('nan'))])
def test_value_codec_rejects_unsupported_or_nonfinite(value):
    with pytest.raises((ValueError, TypeError)):
        codec.encode_payload('VALUE', value)


def test_sparse_exports_never_create_collector_or_forward_unchanged_values():
    async def check():
        prompt, value = source(), item()
        prompt['ns/end']['inputs'].update(value_2=['ns/start', 5], value_10=['process', 2])
        original = copy.deepcopy(prompt)
        job = await prepare(prompt, value)
        assert prompt == original
        bindings = {binding['socket']: binding for binding in job['manifest']}
        assert set(bindings) == {'images', 'masks', 'texts', 'value_2', 'value_10'}
        assert bindings['value_2']['kind'] == bindings['value_10']['kind'] == 'VALUE'
        assert all('Collector' not in node['class_type'] for node in job['workflow'].values())
        assert all(node['class_type'] not in (adapter.runtime.START_CLASS, adapter.runtime.END_CLASS)
                   for node in job['workflow'].values())
        handler._validate_request({key: job[key] for key in ('workflow', 'images', 'media')})
    asyncio.run(check())


@pytest.mark.parametrize('slot', ['value_0', 'value_11', 'value_01', 'values.value_1'])
def test_unknown_end_override_slot_rejected(slot):
    prompt = source()
    prompt['ns/end']['inputs'][slot] = ['ns/start', 4]
    with pytest.raises(ValueError, match='slot'):
        asyncio.run(prepare(prompt))


def test_value_codec_budgets_and_malformed_payload():
    with pytest.raises(ValueError, match='byte limit'):
        codec.encode_payload('VALUE', '雪', max_bytes=2)
    descriptor, payload = codec.encode_payload('VALUE', True)
    with pytest.raises(ValueError):
        codec.decode_payload('VALUE', dict(descriptor, type='INT'), payload)
    descriptor, payload = codec.encode_payload('VALUE', -0.0)
    with pytest.raises(ValueError):
        codec.decode_payload('VALUE', descriptor, struct.pack('<d', math.inf))
    with pytest.raises(ValueError):
        codec.decode_payload('VALUE', descriptor, payload + b'0')
    with pytest.raises(ValueError, match='byte limit'):
        codec.decode_payload('VALUE', descriptor, payload, max_bytes=7)


@pytest.mark.parametrize('fmt', ['exact', 'legacy_png', 'png', 'ffv1', 'jpeg', 'webp', 'h264', 'h265', 'av1'])
def test_end_image_policy_never_compresses_override_tensors(fmt):
    prompt = source()
    prompt['ns/end']['inputs'].update(value_10=['ns/start', 0], transfer_format=fmt, transfer_quality=73)
    job = asyncio.run(prepare(prompt))
    for binding in job['manifest']:
        controls = job['workflow'][binding['node_id']]['inputs']
        if binding['socket'] == 'images' and fmt != 'exact':
            assert controls['transfer_format'] == fmt and controls['transfer_quality'] == 73
        else:
            assert 'transfer_format' not in controls and 'transfer_quality' not in controls
    default = asyncio.run(prepare())
    assert all('transfer_format' not in n['inputs'] and 'transfer_quality' not in n['inputs']
               for n in default['workflow'].values())


def value_response():
    binding = dict(binding_id='binding', collector_id='end', job_id='job', participant_id='pool',
                   node_id='export', socket='value_10', kind='VALUE')
    descriptor, payload = codec.encode_payload('VALUE', 2**63 + 7)
    artifact = contract.artifact_from_payload('artifact', 'VALUE', descriptor, payload)
    return binding, dict(version=1, profile=contract.PROFILE, request_id='request',
        bindings=[dict(binding, count=1)], artifacts=[artifact],
        items=[dict(binding_id='binding', item_index=0, artifact_id='artifact')])


@pytest.mark.parametrize('bad', ['missing', 'extra', 'wrong_kind', 'multiple', 'extra_artifact', 'nonfinite'])
def test_override_response_rejected_atomically(bad):
    binding, response = value_response()
    if bad == 'missing':
        response['bindings'] = []
    elif bad == 'extra':
        response['bindings'].append(dict(binding, binding_id='other', socket='value_9', count=1))
    elif bad == 'wrong_kind':
        response['artifacts'][0]['kind'] = 'STRING'
    elif bad == 'multiple':
        response['bindings'][0]['count'] = 2
    elif bad == 'extra_artifact':
        response['artifacts'].append(dict(response['artifacts'][0], id='extra'))
    else:
        descriptor, _ = codec.encode_payload('VALUE', 1.0)
        response['artifacts'][0] = contract.artifact_from_payload(
            'artifact', 'VALUE', descriptor, struct.pack('<d', float('nan')))
    with pytest.raises(ValueError):
        asyncio.run(contract.decode_response(response, request_id='request', bindings=[binding]))


def test_override_request_and_result_budgets():
    prompt = source()
    prompt['ns/end']['inputs']['value_10'] = '雪' * 100
    with patch.object(contract, 'MAX_ARTIFACT_BYTES', 128):
        with pytest.raises(ValueError, match='byte limit'):
            asyncio.run(prepare(prompt))
    binding, response = value_response()
    with pytest.raises(ValueError, match='total byte limit'):
        asyncio.run(contract.decode_response(response, request_id='request', bindings=[binding], max_total_bytes=1))


def test_collector_compiler_keeps_only_its_public_sockets():
    from importlib import import_module
    compiler = import_module(adapter.__package__.rsplit('.', 1)[0] + '.api.orchestration.serverless_media_prompt')
    assert compiler.COLLECTOR_SOCKETS == ('images', 'audio')
    graph = {'source': {'class_type': 'Source', 'inputs': {}},
             'collector': {'class_type': 'DistributedCollector', 'inputs': {
                 'images': ['source', 0], 'value_10': ['source', 1], 'masks': ['source', 2],
                 'multi_job_id': 'job', 'worker_id': 'pool'}}}
    _, bindings = compiler.compile_media_prompt(graph, 'request')
    assert [binding['socket'] for binding in bindings] == ['images']
