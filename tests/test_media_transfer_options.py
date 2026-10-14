"""Collector controls -> compiler -> endpoint validation, no network or GPU."""
import copy
import json

import pytest

from test_serverless_media_prompt import compiler, _prompt
from test_media_handler import handler
from test_media_collector import module, collect, native_media

FORMATS = ('exact', 'legacy_png', 'png', 'ffv1', 'jpeg', 'webp', 'h264', 'h265', 'av1')


def compile_request(fmt='exact', quality=90):
    prompt = _prompt()
    prompt['source:1'] = {'class_type': 'Source', 'inputs': {}}
    prompt['collector:1']['inputs'].update(transfer_format=fmt, transfer_quality=quality,
                                          masks=['source:1', 1], texts=['source:1', 2])
    original = copy.deepcopy(prompt)
    workflow, outputs = compiler.compile_media_prompt(prompt, 'request:1')
    assert prompt == original
    return {'workflow': workflow, 'images': [], 'media': {
        'version': 1, 'profile': 'comfy-media-v1', 'request_id': 'request:1',
        'artifacts': [], 'inputs': [], 'outputs': outputs}}


@pytest.mark.parametrize('fmt', FORMATS)
def test_compiler_and_handler_preserve_image_policy_only(fmt):
    data = compile_request(fmt, 73)
    for binding in data['media']['outputs']:
        inputs = data['workflow'][binding['node_id']]['inputs']
        if binding['kind'] == 'IMAGE' and fmt != 'exact':
            assert inputs['transfer_format'] == fmt
            assert inputs['transfer_quality'] == 73
        else:
            assert 'transfer_format' not in inputs and 'transfer_quality' not in inputs
    handler._validate_request(data)


@pytest.mark.parametrize('fmt,quality', [('JPEG',90),('',90),('jpeg',True),('jpeg',0),
                                          ('jpeg',101),('jpeg','90'),(['source:1',0],90)])
def test_compiler_rejects_invalid_or_linked_controls(fmt, quality):
    with pytest.raises(ValueError):
        compile_request(fmt, quality)


@pytest.mark.parametrize('field,value', [('transfer_format','unknown'),('transfer_quality',0),
                                        ('transfer_quality',True),('transfer_quality','90')])
def test_handler_rejects_tampered_policy(field, value):
    data = compile_request('jpeg')
    binding = next(b for b in data['media']['outputs'] if b['kind'] == 'IMAGE')
    data['workflow'][binding['node_id']]['inputs'][field] = value
    with pytest.raises(ValueError):
        handler._validate_request(data)


def test_handler_rejects_image_controls_on_mask_export():
    data = compile_request()
    binding = next(b for b in data['media']['outputs'] if b['kind'] == 'MASK')
    data['workflow'][binding['node_id']]['inputs']['transfer_format'] = 'jpeg'
    with pytest.raises(ValueError):
        handler._validate_request(data)


@pytest.mark.parametrize('fmt', FORMATS)
def test_master_passthrough_stays_native_for_every_format(module, fmt):
    media = native_media()
    actual = collect(module, **media, transfer_format=[fmt], transfer_quality=[73])
    for values, original in zip(actual, media.values()):
        assert values is original


@pytest.mark.parametrize('quality', [True, 0, 101, '90', [90,91]])
def test_node_rejects_bad_quality_even_without_remote_participants(module, quality):
    with pytest.raises(ValueError):
        collect(module, texts=['hello'], transfer_quality=quality)
