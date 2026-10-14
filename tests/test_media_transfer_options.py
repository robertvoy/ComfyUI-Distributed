"""Collector controls -> compiler -> endpoint validation, no network or GPU."""
import copy
import json

import pytest

from test_serverless_media_prompt import compiler, _prompt
from test_media_handler import handler
from test_collector_typed_media import collector_module, native_media, assert_exact

FORMATS = ('legacy_png', 'png', 'ffv1', 'jpeg', 'webp', 'h264', 'h265', 'av1', 'exact')


def compile_request(fmt='legacy_png', quality=90):
    prompt = _prompt()
    prompt['source:1'] = {'class_type': 'Source', 'inputs': {}}
    prompt['collector:1']['inputs'].update(transfer_format=fmt, transfer_quality=quality,
                                          audio=['source:1', 1])
    original = copy.deepcopy(prompt)
    workflow, outputs = compiler.compile_media_prompt(prompt, 'request:1')
    assert prompt == original
    return {'workflow': workflow, 'images': [], 'media': {
        'version': 1, 'profile': 'comfy-media-v1', 'request_id': 'request:1',
        'artifacts': [], 'inputs': [], 'outputs': outputs}}


@pytest.mark.parametrize('fmt', FORMATS)
def test_compiler_and_handler_preserve_image_policy_only(fmt):
    data = compile_request(fmt, 73)
    assert {b['socket'] for b in data['media']['outputs']} == {'images', 'audio'}
    for binding in data['media']['outputs']:
        inputs = data['workflow'][binding['node_id']]['inputs']
        if binding['kind'] == 'IMAGE' and fmt != 'exact':
            assert inputs['transfer_format'] == ('png' if fmt == 'legacy_png' else fmt)
            assert inputs['transfer_quality'] == 73
        else:
            assert 'transfer_format' not in inputs and 'transfer_quality' not in inputs
    handler._validate_request(data)


def test_saved_collector_defaults_compile_to_png_not_exact():
    workflow, bindings = compiler.compile_media_prompt(_prompt(), 'request:1')
    inputs = workflow[bindings[0]['node_id']]['inputs']
    assert (inputs['transfer_format'], inputs['transfer_quality']) == ('png', 90)


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


@pytest.mark.parametrize('socket,kind', [('audio', 'AUDIO'), ('masks', 'MASK'), ('texts', 'STRING'), ('video', 'VIDEO')])
def test_handler_rejects_image_controls_on_nonimage_bridge_exports(socket, kind):
    # The bridge still supports all five kinds; these are direct bridge exports,
    # not nonexistent sockets on DistributedCollector.
    data = compile_request('exact')
    binding = next(b for b in data['media']['outputs'] if b['kind'] == 'AUDIO')
    binding.update(socket=socket, kind=kind)
    inputs = data['workflow'][binding['node_id']]['inputs']
    inputs.pop('audio')
    inputs.update({socket: ['source:1', 1], 'binding_json': json.dumps(binding)})
    handler._validate_request(data)
    inputs['transfer_format'] = 'jpeg'
    with pytest.raises(ValueError):
        handler._validate_request(data)


@pytest.mark.parametrize('fmt', FORMATS)
def test_master_passthrough_stays_native_for_every_format(collector_module, fmt):
    media = native_media()
    actual = collector_module.DistributedCollectorNode().run(
        images=[media['images']], audio=[media['audio']], pass_through=[True],
        transfer_format=[fmt], transfer_quality=[73])
    assert actual[0] is media['images']
    assert actual[1] is media['audio']
    assert_exact(actual[0], media['images'])


def test_collector_widget_contract_keeps_saved_defaults_and_appends_exact(collector_module):
    optional = collector_module.DistributedCollectorNode.INPUT_TYPES()['optional']
    assert list(optional) == ['images', 'audio', 'transfer_format', 'transfer_quality']
    assert tuple(optional['transfer_format'][0]) == FORMATS
    assert optional['transfer_format'][1]['default'] == 'legacy_png'
    assert optional['transfer_quality'][1]['default'] == 90
