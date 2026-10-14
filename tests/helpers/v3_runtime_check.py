"""Exercise the actual loader, schemas and executor in a fresh CPU process.

No HTTP listener, workers, model downloads or live installation changes.
The original contracts were captured from 32ac027 using the same core.
"""
import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(COMFY_ROOT))
os.environ['COMFYUI_IS_WORKER'] = '1'
import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.disable_assets = True
from app.assets.manager import default_asset_manager
from comfy_api.v0_0_2 import io
import execution
import nodes
import server
import torch

BASELINE = json.loads((ROOT / 'tests/fixtures/v1_node_contracts.json').read_text())

# Inputs intentionally added after the V1 baseline was captured. They must be
# optional, appended after the baseline inputs, and keep legacy behaviour as
# their default so saved workflows are unaffected.
ADDED_OPTIONAL_INPUTS = {
    'DistributedCollector': {
        'transfer_format': [
            ['legacy_png', 'png', 'ffv1', 'jpeg', 'webp', 'h264', 'h265', 'av1', 'exact'],
            {'default': 'legacy_png'},
        ],
        'transfer_quality': ['INT', {'default': 90, 'min': 1, 'max': 100, 'step': 1}],
    },
}

# Nodes added after the V1 baseline; not part of saved-workflow contracts.
# DistributedWorkerSink is internal (dev-only) and only inserted by the master.
ADDED_NODES = {'DistributedWorkerSink', 'DistributedImageListStart', 'DistributedImageListEnd',
               'DistributedImageListGetValue'}


def normalized_input(value):
    kind = value[0]
    opts = dict(value[1]) if len(value) > 1 else {}
    if kind == 'STRING':
        # V3 serializes the same default single-line widget explicitly.
        opts.setdefault('multiline', False)
    if kind == 'COMBO':
        kind = opts['options']
        opts = {key: val for key, val in opts.items() if key != 'options'}
    if isinstance(kind, list):
        # Single-selection is the V1 dropdown default as well.
        opts.setdefault('multiselect', False)
    return [kind, opts]


async def map_node(cls, values, extra=None):
    """Use the actual input parser and sanitized V3 executor clones."""
    prepared, missing, hidden = execution.get_input_data(values, cls, unique_id='probe', extra_data=extra or {})
    assert not missing, missing
    # Use a worker thread for synchronous nodes: collector/upscale bridge back
    # to PromptServer's running loop, just as ComfyUI's prompt worker does.
    def run():
        return asyncio.run(execution._async_map_node_over_list(
            'v3-acceptance', 'probe', cls, prepared, cls.FUNCTION, v3_data=hidden))
    return await asyncio.to_thread(run)


def check_saved_workflows(mapping):
    checked = 0
    occurrences = 0
    for path in sorted((ROOT / 'workflows').glob('*.json')):
        workflow = json.loads(path.read_text())
        local_nodes = {node['id']: node for node in workflow['nodes']}
        for saved in workflow['nodes']:
            if saved['type'] not in mapping:
                continue
            occurrences += 1
            info = mapping[saved['type']].GET_NODE_INFO_V1()
            inputs = {**info['input'].get('required', {}), **info['input'].get('optional', {})}
            for socket in saved.get('inputs', []):
                assert socket['name'] in inputs, (path.name, saved['id'], socket)
                assert socket['type'] == inputs[socket['name']][0], (path.name, socket)
            for index, output in enumerate(saved.get('outputs', [])):
                assert output['type'] == info['output'][index], (path.name, index)
            for link in workflow['links']:
                if link[1] == saved['id']:
                    assert link[2] < len(info['output']), (path.name, link)
                    target = local_nodes[link[3]]['inputs'][link[4]]
                    assert target['type'] == info['output'][link[2]], (path.name, link)
        checked += 1
    assert checked == 5 and occurrences == 10, (checked, occurrences)
    print('SAVED_WORKFLOW_CONTRACTS_OK', checked, occurrences)


async def check_execution(mapping, module, prompt_server, asset_manager):
    result = await map_node(mapping['DistributedSeed'],
                            {'seed': 123, 'is_worker': True, 'worker_id': 'worker_2'})
    assert result[0].result == (126,)
    values = json.dumps({'_type': 'INT', '3': '17'})
    result = await map_node(mapping['DistributedValue'],
                            {'default_value': '4', 'worker_values': values,
                             'is_worker': True, 'worker_id': 'worker_2'})
    assert result[0].result == (17,)
    metadata = {'workflow': {'nodes': [{'id': 'probe', 'widgets_values': []}]}}
    result = await map_node(mapping['DistributedModelName'], {'text': 'model.ckpt'},
                            {'extra_pnginfo': metadata})
    assert result[0].result == ('model.ckpt',)
    assert result[0].ui == {'text': ['model.ckpt']}
    assert metadata['workflow']['nodes'][0]['widgets_values'] == [['model.ckpt']]
    result = await map_node(mapping['DistributedEmptyImage'],
                            {'height': 8, 'width': 8, 'channels': 3})
    empty = result[0].result[0]
    assert empty.shape == (0, 8, 8, 3) and empty.numel() == 0
    images = torch.arange(10 * 8 * 8 * 3, dtype=torch.float32).reshape(10, 8, 8, 3)
    for count in (1, 3, 10):
        result = await map_node(mapping['ImageBatchDivider'], {'images': images, 'divide_by': count})
        assert len(result[0].result) == 10
        assert torch.equal(torch.cat(result[0].result[:count]), images)
    audio = {'waveform': torch.arange(33, dtype=torch.float32).reshape(1, 1, 33), 'sample_rate': 24000}
    result = await map_node(mapping['AudioBatchDivider'], {'audio': audio, 'divide_by': 3})
    assert len(result[0].result) == 10
    assert torch.equal(torch.cat([item['waveform'] for item in result[0].result[:3]], dim=-1), audio['waveform'])
    # INPUT_IS_LIST must preserve all images/audio but unwrap transport scalars.
    collector = mapping['DistributedCollector']
    prepared, missing, hidden = execution.get_input_data(
        {'load_balance': False, 'multi_job_id': '', 'pass_through': True}, collector, 'collector-list')
    assert not missing
    prepared['images'] = [images[:2], images[2:]]
    prepared['audio'] = [audio, audio]
    result = await execution._async_map_node_over_list(
        'v3-acceptance', 'collector-list', collector, prepared, collector.FUNCTION, v3_data=hidden)
    assert torch.equal(result[0].result[0], images)
    assert result[0].result[1]['waveform'].shape[-1] == 66
    # Exercise real async aggregation without an HTTP endpoint or worker.
    queue = asyncio.Queue()
    await queue.put({'worker_id': 'worker_2', 'tensor': images[2:3], 'image_index': 0, 'is_last': True})
    prompt_server.distributed_pending_jobs['v3-aggregate'] = queue
    result = await map_node(collector,
                            {'images': images[:2], 'load_balance': False,
                             'multi_job_id': 'v3-aggregate', 'enabled_worker_ids': '["worker_2"]'})
    assert torch.equal(result[0].result[0], images[:3])
    assert 'v3-aggregate' not in prompt_server.distributed_pending_jobs

    # The registered node must forward the transfer options to the runtime
    # helper, and default to the legacy transfer when they are not set.
    collector_runtime = sys.modules[module.__name__ + '.nodes.collector'].DistributedCollectorNode
    forwarded = []
    def fake_run(self, **kwargs):
        forwarded.append(kwargs)
        return (kwargs['images'], kwargs['audio'])
    with patch.object(collector_runtime, 'run', fake_run):
        await map_node(collector, {'images': images[:1], 'load_balance': False, 'pass_through': True,
                                   'transfer_format': 'av1', 'transfer_quality': 70})
        await map_node(collector, {'images': images[:1], 'load_balance': False, 'pass_through': True})
    # Set values arrive list-wrapped (is_input_list); unset ones use the
    # execute() defaults. The runtime helper unwraps both forms.
    assert [(kw['transfer_format'], kw['transfer_quality']) for kw in forwarded] == [
        (['av1'], [70]), ('legacy_png', 90)], forwarded

    # Prove V3's per-execution class clones do not leak helper instance state;
    # replace only the GPU/model boundary, not the input parser or executor.
    upscale_runtime = sys.modules[module.__name__ + '.nodes.distributed_upscale'].UltimateSDUpscaleDistributed
    seen = []
    def fake_upscale(self, *args):
        assert not hasattr(self, 'acceptance_marker')
        self.acceptance_marker = True
        seen.append(args)
        return (args[0],)
    inputs = {'upscaled_image': images, 'model': object(), 'positive': [], 'negative': [],
              'vae': object(), 'seed': 1, 'steps': 1, 'cfg': 1.0,
              'sampler_name': 'euler', 'scheduler': 'normal', 'denoise': 0.5,
              'tile_width': 64, 'tile_height': 64, 'padding': 0, 'mask_blur': 0,
              'force_uniform_tiles': True, 'tiled_decode': False,
              'multi_job_id': 'tile-job', 'is_worker': True, 'master_url': 'http://master.invalid',
              'enabled_worker_ids': '["worker_2"]', 'worker_id': 'worker_2',
              'tile_indices': '[2,3]', 'dynamic_threshold': 9}
    with patch.object(upscale_runtime, 'run', fake_upscale):
        for _ in range(2):
            result = await map_node(mapping['UltimateSDUpscaleDistributed'], inputs)
            assert result[0].result[0] is images
    assert len(seen) == 2 and seen[0][-7:] == tuple(inputs[name] for name in (
        'multi_job_id', 'is_worker', 'master_url', 'enabled_worker_ids', 'worker_id',
        'tile_indices', 'dynamic_threshold')), seen[0]
    import math
    assert math.isnan(mapping['UltimateSDUpscaleDistributed'].fingerprint_inputs(multi_job_id='tile-job'))
    assert math.isnan(mapping['UltimateSDUpscaleDistributed'].fingerprint_inputs(multi_job_id=''))

    graph = {
        '1': {'class_type': 'EmptyImage', 'inputs': {'height': 8, 'width': 8, 'batch_size': 10, 'color': 0}},
        '2': {'class_type': 'DistributedCollector', 'inputs': {'images': ['1', 0], 'load_balance': False,
                                                              'pass_through': True}},
        '3': {'class_type': 'ImageBatchDivider', 'inputs': {'images': ['2', 0], 'divide_by': 10}},
        '4': {'class_type': 'PreviewImage', 'inputs': {'images': ['3', 9]}},
    }
    valid = await execution.validate_prompt('v3-graph', graph, None)
    assert valid[0], valid
    invalid = {'1': {'class_type': 'UltimateSDUpscaleDistributed',
                     'inputs': {**inputs, 'sampler_name': 'INVALID_ENUM', 'scheduler': 'INVALID_ENUM'}},
               '2': {'class_type': 'PreviewImage', 'inputs': {'images': ['1', 0]}}}
    rejected = await execution.validate_prompt('v3-invalid', invalid, None)
    assert not rejected[0]
    errors = [error for entry in rejected[3].values() for error in entry['errors']]
    enum_errors = [error['extra_info']['input_name'] for error in errors if error['type'] == 'value_not_in_list']
    assert {'sampler_name', 'scheduler'} <= set(enum_errors), errors
    import folder_paths
    scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
    with tempfile.TemporaryDirectory(prefix='v3-preview-', dir=scratch) as temp:
        with patch.object(folder_paths, 'temp_directory', temp):
            executor = execution.PromptExecutor(
                prompt_server, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=asset_manager)
            await asyncio.to_thread(executor.execute, graph, 'v3-graph', {}, valid[2])
            assert executor.success, executor.status_messages
            history = executor.history_result
            record = history['outputs']['4']['images'][0]
            preview = Path(temp) / record.get('subfolder', '') / record['filename']
            with Image.open(preview) as image:
                assert image.size == (8, 8) and image.mode == 'RGB'
                assert image.getextrema() == ((0, 0), (0, 0), (0, 0))

    # A worker prompt terminated by the sink validates and runs through the
    # real executor without writing anything to the temp directory.
    sink_graph = {
        '1': {'class_type': 'EmptyImage', 'inputs': {'height': 8, 'width': 8, 'batch_size': 4, 'color': 0}},
        '2': {'class_type': 'DistributedCollector', 'inputs': {'images': ['1', 0], 'load_balance': False,
                                                              'pass_through': True}},
        '3': {'class_type': 'DistributedWorkerSink', 'inputs': {'value': ['2', 0]}},
    }
    valid = await execution.validate_prompt('v3-sink', sink_graph, None)
    assert valid[0], valid
    assert valid[2] == ['3'], valid[2]
    collected = []
    def record_run(self, **kwargs):
        collected.append(kwargs['images'])
        return (kwargs['images'][0], None)
    with tempfile.TemporaryDirectory(prefix='v3-sink-', dir=scratch) as temp:
        with patch.object(folder_paths, 'temp_directory', temp), patch.object(collector_runtime, 'run', record_run):
            executor = execution.PromptExecutor(
                prompt_server, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=asset_manager)
            await asyncio.to_thread(executor.execute, sink_graph, 'v3-sink', {}, valid[2])
            assert executor.success, executor.status_messages
            assert len(collected) == 1 and collected[0][0].shape == (4, 8, 8, 3), collected
            assert list(Path(temp).rglob('*')) == [], list(Path(temp).rglob('*'))
            assert executor.history_result['outputs'].get('3', {}) == {}, executor.history_result
    print('V3_EXECUTION_OK nine nodes; upscale GPU boundary mocked; preview artifact verified; sink writes nothing')


async def check_image_list_lazy_executor(mapping, module, prompt_server, asset_manager):
    """Execute real native V3 list/lazy flow; master must skip the processing branch."""
    branch_calls = []
    class ImageListProbe:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'image': ('IMAGE',)}}
        RETURN_TYPES = ('IMAGE',)
        FUNCTION = 'run'
        CATEGORY = 'tests'
        def run(self, image):
            branch_calls.append(image)
            return (image,)

    nodes.NODE_CLASS_MAPPINGS['ImageListTestProbe'] = ImageListProbe
    runtime = sys.modules[module.__name__ + '.image_list.runtime']
    original_contexts = set(runtime._MASTER_JOBS)
    image_list_nodes = sys.modules[module.__name__ + '.nodes.image_list']
    workflow_meta = {'nodes': [{'id': 2, 'type': 'DistributedImageListStart'}], 'extra': {'acceptance': True}}
    async def fake_worker_results(_context_id, *, model_callback=None, include_values=False):
        state = runtime.get_master_job(_context_id)
        assert state.prompt_id == 'image-list-lazy', state.prompt_id
        assert state.workflow_meta == workflow_meta, state.workflow_meta
        assert len(prompt_server.app.cleanup_ctx) >= 1, 'lifecycle cleanup was not registered'
        return (
            [torch.full((1, 8, 8, 3), value, dtype=torch.float32) for value in (0.125, 0.5, 0.875)],
            [None, None, None], [None, None, None],
            [{}, {}, {}],
        )

    graph = {
        '1': {'class_type': 'EmptyImage', 'inputs': {'height': 8, 'width': 8, 'batch_size': 3, 'color': 0}},
        '2': {'class_type': 'DistributedImageListStart', 'inputs': {
            'images': ['1', 0], 'image_list_enabled_worker_ids': '["test-worker"]',
            'image_list_master_callback': 'http://127.0.0.1:1',
            'image_list_workflow_meta': workflow_meta,
        }},
        '3': {'class_type': 'ImageListTestProbe', 'inputs': {'image': ['2', 0]}},
        '4': {'class_type': 'DistributedImageListEnd', 'inputs': {
            'images': ['3', 0], 'context': ['2', 3],
        }},
        '5': {'class_type': 'PreviewImage', 'inputs': {'images': ['4', 0]}},
    }
    valid = await execution.validate_prompt('image-list-lazy', graph, None)
    assert valid[0], valid
    validator = runtime.validate_image_list_regions
    assert 'texts' not in graph['4']['inputs']
    assert len(validator(graph)) == 1
    connected_text_graph = json.loads(json.dumps(graph))
    connected_text_graph['6'] = {
        'class_type': 'ImageListTextProbe', 'inputs': {'text': ['2', 4]},
    }
    connected_text_graph['4']['inputs']['texts'] = ['6', 0]
    assert len(validator(connected_text_graph)) == 1
    for literal_text in ('', 'per-item text'):
        literal_text_graph = json.loads(json.dumps(graph))
        literal_text_graph['4']['inputs']['texts'] = literal_text
        try:
            validator(literal_text_graph)
        except ValueError:
            pass
        else:
            raise AssertionError(f'End accepted unconnected literal text: {literal_text!r}')
    print('IMAGE_LIST_END_TEXT_INPUT_OK native forceInput; unconnected omitted; linked accepted; literals rejected')
    import folder_paths
    scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
    try:
        # Resolve genuine Comfy output/temp annotations against the pinned core,
        # not only the controlled folder_paths seam in unit tests.
        media = sys.modules[module.__name__ + '.api.orchestration.media_sync']
        with tempfile.TemporaryDirectory(prefix='media-roots-', dir=scratch) as media_root:
            for kind in ('input', 'output', 'temp'):
                source = Path(media_root) / kind / 'shared/ref.png'
                source.parent.mkdir(parents=True)
                source.write_bytes(kind.encode())
            with patch.object(folder_paths, 'input_directory', str(Path(media_root) / 'input')), \
                 patch.object(folder_paths, 'output_directory', str(Path(media_root) / 'output')), \
                 patch.object(folder_paths, 'temp_directory', str(Path(media_root) / 'temp')):
                for kind in ('output', 'temp'):
                    refs = media._find_media_references({'1': {'inputs': {'image': f'shared/ref.png [{kind}]'}}})
                    assert len(refs) == 1
                    assert media._load_media_file_sync(refs[0])[0] == kind.encode()
        with tempfile.TemporaryDirectory(prefix='image-list-lazy-', dir=scratch) as temp:
            with patch.object(folder_paths, 'temp_directory', temp), \
                 patch.object(image_list_nodes, 'run_image_list_job', fake_worker_results):
                executor = execution.PromptExecutor(
                    prompt_server, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=asset_manager)
                await asyncio.to_thread(executor.execute, graph, 'image-list-lazy', {}, valid[2])
                assert executor.success, executor.status_messages
                assert branch_calls == [], 'master evaluated the lazy per-image branch'
                records = executor.history_result['outputs']['5']['images']
                assert len(records) == 3, records
                for record in records:
                    preview = Path(temp) / record.get('subfolder', '') / record['filename']
                    with Image.open(preview) as image:
                        assert image.size == (8, 8) and image.mode == 'RGB'
    finally:
        nodes.NODE_CLASS_MAPPINGS.pop('ImageListTestProbe', None)
        for context_id in set(runtime._MASTER_JOBS) - original_contexts:
            runtime.remove_master_job(context_id)
    print('IMAGE_LIST_LAZY_EXECUTION_OK real PromptExecutor; branch skipped on master; 3 previews verified')


async def main():
    asset_manager = default_asset_manager()
    prompt_server = server.PromptServer(asyncio.get_running_loop(), asset_manager)
    assert not (ROOT / 'distributed.py').exists(), 'obsolete root bootstrap remains'
    assert await nodes.load_custom_node(str(ROOT)), 'ComfyUI loader rejected the pack'
    module = sys.modules[str(ROOT).replace('.', '_x_')]
    assert not hasattr(module, 'NODE_CLASS_MAPPINGS'), 'V1 map shadows V3 entrypoint'
    extension = await module.comfy_entrypoint()
    classes = await extension.get_node_list()
    mapping = {cls.GET_SCHEMA().node_id: cls for cls in classes}
    assert len(classes) == len(mapping) == len(BASELINE) + len(ADDED_NODES) == 12
    assert set(mapping) == set(BASELINE) | ADDED_NODES
    for node_id in ADDED_NODES:
        assert nodes.NODE_CLASS_MAPPINGS[node_id] is mapping[node_id]
    assert 'DistributedMediaCollector' not in mapping
    collector_info = mapping['DistributedCollector'].GET_NODE_INFO_V1()
    assert collector_info['output'] == ['IMAGE', 'AUDIO'], collector_info
    assert not any(collector_info['output_is_list']), collector_info
    assert list(collector_info['input']['optional']) == [
        'images', 'audio', 'transfer_format', 'transfer_quality']
    sink_info = mapping['DistributedWorkerSink'].GET_NODE_INFO_V1()
    assert sink_info['output_node'] and sink_info['dev_only'], sink_info
    for node_id, cls in mapping.items():
        if node_id in ADDED_NODES:
            continue
        assert issubclass(cls, io.ComfyNode)
        assert nodes.NODE_CLASS_MAPPINGS[node_id] is cls
        old = dict(BASELINE[node_id])
        if node_id in ('ImageBatchDivider', 'AudioBatchDivider'):
            # V1's ByPassTypeTuple advertises '*' when indexed, while its
            # underlying tuple and existing frontend declare IMAGE/AUDIO.
            # Native V3 declares all ten existing typed sockets explicitly.
            assert old['output'] == ['*'] * 10
            old['output'] = ['IMAGE' if node_id == 'ImageBatchDivider' else 'AUDIO'] * 10
        new = json.loads(json.dumps(cls.GET_NODE_INFO_V1()))
        for group in ('required', 'optional'):
            old_inputs = old['input'].get(group, {})
            new_inputs = dict(new['input'].get(group, {}))
            if group == 'optional':
                for name, expected in ADDED_OPTIONAL_INPUTS.get(node_id, {}).items():
                    assert name in new_inputs, (node_id, name, 'added input missing')
                    actual = normalized_input(new_inputs.pop(name))
                    kind, opts = normalized_input(expected)
                    assert actual[0] == kind, (node_id, name, actual)
                    for key, value in opts.items():
                        assert actual[1].get(key) == value, (node_id, name, key, actual)
            assert list(old_inputs) == list(new_inputs), (node_id, group, 'input order')
            for name, original in old_inputs.items():
                assert normalized_input(original) == normalized_input(new_inputs[name]), (node_id, name, original, new_inputs[name])
        for key in ('output', 'output_name', 'output_is_list', 'is_input_list', 'output_node', 'category', 'display_name'):
            assert old[key] == new[key], (node_id, key, old[key], new[key])
        # Standard context lives in cls.hidden; orchestrator metadata remains
        # accepted by its original kwarg name, without creating new widgets.
        signature = inspect.signature(cls.execute)
        for name, field in old['input'].get('hidden', {}).items():
            if isinstance(field, list):
                assert cls.GET_SCHEMA().accept_all_inputs, node_id
                assert name in signature.parameters, (node_id, name)
                assert signature.parameters[name].default == field[1]['default'], (node_id, name)
            else:
                assert name in new['input']['hidden'], (node_id, name)
        expected_hidden = []
        if node_id == 'DistributedModelName':
            expected_hidden.extend(['unique_id', 'extra_pnginfo'])
        if old['output_node']:
            expected_hidden.extend(name for name in ['prompt', 'extra_pnginfo'] if name not in expected_hidden)
        assert list(new['input'].get('hidden', {})) == expected_hidden, (node_id, new['input'].get('hidden'))
    start_schema = mapping['DistributedImageListStart'].GET_SCHEMA()
    end_schema = mapping['DistributedImageListEnd'].GET_SCHEMA()
    assert start_schema.node_id == 'DistributedImageListStart'
    assert start_schema.display_name == 'Distributed Image List Start'
    assert end_schema.node_id == 'DistributedImageListEnd'
    assert end_schema.display_name == 'Distributed Image List End'
    runtime = sys.modules[module.__name__ + '.image_list.runtime']
    assert [input.id for input in end_schema.inputs] == ['images', 'masks', 'texts', 'context'] + list(runtime.VALUE_NAMES)
    assert [output.io_type for output in end_schema.outputs] == [
        'IMAGE', 'MASK', 'STRING', 'DISTRIBUTED_IMAGE_LIST_VALUES']
    assert end_schema.outputs[3].id == end_schema.outputs[3].display_name == 'values'
    assert all(output.is_output_list for output in end_schema.outputs)
    for override in end_schema.inputs[4:]:
        assert override.optional and override.lazy
        assert override.get_io_type() == 'IMAGE,MASK,STRING,INT,FLOAT,BOOLEAN'
    getter = mapping['DistributedImageListGetValue']
    getter_schema = getter.GET_SCHEMA()
    assert getter_schema.display_name == 'Distributed Image List Get Value'
    assert not getter_schema.is_input_list
    assert getter_schema.outputs[0].id == getter_schema.outputs[0].display_name == 'value'
    assert getter_schema.outputs[0].io_type == '*'
    assert getter_schema.inputs[1].options == list(runtime.VALUE_NAMES)
    assert getter_schema.inputs[1].default == 'value_1'
    # Real core maps scalar mappings, never arbitrary tensor batches.
    tensor = torch.ones(3, 2, 2, 3)
    data, missing, hidden = execution.get_input_data({'value_name': 'value_1'}, getter, 'get-value')
    data['values'] = [{'value_1': tensor}, {'value_1': 17}, {'value_1': False}]
    mapped = await execution._async_map_node_over_list('get-value', 'get-value', getter, data,
                                                       getter.FUNCTION, v3_data=hidden)
    mapped = await execution.resolve_map_node_over_list_results(mapped)
    assert len(mapped) == 3 and mapped[0].result[0] is tensor
    assert mapped[1].result == (17,) and mapped[2].result == (False,)
    try:
        getter.execute({}, 'value_1')
        raise AssertionError('missing Get Value key accepted')
    except ValueError as exc:
        assert 'absent' in str(exc)
    texts_input = next(input for input in end_schema.inputs if input.id == 'texts')
    texts_schema = texts_input.as_dict()
    assert texts_schema.get('forceInput') is True, texts_schema
    assert 'default' not in texts_schema, texts_schema
    assert [out.io_type for out in start_schema.outputs] == [
        'IMAGE', 'MASK', 'INT', 'DISTRIBUTED_IMAGE_LIST_CONTEXT'] + ['*'] * 10
    assert [out.display_name for out in start_schema.outputs][4:] == [f'value_{i}' for i in range(1, 11)]
    assert all(out.is_output_list for out in start_schema.outputs)
    assert start_schema.is_input_list and start_schema.accept_all_inputs
    start_info = json.loads(json.dumps(mapping['DistributedImageListStart'].GET_NODE_INFO_V1()))
    assert list(start_info['input']['optional']) == ['images', 'mask', 'values'], start_info['input']
    values_spec = start_info['input']['optional']['values']
    assert values_spec[0] == 'COMFY_AUTOGROW_V3', values_spec
    template = values_spec[1]['template']
    assert template['names'] == [f'value_{i}' for i in range(1, 11)] and template['min'] == 0, template
    assert template['input']['required']['value'][0] == 'IMAGE,MASK,STRING,INT,FLOAT,BOOLEAN', template
    # Autogrow + INPUT_IS_LIST through the real parser/list mapper, values
    # still split 1-or-N, unconnected slots ignored.
    # get_input_data must accept Autogrow names; constants are wrapped once.
    parsed, missing, hidden = execution.get_input_data(
        {'values.value_1': 'shared', 'values.value_3': 4},
        mapping['DistributedImageListStart'], unique_id='autogrow-probe', extra_data={})
    assert not missing and parsed['values.value_1'] == ['shared'], parsed
    # Linked inputs arrive as the upstream output list (one element per value).
    prepared = {'images': [torch.zeros((3, 2, 2, 3))], 'values.value_1': ['shared'],
                'values.value_3': [4, 5, 6]}
    seen = {}
    runtime_mod = sys.modules[module.__name__ + '.image_list.runtime']
    def capture(_images, _mask, values):
        seen['values'] = values
        raise RuntimeError('captured')
    nodes_mod = sys.modules[module.__name__ + '.nodes.image_list']
    with patch.object(nodes_mod, 'normalize_image_list_inputs', capture):
        try:
            await asyncio.to_thread(lambda: asyncio.run(execution._async_map_node_over_list(
                'autogrow-probe', 'autogrow-probe', mapping['DistributedImageListStart'], prepared,
                'execute', v3_data=hidden)))
        except RuntimeError as exc:
            assert 'captured' in str(exc), exc
    assert seen['values'] == {'value_1': ['shared'], 'value_3': [4, 5, 6]}, seen
    rows = runtime_mod.normalize_image_list_inputs(prepared['images'], None, seen['values'])
    assert [row['values'] for row in rows] == [{'value_1': 'shared', 'value_3': n} for n in (4, 5, 6)]
    print('IMAGE_LIST_AUTOGROW_OK real parser; value_N slots shared/per-item')
    assert all(out.is_output_list for out in end_schema.outputs)
    assert end_schema.is_input_list and end_schema.is_output_node
    print('IMAGE_LIST_SCHEMA_OK exact IDs, names, typed list outputs')
    print('SCHEMA_PARITY_OK', len(mapping))
    check_saved_workflows(mapping)
    await check_execution(mapping, module, prompt_server, asset_manager)
    await check_image_list_lazy_executor(mapping, module, prompt_server, asset_manager)
    print('V3_ACCEPTANCE_OK')


asyncio.run(main())
