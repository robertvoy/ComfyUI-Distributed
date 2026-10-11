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
    print('V3_EXECUTION_OK eight nodes; upscale GPU boundary mocked; preview artifact verified')


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
    async def fake_worker_results(_context_id):
        return (
            [torch.full((1, 8, 8, 3), value, dtype=torch.float32) for value in (0.125, 0.5, 0.875)],
            [None, None, None], [None, None, None],
        )

    graph = {
        '1': {'class_type': 'EmptyImage', 'inputs': {'height': 8, 'width': 8, 'batch_size': 3, 'color': 0}},
        '2': {'class_type': 'DistributedImageListStart', 'inputs': {
            'images': ['1', 0], 'reference_image_mode': 'per_item', 'mask_mode': 'per_item',
            'text_mode': 'shared', 'strength': 1.0, 'strength_mode': 'shared',
            'seed': 17, 'seed_mode': 'shared', 'image_list_enabled_worker_ids': '["test-worker"]',
            'image_list_master_callback': 'http://127.0.0.1:1',
        }},
        '3': {'class_type': 'ImageListTestProbe', 'inputs': {'image': ['2', 0]}},
        '4': {'class_type': 'DistributedImageListEnd', 'inputs': {
            'images': ['3', 0], 'context': ['2', 7],
        }},
        '5': {'class_type': 'PreviewImage', 'inputs': {'images': ['4', 0]}},
    }
    valid = await execution.validate_prompt('image-list-lazy', graph, None)
    assert valid[0], valid
    import folder_paths
    scratch = Path(os.environ.get('TMPDIR', Path.home() / '.hermes/cache/scratch'))
    try:
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
    assert len(classes) == len(mapping) == len(BASELINE) + 2 == 10
    assert set(mapping) == set(BASELINE) | {'DistributedImageListStart', 'DistributedImageListEnd'}
    for node_id in BASELINE:
        cls = mapping[node_id]
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
            new_inputs = new['input'].get(group, {})
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
    assert [out.io_type for out in start_schema.outputs] == [
        'IMAGE', 'IMAGE', 'MASK', 'STRING', 'FLOAT', 'INT', 'INT', 'DISTRIBUTED_IMAGE_LIST_CONTEXT']
    assert all(out.is_output_list for out in start_schema.outputs)
    assert start_schema.is_input_list and start_schema.accept_all_inputs
    assert [out.io_type for out in end_schema.outputs] == ['IMAGE', 'MASK', 'STRING']
    assert all(out.is_output_list for out in end_schema.outputs)
    assert end_schema.is_input_list and end_schema.is_output_node
    print('IMAGE_LIST_SCHEMA_OK exact IDs, names, typed list outputs')
    print('SCHEMA_PARITY_OK', len(mapping))
    check_saved_workflows(mapping)
    await check_execution(mapping, module, prompt_server, asset_manager)
    await check_image_list_lazy_executor(mapping, module, prompt_server, asset_manager)
    print('V3_ACCEPTANCE_OK')


asyncio.run(main())
