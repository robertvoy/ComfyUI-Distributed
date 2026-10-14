"""Real CPU PromptExecutor + standalone bridge proof, no provider or service."""
import asyncio
import importlib
import json
from pathlib import Path
import shutil
import struct
import sys
import types

from media_end_to_end_check import bootstrap

ROOT = Path(__file__).resolve().parents[2]


async def check(base):
    import torch
    import execution
    import folder_paths
    import nodes
    import server
    from app.assets.manager import default_asset_manager
    from comfy_api.v0_0_2 import io

    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert execution.comfy.model_management.get_torch_device().type == 'cpu'
    # Deploy ONLY the bridge folder to a separate import root. Import it before
    # master code: it cannot rely on the Distributed image_list package.
    shutil.copytree(ROOT / 'serverless_bridge', base / 'endpoint_bridge')
    sys.path.insert(0, str(base))
    previously_loaded = set(sys.modules)
    bridge = importlib.import_module('endpoint_bridge')
    handler = importlib.import_module('endpoint_bridge.handler')
    extension = await bridge.comfy_entrypoint()
    classes = await extension.get_node_list()
    assert not any('.image_list' in name for name in set(sys.modules) - previously_loaded)
    calls = []

    class CalculateOverrides(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id='CalculateOverrides', inputs=[io.Image.Input('image'), io.Mask.Input('mask'),
                *[io.AnyType.Input(name) for name in ('integer', 'boolean', 'floating', 'text', 'reference', 'reference_mask')]],
                outputs=[io.Image.Output(), io.Mask.Output(), io.String.Output(), io.Int.Output(),
                         io.Float.Output(), io.Boolean.Output()])

        @classmethod
        def execute(cls, image, mask, integer, boolean, floating, text, reference, reference_mask):
            assert type(integer) is int and type(boolean) is bool and type(floating) is float
            calls.append(integer)
            return io.NodeOutput(reference + image, reference_mask + mask,
                                 text + ':calculated', integer + 7, floating / 2, not boolean)

    class MustNotExecute(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id='MustNotExecute', inputs=[io.Image.Input('images')], outputs=[], is_output_node=True)

        @classmethod
        def execute(cls, images):
            raise AssertionError('unrelated output was not pruned')

    for cls in [*classes, CalculateOverrides, MustNotExecute]:
        cls.GET_NODE_INFO_V1()
        nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls

    package = '_serverless_value_master'
    for suffix in ('', '.api', '.api.orchestration', '.image_list'):
        module = types.ModuleType(package + suffix)
        module.__path__ = [str(ROOT.joinpath(*suffix.lstrip('.').split('.')))] if suffix else [str(ROOT)]
        sys.modules[module.__name__] = module
    adapter = importlib.import_module(package + '.image_list.serverless')
    decode = importlib.import_module(package + '.api.orchestration.serverless_typed_results').collect_typed_results
    rt = adapter.runtime
    assets = default_asset_manager()
    ps = server.PromptServer(asyncio.get_running_loop(), assets)

    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}

    image = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4) / 997
    mask = torch.arange(6, dtype=torch.bfloat16).reshape(1, 2, 3) / 16
    values = {'value_1': 2**63 + 123, 'value_2': False, 'value_3': -0.0, 'value_4': ' 雪\r\n',
              'value_5': image.repeat(2, 1, 1, 1), 'value_6': mask.repeat(2, 1, 1),
              'value_7': 'master-only unchanged'}
    item = {'image': image, 'mask': mask, 'values': values, 'index': 0}
    graph = {
        'start': node(rt.START_CLASS),
        'calculate': node('CalculateOverrides', image=['start', 0], mask=['start', 1],
            integer=['start', 4], boolean=['start', 5], floating=['start', 6], text=['start', 7],
            reference=['start', 8], reference_mask=['start', 9]),
        'end': node(rt.END_CLASS, context=['start', 3], images=['start', 0]),
        'preview': node('MustNotExecute', images=['start', 0]),
    }
    names = ('value_1', 'value_2', 'value_4', 'value_6', 'value_8', 'value_10')
    expected = dict(zip(names, (values['value_5'] + image, values['value_6'] + mask,
        values['value_4'] + ':calculated', values['value_1'] + 7, -0.0, True)))
    graph['end']['inputs'].update({name: ['calculate', index] for index, name in enumerate(names)})
    region = {'start_id': 'start', 'end_id': 'end', 'node_ids': set(graph)}
    worker = {'id': 'pool', 'type': 'serverless', 'endpoint_id': 'ep', 'timeout_seconds': 30,
              'jobs_per_run': 16, 'enabled': True, 'media_profile': 'comfy-media-v1'}

    async def run(prompt, request_id):
        job = await adapter.prepare_serverless_item(prompt, region, item, worker,
                                                    request_id=request_id, job_id='job')
        assert not any('Distributed' in n['class_type'] for n in job['workflow'].values())
        workflow, media, legacy, targets = handler._validate_request(
            {key: job[key] for key in ('workflow', 'images', 'media')})
        stage = handler._Staging(folder_paths.get_input_directory(), request_id)

        async def submit(native):
            assert 'preview' not in native
            valid = await execution.validate_prompt(request_id, native, targets)
            assert valid[0], valid
            executor = execution.PromptExecutor(ps, cache_args={'ram': 0, 'ram_inactive': 0}, asset_manager=assets)
            await asyncio.to_thread(executor.execute, native, request_id, {}, targets)
            assert executor.success, executor.status_messages
            return executor.history_result

        async def fetch(artifact):
            transport = artifact['transport']
            path = Path(folder_paths.get_temp_directory()) / transport['subfolder'] / transport['filename']
            return path.read_bytes()

        execution_adapter = types.SimpleNamespace(media=media, targets=targets, stage=stage,
            remaining=lambda: 30, submit_and_wait=submit, fetch=fetch)
        try:
            response = await handler._run(execution_adapter, workflow, legacy, None, None)
            grouped = await decode(response, media, job['manifest'])
        finally:
            stage.close(cleanup=True)
        result = rt.normalize_worker_result(grouped['job']['images'], values={
            name: grouped['job'][name] for name in rt.VALUE_NAMES if name in grouped['job']})
        state = rt.ImageListJob('ctx', 'start', 'end', prompt, [item], ['pool'], '')
        return job, rt._collect_output_lists([result], state=state, include_values=True)[3][0]

    job, actual = await run(graph, 'calculated')
    assert set(b['socket'] for b in job['manifest']) == {'images', *names}
    assert calls == [values['value_1']]
    for name, wanted in {**values, **expected}.items():
        got = actual[name]
        assert type(got) is type(wanted), name
        if isinstance(wanted, torch.Tensor):
            assert got.dtype == wanted.dtype and got.shape == wanted.shape
            assert torch.equal(got.contiguous().view(torch.uint8), wanted.contiguous().view(torch.uint8)), name
        elif isinstance(wanted, float):
            assert struct.pack('!d', got) == struct.pack('!d', wanted), name
        else:
            assert got == wanted, name
    # Direct overrides exercise exact VALUE import injection for bool/int/float/
    # string, native tensor import slots, and no execution of the calculation.
    for name in names:
        graph['end']['inputs'].pop(name)
    graph['end']['inputs'].update({f'value_{i + 1}': ['start', i + 4] for i in range(6)})
    _, direct = await run(graph, 'passthrough')
    assert calls == [values['value_1']]
    assert type(direct['value_1']) is int and direct['value_1'] == values['value_1']
    assert direct['value_2'] is False
    assert struct.pack('!d', direct['value_3']) == struct.pack('!d', -0.0)
    assert direct['value_4'] == values['value_4']
    assert torch.equal(direct['value_5'], values['value_5'])
    assert torch.equal(direct['value_6'], values['value_6'])
    assert not torch.cuda.is_initialized()
    print('IMAGE_LIST_SERVERLESS_VALUES_CPU_OK calculated six types; sparse master overlay; standalone bridge; real PromptExecutor')


if __name__ == '__main__':
    asyncio.run(check(bootstrap()))
