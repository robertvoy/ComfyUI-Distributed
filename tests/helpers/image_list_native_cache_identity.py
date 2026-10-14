"""Offline CPU proof against ComfyUI's real signatures and PromptExecutor.

Exercises one native worker executor and separate private master-slot lifetimes,
including their shared stable Start / fresh End contract.
"""
import asyncio
import copy
import importlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def main(core):
    sys.path.insert(0, str(Path(core).resolve()))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True  # Must precede all model-management imports.
    comfy.cli_args.args.disable_assets = True
    import execution
    import nodes
    import torch
    from comfy_execution.caching import CacheKeySetInputSignature

    name = '_native_cache_identity_probe'
    for suffix in ('', '.image_list', '.nodes', '.utils'):
        package = types.ModuleType(name + suffix)
        package.__path__ = [str(ROOT / suffix.lstrip('.').replace('.', '/'))]
        sys.modules[name + suffix] = package

    def forbidden(*args, **kwargs):
        raise AssertionError('offline native proof attempted a server-loop bridge')

    sys.modules[name + '.utils.async_helpers'] = types.SimpleNamespace(run_async_in_server_loop=forbidden)
    rt = importlib.import_module(name + '.image_list.runtime')
    boundary = importlib.import_module(name + '.nodes.image_list')
    calls = []

    class Operation:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'image': ('IMAGE',), 'mask': ('MASK',),
                                 'text': ('STRING', {'forceInput': True})}}
        RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
        FUNCTION = 'run'

        def run(self, image, mask, text):
            calls.append(text)
            return image + 1, mask + 1, text + ':done'

    nodes.NODE_CLASS_MAPPINGS.update(DistributedImageListStart=boundary.DistributedImageListStart,
                                    DistributedImageListEnd=boundary.DistributedImageListEnd,
                                    Operation=Operation)
    prompt = {
        'start': {'class_type': rt.START_CLASS, 'inputs': {'image_list_execution_id': 'outer-A'}},
        'op': {'class_type': 'Operation', 'inputs': {
            'image': ['start', 0], 'mask': ['start', 1], 'text': ['start', 4]}},
        'end': {'class_type': rt.END_CLASS, 'inputs': {
            'images': ['op', 0], 'masks': ['op', 1], 'texts': ['op', 2],
            'context': ['start', 3], 'value_3': ['start', 2]}},
    }
    data = rt.normalize_image_list_inputs(torch.zeros(1, 2, 3, 3, dtype=torch.float64),
                                          torch.ones(1, 2, 3, dtype=torch.bfloat16),
                                          {'value_1': 'original', 'value_2': 3})[0]
    region = rt.validate_image_list_regions(prompt)[0]

    def prepare(data):
        token = rt.create_worker_task_token()
        context = rt.create_master_job('start', 'end', prompt, [data], ['worker'], '')
        rt.register_assignment(context, data['index'], 'worker', token)
        identity = rt.store_worker_item(token, rt.encode_item(data))
        result = rt.build_worker_prompt(prompt, region, token, context, data['index'], '', item_identity=identity)
        result['start']['inputs']['image_list_local_execution'] = True
        result['end']['inputs']['image_list_local_execution'] = True
        return token, context, result

    async def signature(graph, executor):
        dynamic = execution.DynamicPrompt(graph)
        changed = execution.IsChangedCache('signature-only', dynamic, executor.caches.outputs)
        keys = CacheKeySetInputSignature(dynamic, graph.keys(), changed)
        await keys.add_keys(graph.keys())
        return {node: keys.get_data_key(node) for node in graph}

    async def probe():
        server = types.SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *a, **kw: None)
        executor = execution.PromptExecutor(server, cache_type=execution.CacheType.LRU,
                                            cache_args={'lru': 64, 'ram': 0, 'ram_inactive': 0})
        ta, ca, a = prepare(data)
        prompt['start']['inputs']['image_list_execution_id'] = 'outer-B'
        tb, cb, b = prepare(data)
        assert ta != tb and ca != cb
        ka, kb = await signature(a, executor), await signature(b, executor)
        assert ka['start'] == kb['start'] and ka['op'] == kb['op']
        assert ka['end'] != kb['end']
        assert a['start']['inputs'] == b['start']['inputs']
        # Exercise actual V3 list mapping, hidden.prompt/unique_id, lazy inputs,
        # native node signatures and cache hits, not a reimplementation of them.
        for number, (token, context, graph) in enumerate(((ta, ca, a), (tb, cb, b))):
            valid = await execution.validate_prompt(f'native-{number}', graph, ['end'])
            assert valid[0], valid
            await executor.execute_async(graph, f'native-{number}', {}, ['end'])
            assert executor.success, executor.status_messages
            assert token not in rt._WORKER_ITEMS
            result = rt.get_master_job(context).results[0]
            assert torch.equal(result['image'], data['image'] + 1)
            assert torch.equal(result['mask'], data['mask'] + 1)
            assert result['text'] == 'original:done'
            assert result['values'] == {'value_3': 0}
            cached = next(entry['nodes'] for event, entry in executor.status_messages if event == 'execution_cached')
            if number:
                assert 'start' in cached and 'op' in cached and 'end' not in cached, cached
            assert calls == ['original'], calls
        for failure in ('expired', 'mismatched'):
            token, context, graph = prepare(data)
            expires, staged = rt._WORKER_ITEMS[token]
            if failure == 'expired':
                rt._WORKER_ITEMS[token] = (0, staged)
            else:
                staged['image'] = staged['image'] + 9
            await executor.execute_async(graph, f'rejected-{failure}', {}, ['end'])
            assert not executor.success, failure
            assert not rt.get_master_job(context).received_indexes
            assert token not in rt._WORKER_ITEMS
            cached = next(entry['nodes'] for event, entry in executor.status_messages if event == 'execution_cached')
            assert 'start' in cached and 'op' in cached and 'end' not in cached, cached
            assert calls == ['original'], calls
        for part in ('image', 'mask', 'value', 'index'):
            changed = copy.deepcopy(data)
            if part in ('image', 'mask'):
                changed[part].reshape(-1)[0] += 1
            elif part == 'value':
                changed['values']['value_2'] = 4  # Even an unconnected slot is content.
            else:
                changed['index'] = 1
            token, context, graph = prepare(changed)
            keys = await signature(graph, executor)
            assert keys['start'] != ka['start'] and keys['op'] != ka['op'], part
            await executor.execute_async(graph, f'changed-{part}', {}, ['end'])
            assert executor.success, executor.status_messages
            assert token not in rt._WORKER_ITEMS
            assert rt.get_master_job(context).received_indexes == {changed['index']}
        assert len(calls) == 5, calls
        # Combine the real Start/End identity path with cache transfer between
        # private master slots, not just independent signature/retention probes.
        local = importlib.import_module(name + '.image_list.local_executor')
        before = len(calls)
        for number in range(2):
            token, context, graph = prepare(data)
            slot = local.LocalExecutionSlot()
            try:
                await asyncio.wrap_future(slot.submit(graph, f'local-{number}', 'end'))
                result = rt.get_master_job(context).results[0]
                assert torch.equal(result['image'], data['image'] + 1)
                assert result['values'] == {'value_3': 0}
                assert token not in rt._WORKER_ITEMS
            finally:
                await slot.close()
                rt.remove_master_job(context)
            assert local._RETAINED_OUTPUTS is not None
        assert len(calls) == before + 1, calls
        assert not rt._WORKER_ITEMS
        assert not torch.cuda.is_initialized()
        print('NATIVE_CACHE_IDENTITY_OK stable Start/op keys; fresh End key; '
              '1 operation for 2 same-data jobs; 4 changed-content misses; '
              'expired/mismatched cached inputs rejected; current delivery and cleanup')

    asyncio.run(probe())


if __name__ == '__main__':
    main(sys.argv[1])
