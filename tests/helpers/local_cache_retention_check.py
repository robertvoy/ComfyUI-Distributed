"""CPU-only native Comfy execution across private-slot lifetimes."""
import asyncio
import importlib
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, sys.argv[1])
import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.disable_assets = True
import execution
import nodes

pkg = '_local_retention_probe'
for name, path in [(pkg, ROOT), (pkg + '.image_list', ROOT / 'image_list')]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
local = importlib.import_module(pkg + '.image_list.local_executor')

CALLS = []
OUTPUTS = []

class Counted:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'value': ('INT',)}}
    RETURN_TYPES = ('INT',)
    FUNCTION = 'run'
    def run(self, value):
        if value < 0:
            raise ValueError('intentional retention failure')
        CALLS.append(value)
        return (value,)

class Uncached(Counted):
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

class Capture:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'value': ('INT',), 'delivery': ('STRING',)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'run'
    def run(self, value, delivery):
        OUTPUTS.append((value, delivery))
        return ()

nodes.NODE_CLASS_MAPPINGS.update(RetentionCounted=Counted, RetentionUncached=Uncached, RetentionCapture=Capture)

async def run(slot, value, delivery, uncached=False):
    prompt = {
        'source': {'class_type': 'RetentionUncached' if uncached else 'RetentionCounted', 'inputs': {'value': value}},
        'end': {'class_type': 'RetentionCapture', 'inputs': {'value': ['source', 0], 'delivery': delivery}},
    }
    await asyncio.wrap_future(slot.submit(prompt, delivery, 'end'))

async def main():
    first = local.LocalExecutionSlot()
    await run(first, 1, 'first-1')
    await run(first, 2, 'first-2')
    await first.close()
    assert local._RETAINED_OUTPUTS is not None
    second = local.LocalExecutionSlot()
    await run(second, 1, 'second-1')
    await run(second, 2, 'second-2')
    await run(second, 3, 'changed')
    await second.close()
    assert CALLS == [1, 2, 3], CALLS
    assert OUTPUTS == [(1, 'first-1'), (2, 'first-2'), (1, 'second-1'), (2, 'second-2'), (3, 'changed')], OUTPUTS
    third = local.LocalExecutionSlot()
    await run(third, 4, 'uncached-1', True)
    await run(third, 4, 'uncached-2', True)
    await third.close()
    assert CALLS[-2:] == [4, 4]
    comfy.cli_args.args.cache_none = True
    disabled = local.LocalExecutionSlot()
    await run(disabled, 1, 'disabled-1')
    await run(disabled, 1, 'disabled-2')
    await disabled.close()
    assert CALLS[-2:] == [1, 1]
    assert local._RETAINED_OUTPUTS is None
    comfy.cli_args.args.cache_none = False
    failed = local.LocalExecutionSlot()
    await run(failed, 7, 'before-failure')
    try:
        await run(failed, -1, 'failure')
    except RuntimeError:
        pass
    else:
        raise AssertionError('failure fixture unexpectedly succeeded')
    await failed.close()
    assert local._RETAINED_OUTPUTS is None
    cancelled = local.LocalExecutionSlot()
    await run(cancelled, 8, 'cancelled')
    cancelled.request_stop()
    await cancelled.close()
    assert local._RETAINED_OUTPUTS is None
    oversized = execution.LRUCache(execution.CacheKeySetInputSignature, max_size=local.NATIVE_CACHE_MAX_ENTRIES)
    empty = execution.DynamicPrompt({})
    await oversized.set_prompt(empty, [], execution.IsChangedCache('bound', empty, oversized))
    oversized.min_generation = oversized.generation
    oversized.cache = {n: n for n in range(local.NATIVE_CACHE_MAX_ENTRIES + 1)}
    local._retain_outputs(oversized)
    assert local._RETAINED_OUTPUTS is None
    print('PASS: cross-run native cache reuse, fresh delivery, input change, noncacheable nodes, cache_none, cancellation, retention bound')

asyncio.run(main())
