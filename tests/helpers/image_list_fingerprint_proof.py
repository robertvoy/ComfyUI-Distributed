"""CPU-only proof against real PromptQueue and IsChangedCache; no services."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

core = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(core))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import execution
import nodes
from comfy_execution.graph import DynamicPrompt
sys.path.insert(0, str(root))
from image_list.lifecycle import has_safe_execution_targets


class Source:
    RETURN_TYPES = ('IMAGE',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'seed': ('INT',)}}

    @classmethod
    def IS_CHANGED(cls, seed):
        return f'fingerprint-{seed}'


nodes.NODE_CLASS_MAPPINGS['ProofSource'] = Source
nodes.NODE_CLASS_MAPPINGS['ProofUtility'] = type('Utility', (), {'OUTPUT_NODE': True})
prompt = {
    'source': {'class_type': 'ProofSource', 'inputs': {'seed': 7}},
    'start': {'class_type': 'DistributedImageListStart', 'inputs': {'images': ['source', 0]}},
    'op': {'class_type': 'ProofUtility', 'inputs': {'image': ['start', 0]}},
    'end': {'class_type': 'DistributedImageListEnd', 'inputs': {
        'images': ['op', 0], 'context': ['start', 3]}},
}
server = SimpleNamespace(queue_updated=lambda: None)
queue = execution.PromptQueue(server)
server.prompt_queue = queue
queue.put((0, 'owned', prompt, {}, ['end']))
item, _ = queue.get(timeout=0)
assert item[2] is prompt
snapshot = queue.get_current_queue_volatile()[0][0][2]
assert snapshot is not prompt
assert has_safe_execution_targets(server, 'owned', prompt)
cache = execution.IsChangedCache('owned', DynamicPrompt(prompt), None)
assert asyncio.run(cache.get('source')) == ['fingerprint-7']
assert prompt['source']['is_changed'] == ['fingerprint-7']
assert 'is_changed' not in snapshot['source']
assert has_safe_execution_targets(server, 'owned', prompt), 'core fingerprint invalidated safe roots'
prompt['source']['inputs']['seed'] = 8
assert not has_safe_execution_targets(server, 'owned', prompt), 'changed input accepted'
prompt['source']['inputs']['seed'] = 7
assert not has_safe_execution_targets(server, 'not-owned', prompt)
queue.get_current_queue_volatile()[0][0][4].append('op')
assert not has_safe_execution_targets(server, 'owned', prompt), 'unsafe output root accepted'
print('FINGERPRINT_PROOF_OK')
