"""Execution-target proof must survive core cache annotations, not graph edits."""
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from image_list.lifecycle import has_safe_execution_targets
from test_image_list_master_runtime import graph

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def proof(monkeypatch):
    prompt = graph()
    queued = copy.deepcopy(prompt)
    entry = [0, 'owned', queued, {}, ['end']]
    server = SimpleNamespace(prompt_queue=SimpleNamespace(
        get_current_queue_volatile=lambda: ([entry], [])))
    monkeypatch.setitem(sys.modules, 'nodes', SimpleNamespace(NODE_CLASS_MAPPINGS={
        'Operation': type('Flagged', (), {'OUTPUT_NODE': True})}))
    return prompt, queued, entry, server


@pytest.mark.parametrize('value', [['digest'], [None], float('nan'), {'nested': [1, 2]}])
@pytest.mark.parametrize('side', ['executing', 'queued', 'both'])
def test_top_level_fingerprints_do_not_invalidate_proof(proof, value, side):
    prompt, queued, _, server = proof
    if side in ('executing', 'both'):
        prompt['source']['is_changed'] = value
    if side in ('queued', 'both'):
        queued['source']['is_changed'] = ['different queued fingerprint']
    assert has_safe_execution_targets(server, 'owned', prompt)
    # Comparisons must not strip or rewrite the caller's data.
    if side in ('executing', 'both'):
        assert prompt['source']['is_changed'] is value
    if side in ('queued', 'both'):
        assert queued['source']['is_changed'] == ['different queued fingerprint']


@pytest.mark.parametrize('change', ['input', 'link', 'class', 'node_id', 'nested_is_changed', 'other_metadata'])
def test_fingerprint_tolerance_does_not_hide_graph_changes(proof, change):
    prompt, _, _, server = proof
    prompt['source']['is_changed'] = ['digest']
    if change == 'input':
        prompt['source']['inputs']['seed'] = 2
    elif change == 'link':
        prompt['op']['inputs']['image'] = ['start', 1]
    elif change == 'class':
        prompt['source']['class_type'] = 'OtherSource'
    elif change == 'node_id':
        prompt['new_source'] = prompt.pop('source')
        prompt['start']['inputs']['images'] = ['new_source', 0]
    elif change == 'nested_is_changed':
        prompt['source']['inputs']['is_changed'] = True
    else:
        prompt['source']['_meta'] = {'title': 'changed'}
    assert not has_safe_execution_targets(server, 'owned', prompt)


@pytest.mark.parametrize('roots', [['op', 'end'], [], ['end', 'end'], None, [True]])
def test_fingerprints_never_grant_unscoped_or_invalid_roots(proof, roots):
    prompt, _, entry, server = proof
    prompt['source']['is_changed'] = ['digest']
    entry[4] = roots
    assert not has_safe_execution_targets(server, 'owned', prompt)


def test_real_core_queue_snapshot_and_fingerprint():
    core = Path(os.environ.get('COMFYUI_SOURCE_ROOT', str(Path(sys.executable).parents[2])))
    if not (core / 'execution.py').exists():
        pytest.skip('real ComfyUI source not available beside this interpreter')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tests/helpers/image_list_fingerprint_proof.py'), str(core)],
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FINGERPRINT_PROOF_OK' in result.stdout
