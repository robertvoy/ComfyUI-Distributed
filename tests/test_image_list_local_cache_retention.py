"""Run the real framework's CPU-only private-cache retention regression."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_native_local_cache_retention():
    core = os.environ.get('COMFYUI_SOURCE_ROOT')
    if not core:
        pytest.skip('Set COMFYUI_SOURCE_ROOT for CPU-only real-framework validation')
    helper = Path(__file__).parent / 'helpers' / 'local_cache_retention_check.py'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(helper), core], env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: cross-run native cache reuse' in result.stdout
