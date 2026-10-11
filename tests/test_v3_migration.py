"""Real-framework acceptance; opt in with COMFYUI_SOURCE_ROOT."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_native_v3_runtime():
    comfy_root = os.environ.get('COMFYUI_SOURCE_ROOT')
    if not comfy_root:
        pytest.skip('Set COMFYUI_SOURCE_ROOT to run native ComfyUI V3 acceptance')
    helper = Path(__file__).parent / 'helpers' / 'v3_runtime_check.py'
    result = subprocess.run(
        [sys.executable, str(helper), str(Path(comfy_root).resolve())],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, 'COMFYUI_IS_WORKER': '1'},
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'V3_ACCEPTANCE_OK' in result.stdout


def test_image_list_network_runtime():
    comfy_root = os.environ.get('COMFYUI_SOURCE_ROOT')
    if not comfy_root:
        pytest.skip('Set COMFYUI_SOURCE_ROOT for model-free multi-process HTTP acceptance')
    helper = Path(__file__).parent / 'helpers' / 'image_list_network_check.py'
    result = subprocess.run(
        [sys.executable, str(helper), str(Path(comfy_root).resolve())],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, 'COMFYUI_IS_WORKER': '1'},
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'IMAGE_LIST_NETWORK_OK' in result.stdout
    assert 'IMAGE_LIST_CANCEL_OK' in result.stdout
