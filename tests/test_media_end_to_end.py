"""Opt-in CPU acceptance of Collector IMAGE/AUDIO and the five-kind bridge.

COMFYUI_SOURCE_ROOT must name an existing compatible checkout; run with that
checkout's Python environment. No installation, normal instance, or provider is
used. The child owns its loopback server and every writable Comfy directory.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_typed_media_end_to_end_cpu():
    configured = os.environ.get("COMFYUI_SOURCE_ROOT")
    if configured is None:
        pytest.skip("Set COMFYUI_SOURCE_ROOT for real CPU typed-media end-to-end acceptance")
    core = Path(configured).expanduser().resolve()
    assert configured and all((core / name).is_file() for name in (
        "execution.py", "server.py", "comfy/cli_args.py",
        "comfy_api/latest/_input_impl/video_types.py",
    )), "Configured COMFYUI_SOURCE_ROOT must provide PromptExecutor and native V3 VIDEO"
    scratch = Path(os.environ.get("TMPDIR", str(Path.home() / ".hermes/cache/scratch")))
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ("PYTHONPATH", "COMFYUI_SOURCE_ROOT"):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", COMFYUI_IS_WORKER="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="media-e2e-", dir=scratch) as run_root:
        env["TMPDIR"] = run_root
        result = subprocess.run(
            [sys.executable, str(ROOT / "tests/helpers/media_end_to_end_check.py"),
             str(core), run_root], cwd=run_root, env=env,
            capture_output=True, text=True, timeout=150,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    for marker in ("NATIVE_INPUTS_AND_DIRECT_EXPORT_REFS_OK", "REAL_HTTP_BRIDGE_RESULTS_OK",
                   "MASTER_AND_DELEGATE_ORDER_OK", "ATOMIC_DECODE_REJECTION_OK",
                   "IMAGE_TRANSFER_HTTP_OK",
                   "MEDIA_END_TO_END_CPU_OK"):
        assert marker in result.stdout, result.stdout + result.stderr
    print(result.stdout)
