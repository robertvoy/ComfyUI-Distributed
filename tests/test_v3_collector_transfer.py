"""Registered V3 DistributedCollector: transfer options in the public schema and forwarded to the runtime.

Runs without a ComfyUI checkout by loading nodes/v3.py against a recording
stand-in for comfy_api's io module. tests/test_v3_migration.py covers the same
boundary against the real framework when COMFYUI_SOURCE_ROOT is set.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "dist_v3_transfer_testpkg"


class _Field:
    def __init__(self, kind, id=None, **kwargs):
        self.kind = kind
        self.id = id
        self.kwargs = kwargs


def _io_type(kind):
    return types.SimpleNamespace(
        Input=lambda id, **kw: _Field(kind, id, **kw),
        Output=lambda id=None, **kw: _Field(kind, id, **kw),
    )


def _fake_io():
    io = types.ModuleType("io")

    class ComfyNode:
        pass

    io.ComfyNode = ComfyNode
    io.Schema = lambda **kw: types.SimpleNamespace(**kw)
    io.NodeOutput = lambda *values, **kw: ("NodeOutput", values, kw)
    io.NumberDisplay = types.SimpleNamespace(number="number")
    io.Hidden = types.SimpleNamespace(unique_id="unique_id", extra_pnginfo="extra_pnginfo")
    for name in ("Int", "Float", "String", "Boolean", "Combo", "Image", "Audio",
                 "AnyType", "Model", "Conditioning", "Vae"):
        setattr(io, name, _io_type(name))
    return io


@pytest.fixture
def v3(monkeypatch):
    calls = []

    class CollectorRuntime:
        def run(self, **kwargs):
            calls.append(kwargs)
            return (kwargs["images"], kwargs["audio"])

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        monkeypatch.delitem(sys.modules, name)
    module(PKG, __path__=[])
    module(f"{PKG}.nodes", __path__=[])
    module(f"{PKG}.utils", __path__=[])
    module(f"{PKG}.nodes.utilities")
    module(f"{PKG}.nodes.distributed_upscale", UltimateSDUpscaleDistributed=object)
    module(f"{PKG}.nodes.collector", DistributedCollectorNode=CollectorRuntime)
    module(f"{PKG}.nodes.image_list", DistributedImageListStart=object, DistributedImageListEnd=object)
    module(f"{PKG}.nodes.media_collector", DistributedMediaCollector=object)
    module("comfy", __path__=[])
    module("comfy.samplers", KSampler=types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["normal"]))
    module("comfy_api", __path__=[])
    module("comfy_api.v0_0_2", io=_fake_io())

    codec_spec = importlib.util.spec_from_file_location(f"{PKG}.utils.transfer_codec", ROOT / "utils" / "transfer_codec.py")
    codec = importlib.util.module_from_spec(codec_spec)
    monkeypatch.setitem(sys.modules, f"{PKG}.utils.transfer_codec", codec)
    codec_spec.loader.exec_module(codec)

    spec = importlib.util.spec_from_file_location(f"{PKG}.nodes.v3", ROOT / "nodes" / "v3.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, f"{PKG}.nodes.v3", mod)
    spec.loader.exec_module(mod)
    return mod, codec, calls


def test_registered_collector_schema_exposes_optional_transfer_inputs(v3):
    mod, codec, _ = v3
    schema = mod.DistributedCollector.define_schema()
    inputs = {field.id: field for field in schema.inputs}

    assert [field.id for field in schema.inputs] == [
        "load_balance", "images", "audio", "transfer_format", "transfer_quality"]
    fmt = inputs["transfer_format"]
    assert fmt.kind == "Combo"
    assert fmt.kwargs["options"] == list(codec.TRANSFER_FORMATS)
    assert fmt.kwargs["default"] == "legacy_png"
    assert fmt.kwargs["optional"] is True
    quality = inputs["transfer_quality"]
    assert quality.kind == "Int"
    assert (quality.kwargs["default"], quality.kwargs["min"], quality.kwargs["max"]) == (90, 1, 100)
    assert quality.kwargs["optional"] is True
    assert schema.is_input_list is True


def test_registered_collector_forwards_transfer_options(v3):
    mod, _, calls = v3

    mod.DistributedCollector.execute(images=["img"], audio=[None], transfer_format=["av1"], transfer_quality=[70],
                                     multi_job_id=["job"], is_worker=[True])

    assert calls[-1]["transfer_format"] == ["av1"]
    assert calls[-1]["transfer_quality"] == [70]
    assert calls[-1]["multi_job_id"] == ["job"]


def test_registered_collector_defaults_to_legacy_transfer(v3):
    mod, _, calls = v3

    mod.DistributedCollector.execute(images=["img"])

    assert calls[-1]["transfer_format"] == "legacy_png"
    assert calls[-1]["transfer_quality"] == 90
