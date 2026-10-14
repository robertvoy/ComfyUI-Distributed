"""CPU-only direct-worker IMAGE return codecs and capability negotiation."""
import asyncio
import base64
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from image_list import runtime as rt
from serverless_bridge import image_codec
from serverless_bridge.image_transfer import FORMATS

ROOT = Path(__file__).resolve().parents[1]


def result(channels=4, dtype=torch.float32):
    image = torch.linspace(-0.1, 1.1, 17 * 19 * channels, dtype=dtype).reshape(1, 17, 19, channels)
    return {"image": image, "mask": torch.full((1, 17, 19), 0.123456789, dtype=torch.float64),
            "text": "exact text: \u00e9\n"}


def legacy_value(value, kind):
    if isinstance(value, str):
        return {"type": kind, "value": value}
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return {"type": kind, "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape), "data": base64.b64encode(raw).decode("ascii")}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_result_and_inputs_are_byte_for_byte_legacy(dtype):
    source = result(dtype=dtype)
    expected = {name: legacy_value(source[name], kind)
                for name, kind in (("image", "IMAGE"), ("mask", "MASK"), ("text", "STRING"))}
    body = rt.encode_request_payload(expected)
    with patch.object(image_codec, "encode_image", side_effect=AssertionError("exact used codec")):
        assert rt.encode_request_payload(rt.encode_result(source)) == body
        assert rt.encode_request_payload(rt.encode_result(source, transfer_format="exact", transfer_quality=90)) == body
    decoded = rt.decode_result(expected)
    assert torch.equal(decoded["image"].view(torch.uint8), source["image"].view(torch.uint8))
    item = {"image": source["image"], "mask": source["mask"], "index": 0, "values": {"value_1": source["image"]}}
    encoded = rt.encode_item(item)
    assert encoded["image"] == expected["image"]
    assert encoded["mask"] == expected["mask"]
    assert encoded["values"]["value_1"] == legacy_value(source["image"], "TENSOR")


@pytest.mark.parametrize("fmt", FORMATS[1:])
@pytest.mark.parametrize("channels", [3, 4])
def test_all_compressed_formats_roundtrip_only_main_image(fmt, channels):
    source = result(channels)
    encoded = rt.encode_result(source, transfer_format=fmt, transfer_quality=90)
    image = encoded["image"]
    assert set(image) == {"type", "encoding", "descriptor", "data"}
    assert image["type"] == "IMAGE"
    assert image["encoding"] == "image-transfer-v1"
    assert image["descriptor"]["format"] == fmt
    assert image["descriptor"]["shape"] == [1, 17, 19, channels]
    assert image["descriptor"]["quality"] == 90
    assert base64.b64decode(image["data"], validate=True)
    decoded = rt.decode_result(json.loads(rt.encode_request_payload(encoded)))
    assert decoded["image"].shape == source["image"].shape
    assert decoded["image"].dtype == source["image"].dtype
    assert decoded["image"].device.type == "cpu"
    assert torch.isfinite(decoded["image"]).all()
    pixels = source["image"][..., :3].float().mul(255).clamp(0, 255).to(torch.uint8).float().div(255)
    if fmt in ("legacy_png", "png", "ffv1"):
        assert torch.equal(decoded["image"][..., :3], pixels)
    else:
        assert (decoded["image"][..., :3] - pixels).abs().mean() < 0.08
    if channels == 4:
        assert torch.equal(decoded["image"][..., 3], source["image"][..., 3])
    assert encoded["mask"] == legacy_value(source["mask"], "MASK")
    assert torch.equal(decoded["mask"], source["mask"])
    assert decoded["text"] == source["text"]


@pytest.mark.parametrize("fmt,quality", [("nope", 90), (["node", 0], 90), ("png", True), ("png", 0), ("exact", 101)])
def test_invalid_options_fail_closed(fmt, quality):
    with pytest.raises(ValueError):
        rt.encode_result(result(), transfer_format=fmt, transfer_quality=quality)


def test_compressed_inputs_masks_and_generic_tensors_are_rejected():
    source = result()
    encoded = rt.encode_result(source, transfer_format="png")
    for kind in ("IMAGE", "MASK", "TENSOR", "STRING"):
        with pytest.raises(ValueError):
            rt.decode_value({**encoded["image"], "type": kind}, kind)
    item = rt.encode_item({"image": source["image"], "index": 0})
    with pytest.raises(ValueError):
        rt.decode_item({**item, "image": encoded["image"]})
    for key, kind in (("mask", "MASK"), ("text", "STRING")):
        with pytest.raises(ValueError):
            rt.decode_result({**encoded, key: {**encoded["image"], "type": kind}})


@pytest.mark.parametrize("shape", [[2, 17, 19, 3], [1, 17, 19, 1], [17, 19, 3], [True, 17, 19, 3], [1, 0, 19, 3]])
def test_invalid_image_shapes_rejected_before_codec_decode(shape):
    encoded = rt.encode_result(result(), transfer_format="png")
    encoded["image"]["descriptor"]["shape"] = shape
    with patch.object(image_codec, "decode_image", side_effect=AssertionError("native codec reached")):
        with pytest.raises(ValueError):
            rt.decode_result(encoded)


@pytest.mark.parametrize("change", [
    {"encoding": "image-transfer-v2"}, {"type": "MASK"}, {"extra": 1}, {"data": "%%%"},
    {"descriptor": None}, {"descriptor": {"encoding": "tensor-le-v1", "shape": [1, 17, 19, 4]}},
])
def test_malformed_compressed_envelopes_rejected(change):
    encoded = rt.encode_result(result(), transfer_format="png")
    encoded["image"].update(change)
    with pytest.raises(ValueError):
        rt.decode_result(encoded)


def test_single_frame_and_image_list_not_bridge_budgets(monkeypatch):
    with pytest.raises(ValueError):
        rt.encode_result({"image": torch.zeros(2, 17, 19, 3)}, transfer_format="png")
    with patch.object(image_codec, "encode_image", wraps=image_codec.encode_image) as encode:
        encoded = rt.encode_result(result(), transfer_format="png")
    assert encode.call_args.kwargs["max_bytes"] == rt.MAX_PAYLOAD_BYTES
    with patch.object(image_codec, "decode_image", wraps=image_codec.decode_image) as decode:
        rt.decode_result(encoded)
    assert decode.call_args.kwargs["max_bytes"] == rt.MAX_PAYLOAD_BYTES
    monkeypatch.setattr(rt, "MAX_PAYLOAD_BYTES", 64)
    with pytest.raises(ValueError, match="limit"):
        rt.encode_result({"image": torch.empty(1, 4, 4, 3, device="meta")}, transfer_format="png")
    with patch.object(image_codec, "decode_image", side_effect=AssertionError("oversize reached codec")):
        with pytest.raises(ValueError, match="limit"):
            rt.decode_result(encoded)


def test_compressed_request_combined_budget(monkeypatch):
    encoded = rt.encode_result(result(), transfer_format="png")
    monkeypatch.setattr(rt, "MAX_REQUEST_BYTES", len(rt.encode_request_payload(encoded)) - 1)
    with pytest.raises(ValueError, match="request"):
        rt.encode_result(result(), transfer_format="png")
    with patch.object(image_codec, "decode_image", side_effect=AssertionError("oversize reached codec")):
        with pytest.raises(ValueError, match="request"):
            rt.decode_result(encoded)


@pytest.mark.parametrize("fmt", FORMATS)
def test_return_formats_preserve_exact_sparse_override_values(fmt):
    from test_image_list_values import assert_values, values

    source = result()
    source["values"] = values()
    source["values"]["value_10"] = source["values"].pop("value_6")
    encoded = rt.encode_result(source, transfer_format=fmt, transfer_quality=73)
    assert set(encoded["values"]) == set(source["values"])
    for name, value in source["values"].items():
        assert encoded["values"][name] == rt.encode_value(value, rt.value_kind(value, name))
    decoded = rt.decode_result(json.loads(rt.encode_request_payload(encoded)))
    assert_values(decoded["values"], source["values"])
    assert torch.equal(decoded["mask"], source["mask"])
    assert decoded["text"] == source["text"]


def test_compressed_aggregate_budget_includes_override_values(monkeypatch):
    source = result()
    encoded_without_values = rt.encode_result(source, transfer_format="png")
    source["values"] = {"value_10": "x" * 500}
    encoded = rt.encode_result(source, transfer_format="png")
    monkeypatch.setattr(rt, "MAX_REQUEST_BYTES", len(rt.encode_request_payload(encoded_without_values)) + 100)
    with pytest.raises(ValueError, match="request"):
        rt.encode_result(source, transfer_format="png")
    with patch.object(image_codec, "decode_image", side_effect=AssertionError("oversize reached codec")):
        with pytest.raises(ValueError, match="request"):
            rt.decode_result(encoded)


@pytest.fixture
def dispatch_module(monkeypatch):
    name = "_image_list_transfer_tests"
    for suffix in ("", ".api", ".api.orchestration", ".utils"):
        module = types.ModuleType(name + suffix)
        module.__path__ = [str(ROOT / suffix.lstrip(".").replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name + suffix, module)
    monkeypatch.setitem(sys.modules, name + ".utils.logging", types.SimpleNamespace(log=lambda *a: None, debug_log=lambda *a: None))
    monkeypatch.setitem(sys.modules, name + ".utils.network", types.SimpleNamespace(
        build_worker_url=lambda worker, path="": "http://" + worker["id"] + path,
        get_client_session=AsyncMock(), probe_worker=AsyncMock()))
    spec = importlib.util.spec_from_file_location(name + ".api.orchestration.dispatch", ROOT / "api/orchestration/dispatch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capability():
    return {"supported": True, "protocol_version": 2, "execution_targets": True,
            "node_ids": ["DistributedImageListStart", "DistributedImageListEnd"]}


class Response:
    status = 200

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self.value


@pytest.mark.parametrize("advertised", [None, [], ["exact"], "png", ["png", 3]])
def test_nonexact_requires_advertised_capability_before_queue(dispatch_module, advertised):
    cap = capability()
    if advertised is not None:
        cap["image_transfer_formats"] = advertised
    session = types.SimpleNamespace(get=Mock(return_value=Response(cap)), post=Mock())
    dispatch_module.get_client_session = AsyncMock(return_value=session)
    with pytest.raises(RuntimeError, match="upgrade.*no work was submitted"):
        asyncio.run(dispatch_module.preflight_image_list_workers(
            [{"id": "worker"}], required_transfer_formats=["png"]))
    assert session.get.call_count == 1
    session.post.assert_not_called()


@pytest.mark.parametrize("requested", [(), ("exact",), ("png", "jpeg")])
def test_exact_legacy_or_advertised_nonexact_capability_succeeds(dispatch_module, requested):
    cap = capability()
    if "png" in requested:
        cap["image_transfer_formats"] = list(FORMATS)
    session = types.SimpleNamespace(get=Mock(side_effect=[Response(cap), Response({"queue_running": [], "queue_pending": []})]))
    dispatch_module.get_client_session = AsyncMock(return_value=session)
    assert asyncio.run(dispatch_module.preflight_image_list_workers(
        [{"id": "worker"}], required_transfer_formats=requested)) == {"worker": True}


def test_capability_endpoint_advertises_shared_formats(monkeypatch):
    from aiohttp import web
    name = "_transfer_routes"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, name + ".api", types.ModuleType(name + ".api"))
    monkeypatch.setitem(sys.modules, name + ".image_list", types.ModuleType(name + ".image_list"))
    monkeypatch.setitem(sys.modules, name + ".image_list.runtime", rt)
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=web.RouteTableDef()))))
    spec = importlib.util.spec_from_file_location(name + ".api.image_list_routes", ROOT / "api/image_list_routes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = asyncio.run(module.image_list_capabilities_endpoint(None))
    caps = json.loads(response.text)
    assert caps["image_transfer_formats"] == list(FORMATS)
    assert all(caps[name] is True for name in ("supported", "execution_targets", "value_results"))
    assert "DistributedImageListGetValue" in caps["node_ids"]


@pytest.mark.parametrize("reject_call", [None, 1, 2, 3])
@pytest.mark.parametrize("with_values", [False, True])
def test_runtime_initial_and_each_nonexact_assignment_preflight(reject_call, with_values):
    from test_image_list_runtime_boundaries import RuntimeBoundaryTests

    async def exercise():
        fixture = RuntimeBoundaryTests()
        await fixture.asyncSetUp()
        try:
            context = fixture.create(2)
            state = fixture.rt.get_master_job(context)
            state.original_prompt[state.end_id]["inputs"].update(transfer_format="png", transfer_quality=87)
            if with_values:
                state.original_prompt[state.end_id]["inputs"]["value_10"] = [state.start_id, 2]
                for item in state.items:
                    item["values"] = {"value_1": "retained"}
                original_set = fixture.rt.set_master_result

                def set_with_override(context, index, token, result):
                    return original_set(context, index, token, {**result, "values": {"value_10": index}})

                fixture.rt.set_master_result = set_with_override
            calls = []

            async def preflight(workers, **kwargs):
                calls.append(kwargs)
                assert kwargs["required_transfer_formats"] == ("png",)
                assert kwargs.get("require_value_results", False) is with_values
                if len(calls) == reject_call:
                    raise RuntimeError("upgrade worker; no work was submitted")
                return {"worker": True}

            sys.modules["_image_list_boundaries.api.orchestration.dispatch"].preflight_image_list_workers = preflight
            if reject_call:
                with pytest.raises(RuntimeError, match="upgrade"):
                    await fixture.rt.run_image_list_job(context, include_values=with_values)
                assert len(fixture.uploads) == max(0, reject_call - 2)
                assert len(fixture.prompts) == max(0, reject_call - 2)
            else:
                result_lists = await fixture.rt.run_image_list_job(context, include_values=with_values)
                assert len(calls) == 3
                assert len(fixture.uploads) == 2
                if with_values:
                    assert result_lists[3] == [{"value_1": "retained", "value_10": index} for index in range(2)]
            assert not fixture.rt._MASTER_JOBS
        finally:
            # Do not call doCleanups(): this borrowed IsolatedAsyncioTestCase has
            # no runner, but its sys.modules patch still must be restored.
            fixture.patch_modules.stop()

    asyncio.run(exercise())
