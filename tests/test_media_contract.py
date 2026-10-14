"""Offline contract tests: malformed results never reach transport/delivery."""
import asyncio
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("media_contract_test", ROOT / "serverless_bridge" / "contract.py")
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


def binding(**changes):
    return dict(binding_id="binding-1", collector_id="100", job_id="job-1",
                participant_id="serverless:worker:0", node_id="export-1", socket="texts", kind="STRING", **changes)


def response():
    artifact = media.artifact_from_payload("a", "STRING", {"encoding": "utf8-v1"}, b" \n")
    return {"version": 1, "profile": media.PROFILE, "request_id": "request-1",
            "bindings": [dict(binding(), count=1)], "artifacts": [artifact],
            "items": [{"binding_id": "binding-1", "item_index": 0, "artifact_id": "a"}]}


def validate(value):
    return media.validate_response(value, request_id="request-1", bindings=[binding()])


def test_valid_manifest_and_exact_inline_payload():
    value = response()
    ordered = validate(value)
    assert asyncio.run(media.resolve_artifact(ordered["binding-1"][0])) == b" \n"


def test_order_uses_indices_not_arrival():
    value = response()
    value["bindings"][0]["count"] = 2
    value["artifacts"].append(media.artifact_from_payload("b", "STRING", {"encoding": "utf8-v1"}, b"second"))
    value["items"] = [{"binding_id": "binding-1", "item_index": 1, "artifact_id": "b"}, value["items"][0]]
    assert [item["id"] for item in validate(value)["binding-1"]] == ["a", "b"]


@pytest.mark.parametrize("field,value", [("version", True), ("version", 2), ("profile", "legacy"), ("request_id", "another")])
def test_envelope_identity(field, value):
    obj = response()
    obj[field] = value
    with pytest.raises(ValueError):
        validate(obj)


@pytest.mark.parametrize("field", ["collector_id", "participant_id", "job_id", "node_id", "binding_id"])
def test_binding_identity(field):
    value = response()
    value["bindings"][0][field] = "another"
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1", 65])
def test_invalid_count(count):
    value = response()
    value["bindings"][0]["count"] = count
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("mutation", ["missing_binding", "missing_item", "extra_item", "duplicate_binding", "duplicate_artifact", "unused_artifact", "wrong_type", "wrong_index"])
def test_incomplete_or_ambiguous_manifest(mutation):
    value = response()
    if mutation == "missing_binding":
        value["bindings"] = []
    elif mutation == "missing_item":
        value["items"] = []
    elif mutation == "extra_item":
        value["items"] *= 2
    elif mutation == "duplicate_binding":
        value["bindings"] *= 2
    elif mutation == "duplicate_artifact":
        value["artifacts"] *= 2
    elif mutation == "unused_artifact":
        artifact = copy.deepcopy(value["artifacts"][0])
        artifact["id"] = "unused"
        value["artifacts"].append(artifact)
    elif mutation == "wrong_type":
        value["artifacts"][0]["kind"] = "MASK"
    elif mutation == "wrong_index":
        value["items"][0]["item_index"] = True
    with pytest.raises(ValueError):
        validate(value)


def test_duplicate_item_indices():
    value = response()
    value["bindings"][0]["count"] = 2
    value["items"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        validate(value)


def test_total_budget():
    with pytest.raises(ValueError, match="total byte"):
        media.validate_response(response(), request_id="request-1", bindings=[binding()], max_total_bytes=1)


def test_checksum_and_base64_validation():
    artifact = response()["artifacts"][0]
    artifact["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(media.resolve_artifact(artifact))
    artifact["transport"]["data_base64"] = "!!!!"
    with pytest.raises(ValueError, match="base64"):
        asyncio.run(media.resolve_artifact(artifact))


def test_empty_text_is_a_value():
    artifact = media.artifact_from_payload("empty", "STRING", {"encoding": "utf8-v1"}, b"")
    assert asyncio.run(media.resolve_artifact(artifact)) == b""


@pytest.mark.parametrize("url,expires", [("http://example.com/a", "2099-01-01T00:00:00Z"),
    ("https://user:secret@example.com/a", "2099-01-01T00:00:00Z"),
    ("https://example.com/a", "2000-01-01T00:00:00Z"),
    ("https://example.com/a", "2099-01-01"),
    ("https://example.com:1234/a", "2099-01-01T00:00:00Z")])
def test_invalid_url_envelopes(url, expires):
    with pytest.raises(ValueError):
        media.artifact_from_payload("a", "STRING", {"encoding": "utf8-v1"}, b"a",
            transport={"type": "url", "url": url, "expires_at": expires})


def test_url_resolution_is_bounded_and_hash_verified():
    artifact = media.artifact_from_payload("a", "STRING", {"encoding": "utf8-v1"}, b"hello",
        transport={"type": "url", "url": "https://example.com/a", "expires_at": "2099-01-01T00:00:00Z"})
    calls = []
    async def download(url, *, max_bytes):
        calls.append((url, max_bytes))
        return b"hello"
    assert asyncio.run(media.resolve_artifact(artifact, download=download)) == b"hello"
    assert calls == [("https://example.com/a", 5)]


def test_descriptor_depth_and_nonfinite_rejected():
    value = {}
    cursor = value
    for _ in range(15):
        cursor["next"] = {}
        cursor = cursor["next"]
    for descriptor in [value, {"x": float("nan")}, {"x": object()}]:
        with pytest.raises(ValueError):
            media.artifact_from_payload("a", "STRING", descriptor, b"")
