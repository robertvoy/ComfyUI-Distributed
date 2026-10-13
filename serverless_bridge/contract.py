"""Versioned, bounded media envelopes shared by master and endpoint bridge.

No provider or ComfyUI bootstrap imports. Network I/O is supplied by the caller,
which must enforce public-HTTPS resolution, redirect and streaming byte limits.
"""
import base64
import binascii
import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

PROFILE = "comfy-media-v1"
VERSION = 1
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARTIFACTS = 128
MAX_ITEMS = 64
MAX_DESCRIPTOR_BYTES = 128 * 1024
SOCKET_KINDS = {"images": "IMAGE", "audio": "AUDIO", "masks": "MASK", "texts": "STRING", "video": "VIDEO"}
BINDING_FIELDS = {"binding_id", "collector_id", "job_id", "participant_id", "node_id", "socket", "kind"}
_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,200}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")


def identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid media identifier")
    return value


def bounded_int(value, maximum, *, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Invalid media count or size")
    return value


def validate_json(value, *, max_bytes=MAX_DESCRIPTOR_BYTES):
    """Bound metadata depth and size without accepting non-finite JSON numbers."""
    def walk(item, depth=0):
        if depth > 12:
            raise ValueError("Media metadata nesting exceeds limit")
        if isinstance(item, dict):
            if len(item) > 4096 or any(not isinstance(k, str) for k in item):
                raise ValueError("Invalid media metadata object")
            for child in item.values():
                walk(child, depth + 1)
        elif isinstance(item, list):
            if len(item) > 4096:
                raise ValueError("Media metadata list exceeds limit")
            for child in item:
                walk(child, depth + 1)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("Invalid media metadata value")
    walk(value)
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("Invalid media JSON") from None
    if len(raw) > max_bytes:
        raise ValueError("Media JSON exceeds byte limit")
    return raw


def validate_binding(binding):
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
        raise ValueError("Invalid media binding")
    for field in BINDING_FIELDS - {"socket", "kind"}:
        identifier(binding[field])
    if SOCKET_KINDS.get(binding["socket"]) != binding["kind"]:
        raise ValueError("Media socket/type mismatch")
    return binding


def validate_bindings(bindings):
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= MAX_ARTIFACTS:
        raise ValueError("Invalid media bindings count")
    result = {}
    destinations = set()
    for binding in bindings:
        validate_binding(binding)
        bid = binding["binding_id"]
        destination = (binding["job_id"], binding["participant_id"], binding["collector_id"], binding["socket"])
        if bid in result or destination in destinations:
            raise ValueError("Duplicate media binding")
        result[bid] = binding
        destinations.add(destination)
    return result


def _validate_url_transport(transport):
    if set(transport) != {"type", "url", "expires_at"}:
        raise ValueError("Invalid media URL transport")
    url = transport["url"]
    if not isinstance(url, str) or len(url) > 8192:
        raise ValueError("Invalid media URL")
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError()
        if parsed.port not in (None, 443):
            raise ValueError()
        expiry = datetime.fromisoformat(transport["expires_at"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid or expired media URL") from None


def validate_artifact(artifact, *, max_bytes=MAX_ARTIFACT_BYTES):
    fields = {"id", "kind", "descriptor", "byte_length", "sha256", "transport"}
    if not isinstance(artifact, dict) or set(artifact) != fields:
        raise ValueError("Invalid media artifact")
    identifier(artifact["id"])
    if artifact["kind"] not in SOCKET_KINDS.values():
        raise ValueError("Unsupported media kind")
    if not isinstance(artifact["descriptor"], dict):
        raise ValueError("Invalid media descriptor")
    validate_json(artifact["descriptor"])
    size = bounded_int(artifact["byte_length"], max_bytes)
    digest = artifact["sha256"]
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("Invalid media checksum")
    transport = artifact["transport"]
    if not isinstance(transport, dict):
        raise ValueError("Invalid media transport")
    if transport.get("type") == "inline":
        if set(transport) != {"type", "data_base64"} or not isinstance(transport["data_base64"], str):
            raise ValueError("Invalid inline media transport")
        if len(transport["data_base64"]) != 4 * ((size + 2) // 3):
            raise ValueError("Inline media length mismatch")
    elif transport.get("type") == "url":
        _validate_url_transport(transport)
    else:
        raise ValueError("Unsupported media transport")
    return artifact


def artifact_from_payload(artifact_id, kind, descriptor, payload, *, transport=None, max_bytes=MAX_ARTIFACT_BYTES):
    if not isinstance(payload, bytes):
        raise ValueError("Media payload must be bytes")
    bounded_int(len(payload), max_bytes)
    result = {"id": artifact_id, "kind": kind, "descriptor": descriptor,
              "byte_length": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
              "transport": transport if transport is not None else {
                  "type": "inline", "data_base64": base64.b64encode(payload).decode("ascii")}}
    return validate_artifact(result, max_bytes=max_bytes)


async def resolve_artifact(artifact, *, download=None, max_bytes=MAX_ARTIFACT_BYTES):
    """download(url, *, max_bytes) must return bounded bytes; never log its URL."""
    validate_artifact(artifact, max_bytes=max_bytes)
    transport = artifact["transport"]
    if transport["type"] == "inline":
        try:
            payload = base64.b64decode(transport["data_base64"], validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Invalid media base64") from None
    else:
        if download is None:
            raise ValueError("Media URL download is not configured")
        payload = await download(transport["url"], max_bytes=artifact["byte_length"])
    if not isinstance(payload, bytes) or len(payload) != artifact["byte_length"]:
        raise ValueError("Media byte length mismatch")
    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise ValueError("Media checksum mismatch")
    return payload


def validate_response(response, *, request_id, bindings, max_total_bytes=MAX_TOTAL_BYTES):
    """Validate ALL identity/completeness before downloading or delivering anything."""
    expected = validate_bindings(bindings)
    identifier(request_id)
    if not isinstance(response, dict) or set(response) != {"version", "profile", "request_id", "bindings", "artifacts", "items"}:
        raise ValueError("Invalid media response envelope")
    if type(response["version"]) is not int or response["version"] != VERSION or response["profile"] != PROFILE:
        raise ValueError("Unsupported media response version/profile")
    if response["request_id"] != request_id:
        raise ValueError("Media request identity mismatch")
    returned = response["bindings"]
    if not isinstance(returned, list) or len(returned) != len(expected):
        raise ValueError("Missing or extra media bindings")
    counts = {}
    for row in returned:
        if not isinstance(row, dict) or set(row) != BINDING_FIELDS | {"count"}:
            raise ValueError("Invalid result binding")
        binding = {key: row[key] for key in BINDING_FIELDS}
        validate_binding(binding)
        bid = binding["binding_id"]
        if bid in counts or expected.get(bid) != binding:
            raise ValueError("Media binding identity mismatch")
        # Connected outputs must produce at least one value. Empty STRING is a value.
        counts[bid] = bounded_int(row["count"], MAX_ITEMS, minimum=1)
    artifacts = response["artifacts"]
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= MAX_ARTIFACTS:
        raise ValueError("Invalid media artifacts count")
    indexed = {}
    total = 0
    for artifact in artifacts:
        validate_artifact(artifact)
        aid = artifact["id"]
        if aid in indexed:
            raise ValueError("Duplicate media artifact")
        total += artifact["byte_length"]
        if total > max_total_bytes:
            raise ValueError("Media response exceeds total byte limit")
        indexed[aid] = artifact
    items = response["items"]
    if not isinstance(items, list) or len(items) != sum(counts.values()):
        raise ValueError("Missing or extra media items")
    ordered = {bid: [None] * count for bid, count in counts.items()}
    used = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"binding_id", "item_index", "artifact_id"}:
            raise ValueError("Invalid media item")
        bid, aid = identifier(item["binding_id"]), identifier(item["artifact_id"])
        if bid not in counts or aid not in indexed:
            raise ValueError("Unknown media item identity")
        index = bounded_int(item["item_index"], counts[bid] - 1)
        if ordered[bid][index] is not None or indexed[aid]["kind"] != expected[bid]["kind"]:
            raise ValueError("Duplicate or mistyped media item")
        ordered[bid][index] = indexed[aid]
        used.add(aid)
    if used != set(indexed) or any(value is None for values in ordered.values() for value in values):
        raise ValueError("Incomplete media result manifest")
    return ordered


async def decode_response(response, *, request_id, bindings, download=None, max_total_bytes=MAX_TOTAL_BYTES):
    ordered = validate_response(response, request_id=request_id, bindings=bindings, max_total_bytes=max_total_bytes)
    # Resolve and decode before exposing a single result to the collector queue.
    from .codec import decode_payload
    cache = {}
    decoded = {}
    for bid, artifacts in ordered.items():
        values = []
        for artifact in artifacts:
            aid = artifact["id"]
            if aid not in cache:
                payload = await resolve_artifact(artifact, download=download)
                if artifact["kind"] == "VIDEO":
                    from .video_codec import decode_video
                    cache[aid] = decode_video(artifact["descriptor"], payload)
                else:
                    cache[aid] = decode_payload(artifact["kind"], artifact["descriptor"], payload)
            values.append(cache[aid])
        decoded[bid] = values
    return decoded
