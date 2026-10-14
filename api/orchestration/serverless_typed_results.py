"""Explicit media-v1 preflight and atomic, lossless result materialization."""
from urllib.parse import urljoin

import aiohttp

from ...serverless_bridge import contract
from ...serverless_bridge.contract import MAX_ARTIFACT_BYTES, PROFILE, validate_bindings
from .serverless_media import (
    MAX_REDIRECTS, _CHUNK_SIZE, _DOWNLOAD_TIMEOUT_SECONDS,
    _PublicResolver, _validate_artifact_url,
)


def validate_prepared_job(prepared):
    """Fail before journalling/submission; legacy jobs keep their original path."""
    profile = prepared["worker"].get("media_profile", "legacy-image-v1")
    media = prepared.get("media")
    if profile == "legacy-image-v1" and media is None:
        return False
    if profile != PROFILE or media is None:
        raise ValueError("Serverless media profile/request mismatch")
    from .runpod_client import serialize_request
    # The shared endpoint validator also checks exporter controls and input refs.
    serialize_request(prepared["workflow"], prepared["images"],
                      prepared["worker"]["timeout_seconds"], media=media)
    expected = validate_bindings(prepared["manifest"])
    if expected != validate_bindings(media["outputs"]):
        raise ValueError("Serverless media manifest identity mismatch")
    if any(binding["participant_id"] != prepared["worker"]["id"] for binding in expected.values()):
        raise ValueError("Serverless media participant identity mismatch")
    return True


async def download_public_artifact(url, *, max_bytes=MAX_ARTIFACT_BYTES):
    """Contract-sized public HTTPS download, never using a provider session/key."""
    if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_ARTIFACT_BYTES:
        raise ValueError("Media artifact exceeds the permitted size")
    current_url = _validate_artifact_url(url)
    connector = aiohttp.TCPConnector(resolver=_PublicResolver(), use_dns_cache=False, limit=4)
    timeout = aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT_SECONDS, connect=10, sock_read=15)
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False,
                                         cookie_jar=aiohttp.DummyCookieJar()) as session:
            for redirect_count in range(MAX_REDIRECTS + 1):
                async with session.get(current_url, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or redirect_count >= MAX_REDIRECTS:
                            raise ValueError("Media artifact redirect is invalid")
                        current_url = _validate_artifact_url(urljoin(current_url, location))
                        continue
                    if response.status != 200:
                        raise ValueError("Media artifact download failed")
                    payload = bytearray()
                    async for chunk in response.content.iter_chunked(_CHUNK_SIZE):
                        if not isinstance(chunk, bytes) or len(payload) + len(chunk) > max_bytes:
                            raise ValueError("Media artifact exceeds the permitted size")
                        payload.extend(chunk)
                    return bytes(payload)
    except Exception:
        # Signed URLs and transport exception details must not reach the ledger.
        raise ValueError("Media artifact download failed or exceeded its byte limit") from None
    raise ValueError("Media artifact download failed")


async def collect_typed_results(output, media, manifest, *, download=None):
    """Decode every binding before returning any collector's native value lists."""
    if not isinstance(output, dict) or output.get("error") or output.get("errors"):
        raise ValueError("Invalid serverless media output")
    decoded = await contract.decode_response(output.get("media"), request_id=media["request_id"],
                                    bindings=manifest,
                                    download=download if download is not None else download_public_artifact)
    grouped = {}
    for binding in manifest:
        values = decoded[binding["binding_id"]]
        if not isinstance(values, list) or not values:
            raise ValueError("Missing serverless media values")
        grouped.setdefault(binding["job_id"], {}).setdefault(binding["socket"], []).extend(values)
    return grouped
