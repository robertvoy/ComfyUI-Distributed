"""Bounded image transport for generic Runpod-backed ComfyUI jobs."""

import asyncio
import base64
import binascii
import hashlib
import io
import ipaddress
import os
import re
import socket
import stat
import warnings
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

import aiohttp
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
MAX_INPUT_TOTAL_BYTES = 10_000_000
MAX_OUTPUT_FILES = 1024
MAX_OUTPUT_IMAGE_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_TOTAL_BYTES = 256 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_OUTPUT_ARRAY_BYTES = 256 * 1024 * 1024
MAX_REDIRECTS = 3
_DOWNLOAD_TIMEOUT_SECONDS = 60
_CHUNK_SIZE = 64 * 1024

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_UNSUPPORTED_IMAGE_EXTENSIONS = {".gif", ".tif", ".tiff", ".avif", ".heic", ".ico", ".svg"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_MEDIA_EXTENSIONS = _IMAGE_EXTENSIONS | _UNSUPPORTED_IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS
_MEDIA_ANNOTATION = re.compile(r"\s+\[([A-Za-z0-9_-]{1,64})\]$")
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z", re.ASCII)
_SAFE_FILENAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,255}\Z", re.ASCII)


class ServerlessMediaError(ValueError):
    """Sanitized input, artifact, or download validation failure."""


def _folder_roots(folder_paths):
    roots = {}
    for kind, getter_name in (
        ("input", "get_input_directory"),
        ("output", "get_output_directory"),
        ("temp", "get_temp_directory"),
    ):
        getter = getattr(folder_paths, getter_name, None)
        if not callable(getter):
            raise ServerlessMediaError("ComfyUI media directories are unavailable")
        try:
            roots[kind] = Path(getter()).resolve(strict=True)
        except Exception:
            raise ServerlessMediaError("ComfyUI media directories are unavailable") from None
    return roots


def _inside_roots(path, roots):
    for root in roots.values():
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _normalize_media_reference(value):
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ServerlessMediaError("Image input reference is invalid")
    value = value.replace("\\", "/")
    explicit_root = None
    annotation = _MEDIA_ANNOTATION.search(value)
    if annotation:
        marker = annotation.group(1).lower()
        if marker not in {"input", "output", "temp"}:
            raise ServerlessMediaError("Image input annotation is unsupported")
        explicit_root = marker
        value = value[: annotation.start()]
    if "://" in value:
        raise ServerlessMediaError("Remote media references are not supported as image inputs")
    if re.match(r"^[A-Za-z]:", value) or value.startswith("//"):
        raise ServerlessMediaError("Image input path is outside the allowed ComfyUI media directories")

    path = PurePosixPath(value)
    if any(part == ".." for part in path.parts):
        raise ServerlessMediaError("Image input path traversal is not allowed")
    if not path.parts or value in {".", ""}:
        raise ServerlessMediaError("Image input reference is invalid")

    if explicit_root:
        value = f"{value} [{explicit_root}]"
    suffix = Path(path.name).suffix.lower()
    return value, suffix


def _resolve_input_path(value, suffix, folder_paths, roots):
    if suffix not in _IMAGE_EXTENSIONS:
        if suffix in _MEDIA_EXTENSIONS:
            raise ServerlessMediaError("Audio and video inputs are not supported by serverless image transport")
        raise ServerlessMediaError("Unsupported serverless image file type")

    try:
        source = Path(value)
        if source.is_absolute():
            annotation = _MEDIA_ANNOTATION.search(value)
            unannotated = value[:annotation.start()] if annotation else value
            resolved = Path(unannotated).resolve(strict=True)
        else:
            resolver = getattr(folder_paths, "get_annotated_filepath", None)
            if not callable(resolver):
                raise ServerlessMediaError("ComfyUI media path resolver is unavailable")
            resolved = Path(resolver(value)).resolve(strict=True)
    except ServerlessMediaError:
        raise
    except Exception:
        raise ServerlessMediaError("Image input file is missing or invalid") from None

    if not _inside_roots(resolved, roots):
        raise ServerlessMediaError("Image input path is outside the allowed ComfyUI media directories")
    try:
        mode = resolved.stat().st_mode
    except OSError:
        raise ServerlessMediaError("Image input file is missing or invalid") from None
    if not stat.S_ISREG(mode):
        raise ServerlessMediaError("Image input is not a regular file")
    return resolved


def _iter_image_references(workflow):
    if not isinstance(workflow, dict):
        raise ServerlessMediaError("Serverless workflow must be an API graph")
    references = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        if class_type == "LoadImage" and not isinstance(inputs, dict):
            raise ServerlessMediaError("LoadImage requires a literal image filename; linked image inputs are unsupported")
        if not isinstance(inputs, dict):
            continue
        if class_type == "LoadImage":
            value = inputs.get("image")
            if not isinstance(value, str):
                raise ServerlessMediaError("LoadImage requires a literal image filename; linked image inputs are unsupported")
            references.append((node_id, inputs, "image", value, True))

        for key in ("image", "file"):
            if key == "image" and class_type == "LoadImage":
                continue
            value = inputs.get(key)
            if not isinstance(value, str):
                continue
            normalized, suffix = _normalize_media_reference(value)
            if suffix in _IMAGE_EXTENSIONS:
                references.append((node_id, inputs, key, normalized, False))
            elif suffix in _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS:
                raise ServerlessMediaError("Audio and video inputs are not supported by serverless image transport")
            elif suffix in _UNSUPPORTED_IMAGE_EXTENSIONS or (key == "image" and suffix):
                raise ServerlessMediaError("Unsupported serverless image file type")

        for key in ("audio", "video"):
            value = inputs.get(key)
            if not isinstance(value, str):
                continue
            _normalized, suffix = _normalize_media_reference(value)
            if suffix in _MEDIA_EXTENSIONS:
                raise ServerlessMediaError("Audio and video inputs are not supported by serverless image transport")
    return references


def _read_limited(path, expected_size):
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_INPUT_IMAGE_BYTES:
                raise ServerlessMediaError("Image input exceeds the permitted size")
            if info.st_size > MAX_INPUT_TOTAL_BYTES:
                raise ServerlessMediaError(
                    f"Total raw serverless image input size exceeds {MAX_INPUT_TOTAL_BYTES:,} bytes; "
                    "base64 encoding adds about 33%, and the complete request body is also limited to 10,000,000 bytes"
                )
            data = handle.read(MAX_INPUT_TOTAL_BYTES + 1)
    except ServerlessMediaError:
        raise
    except OSError:
        raise ServerlessMediaError("Image input file could not be read safely") from None
    if len(data) > MAX_INPUT_TOTAL_BYTES:
        raise ServerlessMediaError(
            f"Total raw serverless image input size exceeds {MAX_INPUT_TOTAL_BYTES:,} bytes; "
            "base64 encoding adds about 33%, and the complete request body is also limited to 10,000,000 bytes"
        )
    if len(data) != expected_size:
        raise ServerlessMediaError("Image input changed while being read")
    return data


async def prepare_images(workflow: dict) -> list[dict]:
    """Pack referenced local images and rewrite only their graph input references.

    ``workflow`` is expected to be a private API-graph copy owned by the caller.
    Returned ``name`` values are flat generated names suitable for ``input.images``.
    """
    references = _iter_image_references(workflow)
    if not references:
        return []

    # Import only in ComfyUI, keeping this module importable in the serverless client.
    try:
        import folder_paths
    except Exception:
        raise ServerlessMediaError("ComfyUI media directories are unavailable") from None

    roots = _folder_roots(folder_paths)
    resolved_refs = []
    unique_paths = {}
    total_source_bytes = 0
    for node_id, inputs, key, value, _is_load_image in references:
        normalized, suffix = _normalize_media_reference(value)
        if key == "image" and suffix not in _IMAGE_EXTENSIONS:
            if suffix in _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS:
                raise ServerlessMediaError("Audio and video inputs are not supported by serverless image transport")
            raise ServerlessMediaError("Unsupported serverless image file type")
        path = _resolve_input_path(normalized, suffix, folder_paths, roots)
        if path not in unique_paths:
            try:
                size = path.stat().st_size
            except OSError:
                raise ServerlessMediaError("Image input file is missing or invalid") from None
            if size < 0 or size > MAX_INPUT_IMAGE_BYTES:
                raise ServerlessMediaError("Image input exceeds the permitted size")
            total_source_bytes += size
            if total_source_bytes > MAX_INPUT_TOTAL_BYTES:
                raise ServerlessMediaError(
                    f"Total raw serverless image input size is {total_source_bytes:,} bytes; "
                    f"the input read guard is {MAX_INPUT_TOTAL_BYTES:,} bytes. Base64 encoding "
                    "adds about 33%, and the complete request body is also limited to 10,000,000 bytes"
                )
            unique_paths[path] = {"size": size, "suffix": suffix}
        resolved_refs.append((inputs, key, path))

    packed_by_name = {}
    path_to_name = {}
    for path, metadata in unique_paths.items():
        data = _read_limited(path, metadata["size"])
        digest = hashlib.sha256(data).hexdigest()
        name = f"serverless_{digest}{metadata['suffix']}"
        path_to_name[path] = name
        packed_by_name.setdefault(name, {"name": name, "image": base64.b64encode(data).decode("ascii")})

    for inputs, key, path in resolved_refs:
        inputs[key] = path_to_name[path]
    return [packed_by_name[name] for name in sorted(packed_by_name)]


def _validate_artifact_url(url):
    if not isinstance(url, str) or not url or any(ord(char) < 0x20 or char.isspace() for char in url):
        raise ServerlessMediaError("Serverless image artifact URL is invalid")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ServerlessMediaError("Serverless image artifact URL is invalid") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise ServerlessMediaError("Serverless image artifact URL must be a public HTTPS URL")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "metadata", "metadata.google.internal"} or host.endswith((".localhost", ".local", ".internal")):
        raise ServerlessMediaError("Serverless image artifact URL must be a public HTTPS URL")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url
    if not address.is_global:
        raise ServerlessMediaError("Serverless image artifact URL must resolve to a public address")
    return url


class _PublicResolver(aiohttp.abc.AbstractResolver):
    """Resolve once, reject non-public answers, and pin the checked address."""

    async def _getaddrinfo(self, host, port, family):
        return await asyncio.get_running_loop().getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM
        )

    async def resolve(self, host, port=0, family=socket.AF_INET):
        try:
            results = await self._getaddrinfo(host, port, family)
        except (OSError, asyncio.TimeoutError):
            raise OSError("Public artifact host could not be resolved") from None
        if not results:
            raise OSError("Public artifact host could not be resolved")

        resolved = []
        seen = set()
        for result_family, _socktype, proto, _canonname, sockaddr in results:
            try:
                address = ipaddress.ip_address(sockaddr[0])
            except (ValueError, TypeError, IndexError):
                raise OSError("Public artifact host resolved to an invalid address") from None
            if not address.is_global:
                raise OSError("Public artifact host resolved to a non-public address")
            key = (str(address), result_family)
            if key in seen:
                continue
            seen.add(key)
            resolved.append({
                "hostname": host,
                "host": str(address),
                "port": port,
                "family": result_family,
                "proto": proto,
                "flags": 0,
            })
        if not resolved:
            raise OSError("Public artifact host could not be resolved")
        return resolved

    async def close(self):
        return None


async def download_public_artifact(url: str, max_bytes: int = MAX_OUTPUT_IMAGE_BYTES) -> bytes:
    """Download a signed public HTTPS artifact without provider credentials."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= MAX_OUTPUT_IMAGE_BYTES:
        raise ServerlessMediaError("Serverless image artifact exceeds the permitted size")
    current_url = _validate_artifact_url(url)
    resolver = _PublicResolver()
    connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False, limit=4)
    timeout = aiohttp.ClientTimeout(
        total=_DOWNLOAD_TIMEOUT_SECONDS,
        connect=10,
        sock_read=15,
    )
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),
        ) as session:
            for redirect_count in range(MAX_REDIRECTS + 1):
                _validate_artifact_url(current_url)
                async with session.get(current_url, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or redirect_count >= MAX_REDIRECTS:
                            raise ServerlessMediaError("Serverless image artifact redirect is invalid")
                        current_url = _validate_artifact_url(urljoin(current_url, location))
                        continue
                    if response.status != 200:
                        raise ServerlessMediaError("Serverless image artifact download failed")
                    payload = bytearray()
                    async for chunk in response.content.iter_chunked(_CHUNK_SIZE):
                        if not isinstance(chunk, bytes):
                            raise ServerlessMediaError("Serverless image artifact response is invalid")
                        if len(payload) + len(chunk) > max_bytes:
                            raise ServerlessMediaError("Serverless image artifact exceeds the permitted size")
                        payload.extend(chunk)
                    if not payload:
                        raise ServerlessMediaError("Serverless image artifact is empty")
                    return bytes(payload)
    except ServerlessMediaError:
        raise
    except Exception:
        raise ServerlessMediaError("Serverless image artifact download failed") from None
    raise ServerlessMediaError("Serverless image artifact download failed")


def _validated_manifest(manifest):
    if not isinstance(manifest, list) or len(manifest) > MAX_OUTPUT_FILES:
        raise ServerlessMediaError("Serverless image manifest is invalid")
    normalized = []
    seen_prefixes = set()
    prefixes = []
    for item in manifest:
        if not isinstance(item, dict):
            raise ServerlessMediaError("Serverless image manifest is invalid")
        if item.get("kind") != "image":
            raise ServerlessMediaError("Only image collectors are supported by serverless media transport")
        for key in ("collector_id", "job_id", "node_id"):
            value = item.get(key)
            if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value):
                raise ServerlessMediaError("Serverless image manifest is invalid")
        prefix = item.get("prefix")
        if not isinstance(prefix, str) or not _PREFIX_RE.fullmatch(prefix) or prefix in {".", ".."}:
            raise ServerlessMediaError("Serverless image manifest contains an unsafe filename prefix")
        folded = prefix.casefold()
        if folded in seen_prefixes:
            raise ServerlessMediaError("Serverless image manifest prefixes must be unique")
        seen_prefixes.add(folded)
        prefixes.append(prefix)
        normalized.append({
            "collector_id": str(item["collector_id"]),
            "job_id": str(item["job_id"]),
            "node_id": str(item["node_id"]),
            "prefix": prefix,
        })
    for first in prefixes:
        for second in prefixes:
            if first != second and second.casefold().startswith(first.casefold() + "_"):
                raise ServerlessMediaError("Serverless image manifest prefixes are ambiguous")
    return normalized


def _artifact_filename_matches(filename, prefixes):
    if not isinstance(filename, str):
        return None
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    matches = [item for item in prefixes if basename.startswith(item["prefix"] + "_")]
    if len(matches) > 1:
        raise ServerlessMediaError("Serverless image artifact matches multiple collectors")
    return matches[0] if matches else None


def _canonical_artifact_identity(entry, filename):
    artifact_path = entry.get("artifact_path")
    artifact_type = entry.get("artifact_type", "output")
    if artifact_path is not None:
        if not isinstance(artifact_path, str) or not artifact_path or "\x00" in artifact_path:
            raise ServerlessMediaError("Serverless image artifact identity is invalid")
        normalized_path = os.path.normcase(os.path.normpath(artifact_path.replace("\\", "/")))
    else:
        subfolder = entry.get("subfolder", "")
        if not isinstance(subfolder, str) or "\x00" in subfolder:
            raise ServerlessMediaError("Serverless image artifact identity is invalid")
        normalized_path = os.path.normcase(os.path.normpath(f"{subfolder}/{filename}"))
    return str(artifact_type), normalized_path


def _decode_base64_data(value, max_bytes=MAX_OUTPUT_IMAGE_BYTES):
    if not isinstance(value, str) or not value:
        raise ServerlessMediaError("Serverless image artifact data is invalid")
    max_bytes = min(max_bytes, MAX_OUTPUT_IMAGE_BYTES)
    max_encoded = ((max_bytes + 2) // 3) * 4
    if len(value) > max_encoded:
        raise ServerlessMediaError("Serverless image artifact exceeds the permitted size")
    padding = len(value) - len(value.rstrip("="))
    estimated_size = (len(value) // 4) * 3 - padding
    if estimated_size > max_bytes:
        raise ServerlessMediaError("Serverless image artifact exceeds the permitted size")
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ServerlessMediaError("Serverless image artifact data is invalid") from None
    if not payload:
        raise ServerlessMediaError("Serverless image artifact is empty")
    if len(payload) > max_bytes:
        raise ServerlessMediaError("Serverless image artifact exceeds the permitted size")
    return payload


def _decode_image_array(payload, total_array_bytes):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(payload))
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ServerlessMediaError("Serverless image dimensions exceed the permitted limit")
            has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
            mode = "RGBA" if has_alpha else "RGB"
            output_bytes = width * height * (4 if has_alpha else 3) * 4
            if total_array_bytes + output_bytes > MAX_OUTPUT_ARRAY_BYTES:
                raise ServerlessMediaError("Decoded serverless images exceed the permitted memory limit")
            converted = image.convert(mode)
            converted.load()
            array = np.array(converted, dtype=np.float32, copy=True)
            np.multiply(array, 1.0 / 255.0, out=array)
            return array, output_bytes
    except ServerlessMediaError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ServerlessMediaError("Serverless image artifact could not be decoded") from None
    except Exception:
        raise ServerlessMediaError("Serverless image artifact could not be decoded") from None


async def collect_images(output: dict, manifest: list, download=None) -> dict[str, list[np.ndarray]]:
    """Validate and decode expected image artifacts, grouped by serverless job ID."""
    if not isinstance(output, dict):
        raise ServerlessMediaError("Serverless job output is invalid")
    nested = output.get("output")
    data = nested if isinstance(nested, dict) else output
    for candidate in (output, data):
        if candidate.get("error") or candidate.get("errors"):
            raise ServerlessMediaError("Serverless job reported an execution error")

    expected = _validated_manifest(manifest)
    if not expected:
        return {}
    artifacts = data.get("images", [])
    if not isinstance(artifacts, list) or len(artifacts) > MAX_OUTPUT_FILES:
        raise ServerlessMediaError("Serverless job returned too many image files")

    selected = {item["prefix"]: [] for item in expected}
    identities = set()
    total_payload_bytes = 0
    for entry in artifacts:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        manifest_entry = _artifact_filename_matches(filename, expected)
        if manifest_entry is None:
            continue
        normalized_filename = filename.replace("\\", "/")
        if "/" in normalized_filename or not _SAFE_FILENAME_RE.fullmatch(filename):
            raise ServerlessMediaError("Serverless image artifact filename is invalid")
        if Path(filename).suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ServerlessMediaError("Serverless image artifact has an unsupported file type")
        if "artifact_type" in entry and entry["artifact_type"] != "output":
            raise ServerlessMediaError("Expected serverless image artifact was not an output image")
        identity = _canonical_artifact_identity(entry, filename)
        if identity in identities:
            raise ServerlessMediaError("Duplicate serverless image artifact identity")
        identities.add(identity)
        match = re.match(rf"^{re.escape(manifest_entry['prefix'])}_(\d+)(?:_|\.)", filename)
        if match is None:
            raise ServerlessMediaError("Expected serverless image filename counter is invalid")
        artifact_type = entry.get("type")
        remaining_bytes = MAX_OUTPUT_TOTAL_BYTES - total_payload_bytes
        if remaining_bytes <= 0:
            raise ServerlessMediaError("Serverless image artifacts exceed the permitted total size")
        if artifact_type == "base64":
            payload = _decode_base64_data(entry.get("data"), remaining_bytes)
        elif artifact_type in {"gcs_url", "s3_url", "url"}:
            url = entry.get("data") or entry.get("url")
            url = _validate_artifact_url(url)
            try:
                if download is None:
                    payload = await download_public_artifact(
                        url, max_bytes=min(MAX_OUTPUT_IMAGE_BYTES, remaining_bytes)
                    )
                else:
                    payload = await download(url)
            except ServerlessMediaError:
                raise
            except Exception:
                raise ServerlessMediaError("Serverless image artifact download failed") from None
            if not isinstance(payload, bytes) or not payload:
                raise ServerlessMediaError("Serverless image artifact response is invalid")
            if len(payload) > MAX_OUTPUT_IMAGE_BYTES:
                raise ServerlessMediaError("Serverless image artifact exceeds the permitted size")
        else:
            raise ServerlessMediaError("Serverless image artifact transport type is unsupported")
        total_payload_bytes += len(payload)
        if total_payload_bytes > MAX_OUTPUT_TOTAL_BYTES:
            raise ServerlessMediaError("Serverless image artifacts exceed the permitted total size")
        selected[manifest_entry["prefix"]].append((int(match.group(1)), filename, identity, payload))

    if any(not selected[item["prefix"]] for item in expected):
        raise ServerlessMediaError("Serverless job is missing an expected collector image")

    result = {}
    total_array_bytes = 0
    for manifest_entry in expected:
        prefix = manifest_entry["prefix"]
        rows = sorted(selected[prefix], key=lambda row: (row[0], row[1], row[2]))
        arrays = []
        shape = None
        for _counter, _filename, _identity, payload in rows:
            array, added_bytes = _decode_image_array(payload, total_array_bytes)
            total_array_bytes += added_bytes
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ServerlessMediaError("Images from one serverless collector have inconsistent dimensions")
            arrays.append(array)
        result.setdefault(manifest_entry["job_id"], []).extend(arrays)
    return result
