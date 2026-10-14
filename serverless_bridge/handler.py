"""Pure HTTP executor for media-v1 requests against a local ComfyUI instance.

No ComfyUI, tensor, or provider imports. ``download(url, *, max_bytes)`` must
provide public-HTTPS/redirect/DNS protection and bounded streaming. Optional
``publish(metadata, payload)`` returns a contract-valid, expiring HTTPS transport.

Only identified invocation-owned files are deleted. Optional trusted local
``temp_dir`` enables exporter cleanup after terminal history and collection.
Without it, exporter retention belongs to the ComfyUI temp-directory lifecycle.
The pinned native /interrupt route is not atomic: running-prompt cancellation
fails closed and retains files instead of risking an unrelated successor prompt.
"""
import asyncio
import base64
import binascii
import copy
import hashlib
import json
import math
import os
import re
import stat
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

import aiohttp

from .contract import (
    BINDING_FIELDS, MAX_ARTIFACT_BYTES, MAX_ARTIFACTS, MAX_TOTAL_BYTES,
    PROFILE, VERSION, artifact_from_payload, identifier, resolve_artifact,
    validate_artifact, validate_bindings, validate_json, validate_response,
)

MAX_RESPONSE_JSON_BYTES = 10_000_000
MAX_HTTP_JSON_BYTES = 32 * 1024 * 1024
POLL_INTERVAL_SECONDS = 0.25
CANCEL_GRACE_SECONDS = 10.0
DRAIN_GRACE_SECONDS = 10.0
HTTP_TIMEOUT_SECONDS = 15.0
_UNCERTAIN_WORK = set()
GET_ATTEMPTS = 3
_IMPORT_SLOTS = {"IMAGE": 0, "AUDIO": 1, "MASK": 2, "STRING": 3, "VIDEO": 4, "VALUE": 5}
# Used ONLY to run the shared manifest validator before /view. Never fetched.
_VALIDATION_TRANSPORT = {"type": "url", "url": "https://media-validation.invalid/artifact",
                         "expires_at": "9999-01-01T00:00:00Z"}


class MediaExecutionUncertain(RuntimeError):
    """The local prompt may still execute; staged files were deliberately retained."""


class _HTTPError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"Local Comfy HTTP request failed (status {status})")
        self.status = status


def _scope_id(value):
    identifier(value)
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Unsafe media scope identifier")
    return value


def _json_bytes(value):
    return json.dumps(value, allow_nan=False).encode("utf-8")


def _link(value, workflow):
    if (not isinstance(value, list) or len(value) != 2 or
            not isinstance(value[0], str) or value[0] not in workflow or
            type(value[1]) is not int or value[1] < 0):
        raise ValueError("Invalid workflow socket link")
    return value[0]


def _validate_request(input_data):
    if (not isinstance(input_data, dict) or not {"workflow", "media"} <= set(input_data)
            or set(input_data) - {"workflow", "media", "images"}):
        raise ValueError("Invalid media request envelope")
    media = input_data["media"]
    if (not isinstance(media, dict) or set(media) !=
            {"version", "profile", "request_id", "artifacts", "inputs", "outputs"}):
        raise ValueError("Invalid media request controls")
    if type(media["version"]) is not int or media["version"] != VERSION or media["profile"] != PROFILE:
        raise ValueError("Unsupported media request version/profile")
    _scope_id(media["request_id"])
    bindings = validate_bindings(media["outputs"])
    targets = []
    for binding in bindings.values():
        for field in BINDING_FIELDS - {"socket", "kind"}:
            _scope_id(binding[field])
        targets.append(binding["node_id"])
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate media exporter node")

    workflow = input_data["workflow"]
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Invalid API workflow")
    validate_json(workflow, max_bytes=16 * 1024 * 1024)
    dependencies = {}
    for node_id, node in workflow.items():
        _scope_id(node_id)
        if (not isinstance(node, dict) or not isinstance(node.get("class_type"), str)
                or not node["class_type"] or not isinstance(node.get("inputs"), dict)):
            raise ValueError("Invalid API workflow node")
        if node["class_type"] == "MediaImportV1":
            raise ValueError("Request may not supply its own media import filenames")
        if node["class_type"] == "MediaExportV1" and node_id not in targets:
            raise ValueError("Undeclared media exporter")
        dependencies[node_id] = set()
        for name, value in node["inputs"].items():
            _scope_id(name)
            if isinstance(value, list):
                dependencies[node_id].add(_link(value, workflow))
    for binding in bindings.values():
        node = workflow.get(binding["node_id"])
        if node is None or node["class_type"] != "MediaExportV1":
            raise ValueError("Missing declared MediaExportV1 node")
        controls = node["inputs"]
        from .image_transfer import DEFAULT_FORMAT, DEFAULT_QUALITY, validate_options
        required = {"request_id", "binding_json", binding["socket"]}
        optional = {"transfer_format", "transfer_quality"} if binding["kind"] == "IMAGE" else set()
        if not required <= set(controls) or set(controls) - required - optional:
            raise ValueError("Media exporter must have exactly its declared socket and controls")
        validate_options(controls.get("transfer_format", DEFAULT_FORMAT),
                         controls.get("transfer_quality", DEFAULT_QUALITY))
        if controls["request_id"] != media["request_id"]:
            raise ValueError("Media exporter request identity mismatch")
        text = controls["binding_json"]
        if not isinstance(text, str) or len(text) > 128 * 1024:
            raise ValueError("Invalid media exporter binding JSON")
        try:
            declared = json.loads(text)
        except (ValueError, RecursionError):
            raise ValueError("Invalid media exporter binding JSON") from None
        if declared != binding:
            raise ValueError("Media exporter binding identity mismatch")
        _link(controls[binding["socket"]], workflow)
    # The execution roots are exactly the declared exporters, not every class
    # with OUTPUT_NODE=True. Required upstream output utilities may remain.
    visited = set()
    pending = list(targets)
    while pending:
        node_id = pending.pop()
        if node_id not in visited:
            visited.add(node_id)
            pending.extend(dependencies[node_id])
    if visited != set(workflow):
        raise ValueError("Workflow contains nodes outside declared output roots")
    if any(set(targets) & sources for sources in dependencies.values()):
        raise ValueError("A media exporter cannot be an upstream dependency")
    # Kahn traversal bounds cycle checking without relying on recursion depth.
    counts = {node: len(sources) for node, sources in dependencies.items()}
    consumers = {node: [] for node in workflow}
    for node, sources in dependencies.items():
        for source in sources:
            consumers[source].append(node)
    pending = [node for node, count in counts.items() if not count]
    processed = 0
    while pending:
        processed += 1
        for consumer in consumers[pending.pop()]:
            counts[consumer] -= 1
            if not counts[consumer]:
                pending.append(consumer)
    if processed != len(workflow):
        raise ValueError("Workflow contains a dependency cycle")

    artifacts = media["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) > MAX_ARTIFACTS:
        raise ValueError("Invalid input artifacts count")
    indexed, total = {}, 0
    for artifact in artifacts:
        validate_artifact(artifact)
        aid = _scope_id(artifact["id"])
        if aid in indexed:
            raise ValueError("Duplicate input artifact")
        indexed[aid] = artifact
        total += artifact["byte_length"]
    inputs = media["inputs"]
    if not isinstance(inputs, list) or len(inputs) > 4096:
        raise ValueError("Invalid input references")
    destinations, used = set(), set()
    exporters = {binding["node_id"]: binding for binding in bindings.values()}
    for ref in inputs:
        if not isinstance(ref, dict) or set(ref) != {"node_id", "input_name", "artifact_id"}:
            raise ValueError("Invalid media input reference")
        for value in ref.values():
            _scope_id(value)
        node_id, name, aid = ref["node_id"], ref["input_name"], ref["artifact_id"]
        if (node_id not in workflow or name not in workflow[node_id]["inputs"]
                or aid not in indexed or (node_id, name) in destinations):
            raise ValueError("Unknown or duplicate media input reference")
        binding = exporters.get(node_id)
        if binding is not None:
            # VALUE exports accept the six explicit runtime types. IMAGE/MASK
            # imports can feed both native consumers and a sparse override.
            allowed = {"VALUE", "IMAGE", "MASK", "STRING"} if binding["kind"] == "VALUE" else {binding["kind"]}
            if name != binding["socket"] or indexed[aid]["kind"] not in allowed:
                raise ValueError("Exporter input must match its declared media socket and kind")
        destinations.add((node_id, name))
        used.add(aid)
    if used != set(indexed):
        raise ValueError("Every input artifact must be used")

    images = input_data.get("images", [])
    if not isinstance(images, list) or len(images) > MAX_ARTIFACTS:
        raise ValueError("Invalid legacy images count")
    names = set()
    legacy = []
    for image in images:
        if not isinstance(image, dict) or set(image) != {"name", "image"}:
            raise ValueError("Invalid legacy image")
        name, encoded = image["name"], image["image"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,254}", name)
                or name in names or not isinstance(encoded, str)):
            raise ValueError("Invalid or duplicate legacy image basename")
        names.add(name)
        if len(encoded) > 4 * ((MAX_ARTIFACT_BYTES + 2) // 3):
            raise ValueError("Legacy image exceeds byte limit")
        # Exact decoded length, including padding, before allocating bytes.
        size = (len(encoded) // 4) * 3 - (len(encoded) - len(encoded.rstrip("=")))
        if len(encoded) % 4 or size < 0 or size > MAX_ARTIFACT_BYTES:
            raise ValueError("Invalid legacy image base64 length")
        total += size
        matches = [node_id for node_id, node in workflow.items()
                   if node["class_type"] == "LoadImage" and node["inputs"].get("image") == name]
        if not matches or any((node_id, "image") in destinations for node_id in matches):
            raise ValueError("Unused or conflicting legacy image binding")
        legacy.append((encoded, size, matches))
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Media inputs exceed total byte limit")
    return copy.deepcopy(workflow), copy.deepcopy(media), legacy, targets


class _Staging:
    """Exclusive random files under a hashed namespace; never delete siblings."""
    def __init__(self, input_dir, request_id):
        self.root = Path(input_dir).resolve()
        self.key = hashlib.sha256(request_id.encode()).hexdigest()
        self.relative = f"distributed-media-v1/{self.key}"
        self.names = []
        self.fd = None

    def _open(self):
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root
        for component in ("distributed-media-v1", self.key):
            directory = directory / component
            if directory.is_symlink():
                raise ValueError("Unsafe media staging directory symlink")
            directory.mkdir(exist_ok=True, mode=0o700)
            if not directory.resolve().is_relative_to(self.root):
                raise ValueError("Unsafe media staging directory")
        self.fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def write(self, payload):
        if self.fd is None:
            self._open()
        name = uuid.uuid4().hex + ".bin"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        self.names.append(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        return f"{self.relative}/{name}"

    def close(self, *, cleanup):
        if self.fd is not None:
            try:
                if cleanup:
                    for name in self.names:
                        try:
                            os.unlink(name, dir_fd=self.fd)
                        except FileNotFoundError:
                            pass
            finally:
                os.close(self.fd)
                self.fd = None
            if cleanup:
                # Never recursively remove a shared namespace or another run.
                for directory in (self.root / self.relative, self.root / "distributed-media-v1"):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass


class _OwnedExports:
    """Best-effort retirement of validated terminal-history files, never a sweep.

    Walk every directory component with openat/O_NOFOLLOW, then pin the directory
    fd and each regular file's identity. Missing, linked, replaced, or unidentifiable
    files stay with the endpoint lifecycle. Never use history-supplied path strings.
    """
    def __init__(self, temp_dir, request_id):
        self.root = Path(temp_dir).absolute() if temp_dir is not None else None
        self.key = hashlib.sha256(request_id.encode()).hexdigest()
        self.fd = None
        self.files = {}

    def _open(self):
        if self.fd is not None:
            return True
        if self.root is None or ".." in self.root.parts:
            return False
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = None
        try:
            fd = os.open(self.root.anchor, flags)
            for component in (*self.root.parts[1:], "distributed-media-v1", self.key):
                child = os.open(component, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            self.fd, fd = fd, None
            return True
        except OSError:
            return False
        finally:
            if fd is not None:
                os.close(fd)

    def _identity(self, name):
        info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return None
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def identify(self, entry, media):
        if self.root is None or not isinstance(entry.get("outputs"), dict):
            return
        # A failed prompt may contain valid completed exporters alongside missing
        # or malformed ones. Claim only independently validated declared records.
        for binding in media["outputs"]:
            node_id = binding["node_id"]
            try:
                _, files = _manifest({"outputs": {node_id: entry["outputs"].get(node_id)}},
                                     dict(media, outputs=[binding]))
            except ValueError:
                continue
            if not files or not self._open():
                continue
            for artifact in files:
                name = artifact["id"] + ".bin"  # _manifest enforces UUID hex.
                try:
                    identity = self._identity(name)
                except OSError:
                    continue
                if identity is not None:
                    self.files.setdefault(name, identity)

    def close(self, *, cleanup):
        if self.fd is None:
            return
        try:
            if cleanup:
                for name, identity in self.files.items():
                    try:
                        if self._identity(name) == identity:
                            os.unlink(name, dir_fd=self.fd)
                    except OSError:
                        pass  # Endpoint lifecycle owns inaccessible leftovers.
        finally:
            os.close(self.fd)
            self.fd = None
        # Deliberately retain directories: they may belong to concurrent runs.


async def _body(response, maximum):
    length = response.content_length
    if length is not None and length > maximum:
        raise ValueError("Local Comfy response exceeds declared byte length")
    body = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        if len(body) + len(chunk) > maximum:
            raise ValueError("Local Comfy response exceeds declared byte length")
        body.extend(chunk)
    return bytes(body)


class _Execution:
    def __init__(self, session, url, stage, media, targets, deadline, exports):
        self.session, self.url, self.stage = session, url, stage
        self.media, self.targets = media, targets
        self.deadline = deadline
        self.run_id = uuid.uuid4().hex
        self.prompt_id = None
        self.attempted = False
        self.terminal = False
        self.exports = exports
        self.interruption_blocked = False

    def remaining(self):
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError("Media execution deadline expired")
        return remaining

    async def json_request(self, method, path, *, body=None, permit_rejection=False):
        async with self.session.request(method, self.url + path, json=body,
                timeout=aiohttp.ClientTimeout(total=min(HTTP_TIMEOUT_SECONDS, self.remaining())),
                allow_redirects=False) as response:
            raw = await _body(response, MAX_HTTP_JSON_BYTES)
            # Native queue-delete and interrupt routes return an empty 200.
            if response.status == 200 and not raw and method == "POST" and path in {"/queue", "/interrupt"}:
                return {}
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError):
                if response.status != 200:
                    raise _HTTPError(response.status) from None
                raise ValueError("Invalid local Comfy JSON response") from None
            if response.status != 200:
                # Native /prompt rejects the entire graph with a 400 error.
                # Other failures may hide acceptance and are never retried.
                if (permit_rejection and response.status == 400 and isinstance(value, dict)
                        and value.get("error") and "prompt_id" not in value):
                    self.terminal = True
                raise _HTTPError(response.status)
            if not isinstance(value, dict):
                raise ValueError("Invalid local Comfy JSON object")
            return value

    async def get_json(self, path):
        for attempt in range(GET_ATTEMPTS):
            try:
                return await self.json_request("GET", path)
            except (_HTTPError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if isinstance(exc, _HTTPError) and exc.status != 429 and exc.status < 500:
                    raise
                if attempt == GET_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(POLL_INTERVAL_SECONDS * (2 ** attempt))

    async def history(self):
        response = await self.get_json("/history/" + quote(self.prompt_id, safe=""))
        entry = response.get(self.prompt_id)
        if entry is None:
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("status"), dict):
            raise ValueError("Malformed Comfy prompt history")
        status = entry["status"]
        messages = status.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Malformed Comfy history messages")
        failed = status.get("status_str") in {"error", "interrupted"} or any(
            isinstance(message, list) and message and message[0] in {"execution_error", "execution_interrupted"}
            for message in messages)
        if failed or status.get("completed") is True:
            self.terminal = True
            self.exports.identify(entry, self.media)
            if failed or status.get("status_str") != "success":
                raise RuntimeError("Local Comfy prompt execution failed")
            return entry
        return None

    async def submit_and_wait(self, workflow):
        body = {"prompt": workflow, "client_id": self.run_id,
                "extra_data": {"distributed_media_run_id": self.run_id},
                "partial_execution_targets": list(self.targets)}
        self.remaining()  # CPU staging must not spend past the absolute deadline.
        self.attempted = True
        response = await self.json_request("POST", "/prompt", body=body, permit_rejection=True)
        self.prompt_id = _scope_id(response.get("prompt_id"))
        if response.get("node_errors"):
            raise RuntimeError("Local Comfy accepted an incomplete set of execution targets")
        while True:
            entry = await self.history()
            if entry is not None:
                return entry
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def settle(self):
        """Bounded recovery; caller shields this from task cancellation."""
        deleted = False
        while True:
            if self.prompt_id:
                try:
                    await self.history()
                except RuntimeError:
                    if not self.terminal:
                        raise
                if self.terminal:
                    return
            queue = await self.get_json("/queue")
            groups = []
            for key in ("queue_running", "queue_pending"):
                rows = queue.get(key)
                if not isinstance(rows, list):
                    raise ValueError("Malformed Comfy queue")
                group = {}
                for row in rows:
                    if not isinstance(row, list) or len(row) < 4 or not isinstance(row[3], dict):
                        raise ValueError("Malformed Comfy queue entry")
                    pid = _scope_id(row[1])
                    if pid in group:
                        raise ValueError("Duplicate Comfy queue identity")
                    group[pid] = row[3]
                groups.append(group)
            running, pending = groups
            if self.prompt_id is None:
                own = [pid for group in groups for pid, extra in group.items()
                       if extra.get("distributed_media_run_id") == self.run_id
                       and extra.get("client_id") == self.run_id]
                if len(own) > 1:
                    raise ValueError("Ambiguous Comfy queue ownership")
                if own:
                    self.prompt_id = own[0]
            if self.prompt_id in pending and not deleted:
                await self.json_request("POST", "/queue", body={"delete": [self.prompt_id]})
                deleted = True
            elif self.prompt_id in running:
                # Pinned ComfyUI server.py checks a snapshot then globally
                # interrupts, outside the queue mutex. Supplying prompt_id does
                # NOT close that race. No safe capability negotiation exists on
                # this route; do not invoke it, even after our own queue read.
                self.interruption_blocked = True
                raise RuntimeError("Native prompt-scoped interruption unavailable")
            elif deleted and self.prompt_id not in running and self.prompt_id not in pending:
                self.terminal = True  # queue deletion confirmed by exact read-back
                return
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def fetch(self, artifact):
        transport = artifact["transport"]
        params = {"filename": transport["filename"], "subfolder": transport["subfolder"], "type": "temp"}
        for attempt in range(GET_ATTEMPTS):
            try:
                async with self.session.get(self.url + "/view", params=params, allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=min(HTTP_TIMEOUT_SECONDS, self.remaining()))) as response:
                    if response.status != 200:
                        raise _HTTPError(response.status)
                    payload = await _body(response, artifact["byte_length"])
                break
            except (_HTTPError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if isinstance(exc, _HTTPError) and exc.status != 429 and exc.status < 500:
                    raise
                if attempt == GET_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(POLL_INTERVAL_SECONDS * (2 ** attempt))
        if len(payload) != artifact["byte_length"]:
            raise ValueError("Media output byte length mismatch")
        if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise ValueError("Media output checksum mismatch")
        return payload


def _manifest(entry, media):
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Missing media history outputs")
    expected_ids = {binding["node_id"] for binding in media["outputs"]}
    if any(node_id not in expected_ids and isinstance(value, dict) and "media_exports" in value
           for node_id, value in outputs.items()):
        raise ValueError("Unexpected media export history")
    response = {"version": VERSION, "profile": PROFILE, "request_id": media["request_id"],
                "bindings": [], "artifacts": [], "items": []}
    files = []
    subfolder = "distributed-media-v1/" + hashlib.sha256(media["request_id"].encode()).hexdigest()
    for binding in media["outputs"]:
        node = outputs.get(binding["node_id"], {})
        records = node.get("media_exports") if isinstance(node, dict) else None
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("Missing or duplicate media export result")
        record = records[0]
        if (not isinstance(record, dict) or set(record) != {"request_id", "binding", "artifacts", "items"}
                or record["request_id"] != media["request_id"]):
            raise ValueError("Media export request identity mismatch")
        if not isinstance(record["artifacts"], list) or len(record["artifacts"]) > MAX_ARTIFACTS:
            raise ValueError("Invalid exported artifacts count")
        converted = []
        for artifact in record["artifacts"]:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("transport"), dict):
                raise ValueError("Invalid exported artifact")
            transport = artifact["transport"]
            aid = artifact.get("id")
            if (not isinstance(aid, str) or not re.fullmatch(r"[a-f0-9]{32}", aid)
                    or transport != {"type": "comfy-file", "filename": aid + ".bin",
                                     "subfolder": subfolder, "folder_type": "temp"}):
                raise ValueError("Unsafe or unowned exported media file")
            converted.append(dict(artifact, transport=dict(_VALIDATION_TRANSPORT)))
        partial = dict(response, bindings=[record["binding"]], artifacts=converted, items=record["items"])
        validate_response(partial, request_id=media["request_id"], bindings=[binding])
        response["bindings"].append(copy.deepcopy(record["binding"]))
        response["artifacts"].extend(converted)
        response["items"].extend(copy.deepcopy(record["items"]))
        files.extend(copy.deepcopy(record["artifacts"]))
    # Validate the whole manifest BEFORE the first /view or publisher callback.
    validate_response(response, request_id=media["request_id"], bindings=media["outputs"])
    return response, files


def _publication_plan(response, files, publish):
    skeleton = dict(response, artifacts=[dict(a, transport={"type": "inline", "data_base64": ""}) for a in files])
    base_size = len(_json_bytes({"media": skeleton}))
    sizes = {a["id"]: 4 * ((a["byte_length"] + 2) // 3) for a in files}
    predicted = base_size + sum(sizes.values())
    published = set()
    if predicted > MAX_RESPONSE_JSON_BYTES and publish is None:
        raise ValueError("Media response exceeds inline JSON limit; a publisher is required")
    # Reserve enough space for a maximum-size escaped URL, including metadata.
    # Small remaining inline values preserve their exact bytes and ordering.
    for aid in sorted(sizes, key=sizes.get, reverse=True):
        if predicted <= MAX_RESPONSE_JSON_BYTES:
            break
        if sizes[aid] > 65_536:
            published.add(aid)
            predicted -= sizes[aid] - 65_536
    if predicted > MAX_RESPONSE_JSON_BYTES:
        raise ValueError("Media response metadata exceeds JSON limit")
    return published


async def _run(execution, workflow, legacy, download, publish):
    media = execution.media
    imports = {}
    for artifact in media["artifacts"]:
        execution.remaining()
        payload = await resolve_artifact(artifact, download=download)
        execution.remaining()
        filename = execution.stage.write(payload)
        node_id = "distributed-media-import:" + uuid.uuid4().hex
        while node_id in workflow:
            node_id = "distributed-media-import:" + uuid.uuid4().hex
        metadata = {key: value for key, value in artifact.items() if key != "transport"}
        workflow[node_id] = {"class_type": "MediaImportV1", "inputs": {
            "artifact_json": json.dumps(metadata, allow_nan=False), "filename": filename}}
        imports[artifact["id"]] = [node_id, _IMPORT_SLOTS[artifact["kind"]]]
    for ref in media["inputs"]:
        workflow[ref["node_id"]]["inputs"][ref["input_name"]] = list(imports[ref["artifact_id"]])
    for encoded, size, matches in legacy:
        execution.remaining()
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Invalid legacy image base64") from None
        if len(payload) != size:
            raise ValueError("Legacy image byte length mismatch")
        execution.remaining()
        filename = execution.stage.write(payload)
        for node_id in matches:
            workflow[node_id]["inputs"]["image"] = filename
    # Replacements can disconnect file loaders whose original endpoint-local
    # paths are unavailable. Keep exactly the post-rewrite exporter closure;
    # shared ancestors still needed by another branch must survive.
    reachable, pending = set(), list(execution.targets)
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(_link(value, workflow) for value in workflow[node_id]["inputs"].values()
                       if isinstance(value, list))
    workflow = {node_id: node for node_id, node in workflow.items() if node_id in reachable}
    entry = await execution.submit_and_wait(workflow)
    response, files = _manifest(entry, media)
    published = _publication_plan(response, files, publish)
    artifacts = []
    for artifact in files:
        payload = await execution.fetch(artifact)
        transport = None
        if artifact["id"] in published:
            metadata = {key: copy.deepcopy(value) for key, value in artifact.items() if key != "transport"}
            transport = await publish(metadata, payload)
            if not isinstance(transport, dict) or transport.get("type") != "url":
                raise ValueError("Publisher must return an expiring HTTPS URL transport")
        artifacts.append(artifact_from_payload(artifact["id"], artifact["kind"], artifact["descriptor"],
                                               payload, transport=transport))
    response["artifacts"] = artifacts
    validate_response(response, request_id=media["request_id"], bindings=media["outputs"])
    result = {"media": response}
    if len(_json_bytes(result)) > MAX_RESPONSE_JSON_BYTES:
        raise ValueError("Media response exceeds JSON byte limit")
    execution.remaining()
    return result


def _release_uncertain_work(task):
    _UNCERTAIN_WORK.discard(task)
    try:
        task.result()
    except BaseException:
        pass


async def _drain_work(work):
    """Cancel once; a misbehaving callback cannot make teardown unbounded."""
    if not work.done():
        work.cancel()
    deadline = asyncio.get_running_loop().time() + DRAIN_GRACE_SECONDS
    while not work.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            # Keep a strong reference until it exits, but never delete its files
            # asynchronously: uncertain ownership belongs to endpoint recovery.
            _UNCERTAIN_WORK.add(work)
            work.add_done_callback(_release_uncertain_work)
            return False
        try:
            # Unlike wait_for, wait's timeout does not wait for cancellation.
            await asyncio.wait({work}, timeout=remaining)
        except asyncio.CancelledError:
            continue
    _release_uncertain_work(work)
    return True


async def execute_media_request(input_data, *, comfy_url, input_dir, temp_dir=None,
                                timeout_seconds=1800, session=None, download=None, publish=None) -> dict:
    """Validate, stage, submit once, collect exact media, and retire owned inputs.

    ``timeout_seconds`` covers staging, HTTP, polling, and publishing. A separate
    bounded cancellation grace is allowed ONLY to establish the local prompt is
    terminal. On uncertain acceptance/termination, raise MediaExecutionUncertain
    and retain input files rather than racing an executing prompt. Never retry an
    ambiguous submission, even when the exception is a timeout.
    Async callback teardown has its own bounded grace; non-cooperative callbacks
    also raise MediaExecutionUncertain and retain files, even after terminal history.

    ``temp_dir`` is a trusted endpoint setting for this ComfyUI's local temp root,
    never a request-supplied path. Only validated exporter UUID.bin files beneath
    distributed-media-v1/sha256(request_id) are retired, after all collection and
    publisher work has unwound. Unidentified files remain lifecycle-owned. With
    ``temp_dir=None`` all exporter files remain lifecycle-owned as before.

    The pinned native /interrupt route has a check/global-interrupt race. Running
    prompts therefore fail closed with MediaExecutionUncertain, without posting
    /interrupt. A prompt-scoped request body alone is not a safety guarantee.
    """
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("Invalid media execution timeout")
    if not isinstance(comfy_url, str):
        raise ValueError("Invalid local Comfy URL")
    try:
        parsed = urlsplit(comfy_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError()
        parsed.port
    except ValueError:
        raise ValueError("Invalid local Comfy URL") from None
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    workflow, media, legacy, targets = _validate_request(input_data)
    stage = _Staging(input_dir, media["request_id"])
    exports = _OwnedExports(temp_dir, media["request_id"])
    owned_session = session is None
    if owned_session:
        session = aiohttp.ClientSession()
    execution = _Execution(session, comfy_url.rstrip("/"), stage, media, targets, deadline, exports)
    work = None
    try:
        remaining = execution.remaining()
        work = asyncio.create_task(_run(execution, workflow, legacy, download, publish))
        return await asyncio.wait_for(asyncio.shield(work), remaining)
    except BaseException as exc:
        # Retire files only after readers unwind. Cooperative async callbacks
        # normally drain immediately; an uncooperative one requires quarantine.
        if work is not None and not await _drain_work(work):
            raise MediaExecutionUncertain(
                "Media callback teardown unconfirmed; owned files retained; "
                "quarantine this endpoint before reuse; do not resubmit automatically"
            ) from exc
        if execution.attempted and not execution.terminal:
            execution.deadline = asyncio.get_running_loop().time() + CANCEL_GRACE_SECONDS
            recovery = asyncio.create_task(asyncio.wait_for(execution.settle(), CANCEL_GRACE_SECONDS))
            # Shield the bounded recovery even from repeated task.cancel() calls.
            while not recovery.done():
                try:
                    await asyncio.shield(recovery)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                recovery.result()
            except BaseException:
                pass
            if not execution.terminal:
                raise MediaExecutionUncertain(
                    "Local Comfy prompt terminal state unconfirmed; owned staged files retained "
                    f"under {stage.relative}; prompt_id={execution.prompt_id or 'unknown'}; "
                    "do not resubmit this request automatically"
                    + ("; native prompt-scoped interruption unavailable (non-atomic /interrupt route)"
                       if execution.interruption_blocked else "")
                ) from exc
        raise
    finally:
        try:
            try:
                exports.close(cleanup=(work is None or work.done()) and execution.terminal)
            finally:
                stage.close(cleanup=(work is None or work.done()) and
                            (not execution.attempted or execution.terminal))
        finally:
            if owned_session:
                await session.close()
