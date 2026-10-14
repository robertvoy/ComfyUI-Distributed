"""CPU-only endpoint adapter tests against an actual loopback Comfy HTTP fake."""
import asyncio
import base64
import copy
import hashlib
import importlib
import json
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "media_handler_testpkg"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "serverless_bridge")]
sys.modules[PACKAGE] = package
handler = importlib.import_module(PACKAGE + ".handler")
contract = importlib.import_module(PACKAGE + ".contract")


def request(kind="STRING", *, incoming=True):
    socket = {value: key for key, value in contract.SOCKET_KINDS.items()}[kind]
    binding = dict(binding_id="binding:1", collector_id="collector:1", job_id="job/42",
                   participant_id="worker:0", node_id="export:1", socket=socket, kind=kind)
    artifact = contract.artifact_from_payload("input:1", kind, {"encoding": "test"}, b"input bytes")
    return {"workflow": {
        "source:1": {"class_type": "Source", "inputs": {"value": "literal"}},
        "export:1": {"class_type": "MediaExportV1", "inputs": {
            "request_id": "request:1", "binding_json": json.dumps(binding), socket: ["source:1", 0]}},
    }, "media": {"version": 1, "profile": contract.PROFILE, "request_id": "request:1",
                  "artifacts": [artifact] if incoming else [],
                  "inputs": [{"node_id": "source:1", "input_name": "value", "artifact_id": "input:1"}] if incoming else [],
                  "outputs": [binding]}}


class Comfy:
    def __init__(self, root, data=None, *, payload=b"result bytes"):
        self.root = root
        self.data = data or request()
        self.payload = payload
        self.submitted = []
        self.views = []
        self.deletes = []
        self.interrupts = []
        self.polls = 0
        self.active = False
        self.mode = "success"
        self.history_failures = 0
        self.record_mutation = None
        self.prompt_seen = asyncio.Event()
        self.terminal_seen = False

    def staged(self):
        return list(self.root.glob("distributed-media-v1/*/*.bin"))

    def record(self):
        binding = self.data["media"]["outputs"][0]
        aid = "a" * 32
        artifact = contract.artifact_from_payload(aid, binding["kind"], {"encoding": "test"}, self.payload)
        artifact["transport"] = {"type": "comfy-file", "filename": aid + ".bin",
                                 "subfolder": "distributed-media-v1/" + hashlib.sha256(b"request:1").hexdigest(),
                                 "folder_type": "temp"}
        value = {"request_id": "request:1", "binding": dict(binding, count=1), "artifacts": [artifact],
                 "items": [{"binding_id": binding["binding_id"], "item_index": 0, "artifact_id": aid}]}
        if self.record_mutation:
            self.record_mutation(value)
        return value

    async def submit(self, req):
        data = await req.json()
        self.submitted.append(data)
        self.active = True
        self.prompt_seen.set()
        if self.data["media"]["artifacts"]:
            assert self.staged()
            link = data["prompt"]["source:1"]["inputs"]["value"]
            imported = data["prompt"][link[0]]
            assert imported["class_type"] == "MediaImportV1"
            metadata = json.loads(imported["inputs"]["artifact_json"])
            assert "transport" not in metadata
            assert (self.root / imported["inputs"]["filename"]).read_bytes() == b"input bytes"
        if self.mode == "ambiguous":
            return web.Response(status=500)
        if self.mode == "partial":
            return web.json_response({"prompt_id": "prompt:1", "node_errors": {"export:1": {"errors": ["bad"]}}})
        return web.json_response({"prompt_id": "prompt:1", "node_errors": {}})

    async def history(self, req):
        self.polls += 1
        if self.history_failures:
            self.history_failures -= 1
            return web.Response(status=503)
        if self.mode in {"queued", "running", "unrelated", "ambiguous"}:
            return web.json_response({})
        if self.data["media"]["artifacts"]:
            assert self.staged(), "staging must survive until terminal confirmation"
        self.active = False
        self.terminal_seen = True
        if self.mode in {"error", "interrupted"}:
            status = {"completed": False, "status_str": "error", "messages": [["execution_interrupted", {}]]}
        else:
            status = {"completed": True, "status_str": "success", "messages": []}
        return web.json_response({"prompt:1": {"status": status, "outputs": {} if self.mode == "missing" else {
            "export:1": {"media_exports": [self.record()]}}}})

    async def view(self, req):
        self.views.append(dict(req.query))
        assert self.terminal_seen
        payload = self.payload if self.mode != "corrupt" else b"X" * len(self.payload)
        if self.mode == "oversized":
            payload += b"X"
        return web.Response(body=payload)

    def entry(self, pid="prompt:1"):
        extra = dict(self.submitted[0].get("extra_data", {}), client_id=self.submitted[0]["client_id"])
        return [1, pid, self.submitted[0]["prompt"], extra, ["export:1"]]

    async def queue(self, req):
        if req.method == "POST":
            body = await req.json()
            assert body == {"delete": ["prompt:1"]}
            self.deletes.append(body)
            assert self.staged()
            self.mode = "deleted"
            self.active = False
            return web.Response(status=200)
        running = [self.entry()] if self.mode == "running" else []
        if self.mode == "unrelated":
            running = [self.entry("not-our-prompt")]
        pending = [self.entry()] if self.mode in {"queued", "ambiguous"} else []
        return web.json_response({"queue_running": running, "queue_pending": pending})

    async def interrupt(self, req):
        assert self.mode == "running"
        assert self.staged()
        self.interrupts.append(await req.json())
        self.mode = "interrupted"
        return web.Response(status=200)


@asynccontextmanager
async def serve(comfy):
    app = web.Application()
    app.add_routes([web.post("/prompt", comfy.submit), web.get("/history/{prompt_id}", comfy.history),
                    web.get("/view", comfy.view), web.get("/queue", comfy.queue),
                    web.post("/queue", comfy.queue), web.post("/interrupt", comfy.interrupt)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("kind,index", [("IMAGE", 0), ("AUDIO", 1), ("MASK", 2), ("STRING", 3), ("VIDEO", 4)])
def test_loopback_submit_poll_view_cleanup_and_caller_ownership(tmp_path, kind, index):
    async def check():
        data = request(kind)
        original = copy.deepcopy(data)
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url, aiohttp.ClientSession() as session:
            result = await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path, session=session)
            assert not session.closed
        contract.validate_response(result["media"], request_id="request:1", bindings=data["media"]["outputs"])
        assert await contract.resolve_artifact(result["media"]["artifacts"][0]) == comfy.payload
        assert len(comfy.submitted) == 1
        assert comfy.submitted[0]["partial_execution_targets"] == ["export:1"]
        assert comfy.submitted[0]["prompt"]["source:1"]["inputs"]["value"][1] == index
        assert comfy.views == [{"filename": "a" * 32 + ".bin", "subfolder": "distributed-media-v1/" + hashlib.sha256(b"request:1").hexdigest(), "type": "temp"}]
        assert data == original
        assert not comfy.staged()
    run(check())


@pytest.mark.parametrize("mode,match", [("corrupt", "checksum"), ("missing", "Missing"), ("error", "execution"), ("oversized", "length")])
def test_terminal_failures_clean_only_after_terminal(tmp_path, mode, match):
    async def check():
        comfy = Comfy(tmp_path)
        comfy.mode = mode
        async with serve(comfy) as url:
            with pytest.raises((ValueError, RuntimeError), match=match):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert len(comfy.submitted) == 1
        assert not comfy.staged()
    run(check())


@pytest.mark.parametrize("mutation", [
    lambda d: d["media"].update(version=True),
    lambda d: d["media"].update(profile="legacy"),
    lambda d: d["media"].update(request_id="../escape"),
    lambda d: d["media"]["outputs"][0].update(node_id="missing"),
    lambda d: d["workflow"]["export:1"].update(class_type="SaveImage"),
    lambda d: d["workflow"]["export:1"]["inputs"].update(request_id="other"),
    lambda d: d["workflow"]["export:1"]["inputs"].update(binding_json="{}"),
    lambda d: d["workflow"]["export:1"]["inputs"].update(images=["source:1", 0]),
    lambda d: d["workflow"]["export:1"]["inputs"].update(texts="literal"),
    lambda d: d["workflow"].update(unrelated={"class_type": "SaveImage", "inputs": {}}),
    lambda d: d["workflow"].update(unsafe={"class_type": "MediaImportV1", "inputs": {}}),
    lambda d: d["media"]["inputs"].append(copy.deepcopy(d["media"]["inputs"][0])),
    lambda d: d["media"]["inputs"][0].update(node_id="missing"),
    lambda d: d["media"]["inputs"][0].update(input_name="missing"),
    lambda d: d["media"].update(inputs=[]),
    lambda d: d["media"]["artifacts"][0].update(sha256="0" * 64),
])
def test_rejection_before_submit(tmp_path, mutation):
    async def check():
        data = request()
        mutation(data)
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url:
            with pytest.raises(ValueError):
                await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.submitted
        assert not comfy.staged()
    run(check())


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(request_id="other"),
    lambda r: r["binding"].update(count=2),
    lambda r: r["binding"].update(collector_id="other"),
    lambda r: r["items"][0].update(item_index=1),
    lambda r: r["artifacts"][0]["transport"].update(filename="../escape"),
    lambda r: r["artifacts"][0]["transport"].update(subfolder="distributed-media-v1/other"),
    lambda r: r["artifacts"][0]["transport"].update(folder_type="input"),
    lambda r: r["artifacts"][0].update(byte_length=contract.MAX_ARTIFACT_BYTES + 1),
])
def test_bad_result_manifest_rejected_before_any_file_read(tmp_path, mutation):
    async def check():
        comfy = Comfy(tmp_path)
        comfy.record_mutation = mutation
        async with serve(comfy) as url:
            with pytest.raises(ValueError):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.views
        assert not comfy.staged()
    run(check())


def test_idempotent_history_retry_without_resubmission(tmp_path):
    async def check():
        comfy = Comfy(tmp_path)
        comfy.history_failures = 2
        async with serve(comfy) as url:
            await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert comfy.polls == 3
        assert len(comfy.submitted) == 1
    run(check())


@pytest.mark.parametrize("mode", ["queued"])
def test_deadline_cancels_only_own_prompt_before_cleanup(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(handler, "POLL_INTERVAL_SECONDS", 0.005)
    async def check():
        comfy = Comfy(tmp_path)
        comfy.mode = mode
        async with serve(comfy) as url:
            with pytest.raises(TimeoutError):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path, timeout_seconds=0.04)
        assert len(comfy.submitted) == 1
        assert bool(comfy.deletes) == (mode == "queued")
        assert bool(comfy.interrupts) == (mode == "running")
        assert not comfy.staged()
    run(check())


def test_unconfirmed_terminal_retains_files_and_never_interrupts_unrelated(tmp_path, monkeypatch):
    monkeypatch.setattr(handler, "POLL_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(handler, "CANCEL_GRACE_SECONDS", 0.06)
    async def check():
        comfy = Comfy(tmp_path)
        comfy.mode = "unrelated"
        async with serve(comfy) as url:
            with pytest.raises(RuntimeError, match="retained"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path, timeout_seconds=0.04)
        assert comfy.staged()
        assert not comfy.interrupts
        assert not comfy.deletes
        assert len(comfy.submitted) == 1
    run(check())


def test_ambiguous_post_is_not_retried_and_own_queue_entry_is_reconciled(tmp_path):
    async def check():
        comfy = Comfy(tmp_path)
        comfy.mode = "ambiguous"
        async with serve(comfy) as url:
            with pytest.raises(RuntimeError):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert len(comfy.submitted) == 1
        assert len(comfy.deletes) == 1
        assert not comfy.staged()
    run(check())


def test_task_cancellation_fails_closed_and_preserves_caller_session(tmp_path):
    async def check():
        comfy = Comfy(tmp_path)
        comfy.mode = "running"
        async with serve(comfy) as url, aiohttp.ClientSession() as session:
            task = asyncio.create_task(handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path, session=session))
            await comfy.prompt_seen.wait()
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(handler.MediaExecutionUncertain, match="prompt-scoped interruption unavailable"):
                await task
            assert not session.closed
        assert not comfy.interrupts
        assert comfy.staged()
    run(check())


def test_legacy_images_are_owned_and_rewritten_without_overwriting_user_files(tmp_path):
    async def check():
        data = request(incoming=False)
        data["workflow"]["source:1"] = {"class_type": "LoadImage", "inputs": {"image": "same.png"}}
        data["images"] = [{"name": "same.png", "image": base64.b64encode(b"new image").decode()}]
        (tmp_path / "same.png").write_bytes(b"user image")
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url:
            await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path)
        filename = comfy.submitted[0]["prompt"]["source:1"]["inputs"]["image"]
        assert filename.startswith("distributed-media-v1/")
        assert filename.endswith(".bin")
        assert (tmp_path / "same.png").read_bytes() == b"user image"
        assert not comfy.staged()
    run(check())


def test_url_input_uses_bounded_injected_downloader(tmp_path):
    async def check():
        data = request()
        data["media"]["artifacts"][0]["transport"] = {"type": "url", "url": "https://example.org/a", "expires_at": "2099-01-01T00:00:00Z"}
        calls = []
        async def download(url, *, max_bytes):
            calls.append((url, max_bytes))
            return b"input bytes"
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url:
            await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path, download=download)
        assert calls == [("https://example.org/a", len(b"input bytes"))]
    run(check())


def test_large_result_requires_publisher_and_validates_returned_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(handler, "MAX_RESPONSE_JSON_BYTES", 100_000)
    async def check():
        comfy = Comfy(tmp_path, payload=b"Z" * 100_000)
        async with serve(comfy) as url:
            with pytest.raises(ValueError, match="publisher"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.views
        assert not comfy.staged()
        published = []
        async def publish(metadata, payload):
            assert "transport" not in metadata
            assert payload == comfy.payload
            published.append(metadata)
            return {"type": "url", "url": "https://example.org/result", "expires_at": "2099-01-01T00:00:00Z"}
        async with serve(comfy) as url:
            result = await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path, publish=publish)
        assert len(published) == 1
        assert result["media"]["artifacts"][0]["transport"]["type"] == "url"
        assert len(json.dumps(result).encode()) <= 100_000
        async def bad_publish(metadata, payload):
            return {"type": "url", "url": "http://example.org/result", "expires_at": "2099-01-01T00:00:00Z"}
        async with serve(comfy) as url:
            with pytest.raises(ValueError):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path, publish=bad_publish)
    run(check())


@pytest.mark.parametrize("name", ["../photo.png", "folder/photo.png", "folder\\photo.png", ".", "..", "/photo.png"])
def test_legacy_basename_rejected_without_submit(tmp_path, name):
    async def check():
        data = request(incoming=False)
        data["images"] = [{"name": name, "image": "YQ=="}]
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url:
            with pytest.raises(ValueError):
                await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.submitted
    run(check())


def test_staging_rejects_symlink_without_writing_or_submit(tmp_path):
    async def check():
        outside = tmp_path / "unrelated"
        outside.mkdir()
        (tmp_path / "distributed-media-v1").symlink_to(outside, target_is_directory=True)
        comfy = Comfy(tmp_path)
        async with serve(comfy) as url:
            with pytest.raises(ValueError, match="symlink"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.submitted
        assert not list(outside.iterdir())
    run(check())


def test_cleanup_preserves_other_runs_in_same_request_namespace(tmp_path):
    async def check():
        directory = tmp_path / "distributed-media-v1" / hashlib.sha256(b"request:1").hexdigest()
        directory.mkdir(parents=True)
        sibling = directory / ("b" * 32 + ".bin")
        sibling.write_bytes(b"other run")
        comfy = Comfy(tmp_path)
        async with serve(comfy) as url:
            await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert sibling.read_bytes() == b"other run"
        assert comfy.staged() == [sibling]
    run(check())


def test_input_total_limit_checked_before_downloader_or_submit(tmp_path):
    async def check():
        data = request()
        data["media"]["artifacts"] = []
        data["media"]["inputs"] = []
        for index in range(3):
            artifact = contract.artifact_from_payload(f"input:{index}", "STRING", {}, b"",
                transport={"type": "url", "url": "https://example.org/input", "expires_at": "2099-01-01T00:00:00Z"})
            artifact["byte_length"] = contract.MAX_ARTIFACT_BYTES
            data["media"]["artifacts"].append(artifact)
            name = f"value:{index}"
            data["workflow"]["source:1"]["inputs"][name] = "literal"
            data["media"]["inputs"].append({"node_id": "source:1", "input_name": name, "artifact_id": artifact["id"]})
        comfy = Comfy(tmp_path, data)
        async def forbidden_download(*args, **kwargs):
            pytest.fail("oversized request reached downloader")
        async with serve(comfy) as url:
            with pytest.raises(ValueError, match="total byte"):
                await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path, download=forbidden_download)
        assert not comfy.submitted
    run(check())


def test_output_total_limit_checked_before_fetch(tmp_path):
    def excessive(record):
        first = record["artifacts"][0]
        record["artifacts"] = []
        record["items"] = []
        record["binding"]["count"] = 3
        for index, char in enumerate("abc"):
            artifact = copy.deepcopy(first)
            artifact["id"] = char * 32
            artifact["byte_length"] = contract.MAX_ARTIFACT_BYTES
            artifact["transport"]["filename"] = artifact["id"] + ".bin"
            record["artifacts"].append(artifact)
            record["items"].append({"binding_id": "binding:1", "item_index": index, "artifact_id": artifact["id"]})
    async def check():
        comfy = Comfy(tmp_path)
        comfy.record_mutation = excessive
        async with serve(comfy) as url:
            with pytest.raises(ValueError, match="total byte"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.views
        assert not comfy.staged()
    run(check())


def test_partial_target_acceptance_is_not_reported_as_success(tmp_path):
    async def check():
        comfy = Comfy(tmp_path)
        comfy.mode = "partial"
        async with serve(comfy) as url:
            with pytest.raises(RuntimeError, match="incomplete"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=tmp_path)
        assert len(comfy.submitted) == 1
        assert not comfy.views
        assert not comfy.staged()
    run(check())


# Native routes are compiled from the pinned source, without importing ComfyUI
# (which would initialize the GPU). All dependencies and HTTP are CPU-only.
@pytest.mark.parametrize("current,requested,race,expected", [
    ("own", "own", False, ["own"]),
    ("other", "own", False, []),
    ("own", "own", True, ["successor"]),
])
def test_pinned_native_interrupt_route_has_snapshot_race(current, requested, race, expected):
    import ast
    import logging
    source = Path("/home/robert/ComfyUI_dev/server.py")
    if not source.is_file():
        pytest.skip("Pinned ComfyUI source unavailable")
    tree = ast.parse(source.read_text())
    routes = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
              and any("/interrupt" in ast.unparse(d) for d in node.decorator_list)]
    assert len(routes) == 1
    route = routes[0]
    route.decorator_list = []
    interrupts = []
    class Queue:
        running = current
        def get_current_queue(self):
            snapshot = [[1, self.running, {}, {}, []]]
            if race:
                self.running = "successor"
            return snapshot, []
    queue = Queue()
    scope = {"json": json, "logging": logging, "web": web,
             "self": types.SimpleNamespace(prompt_queue=queue),
             "nodes": types.SimpleNamespace(interrupt_processing=lambda: interrupts.append(queue.running))}
    exec(compile(ast.Module(body=[route], type_ignores=[]), str(source), "exec"), scope)
    async def check():
        async def body():
            return {"prompt_id": requested}
        response = await scope[route.name](types.SimpleNamespace(json=body))
        assert response.status == 200
    run(check())
    assert interrupts == expected


def exporter_file(root, *, name="a" * 32 + ".bin", payload=b"result bytes"):
    directory = root / "distributed-media-v1" / hashlib.sha256(b"request:1").hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize("mode", ["success", "corrupt", "oversized", "error", "interrupted", "partial"])
def test_owned_export_cleanup_after_terminal_success_or_failure(tmp_path, mode):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        sibling = exporter_file(temp, name="b" * 32 + ".bin", payload=b"unrelated")
        ordinary = exporter_file(temp, name="keep.txt", payload=b"ordinary")
        comfy = Comfy(tmp_path / "input")
        comfy.mode = mode
        view = comfy.view
        async def reading(req):
            assert owned.exists(), "cleanup raced artifact collection"
            return await view(req)
        comfy.view = reading
        async with serve(comfy) as url:
            if mode == "success":
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert comfy.terminal_seen
        assert not owned.exists()
        assert sibling.read_bytes() == b"unrelated"
        assert ordinary.read_bytes() == b"ordinary"
        assert not comfy.staged()
    run(check())


def test_omitted_temp_root_leaves_exporter_lifecycle_ownership(tmp_path):
    async def check():
        owned = exporter_file(tmp_path / "temp")
        comfy = Comfy(tmp_path / "input")
        async with serve(comfy) as url:
            await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root)
        assert owned.read_bytes() == comfy.payload
    run(check())


@pytest.mark.parametrize("mode", ["running", "unrelated"])
def test_uncertain_exports_and_inputs_retained_without_interrupt(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(handler, "POLL_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(handler, "CANCEL_GRACE_SECONDS", 0.04)
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        comfy = Comfy(tmp_path / "input")
        comfy.mode = mode
        async with serve(comfy) as url:
            with pytest.raises(handler.MediaExecutionUncertain):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root,
                                                    temp_dir=temp, timeout_seconds=0.03)
        assert owned.exists()
        assert comfy.staged()
        assert not comfy.interrupts
        assert not comfy.deletes
    run(check())


@pytest.mark.parametrize("component", ["root", "namespace", "request", "file"])
def test_export_cleanup_never_traverses_symlinks(tmp_path, component):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        selected = {"root": temp, "namespace": owned.parent.parent,
                    "request": owned.parent, "file": owned}[component]
        outside = tmp_path / "outside"
        selected.rename(outside)
        selected.symlink_to(outside, target_is_directory=component != "file")
        comfy = Comfy(tmp_path / "input")
        async with serve(comfy) as url:
            await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert selected.is_symlink()
        assert owned.read_bytes() == comfy.payload
    run(check())


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(request_id="other"),
    lambda r: r["binding"].update(collector_id="other"),
    lambda r: r["artifacts"][0]["transport"].update(filename="../escape"),
    lambda r: r["artifacts"][0]["transport"].update(subfolder="distributed-media-v1/other"),
])
def test_unidentified_exports_are_not_deleted(tmp_path, mutation):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        comfy = Comfy(tmp_path / "input")
        comfy.record_mutation = mutation
        async with serve(comfy) as url:
            with pytest.raises(ValueError):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert owned.exists()
        assert not comfy.views
    run(check())


@pytest.mark.parametrize("failure", ["required", "publish"])
def test_publication_failures_cleanup_identified_exports(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(handler, "MAX_RESPONSE_JSON_BYTES", 100_000)
    async def check():
        temp = tmp_path / "temp"
        comfy = Comfy(tmp_path / "input", payload=b"Z" * 100_000)
        owned = exporter_file(temp, payload=comfy.payload)
        async def publish(metadata, payload):
            assert owned.exists()
            raise ValueError("publisher failed")
        async with serve(comfy) as url:
            with pytest.raises(ValueError, match="publisher"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root,
                                                    temp_dir=temp, publish=publish if failure == "publish" else None)
        assert not owned.exists()
    run(check())


def test_repeated_cancel_drains_collection_before_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(handler, "MAX_RESPONSE_JSON_BYTES", 100_000)
    async def check():
        temp = tmp_path / "temp"
        comfy = Comfy(tmp_path / "input", payload=b"Z" * 100_000)
        owned = exporter_file(temp, payload=comfy.payload)
        started, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def publish(metadata, payload):
            started.set()
            try:
                await asyncio.Future()
            finally:
                draining.set()
                await release.wait()
                assert owned.exists(), "cleanup while publisher still reading"
        async with serve(comfy) as url:
            task = asyncio.create_task(handler.execute_media_request(comfy.data, comfy_url=url,
                                       input_dir=comfy.root, temp_dir=temp, publish=publish))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.wait_for(draining.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            assert owned.exists()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not owned.exists()
    run(check())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_stubborn_publisher_teardown_is_bounded_and_retains_files(tmp_path, monkeypatch, cancel_caller):
    monkeypatch.setattr(handler, "MAX_RESPONSE_JSON_BYTES", 100_000)
    monkeypatch.setattr(handler, "DRAIN_GRACE_SECONDS", 0.03)
    async def check():
        temp = tmp_path / "temp"
        comfy = Comfy(tmp_path / "input", payload=b"Z" * 100_000)
        owned = exporter_file(temp, payload=comfy.payload)
        started, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def publish(metadata, payload):
            started.set()
            try:
                await asyncio.Future()
            finally:
                draining.set()
                await release.wait()
                assert owned.exists()
        async with serve(comfy) as url:
            task = asyncio.create_task(handler.execute_media_request(comfy.data, comfy_url=url,
                        input_dir=comfy.root, temp_dir=temp, publish=publish,
                        timeout_seconds=5 if cancel_caller else 0.15))
            try:
                await asyncio.wait_for(started.wait(), 1)
                if cancel_caller:
                    task.cancel()
                await asyncio.wait_for(draining.wait(), 1)
                if cancel_caller:
                    task.cancel()  # must not reset the grace or recancel the reader
                done, _ = await asyncio.wait({task}, timeout=0.5)
                assert task in done, "callback suppressed cancellation indefinitely"
                with pytest.raises(handler.MediaExecutionUncertain, match="teardown"):
                    task.result()
                assert comfy.terminal_seen and owned.exists() and comfy.staged()
                assert handler._UNCERTAIN_WORK
                pending = list(handler._UNCERTAIN_WORK)
                release.set()
                await asyncio.gather(*pending, return_exceptions=True)
                await asyncio.sleep(0)
                assert not handler._UNCERTAIN_WORK
                assert owned.exists() and comfy.staged(), "late callback must not retire files"
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    run(check())


def test_export_replaced_during_collection_is_preserved(tmp_path):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        comfy = Comfy(tmp_path / "input")
        view = comfy.view
        async def replace(req):
            replacement = owned.with_suffix(".replacement")
            replacement.write_bytes(b"unrelated replacement")
            replacement.replace(owned)
            return await view(req)
        comfy.view = replace
        async with serve(comfy) as url:
            await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert owned.read_bytes() == b"unrelated replacement"
    run(check())



@pytest.mark.parametrize("status", ["success", "error"])
def test_partial_history_cleans_only_valid_identified_exporters(tmp_path, status):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        unidentified = exporter_file(temp, name="c" * 32 + ".bin")
        data = request()
        second = dict(data["media"]["outputs"][0], binding_id="binding:2", node_id="export:2", collector_id="collector:2")
        data["media"]["outputs"].append(second)
        data["workflow"]["export:2"] = {"class_type": "MediaExportV1", "inputs": {
            "request_id": "request:1", "binding_json": json.dumps(second), "texts": ["source:1", 0]}}
        comfy = Comfy(tmp_path / "input", data)
        comfy.mode = status
        async with serve(comfy) as url:
            with pytest.raises((ValueError, RuntimeError)):
                await handler.execute_media_request(data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert not owned.exists()
        assert unidentified.exists()
        assert not comfy.views
    run(check())


def test_missing_history_exports_remain_lifecycle_owned(tmp_path):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        comfy = Comfy(tmp_path / "input")
        comfy.mode = "missing"
        async with serve(comfy) as url:
            with pytest.raises(ValueError, match="Missing"):
                await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert owned.exists()
        assert not comfy.staged()
    run(check())


@pytest.mark.parametrize("replacement", ["symlink", "directory", "hardlink"])
def test_nonregular_or_linked_export_files_are_preserved(tmp_path, replacement):
    async def check():
        temp = tmp_path / "temp"
        owned = exporter_file(temp)
        outside = tmp_path / "outside"
        outside.write_bytes(b"unrelated")
        if replacement == "hardlink":
            alias = tmp_path / "alias"
            alias.hardlink_to(owned)
        else:
            owned.unlink()
            if replacement == "symlink":
                owned.symlink_to(outside)
            else:
                owned.mkdir()
        comfy = Comfy(tmp_path / "input")
        async with serve(comfy) as url:
            await handler.execute_media_request(comfy.data, comfy_url=url, input_dir=comfy.root, temp_dir=temp)
        assert owned.exists()
        assert outside.read_bytes() == b"unrelated"
    run(check())



@pytest.mark.parametrize("shared", [False, True])
def test_direct_exporter_input_prunes_only_obsolete_loader_ancestors(tmp_path, shared):
    async def check():
        data = request("VIDEO")
        data["workflow"]["source:1"] = {"class_type": "LoadVideo", "inputs": {"file": "unavailable.mp4"}}
        data["media"]["inputs"] = [{"node_id": "export:1", "input_name": "video", "artifact_id": "input:1"}]
        if shared:
            second = dict(data["media"]["outputs"][0], binding_id="binding:2", node_id="export:2", collector_id="collector:2")
            data["media"]["outputs"].append(second)
            data["workflow"]["export:2"] = {"class_type": "MediaExportV1", "inputs": {
                "request_id": "request:1", "binding_json": json.dumps(second), "video": ["source:1", 0]}}
        fake_data = copy.deepcopy(data)
        fake_data["media"].update(artifacts=[], inputs=[])
        comfy = Comfy(tmp_path, fake_data)
        history = comfy.history
        async def full_history(req):
            response = await history(req)
            body = json.loads(response.text)
            if shared:
                record = comfy.record()
                record["binding"] = dict(second, count=1)
                record["artifacts"][0]["id"] = "b" * 32
                record["artifacts"][0]["transport"]["filename"] = "b" * 32 + ".bin"
                record["items"] = [{"binding_id": second["binding_id"], "item_index": 0, "artifact_id": "b" * 32}]
                body["prompt:1"]["outputs"]["export:2"] = {"media_exports": [record]}
            return web.json_response(body)
        comfy.history = full_history
        async with serve(comfy) as url:
            result = await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path)
        contract.validate_response(result["media"], request_id="request:1", bindings=data["media"]["outputs"])
        submitted = comfy.submitted[0]["prompt"]
        imported, slot = submitted["export:1"]["inputs"]["video"]
        assert slot == 4
        assert submitted[imported]["class_type"] == "MediaImportV1"
        assert json.loads(submitted[imported]["inputs"]["artifact_json"])["kind"] == "VIDEO"
        assert ("source:1" in submitted) is shared
        assert set(submitted) == ({"export:1", imported, "source:1", "export:2"} if shared else {"export:1", imported})
        assert not comfy.staged()
    run(check())


@pytest.mark.parametrize("target,kind", [("request_id", "VIDEO"), ("binding_json", "VIDEO"), ("video", "STRING")])
def test_direct_exporter_input_rejects_controls_and_wrong_kind(tmp_path, target, kind):
    async def check():
        data = request("VIDEO")
        data["media"]["inputs"] = [{"node_id": "export:1", "input_name": target, "artifact_id": "input:1"}]
        data["media"]["artifacts"][0]["kind"] = kind
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url:
            with pytest.raises(ValueError):
                await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path)
        assert not comfy.submitted
        assert not comfy.staged()
    run(check())


def test_slow_download_times_out_before_any_prompt(tmp_path):
    async def check():
        data = request()
        data["media"]["artifacts"][0]["transport"] = {"type": "url", "url": "https://example.org/input", "expires_at": "2099-01-01T00:00:00Z"}
        async def download(*args, **kwargs):
            await asyncio.sleep(60)
        comfy = Comfy(tmp_path, data)
        async with serve(comfy) as url:
            with pytest.raises(TimeoutError):
                await handler.execute_media_request(data, comfy_url=url, input_dir=tmp_path, download=download, timeout_seconds=0.02)
        assert not comfy.submitted
        assert not comfy.staged()
    run(check())
