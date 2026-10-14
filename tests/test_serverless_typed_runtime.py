"""Offline typed submission/delivery regressions; no provider or GPU access."""
import asyncio
import copy
import importlib
import json
import socket
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Load once before integration fixtures snapshot/restore sys.modules. PyTorch
# cannot be re-imported after those fixtures unload its extension-backed modules.
import torch

from test_runpod_client import Response, Session
from test_serverless_runtime import runtime, Secrets

client = importlib.import_module(runtime.__package__ + ".runpod_client")
bridge = runtime.__package__.split(".")[0] + ".serverless_bridge"
contract = importlib.import_module(bridge + ".contract")


def prepared(sockets=("texts",)):
    worker = {"id": "pool:1", "pool_id": "pool", "endpoint_id": "ep1",
              "timeout_seconds": 30, "type": "serverless", "media_profile": contract.PROFILE}
    bindings = [dict(binding_id=f"binding-{i}", collector_id="collector", job_id="collector-job",
                     participant_id=worker["id"], node_id=f"export-{i}", socket=name,
                     kind=contract.SOCKET_KINDS[name]) for i, name in enumerate(sockets)]
    media = dict(version=1, profile=contract.PROFILE, request_id="request-1",
                 artifacts=[], inputs=[], outputs=copy.deepcopy(bindings))
    graph = {"source": {"class_type": "Source", "inputs": {"value": "café雪"}}}
    for binding in bindings:
        graph[binding["node_id"]] = {"class_type": "MediaExportV1", "inputs": {
            "request_id": media["request_id"], "binding_json": json.dumps(binding),
            binding["socket"]: ["source", 0]}}
    return dict(worker=worker, workflow=graph, images=[], manifest=bindings, media=media)


def response(job):
    envelope = dict(version=1, profile=contract.PROFILE, request_id=job["media"]["request_id"],
                    bindings=[], artifacts=[], items=[])
    for index, binding in enumerate(job["manifest"]):
        envelope["bindings"].append(dict(binding, count=2))
        for item_index, text in enumerate(("", " \n雪")):
            aid = f"artifact-{index}-{item_index}"
            descriptor, payload = {"encoding": "utf8-v1"}, text.encode()
            if binding["kind"] != "STRING":
                codec = importlib.import_module(bridge + ".codec")
                values = {"IMAGE": torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 4),
                          "MASK": torch.arange(2, dtype=torch.float16).reshape(1, 1, 2),
                          "AUDIO": {"waveform": torch.arange(4, dtype=torch.float32).reshape(1, 2, 2), "sample_rate": 48000}}
                descriptor, payload = codec.encode_payload(binding["kind"], values[binding["kind"]])
            artifact = contract.artifact_from_payload(aid, binding["kind"], descriptor, payload)
            envelope["artifacts"].append(artifact)
            envelope["items"].insert(0, dict(binding_id=binding["binding_id"], item_index=item_index, artifact_id=aid))
    return {"media": envelope}


class TypedClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_typed_bytes_and_legacy_unchanged(self):
        job = prepared()
        session = Session([Response(payload={"id": "job-1"})])
        api = client.RunpodClient(session, "ep1", "test-only")
        await api.submit(job["workflow"], [], 30, media=job["media"])
        expected = {"input": {"workflow": job["workflow"], "images": [], "media": job["media"]},
                    "policy": {"executionTimeout": 30000, "ttl": 30000}}
        self.assertEqual(session.calls[0][2]["data"], json.dumps(expected, allow_nan=False,
                         ensure_ascii=False, separators=(",", ":")).encode())
        self.assertEqual(len(session.calls), 1)
        expected["input"].pop("media")
        legacy = json.dumps(expected, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertEqual(client.serialize_request(job["workflow"], [], 30), legacy)
        self.assertEqual(client.serialize_request(job["workflow"], [], 30, media=None), legacy)

    async def test_invalid_envelopes_never_post(self):
        for field, value in [("version", True), ("profile", "legacy-image-v1"),
                             ("request_id", "../bad"), ("outputs", []), ("inputs", [{}])]:
            with self.subTest(field=field):
                job = prepared()
                job["media"][field] = value
                session = Session([])
                with self.assertRaises(ValueError):
                    await client.RunpodClient(session, "ep1", "test-only").submit(
                        job["workflow"], [], 30, media=job["media"])
                self.assertFalse(session.calls)
        session = Session([])
        with self.assertRaises(ValueError):
            await client.RunpodClient(session, "ep1", "test-only").submit({}, [], 30, media={})
        self.assertFalse(session.calls)

    async def test_typed_size_boundary_uses_exact_body_before_post(self):
        job = prepared()
        size = len(client.serialize_request(job["workflow"], [], 30, media=job["media"]))
        session = Session([Response(payload={"id": "boundary"})])
        api = client.RunpodClient(session, "ep1", "test-only")
        with patch.object(client, "MAX_REQUEST_BYTES", size):
            await api.submit(job["workflow"], [], 30, media=job["media"])
            job["workflow"]["source"]["inputs"]["value"] += "x"
            with self.assertRaisesRegex(ValueError, "Serialized serverless request body"):
                await api.submit(job["workflow"], [], 30, media=job["media"])
        self.assertEqual(len(session.calls), 1)


class TypedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = types.SimpleNamespace(distributed_pending_jobs={"collector-job": asyncio.Queue()},
                                            distributed_jobs_lock=asyncio.Lock())
        self.manager = runtime.ServerlessManager(self.server, Secrets(), Path(self.temp.name) / "jobs.json",
                                                 AsyncMock(return_value=object()))
        self.manager.poll_seconds = 0
        self.job = prepared()
        self.api = types.SimpleNamespace(submit=AsyncMock(return_value="provider-job"),
            status=AsyncMock(return_value={"status": "COMPLETED", "output": response(self.job)}),
            cancel=AsyncMock(return_value={"status": "CANCELLED"}))
        self.client_patch = patch.object(runtime, "RunpodClient", return_value=self.api)
        self.client_patch.start()

    async def asyncTearDown(self):
        await self.manager.cancel()
        self.client_patch.stop()
        self.temp.cleanup()

    async def run_job(self):
        await self.manager.start("execution", "prompt", [self.job])
        await self.manager.wait_for_execution("execution")

    async def test_preflight_all_jobs_before_any_journal_or_task(self):
        for mutation in ("profile", "missing_media", "manifest", "participant", "exporter", "empty_media"):
            with self.subTest(mutation=mutation):
                bad = prepared()
                if mutation == "profile":
                    bad["worker"]["media_profile"] = "legacy-image-v1"
                elif mutation == "missing_media":
                    bad.pop("media")
                elif mutation == "manifest":
                    bad["manifest"][0]["job_id"] = "other"
                elif mutation == "participant":
                    bad["worker"]["id"] = "other-participant"
                elif mutation == "exporter":
                    bad["workflow"]["export-0"]["inputs"]["request_id"] = "other"
                else:
                    bad["media"] = {}
                with self.assertRaises(ValueError):
                    await self.manager.start("execution", "prompt", [self.job, bad])
                self.assertFalse(self.manager.ledger.all())
                self.assertFalse(self.manager.tasks)
                self.assertFalse(self.manager.runs)
                self.api.submit.assert_not_awaited()

    async def test_text_roundtrip_id_journal_and_payload_cleanup(self):
        async def status(_job):
            self.assertEqual(self.manager.ledger.all()[0]["job_id"], "provider-job")
            return {"status": "COMPLETED", "output": response(self.job)}
        self.api.status.side_effect = status
        await self.run_job()
        self.api.submit.assert_awaited_once_with(self.job["workflow"], [], 30, media=self.job["media"])
        queue = self.server.distributed_pending_jobs["collector-job"]
        self.assertEqual(queue.get_nowait(), {"worker_id": "pool:1", "media": {"texts": ["", " \n雪"]}, "is_last": True})
        self.assertTrue(queue.empty())
        self.assertEqual(self.manager.ledger.all()[0]["status"], "COMPLETED")
        run = next(iter(self.manager.runs.values()))
        for field in ("workflow", "images", "media"):
            self.assertNotIn(field, run)
        self.assertNotIn("data_base64", self.manager.ledger.path.read_text())
        self.api.cancel.assert_not_awaited()

    async def test_all_sockets_grouped_without_native_value_conversion(self):
        self.job = prepared(tuple(contract.SOCKET_KINDS))
        native = {binding["binding_id"]: [object(), {"native": object()}] for binding in self.job["manifest"]}
        with patch.object(contract, "decode_response", AsyncMock(return_value=native)) as decode:
            await self.run_job()
        decode.assert_awaited_once()
        self.assertEqual(decode.call_args.kwargs["request_id"], "request-1")
        self.assertEqual(decode.call_args.kwargs["bindings"], self.job["manifest"])
        queue = self.server.distributed_pending_jobs["collector-job"]
        item = queue.get_nowait()
        self.assertEqual(set(item), {"worker_id", "media", "is_last"})
        self.assertEqual(set(item["media"]), set(contract.SOCKET_KINDS))
        for binding in self.job["manifest"]:
            for actual, expected in zip(item["media"][binding["socket"]], native[binding["binding_id"]]):
                self.assertIs(actual, expected)
        self.assertTrue(queue.empty())

    async def test_identity_and_late_decode_failure_deliver_only_one_error(self):
        self.job = prepared(("texts", "images"))
        for mutation in ("request", "binding", "checksum", "descriptor"):
            with self.subTest(mutation=mutation):
                output = response(self.job)
                if mutation == "request":
                    output["media"]["request_id"] = "other"
                elif mutation == "binding":
                    output["media"]["bindings"][-1]["participant_id"] = "other"
                elif mutation == "checksum":
                    output["media"]["artifacts"][-1]["sha256"] = "0" * 64
                else:
                    output["media"]["artifacts"][-1]["descriptor"] = {}
                self.api.status.return_value = {"status": "COMPLETED", "output": output}
                await self.run_job()
                queue = self.server.distributed_pending_jobs["collector-job"]
                item = queue.get_nowait()
                self.assertIn("serverless_error", item)
                self.assertNotIn("media", item)
                self.assertTrue(queue.empty())
        self.api.cancel.assert_not_awaited()

    async def test_closed_queue_is_not_resurrected(self):
        self.server.distributed_pending_jobs.clear()
        await self.run_job()
        self.assertFalse(self.server.distributed_pending_jobs)
        self.assertEqual(self.manager.ledger.all()[0]["status"], "COMPLETED")

    async def test_real_tensor_audio_and_mask_codecs_preserve_dtype_and_shape(self):
        self.job = prepared(("texts", "images", "audio", "masks"))
        self.api.status.return_value = {"status": "COMPLETED", "output": response(self.job)}
        await self.run_job()
        result = self.server.distributed_pending_jobs["collector-job"].get_nowait()["media"]
        self.assertEqual(result["texts"], ["", " \n雪"])
        self.assertEqual(result["images"][0].dtype, torch.float64)
        self.assertEqual(result["images"][0].shape, (1, 1, 2, 4))
        self.assertEqual(result["masks"][0].dtype, torch.float16)
        self.assertEqual(result["audio"][0]["sample_rate"], 48000)
        self.assertTrue(torch.equal(result["audio"][0]["waveform"], torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)))
        self.assertTrue(all(len(values) == 2 for values in result.values()))

    async def test_run_rechecks_profile_before_submission(self):
        await self.manager.start("execution", "prompt", [self.job])
        self.job["worker"]["media_profile"] = "legacy-image-v1"
        await self.manager.wait_for_execution("execution")
        self.api.submit.assert_not_awaited()
        self.assertEqual(self.manager.ledger.all()[0]["status"], "FAILED")

    async def test_multiple_collectors_receive_one_message_each(self):
        self.job = prepared(("texts", "images"))
        self.job["manifest"][-1]["job_id"] = "second-job"
        self.job["media"]["outputs"] = copy.deepcopy(self.job["manifest"])
        self.job["workflow"]["export-1"]["inputs"]["binding_json"] = json.dumps(self.job["manifest"][-1])
        self.server.distributed_pending_jobs["second-job"] = asyncio.Queue()
        self.api.status.return_value = {"status": "COMPLETED", "output": response(self.job)}
        await self.run_job()
        for job_id, name in (("collector-job", "texts"), ("second-job", "images")):
            queue = self.server.distributed_pending_jobs[job_id]
            self.assertEqual(queue.qsize(), 1)
            item = queue.get_nowait()
            self.assertEqual(set(item["media"]), {name})
            self.assertTrue(item["is_last"])

    async def test_prestart_cancel_releases_media_and_deduplicates_errors(self):
        self.job = prepared(("texts", "images"))
        await self.manager.start("execution", "prompt", [self.job])
        await self.manager.cancel(execution_id="execution")
        self.api.submit.assert_not_awaited()
        run = next(iter(self.manager.runs.values()))
        self.assertNotIn("media", run)
        self.assertEqual(self.server.distributed_pending_jobs["collector-job"].qsize(), 1)
        self.assertEqual(self.manager.ledger.all()[0]["status"], "CANCELLED")

    async def test_ambiguous_submit_keeps_reservation_without_media(self):
        self.api.submit.side_effect = runtime.AmbiguousSubmission("Unknown submission")
        await self.run_job()
        self.api.submit.assert_awaited_once()
        self.api.cancel.assert_not_awaited()
        self.assertEqual(self.manager.ledger.all()[0]["status"], "UNKNOWN")
        self.assertTrue(self.manager.pool_is_active("pool"))
        self.assertNotIn("media", next(iter(self.manager.runs.values())))

    async def test_legacy_still_uses_tensor_messages_and_no_media_kwarg(self):
        self.job.pop("media")
        self.job["worker"]["media_profile"] = "legacy-image-v1"
        self.job["manifest"] = [{"job_id": "collector-job", "prefix": "dist_x", "kind": "image"}]
        with patch.object(runtime, "decode_results", AsyncMock(return_value={"collector-job": ["a", "b"]})):
            await self.run_job()
        self.api.submit.assert_awaited_once_with(self.job["workflow"], [], 30)
        queue = self.server.distributed_pending_jobs["collector-job"]
        self.assertEqual([queue.get_nowait(), queue.get_nowait()], [
            {"worker_id": "pool:1", "tensor": "a", "image_index": 0, "is_last": False},
            {"worker_id": "pool:1", "tensor": "b", "image_index": 1, "is_last": True}])


class TypedDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_bound_cap_and_redirect_policy_without_credentials(self):
        typed = importlib.import_module(runtime.__package__ + ".serverless_typed_results")
        class DownloadResponse(Response):
            headers = {}
            async def chunks(self, _size):
                yield b"ab"
                yield b"cd"
            @property
            def content(self):
                return types.SimpleNamespace(iter_chunked=self.chunks)
        class DownloadSession(Response):
            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return self.response
        session = DownloadSession()
        session.calls = []
        session.response = DownloadResponse()
        with patch.object(typed.aiohttp, "TCPConnector", return_value=object()), \
                patch.object(typed.aiohttp, "ClientSession", return_value=session) as factory:
            self.assertEqual(await typed.download_public_artifact("https://example.com/a", max_bytes=contract.MAX_ARTIFACT_BYTES), b"abcd")
            self.assertFalse(factory.call_args.kwargs["trust_env"])
            self.assertNotIn("headers", factory.call_args.kwargs)
            self.assertFalse(session.calls[0][1]["allow_redirects"])
            with self.assertRaises(ValueError):
                await typed.download_public_artifact("https://example.com/a", max_bytes=3)
            session.response.status = 302
            session.response.headers = {"Location": "https://127.0.0.1/private"}
            with self.assertRaises(ValueError):
                await typed.download_public_artifact("https://example.com/a", max_bytes=4)
        for url in ("http://example.com/a", "https://127.0.0.1/a", "https://user:pass@example.com/a"):
            with self.assertRaises(ValueError):
                await typed.download_public_artifact(url, max_bytes=4)
        resolver = typed._PublicResolver()
        with patch.object(resolver, "_getaddrinfo", AsyncMock(return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])):
            with self.assertRaises(OSError):
                await resolver.resolve("example.com", 443)
