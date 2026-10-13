import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_module():
    spec = importlib.util.spec_from_file_location("dist_runpod_client", ROOT / "api/orchestration/runpod_client.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

client = load_module()


def workflow_for_request_size(target_size, timeout_seconds):
    """Build an ASCII workflow whose compact UTF-8 request body has exact size."""
    workflow = {"1": {"class_type": "Text", "inputs": {"value": ""}}}

    def body_size(candidate):
        payload = {
            "input": {"workflow": candidate, "images": []},
            "policy": {
                "executionTimeout": timeout_seconds * 1000,
                "ttl": timeout_seconds * 1000,
            },
        }
        return len(json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"))

    padding = target_size - body_size(workflow)
    if padding < 0:
        raise AssertionError("target is smaller than the request fixture")
    workflow["1"]["inputs"]["value"] = "x" * padding
    if body_size(workflow) != target_size:
        raise AssertionError("request fixture did not reach the exact target size")
    return workflow

class Response:
    def __init__(self, status=200, payload=None):
        self.status, self.payload = status, payload
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    async def json(self):
        return self.payload

class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_exact_generic_contract_and_once(self):
        session = Session([Response(payload={"id": "job-1", "status": "IN_QUEUE"})])
        api = client.RunpodClient(session, "endpoint123", "secret-value")
        result = await api.submit({"1": {"class_type": "EmptyImage", "inputs": {}}}, [], 1800)
        self.assertEqual(result, "job-1")
        method, url, kwargs = session.calls[0]
        self.assertEqual((method, url), ("POST", "https://api.runpod.ai/v2/endpoint123/run"))
        self.assertIsInstance(kwargs["data"], bytes)
        sent_payload = json.loads(kwargs["data"])
        self.assertEqual(set(sent_payload["input"]), {"workflow", "images"})
        self.assertEqual(sent_payload["policy"]["ttl"], 1800000)
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-value")

    async def test_submit_timeout_is_ambiguous_and_never_retried(self):
        session = Session([TimeoutError("secret-value in unsafe error")])
        api = client.RunpodClient(session, "endpoint123", "secret-value")
        with self.assertRaises(client.AmbiguousSubmission) as ctx:
            await api.submit({}, [], 30)
        self.assertNotIn("secret-value", str(ctx.exception))
        self.assertEqual(len(session.calls), 1)

    async def test_http_submit_failure_not_retried_or_leaked(self):
        for status in [401, 429, 500, 302]:
            session = Session([Response(status, {"error": "secret-value"})])
            api = client.RunpodClient(session, "endpoint123", "secret-value")
            with self.assertRaises(client.RunpodError) as ctx:
                await api.submit({}, [], 30)
            self.assertEqual(len(session.calls), 1)
            self.assertNotIn("secret-value", str(ctx.exception))

    async def test_health_zero_workers_is_valid(self):
        session = Session([Response(payload={"workers": {"idle": 0}, "jobs": {"inQueue": 0}})])
        api = client.RunpodClient(session, "endpoint123", "key")
        self.assertEqual((await api.health())["workers"]["idle"], 0)

    async def test_invalid_provider_id_no_network(self):
        session = Session([])
        with self.assertRaises(ValueError):
            client.RunpodClient(session, "https://attacker.invalid", "key")
        api = client.RunpodClient(session, "endpoint123", "key")
        with self.assertRaises(ValueError):
            await api.status("../other")
        self.assertFalse(session.calls)

    async def test_status_and_cancel_only_target_exact_job(self):
        session = Session([Response(payload={"status": "IN_PROGRESS"}), Response(payload={"status": "CANCELLED"})])
        api = client.RunpodClient(session, "endpoint123", "key")
        await api.status("job-1")
        await api.cancel("job-1")
        self.assertEqual([c[:2] for c in session.calls], [("GET", "https://api.runpod.ai/v2/endpoint123/status/job-1"), ("POST", "https://api.runpod.ai/v2/endpoint123/cancel/job-1")])

    async def test_missing_id_is_ambiguous(self):
        api = client.RunpodClient(Session([Response(payload={"status": "IN_QUEUE"})]), "endpoint123", "key")
        with self.assertRaises(client.AmbiguousSubmission):
            await api.submit({}, [], 30)

    async def test_large_payload_rejected_before_submit(self):
        session = Session([])
        api = client.RunpodClient(session, "endpoint123", "key")
        with self.assertRaises(ValueError):
            await api.submit({}, [{"image": "x" * client.MAX_REQUEST_BYTES}], 30)
        self.assertFalse(session.calls)

    async def test_exact_10mb_serialized_request_is_sent_as_validated_bytes(self):
        session = Session([Response(payload={"id": "boundary-job"})])
        api = client.RunpodClient(session, "endpoint123", "key")
        timeout_seconds = 37
        workflow = workflow_for_request_size(10_000_000, timeout_seconds)

        result = await api.submit(workflow, [], timeout_seconds)

        self.assertEqual(result, "boundary-job")
        sent = session.calls[0][2]["data"]
        self.assertIsInstance(sent, bytes)
        self.assertEqual(len(sent), 10_000_000)
        self.assertEqual(json.loads(sent)["policy"]["executionTimeout"], 37_000)

    async def test_one_byte_over_10mb_is_rejected_without_request_or_payload_leak(self):
        session = Session([Response(payload={"id": "must-not-be-used"})])
        api = client.RunpodClient(session, "endpoint123", "key")
        workflow = workflow_for_request_size(10_000_001, 37)
        workflow["1"]["inputs"]["value"] = "private graph value" + workflow["1"]["inputs"]["value"][19:]

        with self.assertRaisesRegex(ValueError, "10,000,001") as ctx:
            await api.submit(workflow, [], 37)

        self.assertNotIn("private graph value", str(ctx.exception))
        self.assertFalse(session.calls)

    async def test_wire_size_counts_utf8_graph_bytes_and_policy_fields(self):
        session = Session([Response(payload={"id": "unicode-job"})])
        api = client.RunpodClient(session, "endpoint123", "key")
        workflow = {"1": {"class_type": "Text", "inputs": {"value": "café雪"}}}

        await api.submit(workflow, [], 41)

        sent = session.calls[0][2]["data"]
        expected = json.dumps({
            "input": {"workflow": workflow, "images": []},
            "policy": {"executionTimeout": 41_000, "ttl": 41_000},
        }, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.assertEqual(sent, expected)
        self.assertGreater(len(sent), len(sent.decode("utf-8")))
