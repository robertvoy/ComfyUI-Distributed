import importlib.util
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
        self.assertEqual(set(kwargs["json"]["input"]), {"workflow", "images"})
        self.assertEqual(kwargs["json"]["policy"]["ttl"], 1800000)
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
