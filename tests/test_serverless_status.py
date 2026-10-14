import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PKG = "dist_serverless_status_tests"


class StatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        def mod(name, **kwargs):
            result = types.ModuleType(name)
            result.__dict__.update(kwargs)
            return result
        self.manager = types.SimpleNamespace(health=AsyncMock(return_value={"ok": True, "queueRemaining": 0, "message": "Ready (scales from zero)"}), public_jobs=lambda _: [], cancel=AsyncMock(return_value=1))
        self.pool = {"id": "pool", "type": "serverless", "endpoint_id": "ep"}
        modules = {PKG: mod(PKG, __path__=[str(ROOT)]), PKG + ".api": mod("api", __path__=[str(ROOT/'api')]),
                   PKG + ".utils": mod("utils", __path__=[]),
                   PKG + ".utils.config": mod("config", load_config=lambda: {"workers": [self.pool]}),
                   PKG + ".api.orchestration": mod("orchestration", __path__=[str(ROOT/'api/orchestration')]),
                   PKG + ".api.orchestration.serverless_runtime": mod("runtime", get_manager=lambda: self.manager),
                   "server": mod("server", PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=types.SimpleNamespace(get=lambda _: lambda f: f, post=lambda _: lambda f: f))))}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(PKG+'.api.serverless_routes', ROOT/'api/serverless_routes.py')
            self.routes = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.routes)

    async def test_health_only_never_submits_and_supports_scale_to_zero(self):
        request = types.SimpleNamespace(json=AsyncMock(return_value={"worker_id": "pool"}))
        result = await self.routes.test_serverless_endpoint(request)
        self.assertEqual(result.status, 200)
        self.assertTrue(json.loads(result.text)["ok"])
        self.manager.health.assert_awaited_once_with(self.pool, fresh=True)
        self.manager.cancel.assert_not_awaited()

    async def test_unknown_pool_cannot_cancel_other_jobs(self):
        request = types.SimpleNamespace(json=AsyncMock(return_value={"worker_id": "other"}))
        result = await self.routes.cancel_serverless_endpoint(request)
        self.assertEqual(result.status, 404)
        self.manager.cancel.assert_not_awaited()

    async def test_transport_error_does_not_leak_credentials(self):
        self.manager.health.side_effect = RuntimeError("Bearer secret-token https://signed.invalid/?token=value")
        request = types.SimpleNamespace(match_info={"worker_id": "pool"})
        result = await self.routes.serverless_status_endpoint(request)
        self.assertFalse(json.loads(result.text)["ok"])
        self.assertNotIn("secret-token", result.text)
        self.assertNotIn("signed.invalid", result.text)

    async def test_cancel_uses_only_requested_pool(self):
        request = types.SimpleNamespace(json=AsyncMock(return_value={"worker_id": "pool"}))
        result = await self.routes.cancel_serverless_endpoint(request)
        self.assertEqual(result.status, 200)
        self.manager.cancel.assert_awaited_once_with(worker_id="pool")

    async def test_reconcile_only_reads_the_requested_pools_jobs(self):
        self.manager.reconcile = AsyncMock(return_value=2)
        request = types.SimpleNamespace(json=AsyncMock(return_value={"worker_id": "pool"}))
        result = await self.routes.reconcile_serverless_endpoint(request)
        self.assertEqual(result.status, 200)
        self.assertEqual(json.loads(result.text)["resolved"], 2)
        self.manager.reconcile.assert_awaited_once_with(worker_id="pool")
        self.manager.cancel.assert_not_awaited()

    async def test_resolve_requires_explicit_boolean_confirmation_and_record_id(self):
        self.manager.acknowledge_recovery = AsyncMock()
        for payload in ({"worker_id": "pool", "record_id": "record"},
                        {"worker_id": "pool", "record_id": "record", "provider_checked": "true"},
                        {"worker_id": "pool", "provider_checked": True}):
            with self.subTest(payload=payload):
                result = await self.routes.resolve_serverless_endpoint(types.SimpleNamespace(json=AsyncMock(return_value=payload)))
                self.assertEqual(result.status, 400)
        self.manager.acknowledge_recovery.assert_not_awaited()

    async def test_resolve_is_scoped_and_does_not_claim_provider_cancellation(self):
        self.manager.acknowledge_recovery = AsyncMock()
        request = types.SimpleNamespace(json=AsyncMock(return_value={"worker_id": "pool", "record_id": "record", "provider_checked": True}))
        result = await self.routes.resolve_serverless_endpoint(request)
        self.assertEqual(result.status, 200)
        self.manager.acknowledge_recovery.assert_awaited_once_with("pool", "record", provider_checked=True)
        self.assertIn("not provider-confirmed", json.loads(result.text)["message"])
        self.manager.cancel.assert_not_awaited()

    async def test_live_or_mismatched_record_refuses_manual_resolution(self):
        self.manager.acknowledge_recovery = AsyncMock(side_effect=ValueError("Unresolved inactive job not found for this pool"))
        request = types.SimpleNamespace(json=AsyncMock(return_value={"worker_id": "pool", "record_id": "record", "provider_checked": True}))
        result = await self.routes.resolve_serverless_endpoint(request)
        self.assertEqual(result.status, 409)

    async def test_failed_health_still_exposes_recovery_ids(self):
        self.manager.health.side_effect = RuntimeError("private transport error")
        self.manager.public_jobs = lambda _: [{"record_id": "record", "status": "UNKNOWN"}]
        result = await self.routes.serverless_status_endpoint(types.SimpleNamespace(match_info={"worker_id": "pool"}))
        self.assertEqual(json.loads(result.text)["jobs"][0]["record_id"], "record")
