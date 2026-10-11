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
