"""Image-list participation uses the existing master orchestrator setting."""
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch

from test_serverless_integration import PKG, load_modules


class MasterSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.q, modules, self.ps = load_modules()
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.worker = {"id": "worker", "name": "Worker", "type": "local", "enabled": True}
        self.config = {"workers": [self.worker], "settings": {}}
        self.q.load_config = lambda: self.config
        # Model the legacy helper's fallback: image-list must not inherit it.
        self.q.select_active_workers = AsyncMock(side_effect=lambda workers, ws, delegate, **kw:
                                                (workers, delegate if workers else False))
        self.preflight = AsyncMock(return_value={"worker": True})
        sys.modules[PKG + ".api.orchestration.dispatch"].preflight_image_list_workers = self.preflight
        self.q.queue_prompt_payload = AsyncMock(return_value={"prompt_id": "master-prompt", "number": 1})
        self.prompt = {
            "1": {"class_type": "EmptyImage", "inputs": {}},
            "2": {"class_type": "DistributedImageListStart", "inputs": {"images": ["1", 0]}},
            "3": {"class_type": "DistributedImageListEnd", "inputs": {"images": ["2", 0], "context": ["2", 3]}},
        }

    async def queue(self, **kwargs):
        return await self.q.orchestrate_distributed_execution(self.prompt, None, "client", **kwargs)

    def prepared_inputs(self):
        return self.q.queue_prompt_payload.call_args.args[0]["2"]["inputs"]

    async def test_default_includes_master_and_direct_workers(self):
        result = await self.queue()
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], True)
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), ["worker"])
        self.assertEqual(result[2], 1)  # Public worker_count still counts remote participants only.
        self.preflight.assert_awaited_once_with(
            [{"id": "worker", "name": "Worker", "type": "local", "host": None, "port": 8188}],
            require_idle=False)
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.assertEqual(self.ps.distributed_pending_jobs, {})

    async def test_orchestrator_setting_excludes_master_processing(self):
        self.config["settings"]["master_delegate_only"] = True
        await self.queue()
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], False)
        self.assertTrue(self.preflight.call_args.kwargs["require_idle"])
        # Upstream source stays on master even in orchestrator-only mode.
        self.assertIn("1", self.q.queue_prompt_payload.call_args.args[0])

    async def test_explicit_override_wins_over_config_in_both_directions(self):
        for configured, requested in [(True, False), (False, True)]:
            with self.subTest(configured=configured, requested=requested):
                self.config["settings"]["master_delegate_only"] = configured
                await self.queue(delegate_master=requested)
                self.assertIs(self.prepared_inputs()["image_list_master_participates"], not requested)

    async def test_master_only_with_explicit_empty_worker_selection(self):
        result = await self.queue(enabled_worker_ids=[])
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], True)
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), [])
        self.assertEqual(result[2], 0)
        self.preflight.assert_not_awaited()

    async def test_master_only_when_no_workers_are_configured(self):
        self.config["workers"] = []
        await self.queue()
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], True)

    async def test_unavailable_workers_do_not_prevent_enabled_master(self):
        self.q.select_active_workers.return_value = ([], False)
        self.q.select_active_workers.side_effect = None
        self.preflight.return_value = {}
        await self.queue()
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), [])
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], True)

    async def test_busy_workers_do_not_prevent_enabled_master(self):
        self.preflight.return_value = {}
        await self.queue()
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), [])

    async def test_orchestrator_without_workers_fails_before_queue(self):
        self.config["settings"]["master_delegate_only"] = True
        with self.assertRaisesRegex(RuntimeError, "orchestrator|Orchestrator"):
            await self.queue(enabled_worker_ids=[])
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_orchestrator_never_inherits_legacy_offline_master_fallback(self):
        self.config["settings"]["master_delegate_only"] = True
        self.q.select_active_workers.side_effect = None
        self.q.select_active_workers.return_value = ([], False)
        self.preflight.return_value = {}
        with self.assertRaisesRegex(RuntimeError, "orchestrator|Orchestrator"):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_serverless_only_uses_master_without_provider_dispatch(self):
        self.config["workers"] = [{"id": "pool", "type": "serverless", "enabled": True}]
        result = await self.queue()
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], True)
        self.assertEqual(result[2], 0)
        self.q.dispatch_worker_prompt.assert_not_awaited()

    async def test_incompatible_worker_is_not_silently_hidden_by_master(self):
        self.preflight.side_effect = RuntimeError("upgrade worker")
        with self.assertRaisesRegex(RuntimeError, "upgrade worker"):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()
