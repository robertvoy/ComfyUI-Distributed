"""Image-list participation uses the existing master orchestrator setting."""
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

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

    async def test_prepared_master_prunes_side_outputs_without_mutating_input(self):
        import copy
        self.prompt['op'] = {'class_type': 'Flagged', 'inputs': {'image': ['2', 0]}}
        self.prompt['3']['inputs']['images'] = ['op', 0]
        self.prompt['preview'] = {'class_type': 'Flagged', 'inputs': {'image': ['op', 0]}}
        self.prompt['save'] = {'class_type': 'Flagged', 'inputs': {'image': ['3', 0]}}
        original = copy.deepcopy(self.prompt)
        registry = type('Registry', (), {'NODE_CLASS_MAPPINGS': {
            'Flagged': type('Flagged', (), {'OUTPUT_NODE': True})}})
        with patch.dict(sys.modules, {'nodes': registry}):
            await self.queue()
        queued = self.q.queue_prompt_payload.call_args
        self.assertEqual(queued.kwargs['partial_execution_targets'], ['3', 'save'])
        self.assertIn('op', queued.args[0])
        self.assertNotIn('preview', queued.args[0])
        self.assertEqual(self.prompt, original)

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

    async def test_end_overrides_require_value_capability_before_master_queue(self):
        self.prompt['3']['inputs']['value_10'] = ['2', 2]
        await self.queue()
        self.assertIs(self.preflight.call_args.kwargs['require_value_results'], True)
        self.q.queue_prompt_payload.reset_mock()
        self.preflight.side_effect = RuntimeError('value_results unsupported')
        with self.assertRaisesRegex(RuntimeError, 'value_results'):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_orchestrator_setting_excludes_master_processing(self):
        self.config["settings"]["master_delegate_only"] = True
        await self.queue()
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], False)
        self.assertIs(self.preflight.call_args.kwargs["require_idle"], False)
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

    def use_real_preflight(self, queues):
        """Exercise capability/queue parsing with read-only fake HTTP responses."""
        from test_dispatch_selection import dispatch, _FakeSession, _valid_capability

        responses = {(worker_id, "capabilities"): (200, _valid_capability())
                     for worker_id in queues}
        responses.update({(worker_id, "queue"): (200, queue)
                          for worker_id, queue in queues.items()})
        session = _FakeSession(responses)
        for target, value in (
            ("get_client_session", AsyncMock(return_value=session)),
            ("build_worker_url", lambda worker, path: f"http://{worker['id']}{path}"),
        ):
            patched = patch.object(dispatch, target, value)
            patched.start()
            self.addCleanup(patched.stop)
        self.preflight = AsyncMock(wraps=dispatch.preflight_image_list_workers)
        sys.modules[PKG + ".api.orchestration.dispatch"].preflight_image_list_workers = self.preflight
        return responses

    async def run_prepared_selection(self):
        """Run real runtime selection up to the scheduler, without any item I/O."""
        rt = sys.modules[PKG + ".image_list.runtime"]
        inputs = self.prepared_inputs()
        image = object()
        context = rt.create_master_job(
            "2", "3", self.q.queue_prompt_payload.call_args.args[0],
            [{"image": image, "index": 0}],
            json.loads(inputs["image_list_enabled_worker_ids"]),
            inputs["image_list_master_callback"],
            master_participates=inputs["image_list_master_participates"],
            explicit_targets=True,
        )
        self.addCleanup(rt.remove_master_job, context)
        self.scheduler = AsyncMock(return_value=[{"image": image}])
        self.local_slot = Mock(return_value=types.SimpleNamespace(close=AsyncMock()))
        self.runtime_session = AsyncMock(return_value=object())
        config = sys.modules[PKG + ".utils.config"]
        network = sys.modules[PKG + ".utils.network"]
        with patch.object(config, "load_config", lambda: self.config), \
             patch.object(config, "get_worker_timeout_seconds", lambda: 60, create=True), \
             patch.object(config, "get_image_list_job_timeout_seconds", lambda: 0, create=True), \
             patch.object(network, "get_client_session", self.runtime_session, create=True), \
             patch.object(network, "build_worker_url", Mock(), create=True), \
             patch.object(rt, "dynamic_schedule", self.scheduler), \
             patch.dict(sys.modules, {PKG + ".image_list.local_executor": types.SimpleNamespace(
                 LocalExecutionSlot=self.local_slot)}):
            await rt.run_image_list_job(context)
        self.assertIsNone(rt.get_master_job(context))
        return self.scheduler.call_args.args[1]

    async def test_busy_worker_at_enqueue_is_available_to_scheduler_when_idle_at_start(self):
        responses = self.use_real_preflight({
            "worker": {"queue_running": [[1, "prompt-a"]], "queue_pending": []},
        })
        result = await self.queue()
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), ["worker"])
        self.assertEqual(result[2], 1)  # Candidates, not a promise of execution-time availability.
        self.q.dispatch_worker_prompt.assert_not_awaited()

        responses[("worker", "queue")] = (200, {"queue_running": [], "queue_pending": []})
        # A newly enabled worker is not implicitly added to this queued selection.
        self.config["workers"].append({"id": "later", "enabled": True})
        slots = await self.run_prepared_selection()
        self.assertEqual([slot for slot in slots if isinstance(slot, str)], ["worker"])
        self.assertEqual(len(slots), 2)
        self.assertEqual(self.preflight.await_count, 2)
        self.assertEqual(self.preflight.await_args_list[0].args, self.preflight.await_args_list[1].args)

    async def test_worker_still_busy_at_start_is_excluded_from_scheduler(self):
        self.use_real_preflight({
            "worker": {"queue_running": [], "queue_pending": [[1, "prompt-a"]]},
        })
        await self.queue()
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), ["worker"])
        slots = await self.run_prepared_selection()
        self.assertEqual(len(slots), 1)
        self.assertNotIn("worker", slots)
        self.assertEqual(self.preflight.await_count, 2)
        self.runtime_session.assert_not_awaited()

    async def test_explicit_empty_selection_stays_master_only_at_start(self):
        self.use_real_preflight({"worker": {"queue_running": [], "queue_pending": []}})
        await self.queue(enabled_worker_ids=[])
        slots = await self.run_prepared_selection()
        self.assertEqual(len(slots), 1)
        self.assertNotIn("worker", slots)
        self.preflight.assert_not_awaited()
        self.runtime_session.assert_not_awaited()

    async def test_busy_incompatible_worker_fails_before_master_queue(self):
        responses = self.use_real_preflight({
            "worker": {"queue_running": [[1, "prompt-a"]], "queue_pending": []},
        })
        capability = responses[("worker", "capabilities")][1]
        for field, invalid, message in (
            ("protocol_version", 1, "incompatible"),
            ("execution_targets", False, "execution_targets"),
            ("value_results", False, "value_results"),
        ):
            with self.subTest(field=field), patch.dict(capability, {field: invalid}):
                if field == "value_results":
                    self.prompt["3"]["inputs"]["value_10"] = ["2", 2]
                with self.assertRaisesRegex(RuntimeError, message):
                    await self.queue()
                self.q.queue_prompt_payload.assert_not_awaited()
                self.q.dispatch_worker_prompt.assert_not_awaited()

    async def test_orchestrator_with_all_workers_busy_queues_for_idle_start(self):
        responses = self.use_real_preflight({
            "worker": {"queue_running": [[1, "prompt-a"]], "queue_pending": []},
        })
        result = await self.queue(delegate_master=True)
        self.q.queue_prompt_payload.assert_awaited_once()
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), ["worker"])
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], False)
        self.assertEqual(result[2], 1)
        self.q.dispatch_worker_prompt.assert_not_awaited()

        responses[("worker", "queue")] = (200, {"queue_running": [], "queue_pending": []})
        self.assertEqual(await self.run_prepared_selection(), ["worker"])
        self.local_slot.assert_not_called()
        self.assertEqual(self.preflight.await_count, 2)
        self.assertIs(self.preflight.await_args_list[0].kwargs["require_idle"], False)
        self.assertIs(self.preflight.await_args_list[1].kwargs["require_idle"], True)

    async def test_orchestrator_rechecks_availability_without_master_fallback(self):
        responses = self.use_real_preflight({"worker": {"queue_running": [], "queue_pending": []}})
        await self.queue(delegate_master=True)
        responses[("worker", "queue")] = (200, {"queue_running": [[1, "other"]], "queue_pending": []})
        with self.assertRaisesRegex(RuntimeError, "No selected feature-compatible direct worker is idle"):
            await self.run_prepared_selection()
        self.assertEqual(self.preflight.await_count, 2)
        self.scheduler.assert_not_awaited()
        self.local_slot.assert_not_called()

    async def test_orchestrator_retains_busy_candidates_when_another_worker_is_idle(self):
        self.config["workers"].append({"id": "idle", "enabled": True})
        responses = self.use_real_preflight({
            "worker": {"queue_running": [[1, "prompt-a"]], "queue_pending": []},
            "idle": {"queue_running": [], "queue_pending": []},
        })
        await self.queue(delegate_master=True)
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), ["worker", "idle"])
        responses[("worker", "queue")] = (200, {"queue_running": [], "queue_pending": []})
        responses[("idle", "queue")] = (200, {"queue_running": [[1, "other"]], "queue_pending": []})
        self.assertEqual(await self.run_prepared_selection(), ["worker"])
        self.local_slot.assert_not_called()

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

    async def test_serverless_pool_is_retained_with_participating_master(self):
        self.config["workers"] = [{
            "id": "pool", "name": "Cloud", "type": "serverless", "enabled": True,
            "endpoint_id": "endpoint", "media_profile": "comfy-media-v1",
        }]
        result = await self.queue()
        self.assertIs(self.prepared_inputs()["image_list_master_participates"], True)
        self.assertEqual(json.loads(self.prepared_inputs()["image_list_enabled_worker_ids"]), ["pool"])
        self.assertEqual(result[2], 1)
        self.preflight.assert_not_awaited()
        self.q.dispatch_worker_prompt.assert_not_awaited()
        self.assertEqual(self.ps.distributed_pending_jobs, {})

    async def test_legacy_serverless_pool_is_rejected_not_silently_ignored(self):
        self.config["workers"] = [{
            "id": "pool", "name": "Cloud", "type": "serverless", "enabled": True,
            "endpoint_id": "endpoint",
        }]
        with self.assertRaisesRegex(ValueError, "comfy-media-v1"):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()
        self.preflight.assert_not_awaited()

    async def test_incompatible_worker_is_not_silently_hidden_by_master(self):
        self.preflight.side_effect = RuntimeError("upgrade worker")
        with self.assertRaisesRegex(RuntimeError, "upgrade worker"):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()
