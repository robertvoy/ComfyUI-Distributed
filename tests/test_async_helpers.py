import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class _PromptQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _load_async_helpers_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "async_helpers.py"
    package_name = "dist_async_helpers_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    execution_module = types.ModuleType("execution")

    async def _validate_prompt(prompt_id, prompt, partial_execution_targets):
        return (True, None, ["9"], {})

    execution_module.validate_prompt = _validate_prompt
    execution_module.SENSITIVE_EXTRA_DATA_KEYS = []
    sys.modules["execution"] = execution_module

    prompt_server = types.SimpleNamespace(
        trigger_on_prompt=lambda payload: payload,
        number=12,
        prompt_queue=_PromptQueue(),
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=prompt_server)
    sys.modules["server"] = server_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.get_server_loop = lambda: None
    sys.modules[f"{package_name}.utils.network"] = network_module

    spec = importlib.util.spec_from_file_location(f"{package_name}.utils.async_helpers", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module, prompt_server


async_helpers, prompt_server = _load_async_helpers_module()


class QueuePromptPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_insert_if_is_checked_at_insertion_after_validation_yields(self):
        module, prompt_server = _load_async_helpers_module()
        validating = asyncio.Event()
        resume = asyncio.Event()
        state = {"cancelled": False}

        async def slow_validate(prompt_id, prompt, partial_execution_targets):
            validating.set()
            await resume.wait()
            return (True, None, ["9"], {})

        module.execution.validate_prompt = slow_validate
        task = asyncio.ensure_future(module.queue_prompt_payload(
            {"9": {"class_type": "SaveImage", "inputs": {}}}, None, "client", prompt_id="run-1",
            insert_if=lambda: not state["cancelled"],
        ))
        await validating.wait()
        state["cancelled"] = True  # the queue is cleared while validation is in progress
        resume.set()
        with self.assertRaises(module.QueueInsertionCancelled):
            await task
        self.assertEqual(prompt_server.prompt_queue.items, [])
        self.assertEqual(prompt_server.number, 12)  # no queue number consumed either

        state["cancelled"] = False
        resume.set()
        prompt_id = await module.queue_prompt_payload(
            {"9": {"class_type": "SaveImage", "inputs": {}}}, None, "client", prompt_id="run-1",
            insert_if=lambda: not state["cancelled"],
        )
        self.assertEqual(prompt_id, "run-1")
        self.assertEqual([item[1] for item in prompt_server.prompt_queue.items], ["run-1"])

    async def test_queue_prompt_payload_includes_create_time_and_client_metadata(self):
        result = await async_helpers.queue_prompt_payload(
            {"1": {"class_type": "Node"}},
            workflow_meta={"id": "workflow-1"},
            client_id="client-1",
            include_queue_metadata=True,
        )

        self.assertIsInstance(result["prompt_id"], str)
        self.assertTrue(result["prompt_id"])
        self.assertEqual(result["number"], 12)
        self.assertEqual(result["node_errors"], {})

        self.assertEqual(prompt_server.number, 13)
        self.assertEqual(len(prompt_server.prompt_queue.items), 1)
        queued_item = prompt_server.prompt_queue.items[0]
        self.assertEqual(queued_item[0], 12)
        extra_data = queued_item[3]
        self.assertEqual(extra_data["client_id"], "client-1")
        self.assertIn("create_time", extra_data)
        self.assertIsInstance(extra_data["create_time"], int)
        self.assertGreater(extra_data["create_time"], 0)
        self.assertEqual(extra_data["extra_pnginfo"]["workflow"], {"id": "workflow-1"})


class ExplicitTargetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module, self.server = _load_async_helpers_module()
        self.prompt = {"9": {"class_type": "Output"}, "10": {"class_type": "Output"}}
        self.validate = AsyncMock(return_value=(True, None, ["9"], {}))
        self.module.execution.validate_prompt = self.validate

    async def test_targets_survive_callback_omission_and_are_queued_exactly(self):
        seen = []
        def callback(payload):
            seen.append(payload["partial_execution_targets"][:])
            payload["partial_execution_targets"].append("10")
            return {"prompt": payload["prompt"]}
        self.server.trigger_on_prompt = callback
        targets = ["9"]
        await self.module.queue_prompt_payload(self.prompt, partial_execution_targets=targets)
        self.assertEqual(seen, [["9"]])
        self.assertEqual(targets, ["9"])
        self.assertEqual(self.validate.call_args.args[2], ["9"])
        self.assertEqual(self.server.prompt_queue.items[0][4], ["9"])

    async def test_invalid_requests_never_reach_validation_or_queue(self):
        for targets in ([], "9", [9], [True], ["9", "9"], ["missing"], {}):
            with self.subTest(targets=targets):
                with self.assertRaises(self.module.PromptValidationError):
                    await self.module.queue_prompt_payload(self.prompt, partial_execution_targets=targets)
        self.validate.assert_not_awaited()
        self.assertEqual(self.server.prompt_queue.items, [])

    async def test_removed_target_after_callback_is_rejected(self):
        self.server.trigger_on_prompt = lambda payload: {"prompt": {"10": self.prompt["10"]}}
        with self.assertRaises(self.module.PromptValidationError):
            await self.module.queue_prompt_payload(self.prompt, partial_execution_targets=["9"])
        self.validate.assert_not_awaited()
        self.assertEqual(self.server.prompt_queue.items, [])

    async def test_partial_subset_extra_roots_and_target_errors_fail_closed(self):
        for roots, errors in ((["9"], {}), (["9", "10", "extra"], {}),
                              (["9", "10"], {"9": {"errors": [{"message": "bad"}]}}),
                              (["9", "10"], {"dep": {"dependent_outputs": ["10"], "errors": []}})):
            with self.subTest(roots=roots, errors=errors):
                self.validate.return_value = (True, None, roots, errors)
                with self.assertRaises(self.module.PromptValidationError):
                    await self.module.queue_prompt_payload(self.prompt, partial_execution_targets=["9", "10"])
        self.assertEqual(self.server.prompt_queue.items, [])

    async def test_none_preserves_legacy_payload_and_validation(self):
        seen = []
        self.server.trigger_on_prompt = lambda payload: seen.append(payload.copy()) or payload
        await self.module.queue_prompt_payload(self.prompt)
        self.assertNotIn("partial_execution_targets", seen[0])
        self.assertIsNone(self.validate.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
