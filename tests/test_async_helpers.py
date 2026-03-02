import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _FakePromptQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakePromptServer:
    def __init__(self):
        self.number = 0
        self.prompt_queue = _FakePromptQueue()

    def trigger_on_prompt(self, payload):
        return payload


def _bootstrap_package(package_name):
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.get_server_loop = asyncio.get_event_loop
    sys.modules[f"{package_name}.utils.network"] = network_module

    fake_prompt_server = _FakePromptServer()

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=fake_prompt_server)
    sys.modules["server"] = server_module

    execution_module = types.ModuleType("execution")

    async def _validate_prompt(_prompt_id, _prompt, _partial_targets):
        return (True, None, ["1"], {})

    execution_module.validate_prompt = _validate_prompt
    execution_module.SENSITIVE_EXTRA_DATA_KEYS = ()
    sys.modules["execution"] = execution_module

    return fake_prompt_server


def _load_async_helpers_module():
    package_name = "dist_async_helpers_testpkg"
    fake_prompt_server = _bootstrap_package(package_name)
    module_path = Path(__file__).resolve().parents[1] / "utils/async_helpers.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.utils.async_helpers",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, fake_prompt_server


class AsyncHelpersQueuePromptPayloadTests(unittest.TestCase):
    def test_queue_prompt_payload_includes_create_time_metadata(self):
        async_helpers_module, fake_prompt_server = _load_async_helpers_module()

        prompt_id = asyncio.run(
            async_helpers_module.queue_prompt_payload(
                {"1": {"class_type": "KSampler", "inputs": {}}},
                workflow_meta={"id": "workflow-1"},
                client_id="client-1",
            )
        )

        self.assertIsInstance(prompt_id, str)
        self.assertTrue(fake_prompt_server.prompt_queue.items)

        queued_item = fake_prompt_server.prompt_queue.items[-1]
        self.assertEqual(len(queued_item), 6)
        self.assertEqual(queued_item[1], prompt_id)

        extra_data = queued_item[3]
        self.assertEqual(extra_data["client_id"], "client-1")
        self.assertEqual(extra_data["extra_pnginfo"]["workflow"]["id"], "workflow-1")
        self.assertIn("create_time", extra_data)
        self.assertIsInstance(extra_data["create_time"], int)
        self.assertGreater(extra_data["create_time"], 0)


if __name__ == "__main__":
    unittest.main()
