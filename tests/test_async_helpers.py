import importlib.util
import sys
import types
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
