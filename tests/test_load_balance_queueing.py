import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def _load_queue_orchestration_module():
    module_path = Path(__file__).resolve().parents[1] / "api" / "queue_orchestration.py"
    package_name = "dist_queue_orchestration_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    api_pkg = types.ModuleType(f"{package_name}.api")
    api_pkg.__path__ = []
    sys.modules[f"{package_name}.api"] = api_pkg

    orchestration_pkg = types.ModuleType(f"{package_name}.api.orchestration")
    orchestration_pkg.__path__ = []
    sys.modules[f"{package_name}.api.orchestration"] = orchestration_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    prompt_server_instance = types.SimpleNamespace(
        distributed_pending_jobs={},
        distributed_worker_reservations={},
    )
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=prompt_server_instance)
    sys.modules["server"] = server_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers_module.queue_prompt_payload = AsyncMock(
        return_value={"prompt_id": "prompt-master", "number": 1, "node_errors": {}}
    )
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: {
        "workers": [{"id": "w1", "name": "Worker 1", "enabled": True, "host": "worker-1", "port": 8188}],
        "settings": {"load_balance_idle_poll_interval_seconds": 0},
    }
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.ORCHESTRATION_MEDIA_SYNC_CONCURRENCY = 4
    constants_module.ORCHESTRATION_MEDIA_SYNC_TIMEOUT = 5.0
    constants_module.ORCHESTRATION_WORKER_PROBE_CONCURRENCY = 8
    constants_module.ORCHESTRATION_WORKER_PREP_CONCURRENCY = 4
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_master_url = lambda *_args, **_kwargs: "http://master:8188"
    network_module.build_master_callback_url = lambda *_args, **_kwargs: "http://master:8188"
    sys.modules[f"{package_name}.utils.network"] = network_module

    trace_module = types.ModuleType(f"{package_name}.utils.trace_logger")
    trace_module.trace_debug = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.trace_logger"] = trace_module

    schemas_module = types.ModuleType(f"{package_name}.api.schemas")
    schemas_module.parse_positive_int = lambda value, default: int(value if value is not None else default)
    schemas_module.parse_positive_float = lambda value, default: float(value if value is not None else default)
    sys.modules[f"{package_name}.api.schemas"] = schemas_module

    dispatch_module = types.ModuleType(f"{package_name}.api.orchestration.dispatch")
    dispatch_module.dispatch_worker_prompt = AsyncMock()
    dispatch_module.select_active_workers = AsyncMock(return_value=([{"id": "w1", "name": "Worker 1"}], False))
    dispatch_module.select_least_busy_worker = AsyncMock()
    sys.modules[f"{package_name}.api.orchestration.dispatch"] = dispatch_module

    media_sync_module = types.ModuleType(f"{package_name}.api.orchestration.media_sync")
    media_sync_module.convert_paths_for_platform = lambda prompt, _sep: prompt
    media_sync_module.fetch_worker_path_separator = AsyncMock(return_value=None)
    media_sync_module.sync_worker_media = AsyncMock()
    sys.modules[f"{package_name}.api.orchestration.media_sync"] = media_sync_module

    prompt_transform_module = types.ModuleType(f"{package_name}.api.orchestration.prompt_transform")

    class _PromptIndex:
        def __init__(self, prompt):
            self.prompt = prompt
            self.inputs_by_node = {node_id: node.get("inputs", {}) for node_id, node in prompt.items()}

        def nodes_for_class(self, class_type):
            return [
                node_id
                for node_id, node in self.prompt.items()
                if node.get("class_type") == class_type
            ]

        def copy_prompt(self):
            return {node_id: dict(node) for node_id, node in self.prompt.items()}

    prompt_transform_module.PromptIndex = _PromptIndex
    prompt_transform_module.apply_participant_overrides = lambda prompt, *_args, **_kwargs: prompt
    prompt_transform_module.find_nodes_by_class = lambda prompt, class_type: [
        node_id for node_id, node in prompt.items() if node.get("class_type") == class_type
    ]
    prompt_transform_module.generate_job_id_map = lambda _prompt_index, _prefix: {"10": "job-10"}
    prompt_transform_module.prepare_delegate_master_prompt = lambda prompt, _collector_ids: prompt
    prompt_transform_module.prune_prompt_for_worker = lambda prompt: prompt
    sys.modules[f"{package_name}.api.orchestration.prompt_transform"] = prompt_transform_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api.queue_orchestration",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


queue_orchestration = _load_queue_orchestration_module()


class LoadBalanceQueueingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        queue_orchestration.ensure_distributed_state()
        queue_orchestration.prompt_server.distributed_worker_reservations.clear()
        queue_orchestration.prompt_server.distributed_pending_jobs.clear()

    async def test_load_balance_reservation_selection_refreshes_reservations_between_polls(self):
        queue_orchestration.prompt_server.distributed_worker_reservations["w1"] = 1
        calls = []

        async def fake_select(candidates, **_kwargs):
            reserved_slots = candidates[0].get("reserved_slots")
            calls.append(reserved_slots)
            if reserved_slots:
                await queue_orchestration._release_worker_reservation("w1")
                return None
            return candidates[0]

        with patch.object(queue_orchestration, "select_least_busy_worker", side_effect=fake_select):
            selected = await queue_orchestration._select_and_reserve_load_balance_worker(
                [{"id": "w1", "name": "Worker 1"}],
                "exec-test",
                worker_probe_concurrency=1,
                idle_poll_interval=0,
            )

        self.assertEqual(selected["id"], "w1")
        self.assertEqual(calls, [1, 0])
        self.assertEqual(queue_orchestration.prompt_server.distributed_worker_reservations, {"w1": 1})

    async def test_load_balance_does_not_fallback_to_first_candidate_when_idle_selection_fails(self):
        prompt = {
            "10": {
                "class_type": "DistributedCollector",
                "inputs": {"load_balance": True},
            }
        }

        with patch.object(
            queue_orchestration,
            "_select_and_reserve_load_balance_worker",
            new=AsyncMock(return_value=None),
        ), patch.object(
            queue_orchestration,
            "dispatch_worker_prompt",
            new=AsyncMock(),
        ) as dispatch_mock:
            prompt_id, prompt_number, worker_count, node_errors = await queue_orchestration.orchestrate_distributed_execution(
                prompt,
                workflow_meta={},
                client_id="client-1",
                trace_execution_id="exec-test",
            )

        self.assertEqual((prompt_id, prompt_number, worker_count, node_errors), ("prompt-master", 1, 0, {}))
        dispatch_mock.assert_not_awaited()

    async def test_load_balance_releases_worker_reservation_when_dispatch_fails(self):
        prompt = {
            "10": {
                "class_type": "DistributedCollector",
                "inputs": {"load_balance": True},
            }
        }

        async def select_first_candidate(candidates, **_kwargs):
            return candidates[0]

        with patch.object(
            queue_orchestration,
            "select_least_busy_worker",
            side_effect=select_first_candidate,
        ), patch.object(
            queue_orchestration,
            "dispatch_worker_prompt",
            new=AsyncMock(side_effect=RuntimeError("dispatch failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                await queue_orchestration.orchestrate_distributed_execution(
                    prompt,
                    workflow_meta={},
                    client_id="client-1",
                    trace_execution_id="exec-test",
                )

        self.assertEqual(queue_orchestration.prompt_server.distributed_worker_reservations, {})


if __name__ == "__main__":
    unittest.main()
