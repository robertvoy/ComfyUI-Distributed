import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _bootstrap_package(package_name):
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

    prompt_server_instance = types.SimpleNamespace(address="127.0.0.1", port=8188)
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=prompt_server_instance)
    sys.modules["server"] = server_module

    async_helpers_module = types.ModuleType(f"{package_name}.utils.async_helpers")

    async def _queue_prompt_payload(prompt_obj, workflow_meta, client_id):
        _ = (prompt_obj, workflow_meta, client_id)
        return "prompt-id"

    async_helpers_module.queue_prompt_payload = _queue_prompt_payload
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers_module

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.load_config = lambda: {"settings": {}, "workers": []}
    sys.modules[f"{package_name}.utils.config"] = config_module

    constants_module = types.ModuleType(f"{package_name}.utils.constants")
    constants_module.ORCHESTRATION_MEDIA_SYNC_CONCURRENCY = 2
    constants_module.ORCHESTRATION_MEDIA_SYNC_TIMEOUT = 120.0
    constants_module.ORCHESTRATION_WORKER_PROBE_CONCURRENCY = 8
    constants_module.ORCHESTRATION_WORKER_PREP_CONCURRENCY = 4
    sys.modules[f"{package_name}.utils.constants"] = constants_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_master_url = lambda **_kwargs: "http://127.0.0.1:8188"
    sys.modules[f"{package_name}.utils.network"] = network_module

    trace_logger_module = types.ModuleType(f"{package_name}.utils.trace_logger")
    trace_logger_module.trace_debug = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.trace_logger"] = trace_logger_module

    schemas_module = types.ModuleType(f"{package_name}.api.schemas")
    schemas_module.parse_positive_int = lambda value, default: int(value) if str(value).isdigit() and int(value) > 0 else default
    schemas_module.parse_positive_float = (
        lambda value, default: float(value) if isinstance(value, (int, float, str)) and float(value) > 0 else default
    )
    sys.modules[f"{package_name}.api.schemas"] = schemas_module

    dispatch_module = types.ModuleType(f"{package_name}.api.orchestration.dispatch")
    dispatch_module.dispatch_worker_prompt = lambda *args, **kwargs: None
    dispatch_module.rank_workers_by_load = lambda workers, **kwargs: workers
    dispatch_module.select_active_workers = lambda workers, *_args, **_kwargs: (workers, False)
    dispatch_module.select_least_busy_worker = lambda workers, **_kwargs: workers[0] if workers else None
    sys.modules[f"{package_name}.api.orchestration.dispatch"] = dispatch_module

    media_sync_module = types.ModuleType(f"{package_name}.api.orchestration.media_sync")
    media_sync_module.convert_paths_for_platform = lambda prompt, _sep: prompt
    media_sync_module.fetch_worker_path_separator = lambda *_args, **_kwargs: "/"
    media_sync_module.sync_worker_media = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.api.orchestration.media_sync"] = media_sync_module

    prompt_transform_module = types.ModuleType(f"{package_name}.api.orchestration.prompt_transform")
    prompt_transform_module.PromptIndex = object
    prompt_transform_module.apply_participant_overrides = lambda prompt, *_args, **_kwargs: prompt
    prompt_transform_module.find_nodes_by_class = lambda _prompt, _class_name: []
    prompt_transform_module.generate_job_id_map = lambda _index, _prefix: {}
    prompt_transform_module.prepare_delegate_master_prompt = lambda prompt, _ids: prompt
    prompt_transform_module.prune_prompt_for_worker = lambda prompt: prompt
    sys.modules[f"{package_name}.api.orchestration.prompt_transform"] = prompt_transform_module


def _load_module(package_name, module_rel_path, module_name):
    module_path = Path(__file__).resolve().parents[2] / module_rel_path
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_queue_orchestration_module():
    package_name = "dist_queue_orch_testpkg"
    _bootstrap_package(package_name)
    return _load_module(package_name, "api/queue_orchestration.py", "api.queue_orchestration")


queue_orchestration = _load_queue_orchestration_module()


class QueueOrchestrationHelpersTests(unittest.TestCase):
    def test_resolve_enabled_workers_filters_disabled_and_requested(self):
        config = {
            "workers": [
                {"id": "1", "enabled": True, "host": "a", "port": 8188},
                {"id": "2", "enabled": False, "host": "b", "port": 8189},
                {"id": "3", "enabled": True, "host": "c", "port": "bad-port"},
            ]
        }

        enabled_only = queue_orchestration._resolve_enabled_workers(config)
        requested = queue_orchestration._resolve_enabled_workers(config, requested_ids={"2", "3"})

        self.assertEqual([worker["id"] for worker in enabled_only], ["1", "3"])
        self.assertEqual([worker["id"] for worker in requested], ["2", "3"])
        self.assertEqual(requested[1]["port"], 8188)

    def test_resolve_orchestration_limits_uses_defaults_for_invalid_values(self):
        config = {
            "settings": {
                "worker_probe_concurrency": "x",
                "worker_prep_concurrency": "3",
                "media_sync_concurrency": "-1",
                "media_sync_timeout_seconds": "42.5",
            }
        }

        limits = queue_orchestration._resolve_orchestration_limits(config)

        self.assertEqual(limits[0], 8)   # default
        self.assertEqual(limits[1], 3)   # parsed
        self.assertEqual(limits[2], 2)   # default
        self.assertEqual(limits[3], 42.5)

    def test_is_load_balance_enabled_accepts_common_forms(self):
        self.assertTrue(queue_orchestration._is_load_balance_enabled(True))
        self.assertTrue(queue_orchestration._is_load_balance_enabled(1))
        self.assertTrue(queue_orchestration._is_load_balance_enabled(" yes "))
        self.assertFalse(queue_orchestration._is_load_balance_enabled(0))
        self.assertFalse(queue_orchestration._is_load_balance_enabled("off"))

    def test_prompt_requests_load_balance_detects_collector_flag(self):
        prompt_index = types.SimpleNamespace(
            nodes_for_class=lambda class_name: ["10"] if class_name == "DistributedCollector" else [],
            inputs_by_node={"10": {"load_balance": "true"}},
        )
        self.assertTrue(queue_orchestration._prompt_requests_load_balance(prompt_index))

    def test_ensure_distributed_state_initializes_attrs(self):
        prompt_server = types.SimpleNamespace()
        queue_orchestration.ensure_distributed_state(prompt_server)
        self.assertTrue(hasattr(prompt_server, "distributed_pending_jobs"))
        self.assertTrue(hasattr(prompt_server, "distributed_jobs_lock"))
        self.assertIsInstance(prompt_server.distributed_jobs_lock, asyncio.Lock)


if __name__ == "__main__":
    unittest.main()
