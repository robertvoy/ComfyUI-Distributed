import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


def _load_distributed_module(is_worker):
    module_path = Path(__file__).resolve().parents[3] / "distributed.py"
    package_name = "dist_distributed_entry_testpkg"
    calls = {
        "ensure_config_exists": 0,
        "ensure_distributed_state": 0,
        "ensure_tile_jobs_initialized": 0,
        "delayed_auto_launch": 0,
        "register_async_signals": 0,
        "sync_cleanup_registered": 0,
    }

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    # Absolute dependency used by module.
    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace())
    sys.modules["server"] = server_module

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    config_module = types.ModuleType(f"{package_name}.utils.config")
    config_module.ensure_config_exists = lambda: calls.__setitem__("ensure_config_exists", calls["ensure_config_exists"] + 1)
    sys.modules[f"{package_name}.utils.config"] = config_module

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")

    async def _cleanup_client_session():
        return None

    network_module.cleanup_client_session = _cleanup_client_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    workers_pkg = types.ModuleType(f"{package_name}.workers")
    workers_pkg.__path__ = []
    workers_pkg.get_worker_manager = lambda: None
    sys.modules[f"{package_name}.workers"] = workers_pkg

    startup_module = types.ModuleType(f"{package_name}.workers.startup")
    startup_module.delayed_auto_launch = lambda: calls.__setitem__("delayed_auto_launch", calls["delayed_auto_launch"] + 1)
    startup_module.register_async_signals = lambda: calls.__setitem__("register_async_signals", calls["register_async_signals"] + 1)
    startup_module.sync_cleanup = lambda: None
    sys.modules[f"{package_name}.workers.startup"] = startup_module

    upscale_pkg = types.ModuleType(f"{package_name}.upscale")
    upscale_pkg.__path__ = []
    sys.modules[f"{package_name}.upscale"] = upscale_pkg

    job_store_module = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store_module.ensure_tile_jobs_initialized = lambda: calls.__setitem__(
        "ensure_tile_jobs_initialized", calls["ensure_tile_jobs_initialized"] + 1
    )
    sys.modules[f"{package_name}.upscale.job_store"] = job_store_module

    nodes_module = types.ModuleType(f"{package_name}.nodes")
    nodes_module.NODE_CLASS_MAPPINGS = {}
    nodes_module.NODE_DISPLAY_NAME_MAPPINGS = {}
    nodes_module.ImageBatchDivider = object
    nodes_module.DistributedCollectorNode = object
    nodes_module.DistributedSeed = object
    nodes_module.DistributedModelName = object
    nodes_module.DistributedValue = object
    nodes_module.AudioBatchDivider = object
    nodes_module.DistributedEmptyImage = object
    nodes_module.DistributedListSplitter = object
    nodes_module.DistributedListCollector = object
    nodes_module.DistributedBranch = object
    nodes_module.DistributedBranchCollector = object
    nodes_module.AnyType = object
    nodes_module.ByPassTypeTuple = tuple
    nodes_module.any_type = "*"
    sys.modules[f"{package_name}.nodes"] = nodes_module

    api_module = types.ModuleType(f"{package_name}.api")
    sys.modules[f"{package_name}.api"] = api_module

    queue_orch_module = types.ModuleType(f"{package_name}.api.queue_orchestration")
    queue_orch_module.ensure_distributed_state = lambda _ps: calls.__setitem__(
        "ensure_distributed_state", calls["ensure_distributed_state"] + 1
    )
    sys.modules[f"{package_name}.api.queue_orchestration"] = queue_orch_module

    import atexit

    original_register = atexit.register
    atexit.register = lambda fn: calls.__setitem__("sync_cleanup_registered", calls["sync_cleanup_registered"] + 1) or fn

    original_env = os.environ.get("COMFYUI_IS_WORKER")
    try:
        if is_worker:
            os.environ["COMFYUI_IS_WORKER"] = "1"
        else:
            os.environ.pop("COMFYUI_IS_WORKER", None)

        spec = importlib.util.spec_from_file_location(f"{package_name}.distributed", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        atexit.register = original_register
        if original_env is None:
            os.environ.pop("COMFYUI_IS_WORKER", None)
        else:
            os.environ["COMFYUI_IS_WORKER"] = original_env

    return calls


class DistributedEntryTests(unittest.TestCase):
    def test_import_initializes_core_state_for_worker(self):
        calls = _load_distributed_module(is_worker=True)

        self.assertEqual(calls["ensure_config_exists"], 1)
        self.assertEqual(calls["ensure_distributed_state"], 1)
        self.assertEqual(calls["ensure_tile_jobs_initialized"], 1)
        self.assertEqual(calls["delayed_auto_launch"], 0)
        self.assertEqual(calls["register_async_signals"], 0)

    def test_import_triggers_master_startup_hooks(self):
        calls = _load_distributed_module(is_worker=False)

        self.assertEqual(calls["ensure_config_exists"], 1)
        self.assertEqual(calls["ensure_distributed_state"], 1)
        self.assertEqual(calls["ensure_tile_jobs_initialized"], 1)
        self.assertEqual(calls["delayed_auto_launch"], 1)
        self.assertEqual(calls["register_async_signals"], 1)
        self.assertGreaterEqual(calls["sync_cleanup_registered"], 2)


if __name__ == "__main__":
    unittest.main()
