import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_ltx_module():
    module_path = Path(__file__).resolve().parents[1] / "nodes" / "ltx_tiled_sampler.py"
    package_name = "dist_ltx_testpkg"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    for subpkg in ["nodes", "upscale", "utils"]:
        pkg = types.ModuleType(f"{package_name}.{subpkg}")
        pkg.__path__ = []
        sys.modules[f"{package_name}.{subpkg}"] = pkg

    job_models = types.ModuleType(f"{package_name}.upscale.job_models")

    class TileJobState:
        pass

    job_models.TileJobState = TileJobState
    sys.modules[f"{package_name}.upscale.job_models"] = job_models

    async def _noop(*_args, **_kwargs):
        return None

    job_store = types.ModuleType(f"{package_name}.upscale.job_store")
    job_store._cleanup_job = _noop
    job_store._drain_results_queue = _noop
    job_store._get_completed_count = _noop
    job_store._mark_task_completed = _noop
    job_store.ensure_tile_jobs_initialized = lambda: types.SimpleNamespace(
        distributed_tile_jobs_lock=asyncio.Lock(),
        distributed_pending_tile_jobs={},
    )
    job_store.init_static_job_batched = _noop
    sys.modules[f"{package_name}.upscale.job_store"] = job_store

    job_timeout = types.ModuleType(f"{package_name}.upscale.job_timeout")
    job_timeout._check_and_requeue_timed_out_workers = _noop
    sys.modules[f"{package_name}.upscale.job_timeout"] = job_timeout

    async_helpers = types.ModuleType(f"{package_name}.utils.async_helpers")
    async_helpers.run_async_in_server_loop = lambda coro, timeout=None: asyncio.run(coro)
    sys.modules[f"{package_name}.utils.async_helpers"] = async_helpers

    constants = types.ModuleType(f"{package_name}.utils.constants")
    constants.HEARTBEAT_INTERVAL = 10.0
    constants.TILE_WAIT_TIMEOUT = 1.0
    sys.modules[f"{package_name}.utils.constants"] = constants

    logging = types.ModuleType(f"{package_name}.utils.logging")
    logging.debug_log = lambda *_args, **_kwargs: None
    logging.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging

    network = types.ModuleType(f"{package_name}.utils.network")

    async def _missing_session():
        raise RuntimeError("network not available in unit test")

    network.get_client_session = _missing_session
    sys.modules[f"{package_name}.utils.network"] = network

    usdu = types.ModuleType(f"{package_name}.utils.usdu_managment")
    usdu._send_heartbeat_to_master = _noop
    sys.modules[f"{package_name}.utils.usdu_managment"] = usdu

    spec = importlib.util.spec_from_file_location(f"{package_name}.nodes.ltx_tiled_sampler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


ltx = _load_ltx_module()


class LtxTiledSamplerTests(unittest.TestCase):
    def test_compute_tile_starts_with_overlap_covers_edges(self):
        starts, tile_size = ltx._compute_tile_starts(total_size=10, n_tiles=3, overlap=2)

        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + tile_size, 10)
        self.assertGreaterEqual(tile_size, 4)

    def test_pack_unpack_tensors_round_trips_with_safetensors(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 2, 2)

        payload = ltx._pack_tensors({"samples": tensor})
        restored = ltx._unpack_tensors(payload, device="cpu")

        self.assertTrue(torch.equal(restored["samples"], tensor))

    def test_latent_dict_rebuilder_preserves_metadata(self):
        samples = torch.zeros((1, 2, 3, 4, 5))
        latent = {"samples": samples, "noise_mask": "keep-me"}

        extracted, rebuild = ltx._extract_tensor_and_rebuilder(latent)
        rebuilt = rebuild(torch.ones_like(extracted))

        self.assertIs(extracted, samples)
        self.assertEqual(rebuilt["noise_mask"], "keep-me")
        self.assertTrue(torch.equal(rebuilt["samples"], torch.ones_like(samples)))

    def test_node_schema_includes_distributed_hidden_inputs(self):
        schema = ltx.LTXTiledSamplerDistributed.INPUT_TYPES()

        self.assertIn("multi_job_id", schema["hidden"])
        self.assertIn("is_worker", schema["hidden"])
        self.assertIn("master_url", schema["hidden"])
        self.assertIn("enabled_worker_ids", schema["hidden"])
        self.assertIn("LTXTiledSamplerDistributed", ltx.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
