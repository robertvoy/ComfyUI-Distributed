import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_dispatch_module():
    module_path = Path(__file__).resolve().parents[1] / "api" / "orchestration" / "dispatch.py"

    package_name = "dist_dispatch_testpkg"
    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    api_pkg = types.ModuleType(f"{package_name}.api")
    api_pkg.__path__ = []
    sys.modules[f"{package_name}.api"] = api_pkg

    orch_pkg = types.ModuleType(f"{package_name}.api.orchestration")
    orch_pkg.__path__ = []
    sys.modules[f"{package_name}.api.orchestration"] = orch_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    network_module.build_worker_url = lambda *_args, **_kwargs: "http://example.invalid"

    async def _probe_worker(*_args, **_kwargs):
        return None

    network_module.probe_worker = _probe_worker

    async def _fake_session():
        raise RuntimeError("get_client_session should be mocked in these tests")

    network_module.get_client_session = _fake_session
    sys.modules[f"{package_name}.utils.network"] = network_module

    created_aiohttp_stub = False
    if "aiohttp" not in sys.modules:
        created_aiohttp_stub = True
        aiohttp_module = types.ModuleType("aiohttp")

        class _ClientTimeout:
            def __init__(self, total=None):
                self.total = total

        class _ClientConnectorError(Exception):
            pass

        class _WSMsgType:
            TEXT = "TEXT"
            ERROR = "ERROR"
            CLOSED = "CLOSED"

        class _TCPConnector:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class _ClientSession:
            def __init__(self, *args, **kwargs):
                self.closed = False

            async def close(self):
                self.closed = True

        aiohttp_module.ClientTimeout = _ClientTimeout
        aiohttp_module.ClientConnectorError = _ClientConnectorError
        aiohttp_module.WSMsgType = _WSMsgType
        aiohttp_module.TCPConnector = _TCPConnector
        aiohttp_module.ClientSession = _ClientSession
        aiohttp_module.web = types.SimpleNamespace(
            json_response=lambda payload, status=200: {"payload": payload, "status": status}
        )
        sys.modules["aiohttp"] = aiohttp_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api.orchestration.dispatch",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    if created_aiohttp_stub:
        sys.modules.pop("aiohttp", None)
    return module


dispatch = _load_dispatch_module()


class DispatchSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_select_active_workers_filters_offline(self):
        workers = [
            {"id": "w1", "name": "Worker 1"},
            {"id": "w2", "name": "Worker 2"},
            {"id": "w3", "name": "Worker 3"},
        ]

        async def fake_probe(worker):
            return worker["id"] != "w2"

        with patch.object(dispatch, "worker_is_active", side_effect=fake_probe):
            active_workers, delegate_master = await dispatch.select_active_workers(
                workers,
                use_websocket=False,
                delegate_master=False,
                probe_concurrency=3,
            )

        self.assertEqual([w["id"] for w in active_workers], ["w1", "w3"])
        self.assertFalse(delegate_master)

    async def test_select_active_workers_disables_delegate_when_all_offline(self):
        workers = [{"id": "w1", "name": "Worker 1"}]

        async def fake_probe(_worker):
            return False

        with patch.object(dispatch, "worker_is_active", side_effect=fake_probe):
            active_workers, delegate_master = await dispatch.select_active_workers(
                workers,
                use_websocket=False,
                delegate_master=True,
                probe_concurrency=1,
            )

        self.assertEqual(active_workers, [])
        self.assertFalse(delegate_master)

    async def test_select_active_workers_uses_websocket_probe_when_enabled(self):
        workers = [{"id": "w1", "name": "Worker 1"}, {"id": "w2", "name": "Worker 2"}]

        async def fake_http_probe(_worker):
            return False

        async def fake_ws_probe(_worker):
            return True

        with patch.object(dispatch, "worker_is_active", side_effect=fake_http_probe) as http_probe, patch.object(
            dispatch,
            "worker_ws_is_active",
            side_effect=fake_ws_probe,
        ) as ws_probe:
            active_workers, _ = await dispatch.select_active_workers(
                workers,
                use_websocket=True,
                delegate_master=False,
                probe_concurrency=4,
            )

        self.assertEqual([w["id"] for w in active_workers], ["w1", "w2"])
        self.assertEqual(ws_probe.call_count, 2)
        self.assertEqual(http_probe.call_count, 0)

    async def test_probe_concurrency_is_bounded(self):
        workers = [{"id": f"w{i}", "name": f"Worker {i}"} for i in range(6)]
        state = {"in_flight": 0, "max_in_flight": 0}

        async def fake_probe(_worker):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1
            return True

        with patch.object(dispatch, "worker_is_active", side_effect=fake_probe):
            active_workers, _ = await dispatch.select_active_workers(
                workers,
                use_websocket=False,
                delegate_master=False,
                probe_concurrency=2,
            )

        self.assertEqual(len(active_workers), len(workers))
        self.assertLessEqual(state["max_in_flight"], 2)
        self.assertGreaterEqual(state["max_in_flight"], 2)

    async def test_select_least_busy_worker_round_robins_idle_workers(self):
        workers = [
            {"id": "w1", "name": "Worker 1"},
            {"id": "w2", "name": "Worker 2"},
            {"id": "w3", "name": "Worker 3"},
        ]
        queue_map = {"w1": 0, "w2": 0, "w3": 2}

        async def fake_probe(worker_url, timeout=3.0):
            worker_id = worker_url.rsplit("/", 1)[-1]
            return {"exec_info": {"queue_remaining": queue_map[worker_id]}}

        with patch.object(dispatch, "build_worker_url", side_effect=lambda worker: f"http://host/{worker['id']}"), patch.object(
            dispatch,
            "probe_worker",
            side_effect=fake_probe,
        ):
            dispatch._least_busy_rr_index = 0
            selected1 = await dispatch.select_least_busy_worker(workers, probe_concurrency=3)
            selected2 = await dispatch.select_least_busy_worker(workers, probe_concurrency=3)
            selected3 = await dispatch.select_least_busy_worker(workers, probe_concurrency=3)

        self.assertEqual(selected1["id"], "w1")
        self.assertEqual(selected2["id"], "w2")
        self.assertEqual(selected3["id"], "w1")

    async def test_select_least_busy_worker_chooses_smallest_queue_when_all_busy(self):
        workers = [
            {"id": "w1", "name": "Worker 1"},
            {"id": "w2", "name": "Worker 2"},
            {"id": "w3", "name": "Worker 3"},
        ]
        queue_map = {"w1": 5, "w2": 2, "w3": 4}

        async def fake_probe(worker_url, timeout=3.0):
            worker_id = worker_url.rsplit("/", 1)[-1]
            return {"exec_info": {"queue_remaining": queue_map[worker_id]}}

        with patch.object(dispatch, "build_worker_url", side_effect=lambda worker: f"http://host/{worker['id']}"), patch.object(
            dispatch,
            "probe_worker",
            side_effect=fake_probe,
        ):
            selected = await dispatch.select_least_busy_worker(workers, probe_concurrency=2)

        self.assertEqual(selected["id"], "w2")

    async def test_select_least_busy_worker_returns_none_when_all_probes_fail(self):
        workers = [{"id": "w1", "name": "Worker 1"}]

        async def fake_probe(_worker_url, timeout=3.0):
            return None

        with patch.object(dispatch, "build_worker_url", side_effect=lambda worker: f"http://host/{worker['id']}"), patch.object(
            dispatch,
            "probe_worker",
            side_effect=fake_probe,
        ):
            selected = await dispatch.select_least_busy_worker(workers, probe_concurrency=1)

        self.assertIsNone(selected)


    async def test_select_least_busy_worker_waits_for_idle_when_required(self):
        workers = [
            {"id": "w1", "name": "Worker 1"},
            {"id": "w2", "name": "Worker 2"},
        ]
        queue_sequences = {
            "w1": [1, 1],
            "w2": [1, 0],
        }

        async def fake_probe(worker_url, timeout=3.0):
            worker_id = worker_url.rsplit("/", 1)[-1]
            sequence = queue_sequences[worker_id]
            value = sequence.pop(0) if sequence else 0
            return {"exec_info": {"queue_remaining": value}}

        with patch.object(dispatch, "build_worker_url", side_effect=lambda worker: f"http://host/{worker['id']}"), patch.object(
            dispatch,
            "probe_worker",
            side_effect=fake_probe,
        ):
            dispatch._least_busy_rr_index = 0
            selected = await dispatch.select_least_busy_worker(
                workers,
                probe_concurrency=2,
                require_idle=True,
                idle_poll_interval=0,
            )

        self.assertEqual(selected["id"], "w2")


    async def test_select_least_busy_worker_counts_reserved_slots_as_busy(self):
        workers = [
            {"id": "w1", "name": "Worker 1", "reserved_slots": 1},
            {"id": "w2", "name": "Worker 2"},
        ]

        async def fake_probe(worker_url, timeout=3.0):
            return {"exec_info": {"queue_remaining": 0}}

        with patch.object(dispatch, "build_worker_url", side_effect=lambda worker: f"http://host/{worker['id']}"), patch.object(
            dispatch,
            "probe_worker",
            side_effect=fake_probe,
        ):
            dispatch._least_busy_rr_index = 0
            selected = await dispatch.select_least_busy_worker(
                workers,
                probe_concurrency=2,
                require_idle=True,
                idle_poll_interval=0,
                idle_wait_timeout=0,
            )

        self.assertEqual(selected["id"], "w2")


if __name__ == "__main__":
    unittest.main()
