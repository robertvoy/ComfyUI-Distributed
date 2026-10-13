import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


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

        class _ClientError(Exception):
            pass

        class _ClientConnectorError(_ClientError):
            pass

        class _ClientConnectionError(_ClientError):
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
        setattr(aiohttp_module, "ClientError", _ClientError)
        aiohttp_module.ClientConnectorError = _ClientConnectorError
        setattr(aiohttp_module, "ClientConnectionError", _ClientConnectionError)
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


def _load_prompt_transform_module():
    module_path = Path(__file__).resolve().parents[1] / "api" / "orchestration" / "prompt_transform.py"
    package_name = "prompt_transform_worker_selection_testpkg"
    for suffix in ("", ".api", ".api.orchestration", ".utils"):
        package = types.ModuleType(package_name + suffix)
        package.__path__ = []
        sys.modules[package_name + suffix] = package
    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    setattr(logging_module, "debug_log", lambda *_args, **_kwargs: None)
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.api.orchestration.prompt_transform",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompt_transform = _load_prompt_transform_module()


class _FakeResponse:
    def __init__(self, status=200, payload=None, error=None):
        self.status = status
        self.payload = payload
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, timeout=None):
        del timeout
        worker_id = url.removeprefix("http://").split("/", 1)[0]
        endpoint = "capabilities" if url.endswith("/capabilities") else "queue"
        response = self.responses[(worker_id, endpoint)]
        if isinstance(response, BaseException):
            return _FakeResponse(error=response)
        status, payload = response
        return _FakeResponse(status=status, payload=payload)


def _valid_capability():
    return {
        "supported": True,
        "execution_targets": True,
        "protocol_version": 2,
        "node_ids": ["DistributedImageListStart", "DistributedImageListEnd"],
    }


def _valid_queue(running=None, pending=None):
    return {"queue_running": running or [], "queue_pending": pending or []}


class ExplicitTargetTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_preserves_targets_and_omits_default_field(self):
        response = types.SimpleNamespace(raise_for_status=lambda: None)
        class Response:
            async def __aenter__(self): return response
            async def __aexit__(self, *args): pass
        calls = []
        def post(url, **kwargs):
            calls.append(kwargs['json'])
            return Response()
        session = types.SimpleNamespace(post=post)
        with patch.object(dispatch, 'get_client_session', AsyncMock(return_value=session)):
            await dispatch.dispatch_worker_prompt({'id': 'worker'}, {'9': {}}, None,
                                                  partial_execution_targets=['9'])
            await dispatch.dispatch_worker_prompt({'id': 'worker'}, {'9': {}}, None)
        self.assertEqual(calls[0]['partial_execution_targets'], ['9'])
        self.assertNotIn('partial_execution_targets', calls[1])

    async def test_websocket_sends_targets_all_the_way_to_wire(self):
        import json
        sent = []
        async def receive():
            return types.SimpleNamespace(type=dispatch.aiohttp.WSMsgType.TEXT, data=json.dumps({
                'type': 'dispatch_ack', 'request_id': sent[-1]['request_id'],
                'ok': True, 'prompt_id': 'worker-prompt'}))
        ws = types.SimpleNamespace(send_json=AsyncMock(side_effect=sent.append),
                                   receive=receive, close=AsyncMock())
        session = types.SimpleNamespace(ws_connect=AsyncMock(return_value=ws))
        with patch.object(dispatch, 'get_client_session', AsyncMock(return_value=session)):
            result = await dispatch.dispatch_worker_prompt({'id': 'worker'}, {'9': {}}, None,
                use_websocket=True, return_prompt_id=True, partial_execution_targets=['9'])
            await dispatch.dispatch_worker_prompt({'id': 'worker'}, {'9': {}}, None, use_websocket=True)
        self.assertEqual(result, 'worker-prompt')
        self.assertEqual(sent[0]['partial_execution_targets'], ['9'])
        self.assertNotIn('partial_execution_targets', sent[1])


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

    async def test_image_list_preflight_skips_unreachable_worker_and_keeps_healthy_worker(self):
        workers = [{"id": "healthy"}, {"id": "offline"}]
        responses = {
            (worker_id, "capabilities"): (200, _valid_capability())
            for worker_id in ("healthy", "offline")
        }
        responses.update({
            (worker_id, "queue"): (200, _valid_queue())
            for worker_id in ("healthy", "offline")
        })
        responses[("offline", "capabilities")] = dispatch.aiohttp.ClientConnectionError("connection refused")
        session = _FakeSession(responses)

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ), patch.object(dispatch, "log") as log_mock:
            result = await dispatch.preflight_image_list_workers(workers)

        self.assertEqual(result, {"healthy": True})
        self.assertTrue(any("offline" in str(call) for call in log_mock.call_args_list))

    async def test_image_list_preflight_skips_timeout_and_keeps_healthy_worker(self):
        workers = [{"id": "healthy"}, {"id": "slow"}]
        responses: dict = {
            (worker_id, "capabilities"): (200, _valid_capability())
            for worker_id in ("healthy", "slow")
        }
        responses.update({
            (worker_id, "queue"): (200, _valid_queue())
            for worker_id in ("healthy", "slow")
        })
        responses[("slow", "queue")] = asyncio.TimeoutError()
        session = _FakeSession(responses)

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ), patch.object(dispatch, "log") as log_mock:
            result = await dispatch.preflight_image_list_workers(workers)

        self.assertEqual(result, {"healthy": True})
        self.assertTrue(any("slow" in str(call) for call in log_mock.call_args_list))

    async def test_image_list_preflight_excludes_busy_workers(self):
        workers = [{"id": "idle"}, {"id": "busy"}]
        responses = {
            (worker_id, "capabilities"): (200, _valid_capability())
            for worker_id in ("idle", "busy")
        }
        responses.update({
            ("idle", "queue"): (200, _valid_queue()),
            ("busy", "queue"): (200, _valid_queue(running=[{"prompt": "existing"}])),
        })
        session = _FakeSession(responses)

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            result = await dispatch.preflight_image_list_workers(workers)

        self.assertEqual(result, {"idle": True})

    async def test_image_list_preflight_reports_no_idle_workers_when_all_are_unavailable(self):
        workers = [{"id": "offline-a"}, {"id": "offline-b"}]
        responses = {
            (worker_id, "capabilities"): dispatch.aiohttp.ClientConnectionError("unreachable")
            for worker_id in ("offline-a", "offline-b")
        }
        session = _FakeSession(responses)

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            self.assertEqual(
                await dispatch.preflight_image_list_workers(workers, require_idle=False),
                {},
            )
            with self.assertRaisesRegex(RuntimeError, "No selected feature-compatible direct worker is idle"):
                await dispatch.preflight_image_list_workers(workers, require_idle=True)

    async def test_image_list_preflight_requires_exact_execution_targets_support(self):
        for value in (None, False, 1, "true", []):
            with self.subTest(value=value):
                capability = _valid_capability()
                if value is None:
                    capability.pop("execution_targets")
                else:
                    capability["execution_targets"] = value
                session = _FakeSession({("worker", "capabilities"): (200, capability)})
                async def fake_session():
                    return session
                with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
                    dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
                ):
                    with self.assertRaisesRegex(RuntimeError, "execution_targets.*upgrade"):
                        await dispatch.preflight_image_list_workers([{"id": "worker"}])

    async def test_image_list_preflight_fails_closed_for_malformed_capabilities(self):
        malformed = {"supported": True, "protocol_version": 2, "node_ids": None}
        session = _FakeSession({("worker", "capabilities"): (200, malformed)})

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            with self.assertRaisesRegex(RuntimeError, "malformed.*capabilit"):
                await dispatch.preflight_image_list_workers([{"id": "worker"}])

    async def test_image_list_preflight_fails_closed_for_incompatible_protocol(self):
        incompatible = _valid_capability()
        incompatible["protocol_version"] = 1
        session = _FakeSession({("worker", "capabilities"): (200, incompatible)})

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                await dispatch.preflight_image_list_workers([{"id": "worker"}])

    async def test_image_list_preflight_fails_closed_for_missing_required_node(self):
        incompatible = _valid_capability()
        incompatible["node_ids"] = ["DistributedImageListStart"]
        session = _FakeSession({("worker", "capabilities"): (200, incompatible)})

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                await dispatch.preflight_image_list_workers([{"id": "worker"}])

    async def test_image_list_preflight_fails_closed_for_malformed_queue(self):
        responses = {
            ("worker", "capabilities"): (200, _valid_capability()),
            ("worker", "queue"): (200, {"queue_running": [], "queue_pending": None}),
        }
        session = _FakeSession(responses)

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            with self.assertRaisesRegex(RuntimeError, "malformed /queue"):
                await dispatch.preflight_image_list_workers([{"id": "worker"}])

    async def test_image_list_preflight_does_not_hide_unexpected_errors(self):
        session = _FakeSession({("worker", "capabilities"): ValueError("programming error")})

        async def fake_session():
            return session

        with patch.object(dispatch, "get_client_session", side_effect=fake_session), patch.object(
            dispatch, "build_worker_url", side_effect=lambda worker, path: f"http://{worker['id']}{path}"
        ):
            with self.assertRaisesRegex(ValueError, "programming error"):
                await dispatch.preflight_image_list_workers([{"id": "worker"}])


class ImageListDirectWorkerResolutionTests(unittest.TestCase):
    def setUp(self):
        self.config = {"workers": [
            {"id": "enabled", "enabled": True, "host": "enabled-host", "port": 8188},
            {"id": "disabled", "enabled": False, "host": "disabled-host", "port": 8189},
            {"id": "serverless", "enabled": True, "type": "serverless"},
            {"id": "other", "enabled": True, "host": "other-host", "port": 8190},
        ]}

    def test_implicit_selection_uses_only_enabled_direct_workers(self):
        workers = prompt_transform._resolve_direct_workers_for_image_list(self.config, None)
        self.assertEqual([worker["id"] for worker in workers], ["enabled", "other"])

    def test_explicit_selection_includes_exact_disabled_direct_worker(self):
        workers = prompt_transform._resolve_direct_workers_for_image_list(self.config, ["disabled"])
        self.assertEqual([worker["id"] for worker in workers], ["disabled"])

    def test_explicit_empty_selection_returns_no_workers(self):
        self.assertEqual(prompt_transform._resolve_direct_workers_for_image_list(self.config, []), [])

    def test_unknown_explicit_id_raises_instead_of_falling_back(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            prompt_transform._resolve_direct_workers_for_image_list(self.config, ["enabled", "missing"])

    def test_explicit_serverless_selection_is_excluded(self):
        self.assertEqual(prompt_transform._resolve_direct_workers_for_image_list(self.config, ["serverless"]), [])


if __name__ == "__main__":
    unittest.main()
