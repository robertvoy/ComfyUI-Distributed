import asyncio
import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import aiohttp
from aiohttp import web


def _load_dispatch_module():
    module_path = Path(__file__).resolve().parents[1] / "api" / "orchestration" / "dispatch.py"
    package_name = f"image_list_ws_deadline_testpkg_{uuid.uuid4().hex}"
    package_modules = {}

    for suffix in ("", ".api", ".api.orchestration", ".utils"):
        package = types.ModuleType(package_name + suffix)
        package.__path__ = []
        package_modules[package_name + suffix] = package

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    setattr(logging_module, "debug_log", lambda *_args, **_kwargs: None)
    setattr(logging_module, "log", lambda *_args, **_kwargs: None)
    package_modules[logging_module.__name__] = logging_module

    network_module = types.ModuleType(f"{package_name}.utils.network")
    setattr(network_module, "build_worker_url", lambda *_args, **_kwargs: "http://example.invalid")

    async def _unused_probe_worker(*_args, **_kwargs):
        return None

    async def _unused_get_client_session(*_args, **_kwargs):
        raise RuntimeError("get_client_session must be patched by this test")

    setattr(network_module, "probe_worker", _unused_probe_worker)
    setattr(network_module, "get_client_session", _unused_get_client_session)
    package_modules[network_module.__name__] = network_module

    module_name = f"{package_name}.api.orchestration.dispatch"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    package_modules[module_name] = module

    previous_modules = {name: sys.modules.get(name) for name in package_modules}
    try:
        sys.modules.update(package_modules)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


dispatch = _load_dispatch_module()


class WebsocketDispatchDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = None
        self.received_payload = None
        self.dispatch_received = asyncio.Event()
        self.peer_finished = asyncio.Event()
        self.irrelevant_messages_sent = 0

        app = web.Application()
        app.router.add_get("/distributed/worker_ws", self._handle_websocket)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        server = cast(Any, self.site._server)
        port = server.sockets[0].getsockname()[1]
        self.worker_url = f"http://127.0.0.1:{port}"
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def _get_client_session(self):
        return self.session

    async def _handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            payload = await ws.receive_json()
            self.received_payload = payload
            self.dispatch_received.set()

            if self.mode in ("success", "slow-close"):
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": "unrelated-request",
                    "ok": True,
                    "prompt_id": "wrong-prompt",
                })
                await ws.send_json({"type": "heartbeat"})
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": payload["request_id"],
                    "ok": True,
                    "prompt_id": "prompt-123",
                })
                if self.mode == "slow-close":
                    await asyncio.sleep(0.15)
            elif self.mode == "rejection":
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": payload["request_id"],
                    "ok": False,
                    "error": "Invalid prompt",
                    "validation_error": {"node": "bad"},
                    "node_errors": {"bad": "missing input"},
                })
            elif self.mode in ("no-ack", "wait-for-cancel"):
                async def send_irrelevant_messages():
                    while True:
                        try:
                            await ws.send_json({"type": "heartbeat"})
                            self.irrelevant_messages_sent += 1
                            await ws.send_json({
                                "type": "dispatch_ack",
                                "request_id": "unrelated-request",
                                "ok": True,
                                "prompt_id": "not-this-request",
                            })
                            self.irrelevant_messages_sent += 1
                        except (ConnectionError, RuntimeError):
                            return
                        await asyncio.sleep(0.01)

                sender = asyncio.create_task(send_irrelevant_messages())
                try:
                    async for _message in ws:
                        pass
                finally:
                    sender.cancel()
                    try:
                        await sender
                    except asyncio.CancelledError:
                        pass
        finally:
            self.peer_finished.set()
        return ws

    async def _dispatch(self, timeout):
        with patch.object(dispatch, "get_client_session", self._get_client_session):
            return await dispatch._dispatch_via_websocket(
                self.worker_url,
                {"prompt": {"node": {"class_type": "Example"}}, "workflow": {"name": "test"}},
                "client-456",
                timeout=timeout,
            )

    async def test_returns_matching_acknowledgement_prompt_id(self):
        self.mode = "success"

        prompt_id = await self._dispatch(timeout=0.5)

        self.assertEqual(prompt_id, "prompt-123")
        received_payload = self.received_payload
        assert received_payload is not None
        self.assertEqual(received_payload["type"], "dispatch_prompt")
        self.assertTrue(received_payload["request_id"])
        self.assertEqual(received_payload["client_id"], "client-456")
        self.assertEqual(received_payload["prompt"], {"node": {"class_type": "Example"}})
        self.assertEqual(received_payload["workflow"], {"name": "test"})

    async def test_acknowledged_prompt_id_survives_slow_close(self):
        self.mode = "slow-close"
        self.assertEqual(await self._dispatch(timeout=0.03), "prompt-123")

    async def test_matching_rejection_preserves_worker_error_details(self):
        self.mode = "rejection"

        with self.assertRaisesRegex(RuntimeError, "Invalid prompt.*validation_error=.*node_errors="):
            await self._dispatch(timeout=0.5)

    async def test_repeated_irrelevant_messages_do_not_reset_ack_deadline(self):
        self.mode = "no-ack"
        started = asyncio.get_running_loop().time()

        with self.assertRaisesRegex(RuntimeError, "submission status is ambiguous"):
            await asyncio.wait_for(self._dispatch(timeout=0.12), timeout=0.5)

        elapsed = asyncio.get_running_loop().time() - started
        self.assertLess(elapsed, 0.5)
        self.assertGreaterEqual(self.irrelevant_messages_sent, 4)
        await asyncio.wait_for(self.peer_finished.wait(), timeout=0.5)

    async def test_outer_cancellation_propagates_and_closes_peer(self):
        self.mode = "wait-for-cancel"
        task = asyncio.create_task(self._dispatch(timeout=2.0))
        await asyncio.wait_for(self.dispatch_received.wait(), timeout=0.5)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        await asyncio.wait_for(self.peer_finished.wait(), timeout=0.5)


if __name__ == "__main__":
    unittest.main()
