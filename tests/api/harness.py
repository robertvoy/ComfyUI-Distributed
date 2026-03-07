"""Shared stubs/utilities for API route module unit tests."""
from __future__ import annotations

import sys
import types
from typing import Any, Callable


class RoutesStub:
    """Simple decorator-style route registrar used in unit tests."""

    def get(self, _path: str):
        return lambda fn: fn

    def post(self, _path: str):
        return lambda fn: fn


class ClientTimeoutStub:
    def __init__(self, total: float | None = None):
        self.total = total


class WSMsgTypeStub:
    TEXT = "TEXT"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


class WebSocketResponseStub:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    async def prepare(self, _request: Any) -> None:
        return None

    async def send_json(self, _payload: Any) -> None:
        return None

    def __aiter__(self):
        async def _empty():
            if False:
                yield None

        return _empty()


class FormDataStub:
    def add_field(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def reset_package_namespace(package_name: str) -> None:
    """Clear test package modules from sys.modules."""
    prefix = f"{package_name}."
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(prefix):
            del sys.modules[mod_name]


def ensure_namespace_package(module_name: str) -> types.ModuleType:
    """Create a namespace-like package module in sys.modules."""
    module = types.ModuleType(module_name)
    module.__path__ = []
    sys.modules[module_name] = module
    return module


def bootstrap_test_package(
    package_name: str,
    *,
    with_api: bool = True,
    with_utils: bool = True,
    with_workers: bool = False,
    with_upscale: bool = False,
    with_orchestration: bool = False,
) -> None:
    """Create a clean package scaffold for module-level import tests."""
    reset_package_namespace(package_name)
    ensure_namespace_package(package_name)

    if with_api:
        ensure_namespace_package(f"{package_name}.api")
    if with_utils:
        ensure_namespace_package(f"{package_name}.utils")
    if with_workers:
        ensure_namespace_package(f"{package_name}.workers")
    if with_upscale:
        ensure_namespace_package(f"{package_name}.upscale")
    if with_orchestration:
        ensure_namespace_package(f"{package_name}.api.orchestration")


def install_server_stub(prompt_server_instance: Any | None = None) -> Any:
    """Install a minimal `server` module exposing PromptServer.instance.routes."""
    if prompt_server_instance is None:
        prompt_server_instance = types.SimpleNamespace()
    if not hasattr(prompt_server_instance, "routes"):
        prompt_server_instance.routes = RoutesStub()

    server_module = types.ModuleType("server")
    server_module.PromptServer = types.SimpleNamespace(instance=prompt_server_instance)
    sys.modules["server"] = server_module
    return prompt_server_instance


def install_aiohttp_stub(
    json_response_factory: Callable[..., Any],
) -> bool:
    """Install/augment `aiohttp` with common members used by API tests."""
    created = False
    if "aiohttp" not in sys.modules:
        created = True
        aiohttp_module = types.ModuleType("aiohttp")
        sys.modules["aiohttp"] = aiohttp_module

    aiohttp_module = sys.modules["aiohttp"]
    if not hasattr(aiohttp_module, "ClientTimeout"):
        aiohttp_module.ClientTimeout = ClientTimeoutStub
    if not hasattr(aiohttp_module, "WSMsgType"):
        aiohttp_module.WSMsgType = WSMsgTypeStub
    if not hasattr(aiohttp_module, "FormData"):
        aiohttp_module.FormData = FormDataStub

    web_obj = getattr(aiohttp_module, "web", None)
    if web_obj is None:
        web_obj = types.SimpleNamespace()
        aiohttp_module.web = web_obj
    web_obj.json_response = json_response_factory
    if not hasattr(web_obj, "WebSocketResponse"):
        web_obj.WebSocketResponse = WebSocketResponseStub

    return created


def cleanup_optional_module(module_name: str, created: bool) -> None:
    """Remove stubbed module only when this test created it."""
    if created:
        sys.modules.pop(module_name, None)


def install_request_guards_stub(package_name: str) -> None:
    """Install permissive request guard for isolated route tests."""
    request_guards_module = types.ModuleType(f"{package_name}.api.request_guards")

    async def _authorization_error_or_none(_request):
        return None

    request_guards_module.authorization_error_or_none = _authorization_error_or_none
    sys.modules[f"{package_name}.api.request_guards"] = request_guards_module


def install_endpoint_policy_passthrough_stub(package_name: str) -> None:
    """Install endpoint policy stub that executes operation directly."""
    endpoint_policy_module = types.ModuleType(f"{package_name}.api.endpoint_policy")

    async def _run_authorized_endpoint(_request, operation, unexpected_status=500):
        _ = unexpected_status
        return await operation()

    endpoint_policy_module.run_authorized_endpoint = _run_authorized_endpoint
    sys.modules[f"{package_name}.api.endpoint_policy"] = endpoint_policy_module
