from __future__ import annotations

"""
Network and API utilities for ComfyUI-Distributed.
"""
import asyncio
import aiohttp
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from aiohttp import web
from urllib.parse import urlsplit
from .exceptions import DistributedError
from .logging import debug_log

@dataclass
class _NetworkState:
    client_session: aiohttp.ClientSession | None = None


@lru_cache(maxsize=1)
def _network_state() -> _NetworkState:
    return _NetworkState()


def _prompt_server_instance() -> Any:
    import server

    return server.PromptServer.instance

async def get_client_session() -> aiohttp.ClientSession:
    """Get or create a shared aiohttp client session."""
    state = _network_state()
    try:
        asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RuntimeError("get_client_session() requires an active asyncio event loop.") from exc

    if state.client_session is None or state.client_session.closed:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        # Don't set timeout here - set it per request
        state.client_session = aiohttp.ClientSession(connector=connector)
    return state.client_session

async def cleanup_client_session() -> None:
    """Clean up the shared client session."""
    state = _network_state()
    if state.client_session and not state.client_session.closed:
        await state.client_session.close()
        state.client_session = None

async def handle_api_error(request: web.Request, error: Any, status: int = 500) -> web.StreamResponse:
    """Standardized error response handler."""
    _ = request
    if isinstance(error, list):
        messages = [str(item) for item in error]
        debug_log(f"API Error [{status}]: {messages}")
        return web.json_response(
            {
                "status": "error",
                "error": messages,
                "message": "; ".join(messages),
            },
            status=status,
        )

    if isinstance(error, DistributedError):
        debug_log(f"API Error [{status}] ({error.__class__.__name__}): {error}")
        return web.json_response(
            {
                "status": "error",
                "error": str(error),
                "message": str(error),
                "error_type": error.__class__.__name__,
            },
            status=status,
        )

    message = str(error)
    debug_log(f"API Error [{status}]: {message}")
    return web.json_response(
        {
            "status": "error",
            "error": message,
            "message": message,
        },
        status=status,
    )

def get_server_port() -> int:
    """Get the ComfyUI server port."""
    return _prompt_server_instance().port

def get_server_loop() -> asyncio.AbstractEventLoop:
    """Get the ComfyUI server event loop."""
    return _prompt_server_instance().loop


def normalize_host(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    host = value.strip()
    if not host:
        return host
    if "://" in host:
        parsed = urlsplit(host)
        host = parsed.netloc or parsed.path
    return host.split("/")[0]


def build_worker_url(worker: dict[str, Any], endpoint: str = "") -> str:
    """Construct the worker base URL with optional endpoint."""
    host = (worker.get("host") or "").strip()
    port = int(worker.get("port", worker.get("listen_port", 8188)) or 8188)

    if not host:
        host = getattr(_prompt_server_instance(), "address", "127.0.0.1") or "127.0.0.1"

    if host.startswith(("http://", "https://")):
        base = host.rstrip("/")
    else:
        is_cloud = worker.get("type") == "cloud" or host.endswith(".proxy.runpod.net") or port == 443
        scheme = "https" if is_cloud else "http"
        default_port = 443 if scheme == "https" else 80
        port_part = "" if port == default_port else f":{port}"
        base = f"{scheme}://{host}{port_part}"

    return f"{base}{endpoint}"


async def probe_worker(worker_url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    """GET {worker_url}/prompt. Returns parsed JSON or None on any failure."""
    base_url = (worker_url or "").strip().rstrip("/")
    if not base_url:
        return None
    probe_url = base_url if base_url.endswith("/prompt") else f"{base_url}/prompt"
    session = await get_client_session()
    try:
        async with session.get(
            probe_url,
            timeout=aiohttp.ClientTimeout(total=float(timeout)),
        ) as response:
            if response.status != 200:
                debug_log(f"[Distributed] Worker probe non-200 status: {response.status} ({probe_url})")
                return None
            payload = await response.json()
            if isinstance(payload, dict):
                return payload
            debug_log(f"[Distributed] Worker probe returned non-object JSON: {probe_url}")
            return None
    except asyncio.TimeoutError:
        debug_log(f"[Distributed] Worker probe timed out: {probe_url}")
        return None
    except aiohttp.ClientConnectorError:
        debug_log(f"[Distributed] Worker unreachable: {probe_url}")
        return None
    except Exception as exc:
        debug_log(f"[Distributed] Worker probe error ({probe_url}): {exc}")
        return None


def build_master_url(
    config: dict[str, Any] | None = None,
    prompt_server_instance: Any | None = None,
) -> str:
    """Build the best public URL workers should use to reach the master."""
    if config is None:
        from .config import load_config
        config = load_config()

    prompt_server_instance = prompt_server_instance or _prompt_server_instance()
    master_cfg = (config or {}).get("master", {}) or {}
    configured_host = (master_cfg.get("host") or "").strip()
    configured_port = master_cfg.get("port")
    default_port = getattr(prompt_server_instance, "port", 8188) or 8188
    try:
        port = int(configured_port or default_port)
    except (TypeError, ValueError):
        port = int(default_port)

    def _needs_https(hostname):
        hostname = hostname.lower()
        https_domains = (
            ".proxy.runpod.net",
            ".ngrok-free.app",
            ".ngrok-free.dev",
            ".ngrok.io",
            ".trycloudflare.com",
            ".cloudflare.dev",
        )
        return any(hostname.endswith(suffix) for suffix in https_domains)

    if configured_host:
        if configured_host.startswith(("http://", "https://")):
            return configured_host.rstrip("/")

        host = configured_host
        scheme = "https" if _needs_https(host) or port == 443 else "http"
        default_port_for_scheme = 443 if scheme == "https" else 80
        if configured_port is None and scheme == "https" and _needs_https(host):
            port = default_port_for_scheme
        port_part = "" if port == default_port_for_scheme else f":{port}"
        return f"{scheme}://{host}{port_part}"

    address = getattr(prompt_server_instance, "address", "127.0.0.1") or "127.0.0.1"
    if address in ("0.0.0.0", "::"):  # nosec B104 - normalization of wildcard bind host
        address = "127.0.0.1"
    scheme = "https" if port == 443 else "http"
    default_port_for_scheme = 443 if scheme == "https" else 80
    port_part = "" if port == default_port_for_scheme else f":{port}"
    return f"{scheme}://{address}{port_part}"
