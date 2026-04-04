"""
Network and API utilities for ComfyUI-Distributed.
"""
import asyncio
import aiohttp
import re
import server
from aiohttp import web
from .logging import debug_log

# Shared session for connection pooling
_client_session = None

async def get_client_session():
    """Get or create a shared aiohttp client session."""
    global _client_session
    try:
        asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RuntimeError("get_client_session() requires an active asyncio event loop.") from exc

    if _client_session is None or _client_session.closed:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        # Don't set timeout here - set it per request
        _client_session = aiohttp.ClientSession(connector=connector)
    return _client_session

async def cleanup_client_session():
    """Clean up the shared client session."""
    global _client_session
    if _client_session and not _client_session.closed:
        await _client_session.close()
        _client_session = None

async def handle_api_error(request, error, status=500):
    """Standardized error response handler."""
    if isinstance(error, list):
        messages = [str(item) for item in error]
        debug_log(f"API Error [{status}]: {messages}")
        return web.json_response({"errors": messages}, status=status)

    message = str(error)
    debug_log(f"API Error [{status}]: {message}")
    return web.json_response({"error": message}, status=status)

def get_server_port():
    """Get the ComfyUI server port."""
    import server
    return server.PromptServer.instance.port

def get_server_loop():
    """Get the ComfyUI server event loop."""
    import server
    return server.PromptServer.instance.loop


def normalize_host(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    host = value.strip()
    if not host:
        return host
    host = re.sub(r"^https?://", "", host, flags=re.IGNORECASE)
    return host.split("/")[0]


def _split_host_and_port(host):
    if not host:
        return host, None

    if host.startswith("["):
        match = re.match(r"^(\[[^\]]+\])(?::(\d+))?$", host)
        if match:
            parsed_port = int(match.group(2)) if match.group(2) else None
            return match.group(1), parsed_port
        return host, None

    if host.count(":") == 1:
        candidate_host, candidate_port = host.rsplit(":", 1)
        if candidate_port.isdigit():
            return candidate_host, int(candidate_port)

    return host, None


def build_worker_url(worker, endpoint=""):
    """Construct the worker base URL with optional endpoint."""
    host = (worker.get("host") or "").strip()
    port = int(worker.get("port", worker.get("listen_port", 8188)) or 8188)

    if not host:
        host = getattr(server.PromptServer.instance, "address", "127.0.0.1") or "127.0.0.1"

    if host.startswith(("http://", "https://")):
        base = host.rstrip("/")
    else:
        is_cloud = worker.get("type") == "cloud" or host.endswith(".proxy.runpod.net") or port == 443
        scheme = "https" if is_cloud else "http"
        default_port = 443 if scheme == "https" else 80
        port_part = "" if port == default_port else f":{port}"
        base = f"{scheme}://{host}{port_part}"

    return f"{base}{endpoint}"


async def probe_worker(worker_url: str, timeout: float = 5.0) -> dict | None:
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


def build_master_url(config=None, prompt_server_instance=None):
    """Build the best public URL workers should use to reach the master."""
    if config is None:
        from .config import load_config
        config = load_config()

    prompt_server_instance = prompt_server_instance or server.PromptServer.instance
    master_cfg = (config or {}).get("master", {}) or {}
    configured_host = (master_cfg.get("host") or "").strip()
    runtime_port = getattr(prompt_server_instance, "port", 8188) or 8188

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

        host, explicit_port = _split_host_and_port(configured_host)
        port = explicit_port if explicit_port is not None else int(runtime_port)
        scheme = "https" if _needs_https(host) or port == 443 else "http"
        default_port_for_scheme = 443 if scheme == "https" else 80
        if explicit_port is None and scheme == "https" and _needs_https(host):
            port = default_port_for_scheme
        port_part = "" if port == default_port_for_scheme else f":{port}"
        return f"{scheme}://{host}{port_part}"

    address = getattr(prompt_server_instance, "address", "127.0.0.1") or "127.0.0.1"
    if address in ("0.0.0.0", "::"):
        address = "127.0.0.1"
    port = int(runtime_port)
    scheme = "https" if port == 443 else "http"
    default_port_for_scheme = 443 if scheme == "https" else 80
    port_part = "" if port == default_port_for_scheme else f":{port}"
    return f"{scheme}://{address}{port_part}"


def build_master_callback_url(worker, config=None, prompt_server_instance=None):
    """Build the callback URL a specific worker should use to reach the master."""
    prompt_server_instance = prompt_server_instance or server.PromptServer.instance

    worker_type = str((worker or {}).get("type") or "").strip().lower()
    worker_host = normalize_host((worker or {}).get("host"))
    local_hosts = {"", "localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}

    is_local_worker = worker_type == "local" or worker_host in local_hosts
    if is_local_worker:
        port = int(getattr(prompt_server_instance, "port", 8188) or 8188)
        scheme = "https" if port == 443 else "http"
        default_port_for_scheme = 443 if scheme == "https" else 80
        port_part = "" if port == default_port_for_scheme else f":{port}"
        return f"{scheme}://127.0.0.1{port_part}"

    return build_master_url(config=config, prompt_server_instance=prompt_server_instance)
