from __future__ import annotations

from aiohttp import web
import server

from ..utils.cloudflare import cloudflare_tunnel_manager
from ..utils.config import load_config
from ..utils.logging import debug_log, log
from ..utils.network import handle_api_error


@server.PromptServer.instance.routes.get("/distributed/tunnel/status")
async def tunnel_status_endpoint(request: web.Request) -> web.StreamResponse:
    """Return Cloudflare tunnel status and last known details."""
    try:
        status = cloudflare_tunnel_manager.get_status()
        config = load_config()
        master_host = (config.get("master") or {}).get("host")
        return web.json_response({
            "status": "success",
            "tunnel": status,
            "master_host": master_host
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/tunnel/start")
async def tunnel_start_endpoint(request: web.Request) -> web.StreamResponse:
    """Start a Cloudflare tunnel pointing at the current ComfyUI server."""
    try:
        result = await cloudflare_tunnel_manager.start_tunnel()
        config = load_config()
        return web.json_response({
            "status": "success",
            "tunnel": result,
            "master_host": (config.get("master") or {}).get("host")
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/tunnel/stop")
async def tunnel_stop_endpoint(request: web.Request) -> web.StreamResponse:
    """Stop the managed Cloudflare tunnel if running."""
    try:
        result = await cloudflare_tunnel_manager.stop_tunnel()
        config = load_config()
        return web.json_response({
            "status": "success",
            "tunnel": result,
            "master_host": (config.get("master") or {}).get("host")
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)
