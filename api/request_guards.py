"""Shared request guards for distributed API endpoints."""
from __future__ import annotations

from aiohttp import web

from ..utils.config import load_config
from ..utils.network import handle_api_error
from .schemas import is_authorized_request


async def authorization_error_or_none(request: web.Request) -> web.StreamResponse | None:
    """Return a 403 response when distributed API auth fails, else None."""
    if is_authorized_request(request, load_config()):
        return None
    return await handle_api_error(request, "Unauthorized", 403)
