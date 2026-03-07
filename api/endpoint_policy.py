"""Shared endpoint policy helpers for distributed API routes."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from aiohttp import web

from ..utils.network import handle_api_error
from .request_guards import authorization_error_or_none


async def run_authorized_endpoint(
    request: web.Request,
    operation: Callable[[], Awaitable[web.StreamResponse]],
    *,
    unexpected_status: int = 500,
) -> web.StreamResponse:
    """Run endpoint operation with shared auth guard and fallback error mapping."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        return await operation()
    except Exception as exc:
        return await handle_api_error(request, exc, unexpected_status)
