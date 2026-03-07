"""Distributed API package with explicit route bootstrap."""
from __future__ import annotations

from importlib import import_module

_ROUTE_MODULES = (
    "config_routes",
    "tunnel_routes",
    "worker_routes",
    "job_routes",
    "usdu_routes",
)


def bootstrap_routes() -> None:
    """Import route modules once so aiohttp decorators register endpoints."""
    for module_name in _ROUTE_MODULES:
        import_module(f"{__name__}.{module_name}")


__all__ = ["bootstrap_routes"]
