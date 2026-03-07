"""Compatibility facade for legacy imports from `distributed`."""
from __future__ import annotations

from .bootstrap.entrypoint import (
    build_node_mappings,
    initialize_runtime,
)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = build_node_mappings()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "initialize_runtime",
]
