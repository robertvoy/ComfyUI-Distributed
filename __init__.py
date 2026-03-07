"""ComfyUI-Distributed package entrypoint."""
from __future__ import annotations

from .bootstrap.entrypoint import build_node_mappings, initialize_runtime

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = build_node_mappings()
initialize_runtime()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
