"""Bootstrap entrypoints for ComfyUI-Distributed."""

from .entrypoint import build_node_mappings, initialize_runtime

__all__ = ["build_node_mappings", "initialize_runtime"]
