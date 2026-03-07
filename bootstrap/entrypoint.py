"""Single startup owner for node surface and runtime initialization."""
from __future__ import annotations

import atexit
import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def build_node_mappings() -> tuple[dict[str, Any], dict[str, str]]:
    """Build and cache the full exported node mapping surface."""
    from ..nodes import (
        NODE_CLASS_MAPPINGS as distributed_class_mappings,
        NODE_DISPLAY_NAME_MAPPINGS as distributed_display_name_mappings,
    )
    from ..nodes.distributed_upscale import (
        NODE_CLASS_MAPPINGS as upscale_class_mappings,
        NODE_DISPLAY_NAME_MAPPINGS as upscale_display_name_mappings,
    )

    return (
        {**distributed_class_mappings, **upscale_class_mappings},
        {**distributed_display_name_mappings, **upscale_display_name_mappings},
    )


@lru_cache(maxsize=1)
def _runtime_initialized() -> dict[str, bool]:
    return {"done": False}


def initialize_runtime(prompt_server: Any | None = None) -> None:
    """Initialize distributed runtime side effects once."""
    state = _runtime_initialized()
    if state["done"]:
        return

    import server

    from ..api import bootstrap_routes
    from ..api.queue_orchestration import ensure_distributed_state
    from ..upscale.job_store import ensure_tile_jobs_initialized
    from ..utils.config import CONFIG_FILE, ensure_config_exists
    from ..utils.logging import debug_log
    from ..workers.startup import delayed_auto_launch, register_async_signals, sync_cleanup

    ensure_config_exists()
    bootstrap_routes()

    server_instance = prompt_server if prompt_server is not None else server.PromptServer.instance
    ensure_distributed_state(server_instance)
    ensure_tile_jobs_initialized()

    atexit.register(lambda: None)  # placeholder; real cleanup in sync_cleanup
    if not os.environ.get("COMFYUI_IS_WORKER"):
        atexit.register(sync_cleanup)
        delayed_auto_launch()
        register_async_signals()

    state["done"] = True

    node_mappings, _ = build_node_mappings()
    debug_log("Loaded Distributed nodes.")
    debug_log(f"Config file: {CONFIG_FILE}")
    debug_log(f"Available nodes: {list(node_mappings.keys())}")
