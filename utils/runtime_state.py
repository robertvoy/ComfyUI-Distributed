from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DistributedRuntimeState:
    """Typed distributed runtime state bound to PromptServer.instance."""

    distributed_pending_jobs: dict[str, asyncio.Queue] = field(default_factory=dict)
    distributed_jobs_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    distributed_job_allowed_workers: dict[str, set[str]] = field(default_factory=dict)
    distributed_pending_tile_jobs: dict[str, Any] = field(default_factory=dict)
    distributed_tile_jobs_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def get_prompt_server_instance() -> Any:
    import server

    return server.PromptServer.instance


def ensure_distributed_runtime_state(server_instance: Any | None = None) -> DistributedRuntimeState:
    """Return canonical distributed runtime state for the server instance."""
    prompt_server = server_instance or get_prompt_server_instance()
    state = getattr(prompt_server, "distributed_runtime_state", None)
    if not isinstance(state, DistributedRuntimeState):
        state = DistributedRuntimeState()
        setattr(prompt_server, "distributed_runtime_state", state)

    # Backward-compatible aliases for legacy call sites.
    prompt_server.distributed_pending_jobs = state.distributed_pending_jobs
    prompt_server.distributed_jobs_lock = state.distributed_jobs_lock
    prompt_server.distributed_job_allowed_workers = state.distributed_job_allowed_workers
    prompt_server.distributed_pending_tile_jobs = state.distributed_pending_tile_jobs
    prompt_server.distributed_tile_jobs_lock = state.distributed_tile_jobs_lock
    return state
