"""
ComfyUI-Distributed: thin entry point.
All implementation lives in workers/, nodes/, api/.
"""
import atexit
import os

import server

from .utils.config import ensure_config_exists
from .utils.logging import debug_log
from .utils.network import cleanup_client_session
from .workers import get_worker_manager
from .workers.startup import delayed_auto_launch, register_async_signals, sync_cleanup
from .upscale.job_store import ensure_tile_jobs_initialized
from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    ImageBatchDivider,
    DistributedCollectorNode,
    DistributedSeed,
    DistributedModelName,
    DistributedValue,
    AudioBatchDivider,
    DistributedEmptyImage,
    DistributedListSplitter,
    DistributedListCollector,
    DistributedBranch,
    DistributedBranchCollector,
    AnyType,
    ByPassTypeTuple,
    any_type,
)
from . import api  # noqa: F401 - triggers all @routes.* registrations
from .api.queue_orchestration import ensure_distributed_state

ensure_config_exists()

# Aiohttp session cleanup
async def _cleanup_session():
    await cleanup_client_session()


atexit.register(lambda: None)  # placeholder; real cleanup in sync_cleanup

# Initialize distributed job state on prompt_server
prompt_server = server.PromptServer.instance
ensure_distributed_state(prompt_server)
ensure_tile_jobs_initialized()

# Worker startup
if not os.environ.get('COMFYUI_IS_WORKER'):
    atexit.register(sync_cleanup)
    delayed_auto_launch()
    register_async_signals()
