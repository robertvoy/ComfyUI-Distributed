"""Initialize routes, distributed state and the existing worker lifecycle."""
import atexit
import os

import server

from ..utils.config import CONFIG_FILE, ensure_config_exists
from ..utils.logging import debug_log
from ..workers.startup import delayed_auto_launch, register_async_signals, sync_cleanup
from ..upscale.job_store import ensure_tile_jobs_initialized

_initialized = False


def initialize():
    """Called by ComfyExtension.on_load; initialize once per loaded package."""
    global _initialized
    if _initialized:
        return

    ensure_config_exists()
    from .. import api  # noqa: F401 - registers the existing @routes.* handlers
    from ..api.queue_orchestration import ensure_distributed_state

    ensure_distributed_state(server.PromptServer.instance)
    ensure_tile_jobs_initialized()

    if not os.environ.get('COMFYUI_IS_WORKER'):
        atexit.register(sync_cleanup)
        delayed_auto_launch()
        register_async_signals()

    _initialized = True
    debug_log('Loaded Distributed nodes.')
    debug_log(f'Config file: {CONFIG_FILE}')
