import asyncio
import threading
import time
import atexit
import signal
import platform

import server

from ..utils.config import load_config, save_config
from ..utils.logging import debug_log, log
from ..utils.network import normalize_host
from ..utils.cloudflare import cloudflare_tunnel_manager
from ..utils.constants import WORKER_STARTUP_DELAY
from . import get_worker_manager


def auto_launch_workers() -> None:
    """Launch enabled workers if auto_launch_workers is set to true."""
    wm = get_worker_manager()
    try:
        config = load_config()
        if config.get('settings', {}).get('auto_launch_workers', False):
            log("Auto-launch workers is enabled, checking for workers to start...")
            
            # Clear managed_processes before launching new workers
            # This handles cases where the master was killed without proper cleanup
            if config.get('managed_processes'):
                log("Clearing old managed_processes before auto-launch...")
                config['managed_processes'] = {}
                save_config(config)
            
            workers = config.get('workers', [])
            launched_count = 0
            
            for worker in workers:
                if worker.get('enabled', False):
                    worker_id = worker.get('id')
                    worker_name = worker.get('name', f'Worker {worker_id}')
                    
                    # Skip remote workers
                    host = (normalize_host(worker.get('host', 'localhost')) or 'localhost').lower()
                    if host not in ['localhost', '127.0.0.1', '', None]:
                        debug_log(f"Skipping remote worker {worker_name} (host: {host})")
                        continue
                    
                    # Check if already running
                    if str(worker_id) in wm.processes:
                        proc_info = wm.processes[str(worker_id)]
                        if wm.is_process_running(proc_info['pid']):
                            debug_log(f"Worker {worker_name} already running, skipping")
                            continue
                    
                    # Launch the worker
                    try:
                        pid = wm.launch_worker(worker)
                        log(f"Auto-launched worker {worker_name} (PID: {pid})")
                        
                        # Mark as launching in managed processes
                        if str(worker_id) in wm.processes:
                            wm.processes[str(worker_id)]['launching'] = True
                            wm.save_processes()
                        
                        launched_count += 1
                    except Exception as e:
                        log(f"Failed to auto-launch worker {worker_name}: {e}")
            
            if launched_count > 0:
                log(f"Auto-launched {launched_count} worker(s)")
            else:
                debug_log("No workers to auto-launch")
        else:
            debug_log("Auto-launch workers is disabled")
    except Exception as e:
        log(f"Error during auto-launch: {e}")

# Schedule auto-launch after a short delay to ensure server is ready
def delayed_auto_launch() -> None:
    """Delay auto-launch to ensure server is fully initialized."""
    import threading
    timer = threading.Timer(WORKER_STARTUP_DELAY, auto_launch_workers)
    timer.daemon = True
    timer.start()

# Async cleanup function for proper shutdown
async def async_cleanup_and_exit(signum: int | None = None) -> None:
    """Async-friendly cleanup and exit."""
    _ = signum
    wm = get_worker_manager()
    cleanup_error = None
    try:
        config = load_config()
        if config.get('settings', {}).get('stop_workers_on_master_exit', True):
            print("\n[Distributed] Master shutting down, stopping all managed workers...")
            wm.cleanup_all()
        else:
            print("\n[Distributed] Master shutting down, workers will continue running")
            wm.save_processes()
        try:
            await cloudflare_tunnel_manager.stop_tunnel()
        except Exception as tunnel_error:
            log(f"[Distributed] Warning: Cloudflare tunnel did not stop cleanly during shutdown: {tunnel_error}")
    except Exception as e:
        print(f"[Distributed] Error during cleanup: {e}")
        cleanup_error = e
    
    # Stop the event loop gracefully on all platforms.
    loop = asyncio.get_running_loop()
    loop.stop()
    if cleanup_error is not None:
        raise cleanup_error

def register_async_signals() -> None:
    """Register async signal handlers for graceful shutdown."""
    wm = get_worker_manager()
    if platform.system() == "Windows":
        # Windows doesn't support add_signal_handler, use traditional signal handling
        def signal_handler(signum: int, _frame: object) -> None:
            # Schedule the async cleanup in the event loop
            loop = server.PromptServer.instance.loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(async_cleanup_and_exit(signum), loop)
            else:
                # Fallback to sync cleanup if loop isn't running
                cleanup_error = None
                try:
                    config = load_config()
                    if config.get('settings', {}).get('stop_workers_on_master_exit', True):
                        print("\n[Distributed] Master shutting down, stopping all managed workers...")
                        wm.cleanup_all()
                    else:
                        print("\n[Distributed] Master shutting down, workers will continue running")
                        wm.save_processes()
                except Exception as e:
                    print(f"[Distributed] Error during cleanup: {e}")
                    cleanup_error = e
                if cleanup_error is not None:
                    raise RuntimeError("Cleanup failed during Windows fallback signal handling") from cleanup_error
                raise SystemExit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Unix-like systems support add_signal_handler
        loop = server.PromptServer.instance.loop
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(async_cleanup_and_exit(s)))
        
        # SIGHUP is Unix-only
        loop.add_signal_handler(signal.SIGHUP, lambda: asyncio.create_task(async_cleanup_and_exit(signal.SIGHUP)))

def sync_cleanup() -> None:
    """Synchronous wrapper for atexit."""
    wm = get_worker_manager()
    try:
        # For atexit, we don't want to stop the loop or exit
        config = load_config()
        if config.get('settings', {}).get('stop_workers_on_master_exit', True):
            print("\n[Distributed] Master shutting down, stopping all managed workers...")
            wm.cleanup_all()
        else:
            print("\n[Distributed] Master shutting down, workers will continue running")
            wm.save_processes()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(cloudflare_tunnel_manager.stop_tunnel())
            else:
                loop.run_until_complete(cloudflare_tunnel_manager.stop_tunnel())
        except RuntimeError:
            # No running loop; create a temporary one
            asyncio.run(cloudflare_tunnel_manager.stop_tunnel())
        except Exception as tunnel_error:
            log(f"[Distributed] Warning: Cloudflare tunnel did not stop cleanly during sync cleanup: {tunnel_error}")
    except Exception as e:
        print(f"[Distributed] Error during cleanup: {e}")
        raise
