import os
import platform
import uuid

import aiohttp

from ..utils.network import normalize_host, get_client_session
from ..utils.logging import debug_log


async def is_local_worker(worker_config):
    """Check if a worker is running on the same machine as the master."""
    if worker_config.get('type') == 'serverless':
        return False
    host = normalize_host(worker_config.get('host', 'localhost')) or 'localhost'
    if host in ['localhost', '127.0.0.1', '0.0.0.0', ''] or worker_config.get('type') == 'local':
        return True
    
    # For cloud workers, check if on same physical host
    if worker_config.get('type') == 'cloud':
        return await is_same_physical_host(worker_config)
    
    return False

async def is_same_physical_host(worker_config):
    """Compare machine IDs to determine if worker is on same physical host."""
    try:
        # Get master machine ID
        master_machine_id = get_machine_id()
        
        # Fetch worker's machine ID via API
        host = normalize_host(worker_config.get('host', 'localhost')) or 'localhost'
        port = worker_config.get('port', 8188)
        
        session = await get_client_session()
        async with session.get(
            f"http://{host}:{port}/distributed/system_info",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                worker_machine_id = data.get('machine_id')
                return worker_machine_id == master_machine_id
            else:
                debug_log(f"Failed to get system info from worker: HTTP {resp.status}")
                return False
    except Exception as e:
        debug_log(f"Error checking same physical host: {e}")
        return False

def get_machine_id():
    """Get a unique identifier for this machine."""
    # Try multiple methods to get a stable machine ID
    try:
        # Method 1: MAC address-based UUID
        return str(uuid.getnode())
    except Exception:
        try:
            # Method 2: Platform + hostname
            import socket
            return f"{platform.machine()}_{socket.gethostname()}"
        except Exception:
            # Fallback
            return platform.machine()

def is_docker_environment():
    """Check if running inside Docker container."""
    return (os.path.exists('/.dockerenv') or 
            os.environ.get('DOCKER_CONTAINER', False) or
            'docker' in platform.node().lower())

def is_runpod_environment():
    """Check if running in Runpod environment."""
    return (os.environ.get('RUNPOD_POD_ID') is not None or
            os.environ.get('RUNPOD_API_KEY') is not None)
