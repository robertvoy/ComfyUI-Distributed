"""
Shared constants for ComfyUI-Distributed.
"""
import os

# Timeouts (in seconds)
WORKER_JOB_TIMEOUT = 30.0
TILE_COLLECTION_TIMEOUT = 30.0
TILE_WAIT_TIMEOUT = 30.0
PROCESS_TERMINATION_TIMEOUT = 5.0

# Process monitoring
WORKER_CHECK_INTERVAL = 2.0
STATUS_CHECK_INTERVAL = 5.0

# Cloudflare tunnel
TUNNEL_START_TIMEOUT = float(os.environ.get("TUNNEL_START_TIMEOUT", "25"))
CLOUDFLARE_LOG_BUFFER_SIZE = 200

# Network
CHUNK_SIZE = 8192
LOG_TAIL_BYTES = 65536  # 64KB

# File paths
WORKER_LOG_PATTERN = "distributed_worker_*.log"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
GPU_CONFIG_FILE = os.path.join(PROJECT_ROOT, "gpu_config.json")

# Worker management
WORKER_STARTUP_DELAY = 2.0

# Tile transfer
TILE_TRANSFER_TIMEOUT = 30.0

# Process cleanup
PROCESS_WAIT_TIMEOUT = 3.0
QUEUE_INIT_TIMEOUT = 5.0
TILE_SEND_TIMEOUT = 60.0
JOB_INIT_GRACE_PERIOD = 10.0

# Memory operations  
MEMORY_CLEAR_DELAY = 0.5

# Batch processing
MAX_BATCH = int(os.environ.get('COMFYUI_MAX_BATCH', '20'))  # Maximum items per batch to prevent timeouts/OOM (~100MB chunks for 512x512 PNGs)

# Heartbeat monitoring
HEARTBEAT_INTERVAL = float(os.environ.get('COMFYUI_HEARTBEAT_INTERVAL', '10'))  # Heartbeat/check interval in seconds
HEARTBEAT_TIMEOUT = int(os.environ.get('COMFYUI_HEARTBEAT_TIMEOUT', '60'))  # Worker heartbeat timeout in seconds (default 60s)

# USDU result collection
DYNAMIC_MODE_MAX_POLL_TIMEOUT = 10.0

# Static mode job poll loop
JOB_POLL_INTERVAL = 1.0
JOB_POLL_MAX_ATTEMPTS = 20

# Orchestration pipeline
ORCHESTRATION_WORKER_PROBE_CONCURRENCY = int(
    os.environ.get('COMFYUI_ORCHESTRATION_WORKER_PROBE_CONCURRENCY', '8')
)
ORCHESTRATION_WORKER_PREP_CONCURRENCY = int(
    os.environ.get('COMFYUI_ORCHESTRATION_WORKER_PREP_CONCURRENCY', '4')
)
ORCHESTRATION_MEDIA_SYNC_CONCURRENCY = int(
    os.environ.get('COMFYUI_ORCHESTRATION_MEDIA_SYNC_CONCURRENCY', '2')
)
ORCHESTRATION_MEDIA_SYNC_TIMEOUT = float(
    os.environ.get('COMFYUI_ORCHESTRATION_MEDIA_SYNC_TIMEOUT', '120')
)
