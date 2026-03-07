from __future__ import annotations

import json
import asyncio
import os
import time
import platform
import subprocess  # nosec B404 - commands are fixed and never shell-expanded
import socket
from typing import Any

import torch
import aiohttp
from aiohttp import web
import server

from ..utils.config import load_config
from ..utils.logging import debug_log, log
from ..utils.network import (
    build_worker_url,
    get_client_session,
    handle_api_error,
    normalize_host,
    probe_worker,
)
from ..utils.constants import CHUNK_SIZE
from ..workers import get_worker_manager
from .request_guards import authorization_error_or_none
from .schemas import (
    distributed_auth_headers,
    require_fields,
    require_worker_id,
)
from ..workers.detection import (
    get_machine_id,
    is_docker_environment,
    is_runpod_environment,
)
from ..utils.async_helpers import PromptValidationError, queue_prompt_payload


@server.PromptServer.instance.routes.get("/distributed/worker_ws")
async def worker_ws_endpoint(request: web.Request) -> web.StreamResponse:
    """WebSocket endpoint for worker prompt dispatch."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data or "{}")
            except json.JSONDecodeError:
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": None,
                    "ok": False,
                    "error": "Invalid JSON payload.",
                })
                continue

            if data.get("type") != "dispatch_prompt":
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": data.get("request_id"),
                    "ok": False,
                    "error": "Unsupported websocket message type.",
                })
                continue

            prompt = data.get("prompt")
            if not isinstance(prompt, dict):
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": data.get("request_id"),
                    "ok": False,
                    "error": "Field 'prompt' must be an object.",
                })
                continue

            try:
                prompt_id = await queue_prompt_payload(
                    prompt,
                    workflow_meta=data.get("workflow"),
                    client_id=data.get("client_id"),
                )
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": data.get("request_id"),
                    "ok": True,
                    "prompt_id": prompt_id,
                })
            except PromptValidationError as exc:
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": data.get("request_id"),
                    "ok": False,
                    "error": str(exc),
                    "validation_error": exc.validation_error,
                    "node_errors": exc.node_errors,
                })
            except Exception as exc:
                await ws.send_json({
                    "type": "dispatch_ack",
                    "request_id": data.get("request_id"),
                    "ok": False,
                    "error": str(exc),
                })
        elif msg.type == aiohttp.WSMsgType.ERROR:
            log(f"[Distributed] Worker websocket error: {ws.exception()}")

    return ws


@server.PromptServer.instance.routes.post("/distributed/worker/clear_launching")
async def clear_launching_state(request: web.Request) -> web.StreamResponse:
    """Clear the launching flag when worker is confirmed running."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        wm = get_worker_manager()
        data = await request.json()
        missing = require_fields(data, "worker_id")
        if missing:
            return await handle_api_error(request, f"Missing required field(s): {', '.join(missing)}", 400)

        worker_id = str(data.get("worker_id")).strip()
        config = load_config()
        try:
            worker_id = require_worker_id(worker_id, config)
        except ValueError as exc:
            return await handle_api_error(request, exc, 404)
        
        # Clear launching flag in managed processes
        if worker_id in wm.processes:
            if 'launching' in wm.processes[worker_id]:
                del wm.processes[worker_id]['launching']
                wm.save_processes()
                debug_log(f"Cleared launching state for worker {worker_id}")
        
        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e, 500)


def get_network_ips() -> list[str]:
    """Get all network IPs, trying multiple methods."""
    command_timeout = 5.0
    ips = []
    hostname = socket.gethostname()

    # Method 1: Try socket.getaddrinfo
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for info in addr_info:
            ip = info[4][0]
            if ip and ip not in ips and not ip.startswith('::'):  # Skip IPv6 for now
                ips.append(ip)
    except (socket.gaierror, OSError) as exc:
        debug_log(f"get_network_ips: getaddrinfo failed for hostname {hostname}: {exc}")

    # Method 2: Try to connect to external server and get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS
        local_ip = s.getsockname()[0]
        s.close()
        if local_ip not in ips:
            ips.append(local_ip)
    except (OSError, socket.error) as exc:
        debug_log(f"get_network_ips: UDP local IP probe failed: {exc}")

    # Method 3: Platform-specific commands
    try:
        if platform.system() == "Windows":
            # Windows ipconfig
            result = subprocess.run(  # nosec B603 - static command, no user input
                ["ipconfig"],
                capture_output=True,
                text=True,
                timeout=command_timeout,
            )
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'IPv4' in line and i + 1 < len(lines):
                    ip = lines[i].split(':')[-1].strip()
                    if ip and ip not in ips:
                        ips.append(ip)
        else:
            # Unix/Linux/Mac ifconfig or ip addr
            try:
                result = subprocess.run(  # nosec B603 - static command, no user input
                    ["ip", "addr"],
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                )
            except (FileNotFoundError, OSError):
                try:
                    result = subprocess.run(  # nosec B603 - static command, no user input
                        ["ifconfig"],
                        capture_output=True,
                        text=True,
                        timeout=command_timeout,
                    )
                except (FileNotFoundError, OSError):
                    result = None

            import re
            ip_pattern = re.compile(r'inet\s+(\d+\.\d+\.\d+\.\d+)')
            if result is not None:
                for match in ip_pattern.finditer(result.stdout):
                    ip = match.group(1)
                    if ip and ip not in ips:
                        ips.append(ip)
    except (OSError, subprocess.SubprocessError) as exc:
        debug_log(f"get_network_ips: platform command probe failed: {exc}")

    return ips


def get_recommended_ip(ips: list[str]) -> str | None:
    """Choose the best IP for master-worker communication."""
    # Priority order:
    # 1. Private network ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
    # 2. Other non-localhost IPs
    # 3. Localhost as last resort

    private_ips = []
    public_ips = []

    for ip in ips:
        if ip.startswith('127.') or ip == 'localhost':
            continue
        elif (ip.startswith('192.168.')
                or ip.startswith('10.')
                or (ip.startswith('172.') and 16 <= int(ip.split('.')[1]) <= 31)):
            private_ips.append(ip)
        else:
            public_ips.append(ip)

    # Prefer private IPs
    if private_ips:
        # Prefer 192.168 range as it's most common
        for ip in private_ips:
            if ip.startswith('192.168.'):
                return ip
        return private_ips[0]
    elif public_ips:
        return public_ips[0]
    elif ips:
        return ips[0]
    else:
        return None


def _get_cuda_info() -> tuple[int | None, int, int]:
    """Detect CUDA device index and total physical GPU count.

    Returns (cuda_device, cuda_device_count, physical_device_count).
    All three are 0/None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None, 0, 0
    try:
        cuda_device_count = torch.cuda.device_count()
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible.strip():
            visible_devices = [int(d.strip()) for d in cuda_visible.split(',') if d.strip().isdigit()]
            if visible_devices:
                cuda_device = visible_devices[0]
                try:
                    result = subprocess.run(  # nosec B603 - static command, no user input
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    physical_device_count = (
                        len(result.stdout.strip().split('\n'))
                        if result.returncode == 0
                        else max(visible_devices) + 1
                    )
                except (FileNotFoundError, OSError, subprocess.SubprocessError):
                    physical_device_count = max(visible_devices) + 1
                return cuda_device, cuda_device_count, physical_device_count
            else:
                return 0, cuda_device_count, cuda_device_count
        else:
            cuda_device = torch.cuda.current_device()
            return cuda_device, cuda_device_count, cuda_device_count
    except Exception as e:
        debug_log(f"CUDA detection error: {e}")
        return None, 0, 0


def _collect_network_info_sync() -> dict[str, Any]:
    """Collect network/cuda info in a worker thread to avoid blocking route handlers."""
    cuda_device, cuda_device_count, physical_device_count = _get_cuda_info()
    hostname = socket.gethostname()
    all_ips = get_network_ips()
    recommended_ip = get_recommended_ip(all_ips)
    return {
        "hostname": hostname,
        "all_ips": all_ips,
        "recommended_ip": recommended_ip,
        "cuda_device": cuda_device,
        "cuda_device_count": physical_device_count if physical_device_count > 0 else cuda_device_count,
    }


def _read_worker_log_sync(log_file: str, lines_to_read: int) -> dict[str, Any]:
    """Read worker log content from disk in a threadpool worker."""
    file_size = os.path.getsize(log_file)

    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        if lines_to_read > 0 and file_size > 1024 * 1024:
            # Read last N lines efficiently from end of file.
            lines = []
            f.seek(0, 2)
            file_length = f.tell()
            chunk_size = min(CHUNK_SIZE, file_length)

            while len(lines) < lines_to_read and f.tell() > 0:
                current_pos = max(0, f.tell() - chunk_size)
                f.seek(current_pos)
                chunk = f.read(chunk_size)
                chunk_lines = chunk.splitlines()
                if current_pos > 0:
                    chunk_lines = chunk_lines[1:]
                lines = chunk_lines + lines
                f.seek(current_pos)

            content = '\n'.join(lines[-lines_to_read:])
            truncated = len(lines) > lines_to_read
        else:
            content = f.read()
            truncated = False

    return {
        "content": content,
        "file_size": file_size,
        "truncated": truncated,
        "lines_shown": lines_to_read if truncated else content.count('\n') + 1,
    }


def _parse_positive_int_query(
    value: Any,
    default: int,
    minimum: int = 1,
    maximum: int | None = 10000,
) -> int:
    """Parse bounded positive integer query params with sane fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _find_worker_by_id(config: dict[str, Any], worker_id: str) -> dict[str, Any] | None:
    worker_id_str = str(worker_id).strip()
    for worker in config.get("workers", []):
        if str(worker.get("id")).strip() == worker_id_str:
            return worker
    return None


@server.PromptServer.instance.routes.get("/distributed/local_log")
async def get_local_log_endpoint(request: web.Request) -> web.StreamResponse:
    """Return this instance's in-memory ComfyUI log buffer."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        from app.logger import get_logs
    except Exception as e:
        return await handle_api_error(request, f"Failed to import app.logger: {e}", 500)

    try:
        lines_to_read = _parse_positive_int_query(request.query.get("lines"), default=300, maximum=3000)
        logs = get_logs()
        if logs is None:
            return web.json_response(
                {
                    "status": "success",
                    "content": "",
                    "entries": 0,
                    "source": "memory",
                    "truncated": False,
                    "lines_shown": 0,
                }
            )

        entries = list(logs)
        selected_entries = entries[-lines_to_read:]
        content = "".join(
            entry.get("m", "") if isinstance(entry, dict) else str(entry)
            for entry in selected_entries
        )
        lines_shown = content.count("\n") + (1 if content else 0)

        return web.json_response(
            {
                "status": "success",
                "content": content,
                "entries": len(selected_entries),
                "source": "memory",
                "truncated": len(entries) > len(selected_entries),
                "lines_shown": lines_shown,
            }
        )
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/network_info")
async def get_network_info_endpoint(request: web.Request) -> web.StreamResponse:
    """Get network interfaces and recommend best IP for master."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, _collect_network_info_sync)
        
        return web.json_response({
            "status": "success",
            **info,
            "message": "Auto-detected network configuration"
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.get("/distributed/system_info")
async def get_system_info_endpoint(request: web.Request) -> web.StreamResponse:
    """Get system information including machine ID for local worker detection."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        import socket
        
        return web.json_response({
            "status": "success",
            "hostname": socket.gethostname(),
            "machine_id": get_machine_id(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "node": platform.node(),
                "path_separator": os.sep,  # Add path separator
                "os_name": os.name  # Add OS name (posix, nt, etc.)
            },
            "is_docker": is_docker_environment(),
            "is_runpod": is_runpod_environment(),
            "runpod_pod_id": os.environ.get('RUNPOD_POD_ID')
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/launch_worker")
async def launch_worker_endpoint(request: web.Request) -> web.StreamResponse:
    """Launch a worker process from the UI."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        wm = get_worker_manager()
        data = await request.json()
        missing = require_fields(data, "worker_id")
        if missing:
            return await handle_api_error(request, f"Missing required field(s): {', '.join(missing)}", 400)

        worker_id = str(data.get("worker_id")).strip()
        
        # Find worker config
        config = load_config()
        try:
            worker_id = require_worker_id(worker_id, config)
        except ValueError as exc:
            return await handle_api_error(request, exc, 404)
        worker = next((w for w in config.get("workers", []) if str(w.get("id")) == worker_id), None)
        if not worker:
            return await handle_api_error(request, f"Worker {worker_id} not found", 404)
        
        # Ensure consistent string ID
        worker_id_str = worker_id
        
        # Check if already running (managed by this instance)
        if worker_id_str in wm.processes:
            proc_info = wm.processes[worker_id_str]
            process = proc_info.get('process')
            
            # Check if still running
            is_running = False
            if process:
                is_running = process.poll() is None
            else:
                # Restored process without subprocess object
                is_running = wm.is_process_running(proc_info['pid'])
            
            if is_running:
                return await handle_api_error(request, "Worker already running (managed by UI)", 409)
            else:
                # Process is dead, remove it
                del wm.processes[worker_id_str]
                wm.save_processes()
        
        # Launch the worker
        try:
            loop = asyncio.get_running_loop()
            pid = await loop.run_in_executor(None, wm.launch_worker, worker)
            log_file = wm.processes[worker_id_str].get('log_file')
            return web.json_response({
                "status": "success",
                "pid": pid,
                "message": f"Worker {worker['name']} launched",
                "log_file": log_file
            })
        except Exception as e:
            return await handle_api_error(request, f"Failed to launch worker: {str(e)}", 500)
            
    except Exception as e:
        return await handle_api_error(request, e, 400)


@server.PromptServer.instance.routes.post("/distributed/stop_worker")
async def stop_worker_endpoint(request: web.Request) -> web.StreamResponse:
    """Stop a worker process that was launched from the UI."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        wm = get_worker_manager()
        data = await request.json()
        missing = require_fields(data, "worker_id")
        if missing:
            return await handle_api_error(request, f"Missing required field(s): {', '.join(missing)}", 400)

        worker_id = str(data.get("worker_id")).strip()
        config = load_config()
        try:
            worker_id = require_worker_id(worker_id, config)
        except ValueError as exc:
            return await handle_api_error(request, exc, 404)

        success, message = wm.stop_worker(worker_id)
        
        if success:
            return web.json_response({"status": "success", "message": message})
        else:
            return await handle_api_error(
                request,
                message,
                404 if "not managed" in message else 409,
            )
            
    except Exception as e:
        return await handle_api_error(request, e, 400)


@server.PromptServer.instance.routes.get("/distributed/managed_workers")
async def get_managed_workers_endpoint(request: web.Request) -> web.StreamResponse:
    """Get list of workers managed by this UI instance."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        managed = get_worker_manager().get_managed_workers()
        return web.json_response({
            "status": "success",
            "managed_workers": managed
        })
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/local-worker-status")
async def get_local_worker_status_endpoint(request: web.Request) -> web.StreamResponse:
    """Check status of all local workers (localhost/no host specified)."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        config = load_config()
        worker_statuses = {}
        
        for worker in config.get("workers", []):
            # Only check local workers
            host = normalize_host(worker.get("host")) or ""
            if not host or host in ["localhost", "127.0.0.1"]:
                worker_id = worker["id"]
                port = worker["port"]
                
                # Check if worker is enabled
                if not worker.get("enabled", False):
                    worker_statuses[worker_id] = {
                        "online": False,
                        "enabled": False,
                        "processing": False,
                        "queue_count": 0
                    }
                    continue
                
                # Try to connect to worker
                try:
                    worker_url = build_worker_url(worker)
                    data = await probe_worker(worker_url, timeout=2.0)
                    if data is None:
                        worker_statuses[worker_id] = {
                            "online": False,
                            "enabled": True,
                            "processing": False,
                            "queue_count": 0,
                            "error": "Unavailable",
                        }
                        continue
                    queue_remaining = data.get("exec_info", {}).get("queue_remaining", 0)
                    worker_statuses[worker_id] = {
                        "online": True,
                        "enabled": True,
                        "processing": queue_remaining > 0,
                        "queue_count": queue_remaining
                    }
                except asyncio.TimeoutError:
                    worker_statuses[worker_id] = {
                        "online": False,
                        "enabled": True,
                        "processing": False,
                        "queue_count": 0,
                        "error": "Timeout"
                    }
                except Exception as e:
                    worker_statuses[worker_id] = {
                        "online": False,
                        "enabled": True,
                        "processing": False,
                        "queue_count": 0,
                        "error": str(e)
                    }
        
        return web.json_response({
            "status": "success",
            "worker_statuses": worker_statuses
        })
    except Exception as e:
        debug_log(f"Error checking local worker status: {e}")
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/worker_log/{worker_id}")
async def get_worker_log_endpoint(request: web.Request) -> web.StreamResponse:
    """Get log content for a specific worker."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        wm = get_worker_manager()
        worker_id = request.match_info['worker_id']
        
        # Ensure worker_id is string
        worker_id = str(worker_id)
        
        # Check if we manage this worker
        if worker_id not in wm.processes:
            return await handle_api_error(request, f"Worker {worker_id} not managed by UI", 404)
        
        proc_info = wm.processes[worker_id]
        log_file = proc_info.get('log_file')
        
        if not log_file or not os.path.exists(log_file):
            return await handle_api_error(request, "Log file not found", 404)
        
        # Read last N lines (or full file if small)
        lines_to_read = _parse_positive_int_query(request.query.get('lines'), default=1000)
        
        try:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(None, _read_worker_log_sync, log_file, lines_to_read)
            
            return web.json_response({
                "status": "success",
                "content": payload["content"],
                "log_file": log_file,
                "file_size": payload["file_size"],
                "truncated": payload["truncated"],
                "lines_shown": payload["lines_shown"],
            })
            
        except Exception as e:
            return await handle_api_error(request, f"Error reading log file: {str(e)}", 500)
            
    except Exception as e:
        return await handle_api_error(request, e, 500)


@server.PromptServer.instance.routes.get("/distributed/remote_worker_log/{worker_id}")
async def get_remote_worker_log_endpoint(request: web.Request) -> web.StreamResponse:
    """Proxy a remote worker log request to the worker's local in-memory log endpoint."""
    auth_error = await authorization_error_or_none(request)
    if auth_error is not None:
        return auth_error
    try:
        worker_id = str(request.match_info["worker_id"]).strip()
        config = load_config()
        worker = _find_worker_by_id(config, worker_id)
        if not worker:
            return await handle_api_error(request, f"Worker {worker_id} not found", 404)

        # Remote log proxy is only meaningful for remote/cloud workers.
        host = normalize_host(worker.get("host")) or ""
        if not host:
            return await handle_api_error(
                request,
                f"Worker {worker_id} is local; use /distributed/worker_log/{worker_id} instead.",
                400,
            )

        lines_to_read = _parse_positive_int_query(request.query.get("lines"), default=300, maximum=3000)
        worker_url = build_worker_url(worker, "/distributed/local_log")
        session = await get_client_session()
        async with session.get(
            worker_url,
            params={"lines": str(lines_to_read)},
            headers=distributed_auth_headers(config),
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                return await handle_api_error(
                    request,
                    f"Remote worker {worker_id} returned HTTP {resp.status}: {body[:400]}",
                    resp.status,
                )

            try:
                data = await resp.json()
            except Exception as e:
                return await handle_api_error(
                    request,
                    f"Remote worker {worker_id} returned invalid JSON: {e}",
                    502,
                )

        return web.json_response(data)
    except Exception as e:
        return await handle_api_error(request, e, 500)
