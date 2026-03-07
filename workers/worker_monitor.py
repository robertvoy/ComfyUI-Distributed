#!/usr/bin/env python3
"""
Worker process monitor - monitors if the master process is still alive
and terminates the worker if the master dies.
"""
import os
import sys
import time
import platform
import signal
from typing import Any

try:
    from ..utils.process import is_process_alive, launch_process_with_timeout, terminate_process
    from ..utils.constants import WORKER_CHECK_INTERVAL, PROCESS_TERMINATION_TIMEOUT
except ImportError:
    # Fallback when executed as a standalone script.
    from utils.process import is_process_alive, launch_process_with_timeout, terminate_process
    from utils.constants import WORKER_CHECK_INTERVAL, PROCESS_TERMINATION_TIMEOUT

def monitor_and_run(master_pid: int, command: list[str]) -> int:
    """Run command and monitor master process; return process exit code."""
    # Start the actual worker process
    print(f"[Distributed] Launching worker command: {' '.join(command)}")
    worker_process = launch_process_with_timeout(command, timeout_seconds=10.0)
    
    print(f"[Distributed] Started worker PID: {worker_process.pid}")
    print(f"[Distributed] Monitoring master PID: {master_pid}")
    
    # Write worker PID to a file so parent can track it
    monitor_pid = os.getpid()
    pid_info_file = os.environ.get('WORKER_PID_FILE')
    if pid_info_file:
        try:
            with open(pid_info_file, 'w') as f:
                f.write(f"{monitor_pid},{worker_process.pid}")
            print(f"[Distributed] Wrote PID info to {pid_info_file}")
        except Exception as e:
            raise RuntimeError(f"Could not write PID file '{pid_info_file}': {e}") from e
    
    # Define cleanup function
    should_stop = False
    monitor_exit_code = 0

    def cleanup_worker(
        signum: int | None = None,
        frame: Any | None = None,
        exit_code: int = 0,
    ) -> None:
        """Clean up worker process when monitor is terminated."""
        nonlocal should_stop
        nonlocal monitor_exit_code
        _ = frame
        should_stop = True
        monitor_exit_code = int(exit_code)

        if signum:
            print(f"\n[Distributed] Received signal {signum}, terminating worker...")
        else:
            print("\n[Distributed] Terminating worker...")
        
        if worker_process.poll() is None:  # Still running
            try:
                terminate_process(worker_process, timeout=PROCESS_TERMINATION_TIMEOUT)
            except TimeoutError:
                print("[Distributed] Worker did not terminate within timeout.")
        
        print("[Distributed] Worker terminated.")

    def _signal_handler(signum, frame):
        cleanup_worker(signum=signum, frame=frame, exit_code=0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGHUP, _signal_handler)
    
    # Monitor loop
    check_interval = WORKER_CHECK_INTERVAL
    
    try:
        while True:
            if should_stop:
                break

            # Check if worker is still running
            if worker_process.poll() is not None:
                print(f"[Distributed] Worker process exited with code: {worker_process.returncode}")
                monitor_exit_code = int(worker_process.returncode or 0)
                break
            
            # Check if master is still alive
            if not is_process_alive(master_pid):
                print(f"[Distributed] Master process {master_pid} is no longer running. Terminating worker...")
                cleanup_worker(exit_code=0)
                break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        cleanup_worker(exit_code=0)

    return monitor_exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for worker monitor script."""
    args = list(sys.argv[1:] if argv is None else argv)

    # Get master PID from environment
    master_pid = os.environ.get('COMFYUI_MASTER_PID')
    if not master_pid:
        print("[Distributed] Error: COMFYUI_MASTER_PID not set")
        return 1
    
    try:
        master_pid = int(master_pid)
    except ValueError:
        print(f"[Distributed] Error: Invalid master PID: {master_pid}")
        return 1
    
    # Get the actual command to run (all remaining arguments)
    if not args:
        print("[Distributed] Error: No command specified")
        return 1
    
    command = args
    
    # Start monitoring
    return monitor_and_run(master_pid, command)


if __name__ == "__main__":
    raise SystemExit(main())
