import glob
import os
import shlex
import shutil
from typing import Any

from ...utils.logging import debug_log
from ...utils.process import get_python_executable


class LaunchCommandBuilder:
    """Build command-lines for launching worker ComfyUI processes."""

    def _find_windows_terminal(self) -> str | None:
        """Find Windows Terminal executable."""
        possible_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\WindowsApps\Microsoft.WindowsTerminal_*\wt.exe"),
            "wt.exe",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
            if "*" in path:
                matches = glob.glob(path)
                if matches:
                    return matches[0]

        wt_path = shutil.which("wt")
        if wt_path:
            return wt_path
        return None

    def build_launch_command(self, worker_config: dict[str, Any], comfy_root: str) -> list[str]:
        """Build the command to launch a worker."""
        main_py = os.path.join(comfy_root, "main.py")

        if os.path.exists(main_py):
            cmd = [
                get_python_executable(),
                main_py,
                "--port",
                str(worker_config["port"]),
                "--enable-cors-header",
            ]
            debug_log(f"Using main.py: {main_py}")
        else:
            error_msg = f"Could not find main.py in {comfy_root}\n"
            error_msg += f"Searched for: {main_py}\n"
            error_msg += f"Directory contents of {comfy_root}:\n"
            try:
                if os.path.exists(comfy_root):
                    files = os.listdir(comfy_root)[:20]
                    error_msg += "  " + "\n  ".join(files)
                    if len(os.listdir(comfy_root)) > 20:
                        error_msg += f"\n  ... and {len(os.listdir(comfy_root)) - 20} more files"
                else:
                    error_msg += f"  Directory {comfy_root} does not exist!"
            except Exception as exc:
                error_msg += f"  Error listing directory: {exc}"

            error_msg += "\n\nPossible solutions:\n"
            error_msg += "1. Check if ComfyUI is installed in a different location\n"
            error_msg += "2. For Docker: ComfyUI might be in /ComfyUI or /app\n"
            error_msg += "3. Ensure the custom node is installed in the correct location\n"
            raise RuntimeError(error_msg)

        if worker_config.get("extra_args"):
            raw_args = worker_config["extra_args"].strip()
            if raw_args:
                extra_args_list = shlex.split(raw_args)
                forbidden_chars = set(";|>&<`$()[]{}*!?")
                for arg in extra_args_list:
                    if any(char in forbidden_chars for char in arg):
                        forbidden = "".join(forbidden_chars)
                        raise ValueError(f"Invalid characters in extra_args: {arg}. Forbidden: {forbidden}")
                cmd.extend(extra_args_list)

        return cmd
