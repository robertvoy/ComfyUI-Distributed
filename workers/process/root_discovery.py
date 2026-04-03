import os
import sys

from ...utils.logging import debug_log, log


class ComfyRootDiscovery:
    """Resolve the ComfyUI root directory across local and container layouts."""

    def _find_root_from_loaded_modules(self):
        """Use already-imported ComfyUI modules to locate the runtime root."""
        for module_name in ("server", "folder_paths", "main"):
            module = sys.modules.get(module_name)
            module_file = getattr(module, "__file__", None)
            if not module_file:
                continue

            candidate = os.path.dirname(os.path.abspath(module_file))
            if os.path.exists(os.path.join(candidate, "main.py")):
                debug_log(f"Found ComfyUI root via loaded module {module_name}: {candidate}")
                return candidate

        return None

    def find_comfy_root(self):
        # Start from current file location.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_root = os.path.dirname(os.path.dirname(current_dir))

        # Method 1: Check for environment variable override.
        env_root = os.environ.get("COMFYUI_ROOT")
        if env_root and os.path.exists(os.path.join(env_root, "main.py")):
            debug_log(f"Found ComfyUI root via COMFYUI_ROOT environment variable: {env_root}")
            return env_root

        # Method 2: Inspect the already-loaded ComfyUI runtime modules.
        runtime_root = self._find_root_from_loaded_modules()
        if runtime_root:
            return runtime_root

        # Method 3: Try going up from custom_nodes directory.
        if os.path.exists(os.path.join(potential_root, "main.py")):
            debug_log(f"Found ComfyUI root via directory traversal: {potential_root}")
            return potential_root

        # Method 4: Look for common Docker paths.
        docker_paths = [
            "/basedir",
            "/ComfyUI",
            "/app",
            "/workspace/ComfyUI",
            "/comfyui",
            "/opt/ComfyUI",
            "/workspace",
        ]
        for path in docker_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "main.py")):
                debug_log(f"Found ComfyUI root in Docker path: {path}")
                return path

        # Method 5: Search upwards for main.py.
        search_dir = current_dir
        for _ in range(5):
            if os.path.exists(os.path.join(search_dir, "main.py")):
                debug_log(f"Found ComfyUI root via upward search: {search_dir}")
                return search_dir
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent

        # Method 6: Try to import and use folder_paths.
        try:
            import folder_paths

            if hasattr(folder_paths, "base_path") and os.path.exists(
                os.path.join(folder_paths.base_path, "main.py")
            ):
                debug_log(f"Found ComfyUI root via folder_paths: {folder_paths.base_path}")
                return folder_paths.base_path
        except Exception as exc:
            debug_log(f"folder_paths root detection failed: {exc}")

        log("Warning: Could not reliably determine ComfyUI root directory")
        log(f"Current directory: {current_dir}")
        log(f"Initial guess was: {potential_root}")
        return potential_root
