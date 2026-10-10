import glob
import hashlib
import os
import shlex
import shutil

from ...utils.logging import debug_log
from ...utils.process import get_python_executable


class LaunchCommandBuilder:
    """Build command-lines for launching worker ComfyUI processes."""

    def _extend_arg(self, cmd, flag, value):
        if value in (None, "", [], ()):
            return
        cmd.extend([flag, str(value)])

    def _extend_grouped_args(self, cmd, flag, values):
        for group in values or []:
            flattened = [str(item) for item in group if item]
            if flattened:
                cmd.append(flag)
                cmd.extend(flattened)

    def _get_runtime_args(self):
        try:
            from comfy.cli_args import args
            return args
        except Exception as exc:
            debug_log(f"Could not read current ComfyUI CLI args for worker launch: {exc}")
            return None

    def _build_runtime_launch_args(self):
        args = self._get_runtime_args()
        if args is None:
            return []

        inherited = []
        self._extend_arg(inherited, "--listen", getattr(args, "listen", None))
        self._extend_arg(inherited, "--base-directory", getattr(args, "base_directory", None))
        self._extend_arg(inherited, "--temp-directory", getattr(args, "temp_directory", None))
        self._extend_arg(inherited, "--input-directory", getattr(args, "input_directory", None))
        self._extend_arg(inherited, "--output-directory", getattr(args, "output_directory", None))
        self._extend_arg(inherited, "--user-directory", getattr(args, "user_directory", None))
        self._extend_arg(inherited, "--front-end-root", getattr(args, "front_end_root", None))
        self._extend_grouped_args(
            inherited,
            "--extra-model-paths-config",
            getattr(args, "extra_model_paths_config", None),
        )

        if getattr(args, "enable_manager", False):
            inherited.append("--enable-manager")
        if getattr(args, "disable_manager_ui", False):
            inherited.append("--disable-manager-ui")
        if getattr(args, "enable_manager_legacy_ui", False):
            inherited.append("--enable-manager-legacy-ui")
        if getattr(args, "windows_standalone_build", False):
            inherited.append("--windows-standalone-build")
        if getattr(args, "log_stdout", False):
            inherited.append("--log-stdout")

        verbose = getattr(args, "verbose", None)
        if verbose and verbose != "INFO":
            inherited.extend(["--verbose", str(verbose)])

        return inherited

    def _find_windows_terminal(self):
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

    @staticmethod
    def _arg_value(cmd, flag):
        """Return the last value, matching argparse's override semantics."""
        value = None
        for index, arg in enumerate(cmd):
            if arg == flag and index + 1 < len(cmd):
                value = cmd[index + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
        return value

    def _add_worker_database(self, cmd, worker_config, comfy_root):
        args = self._get_runtime_args()
        # The parsed namespace advertises support without launching ComfyUI
        # (or importing another checkout's argument parser).
        if args is None or not hasattr(args, "database_url"):
            return
        if any(arg.split("=", 1)[0] in ("--database-url", "--disable-assets") for arg in cmd):
            return

        user_directory = self._arg_value(cmd, "--user-directory")
        base_directory = self._arg_value(cmd, "--base-directory")
        if not user_directory:
            user_directory = os.path.join(base_directory or comfy_root, "user")
        if not os.path.isabs(user_directory):
            # Popen uses comfy_root as cwd, which can differ from the master's.
            user_directory = os.path.join(comfy_root, user_directory)
        database_directory = os.path.abspath(os.path.join(user_directory, "distributed", "workers"))
        # Hash the immutable ID: names/GPU/ports may change, and IDs must not
        # become filesystem paths or collide after filename sanitization.
        worker_id = str(worker_config["id"])
        filename = hashlib.sha256(worker_id.encode("utf-8")).hexdigest() + ".db"
        database_path = os.path.join(database_directory, filename).replace(os.sep, "/")
        # Use the same URL codec as ComfyUI; raw '?' and '%xx' can otherwise
        # truncate/decode the path and make distinct workers share a database.
        from sqlalchemy.engine import URL, make_url
        database_url = URL.create("sqlite", database=database_path).render_as_string()
        if make_url(database_url).database != database_path:
            raise ValueError(
                "Installed SQLAlchemy cannot safely encode the worker database path; "
                "use a user directory without '?' or an explicit --database-url with a safe path"
            )
        os.makedirs(database_directory, exist_ok=True)
        cmd.extend(["--database-url", database_url])

    def build_launch_command(self, worker_config, comfy_root):
        """Build the command to launch a worker."""
        main_py = os.path.join(comfy_root, "main.py")

        if os.path.exists(main_py):
            cmd = [
                get_python_executable(),
                main_py,
            ]
            cmd.extend(self._build_runtime_launch_args())
            cmd.extend(["--port", str(worker_config["port"])])

            current_args = self._get_runtime_args()
            current_cors = getattr(current_args, "enable_cors_header", None) if current_args else None
            cmd.append("--enable-cors-header")
            if current_cors is not None:
                cmd.append(str(current_cors))

            if "--disable-auto-launch" not in cmd:
                cmd.append("--disable-auto-launch")

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

        if int(self._arg_value(cmd, "--port")) != int(worker_config["port"]):
            raise ValueError("extra_args must not override the configured worker port; set the port in the UI/config instead")
        self._add_worker_database(cmd, worker_config, comfy_root)
        return cmd
