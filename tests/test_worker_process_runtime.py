import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


def _load_process_module(module_filename: str):
    module_path = Path(__file__).resolve().parents[1] / "workers" / "process" / module_filename
    package_name = "dist_proc_testpkg"
    module_name = module_filename[:-3]

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    workers_pkg = types.ModuleType(f"{package_name}.workers")
    workers_pkg.__path__ = []
    sys.modules[f"{package_name}.workers"] = workers_pkg

    process_pkg = types.ModuleType(f"{package_name}.workers.process")
    process_pkg.__path__ = []
    sys.modules[f"{package_name}.workers.process"] = process_pkg

    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = []
    sys.modules[f"{package_name}.utils"] = utils_pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    process_module = types.ModuleType(f"{package_name}.utils.process")
    process_module.get_python_executable = lambda: "/usr/bin/test-python"
    sys.modules[f"{package_name}.utils.process"] = process_module

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.workers.process.{module_name}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


root_discovery_module = _load_process_module("root_discovery.py")
launch_builder_module = _load_process_module("launch_builder.py")


class ComfyRootDiscoveryTests(unittest.TestCase):
    def test_prefers_loaded_comfyui_module_path(self):
        discovery = root_discovery_module.ComfyRootDiscovery()
        server_module = types.SimpleNamespace(__file__="/opt/ComfyUI/server.py")

        def fake_exists(path):
            return path == "/opt/ComfyUI/main.py"

        with patch.dict(sys.modules, {"server": server_module}, clear=False), \
             patch.object(root_discovery_module.os.path, "exists", side_effect=fake_exists), \
             patch.dict(root_discovery_module.os.environ, {}, clear=True):
            self.assertEqual(discovery.find_comfy_root(), "/opt/ComfyUI")


class LaunchCommandBuilderTests(unittest.TestCase):
    def test_inherits_runtime_layout_args_for_desktop(self):
        builder = launch_builder_module.LaunchCommandBuilder()
        runtime_args = Namespace(
            listen="127.0.0.1",
            base_directory="C:/Users/test/ComfyUI",
            temp_directory=None,
            input_directory="C:/Users/test/ComfyUI/input",
            output_directory="C:/Users/test/ComfyUI/output",
            user_directory="C:/Users/test/ComfyUI/user",
            front_end_root="C:/Program Files/ComfyUI/web_custom_versions/desktop_app",
            extra_model_paths_config=[["C:/Users/test/AppData/Roaming/ComfyUI/extra_models_config.yaml"]],
            enable_manager=True,
            disable_manager_ui=False,
            enable_manager_legacy_ui=False,
            windows_standalone_build=True,
            log_stdout=True,
            verbose="INFO",
            enable_cors_header="*",
        )
        comfy_module = types.ModuleType("comfy")
        comfy_cli_args = types.ModuleType("comfy.cli_args")
        comfy_cli_args.args = runtime_args

        worker_config = {
            "port": 9001,
            "extra_args": "--preview-method auto",
        }

        def fake_exists(path):
            return path == "/desktop/ComfyUI/main.py"

        with patch.dict(
            sys.modules,
            {"comfy": comfy_module, "comfy.cli_args": comfy_cli_args},
            clear=False,
        ), patch.object(launch_builder_module.os.path, "exists", side_effect=fake_exists):
            cmd = builder.build_launch_command(worker_config, "/desktop/ComfyUI")

        self.assertEqual(cmd[:2], ["/usr/bin/test-python", "/desktop/ComfyUI/main.py"])
        self.assertIn("--listen", cmd)
        self.assertIn("127.0.0.1", cmd)
        self.assertIn("--base-directory", cmd)
        self.assertIn("C:/Users/test/ComfyUI", cmd)
        self.assertIn("--input-directory", cmd)
        self.assertIn("--output-directory", cmd)
        self.assertIn("--user-directory", cmd)
        self.assertIn("--front-end-root", cmd)
        self.assertIn("--extra-model-paths-config", cmd)
        self.assertIn("C:/Users/test/AppData/Roaming/ComfyUI/extra_models_config.yaml", cmd)
        self.assertIn("--enable-manager", cmd)
        self.assertIn("--windows-standalone-build", cmd)
        self.assertIn("--log-stdout", cmd)
        self.assertIn("--disable-auto-launch", cmd)
        self.assertIn("--enable-cors-header", cmd)
        self.assertIn("*", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("9001", cmd)
        self.assertNotIn("--auto-launch", cmd)


if __name__ == "__main__":
    unittest.main()
