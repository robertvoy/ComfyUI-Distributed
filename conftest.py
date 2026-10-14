# conftest.py — project-level pytest configuration.
#
# Problem: custom_nodes/ComfyUI-Distributed/__init__.py uses relative imports
# (from .nodes.v3 import ...) that fail when pytest tries to import it as a
# standalone module during Package.setup() for the root package node.
#
# Fix: patch Package.setup() to skip the root-package's __init__.py import.
# All actual package context is provided by each test module via
# importlib.util.spec_from_file_location with synthetic stub packages.

from _pytest.python import Package

_orig_pkg_setup = Package.setup


def _patched_pkg_setup(self) -> None:
    # Skip the root package setup — its __init__.py uses relative imports
    # that require a parent package (ComfyUI's plugin loader) which is not
    # available in the test environment.
    if self.path == self.config.rootpath:
        return
    _orig_pkg_setup(self)


Package.setup = _patched_pkg_setup

collect_ignore = [
    "__init__.py",
    "distributed.py",
    "distributed_upscale.py",
]
