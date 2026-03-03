"""Cloudflared binary discovery and download helpers."""

import os
import platform
import shutil
import stat
from urllib import error as urlerror
from urllib import request
from urllib.parse import urlsplit

from ..logging import debug_log


def _get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _get_cloudflared_dir():
    return os.path.join(_get_project_root(), "bin")


def _get_platform_binary_name():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if "arm" in machine:
            return "cloudflared-windows-arm64.exe"
        return "cloudflared-windows-amd64.exe"
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "cloudflared-darwin-arm64"
        return "cloudflared-darwin-amd64"
    if system == "linux":
        if machine in ("arm64", "aarch64"):
            return "cloudflared-linux-arm64"
        return "cloudflared-linux-amd64"

    raise RuntimeError(f"Unsupported platform for cloudflared: {system}/{machine}")


def _get_binary_path(bin_dir=None):
    bin_dir = bin_dir or _get_cloudflared_dir()
    binary_name = "cloudflared.exe" if platform.system().lower() == "windows" else "cloudflared"
    return os.path.join(bin_dir, binary_name)


def _get_release_download_base() -> str:
    """Resolve cloudflared release download base URL from environment/config."""
    configured = (os.environ.get("CLOUDFLARED_RELEASE_BASE") or "").strip()
    if configured:
        return configured.rstrip("/")
    return "github.com/cloudflare/cloudflared/releases/latest/download"


def _download_cloudflared():
    asset = _get_platform_binary_name()
    base = _get_release_download_base()
    if "://" not in base:
        scheme = (os.environ.get("CLOUDFLARED_RELEASE_SCHEME") or "https").strip() or "https"
        base = f"{scheme}://{base}"
    url = f"{base}/{asset}"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Unsupported cloudflared download URL scheme: {parsed.scheme}")

    bin_dir = _get_cloudflared_dir()
    os.makedirs(bin_dir, exist_ok=True)
    target_path = _get_binary_path(bin_dir)

    debug_log(f"Downloading cloudflared from {url}")
    try:
        with request.urlopen(url, timeout=30) as resp:  # nosec B310 - scheme is allowlisted above
            with open(target_path, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urlerror.URLError as exc:
        raise RuntimeError(f"Failed to download cloudflared: {exc}") from exc

    st = os.stat(target_path)
    os.chmod(target_path, st.st_mode | stat.S_IEXEC)
    debug_log(f"Downloaded cloudflared to {target_path}")
    return target_path


def ensure_binary() -> str:
    """Return a usable cloudflared binary path, downloading if necessary."""
    env_path = os.environ.get("CLOUDFLARED_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    local_candidate = _get_binary_path()
    if os.path.exists(local_candidate):
        return local_candidate

    path_binary = shutil.which("cloudflared")
    if path_binary:
        return path_binary

    return _download_cloudflared()
