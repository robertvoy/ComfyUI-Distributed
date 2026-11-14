"""Cloudflare tunnel state persistence helpers."""

from ..config import load_config, save_config
from ..network import normalize_host


def _get_tunnel_config(cfg):
    tunnel_cfg = cfg.get("tunnel", {})
    if isinstance(tunnel_cfg, dict):
        return tunnel_cfg
    return {}


def load_tunnel_state():
    cfg = load_config()
    tunnel_cfg = _get_tunnel_config(cfg)
    master_cfg = cfg.get("master", {}) if isinstance(cfg.get("master", {}), dict) else {}
    return {
        "status": tunnel_cfg.get("status", "stopped"),
        "public_url": tunnel_cfg.get("public_url") or None,
        "pid": tunnel_cfg.get("pid"),
        "pid_create_time": tunnel_cfg.get("pid_create_time"),
        "log_file": tunnel_cfg.get("log_file"),
        "previous_master_host": tunnel_cfg.get("previous_master_host"),
        "master_host": master_cfg.get("host"),
    }


# Sentinel for "leave unchanged"; None is a real value for the pid fields.
_UNSET = object()


def persist_tunnel_state(
    status=None,
    public_url=None,
    pid=_UNSET,
    log_file=None,
    previous_host=None,
    master_host=None,
    pid_create_time=_UNSET,
):
    cfg = load_config()
    tunnel_cfg = _get_tunnel_config(cfg)

    if status is not None:
        tunnel_cfg["status"] = status
    if public_url is not None:
        tunnel_cfg["public_url"] = public_url
    if pid is not _UNSET:
        tunnel_cfg["pid"] = pid
        if pid is None:
            # A cleared PID must not keep the identity of the old process.
            tunnel_cfg["pid_create_time"] = None
    if pid_create_time is not _UNSET:
        tunnel_cfg["pid_create_time"] = pid_create_time
    if log_file is not None:
        tunnel_cfg["log_file"] = log_file
    if previous_host is not None:
        tunnel_cfg["previous_master_host"] = previous_host
    if master_host is not None:
        cfg.setdefault("master", {})["host"] = master_host

    cfg["tunnel"] = tunnel_cfg
    save_config(cfg)


def clear_tunnel_state(log_file=None, previous_host=None, master_host=None):
    persist_tunnel_state(
        status="stopped",
        public_url="",
        pid=None,  # Actually written: a stale PID could later belong to another process.
        log_file=log_file,
        previous_host=previous_host,
        master_host=master_host,
    )


def resolve_restore_master_host(previous_master_host):
    """Determine whether master host should be restored after tunnel stop."""
    cfg = load_config()
    tunnel_cfg = _get_tunnel_config(cfg)
    active_url = tunnel_cfg.get("public_url")
    current_master_host = (cfg.get("master") or {}).get("host")

    if not active_url:
        return None

    active_host = normalize_host(active_url)
    current_host = normalize_host(current_master_host)
    if current_host == active_host:
        return previous_master_host or ""
    return None
