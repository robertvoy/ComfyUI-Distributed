"""Cloudflare tunnel state persistence helpers."""

from dataclasses import dataclass
from typing import Any

from ..config import load_config, save_config
from ..network import normalize_host


@dataclass(frozen=True)
class TunnelStateUpdate:
    status: str | None = None
    public_url: str | None = None
    pid: int | None = None
    log_file: str | None = None
    previous_host: str | None = None
    master_host: str | None = None


def _get_tunnel_config(cfg: dict[str, Any]) -> dict[str, Any]:
    tunnel_cfg = cfg.get("tunnel", {})
    if isinstance(tunnel_cfg, dict):
        return tunnel_cfg
    return {}


def load_tunnel_state() -> dict[str, Any]:
    cfg = load_config()
    tunnel_cfg = _get_tunnel_config(cfg)
    master_cfg = cfg.get("master", {}) if isinstance(cfg.get("master", {}), dict) else {}
    return {
        "status": tunnel_cfg.get("status", "stopped"),
        "public_url": tunnel_cfg.get("public_url") or None,
        "pid": tunnel_cfg.get("pid"),
        "log_file": tunnel_cfg.get("log_file"),
        "previous_master_host": tunnel_cfg.get("previous_master_host"),
        "master_host": master_cfg.get("host"),
    }


def persist_tunnel_state(update: TunnelStateUpdate) -> None:
    cfg = load_config()
    tunnel_cfg = _get_tunnel_config(cfg)

    if update.status is not None:
        tunnel_cfg["status"] = update.status
    if update.public_url is not None:
        tunnel_cfg["public_url"] = update.public_url
    if update.pid is not None:
        tunnel_cfg["pid"] = update.pid
    if update.log_file is not None:
        tunnel_cfg["log_file"] = update.log_file
    if update.previous_host is not None:
        tunnel_cfg["previous_master_host"] = update.previous_host
    if update.master_host is not None:
        cfg.setdefault("master", {})["host"] = update.master_host

    cfg["tunnel"] = tunnel_cfg
    save_config(cfg)


def clear_tunnel_state(
    log_file: str | None = None,
    previous_host: str | None = None,
    master_host: str | None = None,
) -> None:
    persist_tunnel_state(
        TunnelStateUpdate(
            status="stopped",
            public_url="",
            pid=None,
            log_file=log_file,
            previous_host=previous_host,
            master_host=master_host,
        )
    )


def resolve_restore_master_host(previous_master_host: str | None) -> str | None:
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
