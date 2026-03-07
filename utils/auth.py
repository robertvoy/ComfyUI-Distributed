from __future__ import annotations

import hmac
from typing import Any


AUTH_HEADER_NAME = "X-Distributed-Token"


def get_distributed_api_token(config: dict[str, Any] | None) -> str:
    """Return configured distributed API token or empty string when unset."""
    settings = (config or {}).get("settings", {}) if isinstance(config, dict) else {}
    token = settings.get("distributed_api_token") or settings.get("api_token")
    return str(token).strip() if token is not None else ""


def distributed_auth_headers(config: dict[str, Any] | None) -> dict[str, str]:
    """Build outbound auth headers for distributed internal API calls."""
    token = get_distributed_api_token(config)
    if not token:
        return {}
    return {AUTH_HEADER_NAME: token}


def is_authorized_request(request: Any, config: dict[str, Any] | None) -> bool:
    """Validate a request against the configured shared distributed token."""
    expected_token = get_distributed_api_token(config)
    if not expected_token:
        return True

    provided_token = ""
    headers = getattr(request, "headers", None)
    if headers is not None:
        provided_token = str(headers.get(AUTH_HEADER_NAME, "")).strip()

    return hmac.compare_digest(provided_token, expected_token)
