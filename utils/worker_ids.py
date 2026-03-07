import json
import re

_LEGACY_INDEX_TOKEN_RE = re.compile(r"^(?:worker_)?\d+$")


def _require_canonical_worker_id(value: object) -> str:
    worker_id = str(value).strip()
    if not worker_id:
        raise ValueError("enabled_worker_ids cannot contain blank IDs")
    if _LEGACY_INDEX_TOKEN_RE.match(worker_id):
        raise ValueError(
            f"enabled_worker_ids entry '{worker_id}' uses a legacy index token; "
            "provide stable explicit worker IDs"
        )
    return worker_id


def require_enabled_worker_ids(enabled_worker_ids: list[str] | None) -> list[str]:
    """Strictly validate canonical enabled-worker list."""
    if enabled_worker_ids is None:
        return []
    if not isinstance(enabled_worker_ids, list):
        raise ValueError("enabled_worker_ids must decode to a list")

    workers = []
    seen = set()
    for worker_id in enabled_worker_ids:
        worker_id_str = _require_canonical_worker_id(worker_id)
        if worker_id_str in seen:
            continue
        seen.add(worker_id_str)
        workers.append(worker_id_str)
    return workers


def decode_enabled_worker_ids(value: str | list[str] | None) -> list[str]:
    """Decode boundary payload into canonical enabled-worker list."""
    raw = json.loads(value) if isinstance(value, str) else value
    return require_enabled_worker_ids(raw)


def coerce_enabled_worker_ids(enabled_worker_ids: str | list[str] | None) -> list[str]:
    """Best-effort boundary decoding for permissive internal call sites."""
    try:
        return decode_enabled_worker_ids(enabled_worker_ids)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def parse_worker_index(
    worker_id: str,
    enabled_worker_ids: list[str],
) -> int | None:
    """Resolve worker index from canonical ID or legacy index token."""
    workers = list(enabled_worker_ids or [])
    worker_token = str(worker_id).strip()
    if not worker_token:
        return None

    if workers and worker_token in workers:
        return workers.index(worker_token)

    normalized = worker_token[len("worker_") :] if worker_token.startswith("worker_") else worker_token

    try:
        parsed_index = int(normalized)
    except (TypeError, ValueError):
        return None
    if parsed_index < 0:
        return None
    return parsed_index


def resolve_worker_identity(
    worker_id: str,
    enabled_worker_ids: list[str],
) -> tuple[str, int | None]:
    """Return canonical worker id and resolved index when possible."""
    worker_token = str(worker_id).strip()
    worker_index = parse_worker_index(worker_token, enabled_worker_ids)
    workers = list(enabled_worker_ids or [])
    if worker_index is None:
        return worker_token, None
    if 0 <= worker_index < len(workers):
        return workers[worker_index], worker_index
    return worker_token, worker_index


def worker_value_key(
    worker_id: str,
    enabled_worker_ids: list[str],
) -> str:
    """Return a stable key used for worker-value maps."""
    worker_token, worker_index = resolve_worker_identity(worker_id, enabled_worker_ids)
    if worker_index is not None:
        return str(worker_index + 1)
    return worker_token
