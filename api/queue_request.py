from dataclasses import dataclass
from typing import Any

try:
    from ..utils.worker_ids import require_enabled_worker_ids
except ImportError:  # pragma: no cover - supports direct module loading in isolated tests.
    from utils.worker_ids import require_enabled_worker_ids


@dataclass(frozen=True)
class QueueRequestPayload:
    prompt: dict[str, Any]
    workflow_meta: dict[str, Any] | None
    client_id: str
    delegate_master: bool | None
    enabled_worker_ids: list[str]
    trace_execution_id: str | None


def parse_queue_request_payload(data: Any) -> QueueRequestPayload:
    """Parse and validate /distributed/queue payload into a normalized shape."""
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object body")

    prompt = data.get("prompt")
    # Auto-prepare is always on server-side; keep the field for wire compatibility.
    if prompt is None:
        workflow_payload = data.get("workflow")
        if isinstance(workflow_payload, dict):
            candidate_prompt = workflow_payload.get("prompt")
            if isinstance(candidate_prompt, dict):
                prompt = candidate_prompt

    if not isinstance(prompt, dict):
        raise ValueError("Field 'prompt' must be an object")

    workflow_meta = data.get("workflow")
    if workflow_meta is not None and not isinstance(workflow_meta, dict):
        raise ValueError("Field 'workflow' must be an object when provided")

    enabled_ids_raw = data.get("enabled_worker_ids")
    workers_field = data.get("workers")
    if enabled_ids_raw is None and workers_field is not None:
        if not isinstance(workers_field, list):
            raise ValueError("Field 'workers' must be a list when provided")
        enabled_ids_raw = []
        for entry in workers_field:
            worker_id = entry.get("id") if isinstance(entry, dict) else entry
            if worker_id is not None:
                enabled_ids_raw.append(str(worker_id))

    if enabled_ids_raw is None:
        raise ValueError("enabled_worker_ids required")
    else:
        if not isinstance(enabled_ids_raw, list):
            raise ValueError("enabled_worker_ids must be a list of worker IDs")
        try:
            enabled_ids = require_enabled_worker_ids(enabled_ids_raw)
        except ValueError as exc:
            raise ValueError(f"enabled_worker_ids invalid: {exc}") from exc

    delegate_master = data.get("delegate_master")
    if delegate_master is not None and not isinstance(delegate_master, bool):
        raise ValueError("delegate_master must be a boolean when provided")

    client_id = data.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("client_id required")
    client_id = client_id.strip()

    trace_execution_id = data.get("trace_execution_id")
    if trace_execution_id is not None:
        if not isinstance(trace_execution_id, str):
            raise ValueError("trace_execution_id must be a string when provided")
        trace_execution_id = trace_execution_id.strip() or None

    return QueueRequestPayload(
        prompt=prompt,
        workflow_meta=workflow_meta,
        client_id=client_id,
        delegate_master=delegate_master,
        enabled_worker_ids=enabled_ids,
        trace_execution_id=trace_execution_id,
    )
