"""Shared rules for provider capacity and durable recovery protection.

A local task ending is not evidence that provider work ended. RESOLVED is an
explicit operator acknowledgment, never a provider-reported state.
"""

PROVIDER_TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})
TERMINAL = PROVIDER_TERMINAL | {"RESOLVED"}
PROVIDER_STATES = PROVIDER_TERMINAL | {"IN_QUEUE", "IN_PROGRESS"}


def terminal_provider_status(response):
    status = response.get("status") if isinstance(response, dict) else None
    return status if isinstance(status, str) and status in PROVIDER_TERMINAL else None


def is_unresolved(record):
    return record.get("status") not in TERMINAL


def occupies_capacity(record):
    # WAITING is journalled before admission and cannot have been submitted.
    # Every other unresolved state is conservatively reserved, including a
    # submission whose provider ID was lost and an unconfirmed cancellation.
    return is_unresolved(record) and (record.get("status") != "WAITING" or bool(record.get("job_id")))


def belongs_to_pool(record, worker_id):
    return str(record.get("worker_id")) == str(worker_id)
