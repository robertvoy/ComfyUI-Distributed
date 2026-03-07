from typing import Any


def build_worker_identity_hidden_inputs() -> dict[str, tuple[str, dict[str, Any]]]:
    """Hidden-input contract for worker identity context."""
    return {
        "is_worker": ("BOOLEAN", {"default": False}),
        "enabled_worker_ids": ("STRING", {"default": "[]"}),
        "worker_id": ("STRING", {"default": ""}),
    }


def build_distributed_hidden_inputs(
    *,
    include_assigned_branch: bool = False,
    assigned_branch_max: int = 9,
    include_worker_batch_size: bool = False,
    include_pass_through: bool = False,
) -> dict[str, tuple[str, dict[str, Any]]]:
    hidden_inputs: dict[str, tuple[str, dict[str, Any]]] = {
        "multi_job_id": ("STRING", {"default": ""}),
        "master_url": ("STRING", {"default": ""}),
        "delegate_only": ("BOOLEAN", {"default": False}),
    }
    hidden_inputs.update(build_worker_identity_hidden_inputs())
    if include_assigned_branch:
        hidden_inputs["assigned_branch"] = (
            "INT",
            {"default": -1, "min": -1, "max": assigned_branch_max},
        )
    if include_worker_batch_size:
        hidden_inputs["worker_batch_size"] = (
            "INT",
            {"default": 1, "min": 1, "max": 1024},
        )
    if include_pass_through:
        hidden_inputs["pass_through"] = ("BOOLEAN", {"default": False})
    return hidden_inputs
