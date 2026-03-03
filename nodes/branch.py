from typing import Any

from .utilities import any_type


class DistributedBranch:
    @classmethod
    def INPUT_TYPES(cls: type["DistributedBranch"]) -> dict[str, Any]:
        return {
            "required": {
                "input": (any_type,),
                "num_branches": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1}),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
                "assigned_branch": ("INT", {"default": -1, "min": -1, "max": 9}),
                "multi_job_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = tuple([any_type] * 10)
    RETURN_NAMES = tuple([f"branch_{idx + 1}" for idx in range(10)])
    FUNCTION = "branch"
    CATEGORY = "utils"

    def branch(
        self,
        input: Any,
        num_branches: int = 2,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        return tuple([input] * 10)
