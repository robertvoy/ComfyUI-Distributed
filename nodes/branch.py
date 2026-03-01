from .utilities import any_type


class DistributedBranch:
    @classmethod
    def INPUT_TYPES(cls):
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

    def branch(self, input, num_branches=2, is_worker=False, worker_id="", assigned_branch=-1, multi_job_id=""):
        return tuple([input] * 10)
