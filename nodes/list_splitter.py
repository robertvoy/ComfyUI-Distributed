from .utilities import _chunk_bounds


class DistributedListSplitter:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {
                "participant_index": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "total_participants": ("INT", {"default": 1, "min": 1, "max": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "split"
    CATEGORY = "image"

    def split(self, images, participant_index=0, total_participants=1):
        if images is None:
            return ([],)

        image_list = images if isinstance(images, list) else [images]

        split_count = max(1, int(total_participants or 1))
        if split_count <= 1:
            return (list(image_list),)

        bounds = _chunk_bounds(len(image_list), split_count)
        index = int(participant_index or 0)
        if index < 0:
            index = 0
        if index >= split_count:
            index = split_count - 1

        start, end = bounds[index]
        return (list(image_list[start:end]),)
