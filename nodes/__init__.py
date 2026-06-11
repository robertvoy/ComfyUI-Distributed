from .utilities import (
    DistributedSeed,
    DistributedModelName,
    DistributedValue,
    ImageBatchDivider,
    AudioBatchDivider,
    DistributedEmptyImage,
    AnyType,
    ByPassTypeTuple,
    any_type,
)
from .collector import DistributedCollectorNode
from .ltx_tiled_sampler import (
    NODE_CLASS_MAPPINGS as LTX_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as LTX_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    "DistributedCollector": DistributedCollectorNode,
    "DistributedSeed": DistributedSeed,
    "DistributedModelName": DistributedModelName,
    "DistributedValue": DistributedValue,
    "ImageBatchDivider": ImageBatchDivider,
    "AudioBatchDivider": AudioBatchDivider,
    "DistributedEmptyImage": DistributedEmptyImage,
    **LTX_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DistributedCollector": "Distributed Collector",
    "DistributedSeed": "Distributed Seed",
    "DistributedModelName": "Distributed Model Name",
    "DistributedValue": "Distributed Value",
    "ImageBatchDivider": "Image Batch Divider",
    "AudioBatchDivider": "Audio Batch Divider",
    "DistributedEmptyImage": "Distributed Empty Image",
    **LTX_DISPLAY_NAME_MAPPINGS,
}
