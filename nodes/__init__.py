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
from .branch import DistributedBranch
from .branch_collector import DistributedBranchCollector

NODE_CLASS_MAPPINGS = {
    "DistributedCollector": DistributedCollectorNode,
    "DistributedBranch": DistributedBranch,
    "DistributedBranchCollector": DistributedBranchCollector,
    "DistributedSeed": DistributedSeed,
    "DistributedModelName": DistributedModelName,
    "DistributedValue": DistributedValue,
    "ImageBatchDivider": ImageBatchDivider,
    "AudioBatchDivider": AudioBatchDivider,
    "DistributedEmptyImage": DistributedEmptyImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DistributedCollector": "Distributed Collector",
    "DistributedBranch": "Distributed Branch",
    "DistributedBranchCollector": "Distributed Branch Collector",
    "DistributedSeed": "Distributed Seed",
    "DistributedModelName": "Distributed Model Name",
    "DistributedValue": "Distributed Value",
    "ImageBatchDivider": "Image Batch Divider",
    "AudioBatchDivider": "Audio Batch Divider",
    "DistributedEmptyImage": "Distributed Empty Image",
}
