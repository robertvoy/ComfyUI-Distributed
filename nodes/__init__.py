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
from .list_splitter import DistributedListSplitter
from .list_collector import DistributedListCollector
from .branch import DistributedBranch
from .join import DistributedJoin

NODE_CLASS_MAPPINGS = {
    "DistributedCollector": DistributedCollectorNode,
    "DistributedListSplitter": DistributedListSplitter,
    "DistributedListCollector": DistributedListCollector,
    "DistributedBranch": DistributedBranch,
    "DistributedJoin": DistributedJoin,
    "DistributedSeed": DistributedSeed,
    "DistributedModelName": DistributedModelName,
    "DistributedValue": DistributedValue,
    "ImageBatchDivider": ImageBatchDivider,
    "AudioBatchDivider": AudioBatchDivider,
    "DistributedEmptyImage": DistributedEmptyImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DistributedCollector": "Distributed Collector",
    "DistributedListSplitter": "Distributed List Splitter",
    "DistributedListCollector": "Distributed List Collector",
    "DistributedBranch": "Distributed Branch",
    "DistributedJoin": "Distributed Join",
    "DistributedSeed": "Distributed Seed",
    "DistributedModelName": "Distributed Model Name",
    "DistributedValue": "Distributed Value",
    "ImageBatchDivider": "Image Batch Divider",
    "AudioBatchDivider": "Audio Batch Divider",
    "DistributedEmptyImage": "Distributed Empty Image",
}
