"""ComfyUI-Distributed's native V3 extension entrypoint."""
from comfy_api.v0_0_2 import ComfyExtension, io

from .nodes.v3 import NODES
from .runtime.bootstrap import initialize

WEB_DIRECTORY = './web'


class DistributedExtension(ComfyExtension):
    async def on_load(self) -> None:
        initialize()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> DistributedExtension:
    return DistributedExtension()


__all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']
