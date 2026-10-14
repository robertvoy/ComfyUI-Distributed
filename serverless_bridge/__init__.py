"""Standalone typed-media bridge; copy this directory into endpoint custom_nodes.

Importing codecs does not load ComfyUI, providers or Distributed bootstrap.
"""

async def comfy_entrypoint():
    from comfy_api.v0_0_2 import ComfyExtension
    from .nodes import MediaImportV1, MediaExportV1

    class MediaBridgeExtension(ComfyExtension):
        async def get_node_list(self):
            return [MediaImportV1, MediaExportV1]

    return MediaBridgeExtension()
