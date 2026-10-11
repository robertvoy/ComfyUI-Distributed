"""Public V3 Start/End nodes for dynamic per-image image-list execution."""
from __future__ import annotations

import json

from comfy_api.v0_0_2 import io

from ..utils.async_helpers import run_async_in_server_loop
from ..image_list.runtime import (
    create_master_job,
    normalize_image_list_inputs,
    normalize_worker_result,
    run_image_list_job,
    encode_result,
    unwrap_single,
)


class DistributedImageListStart(io.ComfyNode):
    """Materialize a batch once, then expose ordered per-item typed values."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DistributedImageListStart",
            display_name="Distributed Image List Start",
            category="image/distributed",
            description=("Starts one per-image processing region. Optional inputs explicitly choose shared "
                         "(exactly one value) or per-item (exactly one value per image) alignment."),
            inputs=[
                io.Image.Input("images", optional=True, tooltip="Image list or image batch; each frame becomes one worker job."),
                io.Image.Input("reference_image", optional=True),
                io.Combo.Input("reference_image_mode", options=["per_item", "shared"], default="per_item"),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input("mask_mode", options=["per_item", "shared"], default="per_item"),
                io.String.Input("text", optional=True),
                io.Combo.Input("text_mode", options=["per_item", "shared"], default="shared"),
                io.Float.Input("strength", optional=True, default=1.0),
                io.Combo.Input("strength_mode", options=["per_item", "shared"], default="shared"),
                io.Int.Input("seed", optional=True, default=0, min=0, max=0xffffffffffffffff),
                io.Combo.Input("seed_mode", options=["per_item", "shared"], default="shared"),
            ],
            outputs=[
                io.Image.Output(display_name="image", is_output_list=True),
                io.Image.Output(display_name="reference_image", is_output_list=True),
                io.Mask.Output(display_name="mask", is_output_list=True),
                io.String.Output(display_name="text", is_output_list=True),
                io.Float.Output(display_name="strength", is_output_list=True),
                io.Int.Output(display_name="seed", is_output_list=True),
                io.Int.Output(display_name="item_index", is_output_list=True),
                io.Custom("DISTRIBUTED_IMAGE_LIST_CONTEXT").Output(display_name="context", is_output_list=True),
            ],
            is_input_list=True,
            accept_all_inputs=True,
            not_idempotent=True,
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images=None, reference_image=None, reference_image_mode="per_item",
                mask=None, mask_mode="per_item", text=None, text_mode="shared",
                strength=None, strength_mode="shared", seed=None, seed_mode="shared",
                worker_item_token="", worker_context_id="", image_list_enabled_worker_ids="[]",
                image_list_master_callback="", image_list_execution_id="",
                image_list_workflow_meta=None):
        worker_item_token = unwrap_single(worker_item_token)
        worker_context_id = unwrap_single(worker_context_id)
        image_list_workflow_meta = unwrap_single(image_list_workflow_meta)
        if worker_item_token:
            item = run_async_in_server_loop(_consume_worker_item(worker_item_token), timeout=10)
            context_id = worker_context_id
            if not isinstance(context_id, str) or not context_id:
                raise RuntimeError("Worker image-list prompt is missing its paired context")
            return io.NodeOutput(
                [item["image"]], [item.get("reference_image")], [item.get("mask")],
                [item.get("text")], [item.get("strength")], [item.get("seed")],
                [item["index"]], [context_id],
            )

        items = normalize_image_list_inputs(
            images, reference_image, mask, text, strength, seed,
            reference_image_mode, mask_mode, text_mode, strength_mode, seed_mode,
        )
        prompt = getattr(cls.hidden, "prompt", None)
        start_id = str(getattr(cls.hidden, "unique_id", ""))
        if not prompt or not start_id:
            raise RuntimeError("Distributed Image List Start needs a submitted ComfyUI prompt context")
        try:
            worker_ids = json.loads(str(unwrap_single(image_list_enabled_worker_ids or "[]")))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid preflighted image-list worker list") from exc
        if not isinstance(worker_ids, list) or not all(isinstance(item, str) for item in worker_ids):
            raise RuntimeError("Invalid preflighted image-list worker list")
        from ..image_list.runtime import validate_image_list_regions
        regions = validate_image_list_regions(prompt)
        region = next((row for row in regions if row["start_id"] == start_id), None)
        if region is None:
            raise RuntimeError(f"No safe paired Image List End was found for Start node {start_id}")
        context_id = create_master_job(
            start_id, region["end_id"], prompt, items, worker_ids,
            str(unwrap_single(image_list_master_callback or "")),
            image_list_workflow_meta if isinstance(image_list_workflow_meta, dict) else None,
        )
        return io.NodeOutput(
            [item["image"] for item in items],
            [item.get("reference_image") for item in items],
            [item.get("mask") for item in items],
            [item.get("text") for item in items],
            [item.get("strength") for item in items],
            [item.get("seed") for item in items],
            [item["index"] for item in items],
            [context_id for _ in items],
        )


async def _consume_worker_item(token):
    # Item state is local to each worker process; source nodes consume it exactly once.
    from ..image_list.runtime import consume_worker_item
    return consume_worker_item(token)


class DistributedImageListEnd(io.ComfyNode):
    """Collect branch results in source order without evaluating the branch on master."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DistributedImageListEnd",
            display_name="Distributed Image List End",
            category="image/distributed",
            description="Closes a paired image-list region and returns ordered image, mask, and text lists.",
            inputs=[
                io.Image.Input("images", lazy=True),
                io.Mask.Input("masks", optional=True, lazy=True),
                io.String.Input("texts", optional=True, lazy=True, force_input=True),
                io.Custom("DISTRIBUTED_IMAGE_LIST_CONTEXT").Input("context"),
            ],
            outputs=[
                io.Image.Output(display_name="image_list", is_output_list=True),
                io.Mask.Output(display_name="masks", is_output_list=True),
                io.String.Output(display_name="texts", is_output_list=True),
            ],
            is_input_list=True,
            is_output_node=True,
            accept_all_inputs=True,
            not_idempotent=True,
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, images=None, masks=None, texts=None, context=None,
                          worker_result_token="", **kwargs):
        worker_result_token = unwrap_single(worker_result_token)
        if worker_result_token:
            # Decide which optional sockets are connected from the actual worker prompt;
            # absent optional inputs must not be demanded from Comfy's lazy executor.
            prompt = getattr(cls.hidden, "prompt", {}) or {}
            end_inputs = next((node.get("inputs", {}) for node in prompt.values()
                               if isinstance(node, dict)
                               and node.get("class_type") == "DistributedImageListEnd"
                               and node.get("inputs", {}).get("worker_result_token") == worker_result_token), {})
            required = []
            for name, value in (("images", images), ("masks", masks), ("texts", texts)):
                connected = name == "images" or name in end_inputs
                missing = value is None or (isinstance(value, (list, tuple)) and all(v is None for v in value))
                if connected and missing:
                    required.append(name)
            return required
        # Master only needs Start's context. Never request branch outputs here.
        if context is None or (isinstance(context, (tuple, list)) and all(item is None for item in context)):
            return ["context"]
        return []

    @classmethod
    def execute(cls, images=None, masks=None, texts=None, context=None,
                worker_result_token="", worker_item_index=-1, master_callback_url=""):
        worker_result_token = unwrap_single(worker_result_token)
        master_callback_url = unwrap_single(master_callback_url)
        if worker_result_token:
            context_values = [item for item in (context if isinstance(context, (list, tuple)) else [context])
                              if isinstance(item, str) and item]
            if not context_values or len(set(context_values)) != 1:
                raise RuntimeError("Worker Image List End is missing its paired context")
            result = normalize_worker_result(images, masks, texts)
            payload = encode_result(result)
            run_async_in_server_loop(
                submit_worker_result(str(master_callback_url), str(worker_result_token),
                                     context_values[0], int(unwrap_single(worker_item_index)), payload), timeout=60,
            )
            return io.NodeOutput([result["image"]], [result.get("mask")], [result.get("text")])

        context_values = [item for item in (context if isinstance(context, (list, tuple)) else [context])
                          if isinstance(item, str) and item]
        if not context_values or len(set(context_values)) != 1:
            raise RuntimeError("Distributed Image List End requires the paired Start context wire")
        context_id = context_values[0]
        from ..image_list.runtime import ImageListCancelledError
        try:
            # Put the deadline inside the coroutine so timeout cancels owned remote work.
            import asyncio
            result_lists = run_async_in_server_loop(asyncio.wait_for(run_image_list_job(context_id), 3600))
        except ImageListCancelledError:
            import comfy.model_management
            raise comfy.model_management.InterruptProcessingException()
        images_out, masks_out, texts_out = result_lists
        if len(images_out) != len(masks_out) or len(images_out) != len(texts_out):
            raise RuntimeError("Image-list collector received mismatched image, mask, or text result counts")
        return io.NodeOutput(images_out, masks_out, texts_out)


async def submit_worker_result(callback_url, item_token, context_id, item_index, result):
    if not callback_url.startswith(("http://", "https://")):
        raise RuntimeError("Worker image-list callback URL is missing or invalid")
    import aiohttp
    from ..utils.network import get_client_session
    session = await get_client_session()
    payload = {"context_id": context_id, "token": item_token,
               "item_index": item_index, "result": result}
    async with session.post(
        f"{callback_url.rstrip('/')}/distributed/image-list/result",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as response:
        if response.status >= 400:
            body = (await response.text())[:500]
            raise RuntimeError(f"Master rejected image-list result (HTTP {response.status}): {body}")
