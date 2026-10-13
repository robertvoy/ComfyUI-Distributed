"""Public V3 Start/End nodes for dynamic per-image image-list execution."""
from __future__ import annotations

import json

from comfy_api.v0_0_2 import io

from ..utils.async_helpers import run_async_in_server_loop
from ..image_list.runtime import (
    MAX_VALUE_SLOTS,
    VALUE_NAMES,
    VALUES_INPUT,
    create_master_job,
    normalize_image_list_inputs,
    normalize_worker_result,
    run_image_list_job,
    encode_result,
    encode_request_payload,
    unwrap_single,
)


# Types the transport can carry exactly. Anything else (models, latents, ...)
# must be loaded inside the region or shared as a model dependency instead.
_VALUE_TYPES = [io.Image, io.Mask, io.String, io.Int, io.Float, io.Boolean]
assert len(VALUE_NAMES) == MAX_VALUE_SLOTS


class DistributedImageListStart(io.ComfyNode):
    """Materialize a batch once, then expose ordered per-item typed values."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DistributedImageListStart",
            display_name="Distributed Image List Start",
            category="image/distributed",
            description=("Starts one per-image processing region. Each image becomes one job. "
                         "mask and value_N inputs take 1 value (shared by every image) or exactly "
                         "one value per image, paired in order."),
            inputs=[
                io.Image.Input("images", optional=True, tooltip="Image list or image batch; each frame becomes one job."),
                io.Mask.Input("mask", optional=True,
                              tooltip="1 mask shared by every image, or one mask per image."),
                io.Autogrow.Input(VALUES_INPUT, optional=True, template=io.Autogrow.TemplateNames(
                    io.MultiType.Input("value", types=_VALUE_TYPES,
                                       tooltip=("Any image, mask, text, number or boolean. 1 value is shared by "
                                                "every image; one value per image pairs them in order. "
                                                "A batch stays one value.")),
                    names=list(VALUE_NAMES), min=0)),
            ],
            outputs=[
                io.Image.Output(display_name="image", is_output_list=True),
                io.Mask.Output(display_name="mask", is_output_list=True),
                io.Int.Output(display_name="item_index", is_output_list=True),
                io.Custom("DISTRIBUTED_IMAGE_LIST_CONTEXT").Output(display_name="context", is_output_list=True),
                # value_N outputs mirror value_N inputs. The web extension hides
                # unused ones and shows the connected input's type.
                *[io.AnyType.Output(id=name, display_name=name, is_output_list=True) for name in VALUE_NAMES],
            ],
            is_input_list=True,
            accept_all_inputs=True,
            not_idempotent=True,
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images=None, mask=None, values=None,
                worker_item_token="", worker_context_id="", image_list_enabled_worker_ids="[]",
                image_list_master_callback="", image_list_execution_id="",
                image_list_workflow_meta=None, image_list_master_participates=None,
                image_list_local_execution=False):
        worker_item_token = unwrap_single(worker_item_token)
        worker_context_id = unwrap_single(worker_context_id)
        image_list_workflow_meta = unwrap_single(image_list_workflow_meta)
        if worker_item_token:
            if unwrap_single(image_list_local_execution):
                from ..image_list.runtime import consume_worker_item
                item = consume_worker_item(worker_item_token)
            else:
                item = run_async_in_server_loop(_consume_worker_item(worker_item_token), timeout=10)
            context_id = worker_context_id
            if not isinstance(context_id, str) or not context_id:
                raise RuntimeError("Worker image-list prompt is missing its paired context")
            item_values = item.get("values") or {}
            return io.NodeOutput(
                [item["image"]], [item.get("mask")], [item["index"]], [context_id],
                *[[item_values.get(name)] for name in VALUE_NAMES],
            )

        items = normalize_image_list_inputs(images, mask, values)
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
        from comfy_execution.utils import get_executing_context
        from ..image_list.lifecycle import has_safe_execution_targets
        import server
        executing = get_executing_context()
        explicit_targets = has_safe_execution_targets(
            getattr(server.PromptServer, "instance", None),
            executing.prompt_id if executing is not None else None, prompt)
        from ..image_list.runtime import validate_image_list_regions
        regions = validate_image_list_regions(prompt, explicit_targets=explicit_targets)
        region = next((row for row in regions if row["start_id"] == start_id), None)
        if region is None:
            raise RuntimeError(f"No safe paired Image List End was found for Start node {start_id}")
        master_participates = unwrap_single(image_list_master_participates)
        if master_participates is None:
            from ..utils.config import load_config
            master_participates = not bool((load_config().get("settings", {}) or {}).get("master_delegate_only", False))
        if not isinstance(master_participates, bool):
            raise ValueError("image_list_master_participates must be a boolean")
        context_id = create_master_job(
            start_id, region["end_id"], prompt, items, worker_ids,
            str(unwrap_single(image_list_master_callback or "")),
            image_list_workflow_meta if isinstance(image_list_workflow_meta, dict) else None,
            prompt_id=executing.prompt_id if executing is not None else None,
            master_participates=master_participates,
            explicit_targets=explicit_targets,
        )
        return io.NodeOutput(
            [item["image"] for item in items],
            [item.get("mask") for item in items],
            [item["index"] for item in items],
            [context_id for _ in items],
            *[[item["values"].get(name) for item in items] for name in VALUE_NAMES],
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
            hidden=[io.Hidden.prompt, io.Hidden.unique_id, io.Hidden.execution_list],
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
                worker_result_token="", worker_item_index=-1, master_callback_url="",
                image_list_local_execution=False):
        worker_result_token = unwrap_single(worker_result_token)
        master_callback_url = unwrap_single(master_callback_url)
        if worker_result_token:
            context_values = [item for item in (context if isinstance(context, (list, tuple)) else [context])
                              if isinstance(item, str) and item]
            if not context_values or len(set(context_values)) != 1:
                raise RuntimeError("Worker Image List End is missing its paired context")
            result = normalize_worker_result(images, masks, texts)
            if unwrap_single(image_list_local_execution):
                from ..image_list.runtime import set_master_result
                set_master_result(context_values[0], int(unwrap_single(worker_item_index)),
                                  str(worker_result_token), result)
            else:
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
            # The runtime owns the configurable deadline and bounded remote cleanup.
            execution_list = getattr(cls.hidden, "execution_list", None)
            # Local model outputs belong to this still-running outer prompt.
            # Add them to its existing tracker; never start/end another tracker
            # that could unpin a model still needed by the parent.
            model_callback = getattr(execution_list, "output_link_callback", None)
            result_lists = run_async_in_server_loop(run_image_list_job(context_id, model_callback=model_callback))
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
        data=encode_request_payload(payload),
        headers={"Content-Type": "application/json"},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as response:
        if response.status >= 400:
            body = (await response.text())[:500]
            raise RuntimeError(f"Master rejected image-list result (HTTP {response.status}): {body}")
