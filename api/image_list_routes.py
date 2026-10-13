"""Bounded, typed HTTP transport for dynamic image-list worker tasks."""
import json

from aiohttp import web
import server

from ..image_list.runtime import (
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    decode_result,
    delete_worker_item,
    set_master_result,
    store_worker_item,
)

prompt_server = server.PromptServer.instance


async def _json_body(request):
    length = request.content_length
    if length is not None and length > MAX_REQUEST_BYTES:
        raise ValueError(f"Request exceeds the {MAX_REQUEST_BYTES}-byte image-list limit")
    raw = bytearray()
    async for chunk in request.content.iter_chunked(64 * 1024):
        raw.extend(chunk)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError(f"Request exceeds the {MAX_REQUEST_BYTES}-byte image-list limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid image-list JSON body: {exc}") from exc


@prompt_server.routes.get("/distributed/image-list/capabilities")
async def image_list_capabilities_endpoint(_request):
    return web.json_response({
        "supported": True,
        "protocol_version": PROTOCOL_VERSION,
        "node_ids": ["DistributedImageListStart", "DistributedImageListEnd"],
        "transport": "typed-float-tensor-json-v1",
    })


@prompt_server.routes.post("/distributed/image-list/item")
async def image_list_item_endpoint(request):
    try:
        data = await _json_body(request)
        if not isinstance(data, dict):
            raise ValueError("Expected an image-list item object")
        context_id = data.get("context_id")
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("context_id must be a non-empty string")
        item = data.get("item")
        store_worker_item(data.get("token"), item)
        return web.json_response({"status": "stored", "item_index": item["index"]})
    except (ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@prompt_server.routes.delete("/distributed/image-list/item/{token}")
async def image_list_delete_item_endpoint(request):
    token = request.match_info.get("token", "")
    return web.json_response({"status": "deleted", "removed": delete_worker_item(token)})


@prompt_server.routes.post("/distributed/image-list/result")
async def image_list_result_endpoint(request):
    try:
        data = await _json_body(request)
        if not isinstance(data, dict):
            raise ValueError("Expected an image-list result object")
        context_id = data.get("context_id")
        token = data.get("token")
        item_index = data.get("item_index")
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("context_id must be a non-empty string")
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        if isinstance(item_index, bool) or not isinstance(item_index, int) or item_index < 0:
            raise ValueError("item_index must be a non-negative integer")
        result = decode_result(data.get("result"))
        set_master_result(context_id, item_index, token, result)
        return web.json_response({"status": "accepted", "item_index": item_index})
    except (ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
