"""
Async helper utilities for ComfyUI-Distributed.
"""
import asyncio
import threading
import time
import uuid
import execution
import server
from typing import Optional, Any, Coroutine
from .network import get_server_loop

def run_async_in_server_loop(coro: Coroutine, timeout: Optional[float] = None) -> Any:
    """
    Run async coroutine in server's event loop and wait for result.
    
    This is useful when you need to run async code from a synchronous context
    but want to use the server's existing event loop instead of creating a new one.
    
    Args:
        coro: The coroutine to run
        timeout: Optional timeout in seconds
        
    Returns:
        The result of the coroutine
        
    Raises:
        TimeoutError: If the operation times out
        Exception: Any exception raised by the coroutine
    """
    event = threading.Event()
    result = None
    error = None
    
    async def wrapper():
        nonlocal result, error
        try:
            result = await coro
        except Exception as e:
            error = e
        finally:
            event.set()
    
    # Schedule on server's event loop
    loop = get_server_loop()
    asyncio.run_coroutine_threadsafe(wrapper(), loop)
    
    # Wait for completion
    if not event.wait(timeout):
        raise TimeoutError(f"Async operation timed out after {timeout} seconds")
    
    if error:
        raise error
    return result


prompt_server = server.PromptServer.instance


def _summarize_node_errors(node_errors: dict) -> str:
    if not isinstance(node_errors, dict) or not node_errors:
        return ""

    parts = []
    for node_id, entry in node_errors.items():
        if not isinstance(entry, dict):
            continue
        class_type = str(entry.get("class_type") or "UnknownNode")
        for err in entry.get("errors", []):
            if not isinstance(err, dict):
                continue
            message = str(err.get("message") or "validation error")
            details = str(err.get("details") or "").strip()
            parts.append(
                f"{class_type}#{node_id}: {message}{f' ({details})' if details else ''}"
            )
            if len(parts) >= 5:
                return " | ".join(parts)
    return " | ".join(parts)


class PromptValidationError(RuntimeError):
    """Raised when a prompt fails ComfyUI validation with structured context."""

    def __init__(self, error_payload, node_errors=None):
        payload = error_payload if isinstance(error_payload, dict) else {
            "type": "prompt_validation_failed",
            "message": str(error_payload),
            "details": "",
            "extra_info": {},
        }
        self.validation_error = dict(payload)
        self.node_errors = node_errors if isinstance(node_errors, dict) else {}

        if self.node_errors:
            details = str(self.validation_error.get("details") or "").strip()
            if not details:
                summary = _summarize_node_errors(self.node_errors)
                if summary:
                    self.validation_error["details"] = summary

        merged = dict(self.validation_error)
        if self.node_errors:
            merged["node_errors"] = self.node_errors
        super().__init__(f"Invalid prompt: {merged}")


async def queue_prompt_payload(
    prompt_obj,
    workflow_meta=None,
    client_id=None,
    include_queue_metadata=False,
):
    """Validate and queue a prompt via ComfyUI's prompt queue."""
    payload = {"prompt": prompt_obj}
    payload = prompt_server.trigger_on_prompt(payload)
    prompt = payload["prompt"]

    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        error_payload = valid[1] if len(valid) > 1 else "Prompt outputs failed validation"
        node_errors = valid[3] if len(valid) > 3 else {}
        raise PromptValidationError(error_payload, node_errors)

    extra_data = {"create_time": int(time.time() * 1000)}
    if workflow_meta:
        extra_data.setdefault("extra_pnginfo", {})["workflow"] = workflow_meta
    if client_id:
        extra_data["client_id"] = client_id

    sensitive = {}
    for key in getattr(execution, "SENSITIVE_EXTRA_DATA_KEYS", []):
        if key in extra_data:
            sensitive[key] = extra_data.pop(key)

    number = getattr(prompt_server, "number", 0)
    prompt_server.number = number + 1
    prompt_queue_item = (number, prompt_id, prompt, extra_data, valid[2], sensitive)
    prompt_server.prompt_queue.put(prompt_queue_item)

    if include_queue_metadata:
        return {
            "prompt_id": prompt_id,
            "number": number,
            "node_errors": {},
        }

    return prompt_id
