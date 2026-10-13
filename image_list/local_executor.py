"""A queue-free Comfy graph slot, only while the synchronous outer End is blocked.

Do NOT use PromptExecutor.execute[_async] here. Its prompt lifecycle clears the
process interrupt, resets progress handlers, changes RAM-cache callbacks and
starts/ends a model tracker. Those belong to the suspended parent executor.
Instead use core's ExecutionList/execute (including lazy/async/expanded nodes)
with private provider-free caches and a private server facade. Model outputs are
registered with the existing parent tracker, whose lifetime includes this work.

Core progress and GraphBuilder defaults are process-global. Borrow them only
inside the owned execution thread, restore them in finally, and join that thread
before End returns. Cancellation never clears or raises the global interrupt:
it stops locally at node/progress boundaries. A non-cooperative synchronous node
must finish before cleanup can return; abandoning its thread is not safe.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

from .runtime import ImageListCancelledError


_EXECUTION_LOCK = threading.Lock()


class LocalExecutionSlot:
    def __init__(self, *, model_callback=None):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-list-master")
        self._stop = threading.Event()
        self._future = None
        self._caches = None
        self._model_callback = model_callback
        self._closed = False
        self._loop = None
        self._task = None
        self._state_lock = threading.Lock()

    def submit(self, prompt, prompt_id, end_id, workflow_meta=None):
        if self._closed or self._stop.is_set():
            raise ImageListCancelledError("Local image-list slot has stopped")
        if self._future is not None and not self._future.done():
            raise RuntimeError("Local image-list slot already has an in-flight item")
        self._future = self._pool.submit(self._run, prompt, prompt_id, end_id, workflow_meta)
        return self._future

    def request_stop(self):
        self._stop.set()
        # Wakes a graph waiting on an async node as well as stopping synchronous
        # nodes at their next progress/between-node checkpoint.
        with self._state_lock:
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)

    async def close(self):
        self.request_stop()
        try:
            if self._future is not None:
                # Do not cancel this Future: cancelling an asyncio wrapper cannot
                # kill its running thread, and falsely makes cleanup look done.
                waiter = asyncio.wrap_future(self._future)
                while not waiter.done():
                    try:
                        await asyncio.shield(waiter)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break  # The scheduler propagates execution errors.
                if waiter.done() and not waiter.cancelled():
                    waiter.exception()
        finally:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._caches = None
            self._closed = True

    def _run(self, prompt, prompt_id, end_id, workflow_meta):
        # The normal Comfy prompt worker is single-threaded and blocked in End.
        # Fail closed if another local slot is nevertheless executing, rather
        # than interleave writes to core's process-global execution state.
        if not _EXECUTION_LOCK.acquire(blocking=False):
            raise RuntimeError("Another local image-list executor is already active")
        try:
            asyncio.run(self._execute(prompt, prompt_id, end_id, workflow_meta))
        except asyncio.CancelledError as exc:
            raise ImageListCancelledError("Local image-list execution cancelled") from exc
        finally:
            with self._state_lock:
                self._loop = self._task = None
            _EXECUTION_LOCK.release()

    def _check_stopped(self):
        import comfy.model_management
        if self._stop.is_set() or comfy.model_management.processing_interrupted():
            raise ImageListCancelledError("Local image-list execution interrupted")

    async def _execute(self, prompt, prompt_id, end_id, workflow_meta):
        import execution
        import torch
        import comfy.model_management
        import comfy.utils
        from comfy_execution import progress
        from comfy_execution.graph_utils import GraphBuilder

        self._check_stopped()
        with self._state_lock:
            self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        dynamic_prompt = execution.DynamicPrompt(prompt)
        private_progress = progress.ProgressRegistry(prompt_id, dynamic_prompt)
        parent_progress = progress.global_progress_registry
        parent_hook = comfy.utils.PROGRESS_BAR_HOOK
        prefix_keys = ("_default_prefix_root", "_default_prefix_call_index", "_default_prefix_graph_index")
        parent_prefix = {key: getattr(GraphBuilder, key) for key in prefix_keys}
        progress.global_progress_registry = private_progress

        def progress_hook(value, total, preview=None, prompt_id=None, node_id=None):
            self._check_stopped()
            # Never route a child progress/preview event through the parent's
            # client or reset its handlers. Keep the private progress accurate.
            if node_id is not None:
                private_progress.update_progress(node_id, value, total, preview)

        comfy.utils.PROGRESS_BAR_HOOK = progress_hook
        try:
            with torch.inference_mode():
                if self._caches is None:
                    outputs = execution.HierarchicalCache(execution.CacheKeySetInputSignature)
                    objects = execution.HierarchicalCache(execution.CacheKeySetID)
                    self._caches = SimpleNamespace(outputs=outputs, objects=objects, all=[outputs, objects])
                caches = self._caches
                changed = execution.IsChangedCache(prompt_id, dynamic_prompt, caches.outputs)
                for cache in caches.all:
                    await cache.set_prompt(dynamic_prompt, prompt.keys(), changed)
                    cache.clean_unused()
                execution_list = execution.ExecutionList(dynamic_prompt, caches.outputs, self._model_callback)
                execution_list.add_node(end_id)
                server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *a, **kw: None)
                assets = execution.NoAssets(execution.args)
                extra = {"extra_pnginfo": {"workflow": workflow_meta}} if workflow_meta is not None else {}
                executed, pending_subgraphs, pending_async, ui = set(), {}, {}, {}
                while not execution_list.is_empty():
                    self._check_stopped()
                    node_id, error, exc = await execution_list.stage_node_execution()
                    if error is not None:
                        raise RuntimeError(f"Local image-list graph failed: {error}") from exc
                    self._check_stopped()
                    result, error, exc = await execution.execute(
                        server, dynamic_prompt, caches, node_id, extra, executed, prompt_id,
                        execution_list, pending_subgraphs, pending_async, ui, assets)
                    if result == execution.ExecutionResult.FAILURE:
                        if isinstance(exc, (comfy.model_management.InterruptProcessingException,
                                            ImageListCancelledError)):
                            raise ImageListCancelledError("Local image-list execution interrupted") from exc
                        raise RuntimeError(f"Local image-list node {node_id} failed: {error}") from exc
                    if result == execution.ExecutionResult.PENDING:
                        execution_list.unstage_node_execution()
                    else:
                        execution_list.complete_node_execution()
                self._check_stopped()
        finally:
            # Core execute creates async-node and unblock tasks. This is a private
            # event loop, so every other task is owned here. Drain/cancel them
            # before restoring globals, including when the graph failed early.
            with self._state_lock:
                self._task = None  # Stop requests must not interrupt this drain.
            try:
                while True:
                    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
                    if not pending:
                        break
                    for task in pending:
                        task.cancel()
                    drain = asyncio.gather(*pending, return_exceptions=True)
                    while not drain.done():
                        try:
                            await asyncio.shield(drain)
                        except asyncio.CancelledError:
                            continue
            finally:
                progress.global_progress_registry = parent_progress
                comfy.utils.PROGRESS_BAR_HOOK = parent_hook
                for key, value in parent_prefix.items():
                    setattr(GraphBuilder, key, value)
