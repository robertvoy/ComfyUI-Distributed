"""Regression coverage at the scheduler and real HTTP transport boundaries."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from image_list import runtime
from test_image_list import graph_two_sequential_regions, start_node


def load_routes():
    root = Path(__file__).resolve().parents[1]
    name = '_image_list_transport_tests'
    routes = web.RouteTableDef()
    modules = {
        name: types.ModuleType(name),
        name + '.api': types.ModuleType(name + '.api'),
        name + '.image_list': types.ModuleType(name + '.image_list'),
        name + '.image_list.runtime': runtime,
        'server': types.SimpleNamespace(PromptServer=types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=routes))),
    }
    spec = importlib.util.spec_from_file_location(name + '.api.image_list_routes', root / 'api/image_list_routes.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, routes


class CapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_capability_advertises_target_control_without_tensor_protocol_bump(self):
        module, _routes = load_routes()
        response = await module.image_list_capabilities_endpoint(None)
        payload = json.loads(response.text)
        self.assertIs(payload['execution_targets'], True)
        self.assertEqual(payload['protocol_version'], 2)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.module, routes = load_routes()
        app = web.Application()
        app.add_routes(routes)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        runtime._WORKER_ITEMS.clear()

    async def test_fragmented_chunked_item_upload_is_consumable(self):
        token = runtime.create_worker_task_token()
        item = {'image': torch.full((1, 2, 3, 3), 0.25), 'index': 2}
        data = json.dumps({'token': token, 'context_id': 'context', 'item': runtime.encode_item(item)}).encode()
        async def chunks():
            yield data[:8]
            await asyncio.sleep(0.02)
            yield data[8:]
        response = await self.client.post('/distributed/image-list/item', data=chunks())
        self.assertEqual(response.status, 200, await response.text())
        result = runtime.consume_worker_item(token)
        self.assertTrue(torch.equal(result['image'], item['image']))
        self.assertEqual(result['index'], 2)

    async def test_complete_item_upload_and_replay_rejection(self):
        token = runtime.create_worker_task_token()
        data = {'token': token, 'context_id': 'context', 'item': runtime.encode_item(
            {'image': torch.zeros((1, 2, 3, 3)), 'index': 0})}
        response = await self.client.post('/distributed/image-list/item', json=data)
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.post('/distributed/image-list/item', json=data)
        self.assertEqual(response.status, 400)
        runtime.consume_worker_item(token)
        with self.assertRaisesRegex(RuntimeError, 'already consumed'):
            runtime.consume_worker_item(token)

    async def test_chunked_body_limit_is_aggregate(self):
        async def chunks():
            yield b'{"context_id":"'
            await asyncio.sleep(0.01)
            yield b'x' * 60
            yield b'"}'
        with patch.object(self.module, 'MAX_REQUEST_BYTES', 64):
            response = await self.client.post('/distributed/image-list/item', data=chunks())
            self.assertEqual(response.status, 400)
            self.assertIn('exceeds', await response.text())

    async def test_invalid_result_shapes_are_rejected(self):
        for kind, shape in [('image', (1, 2, 3)), ('mask', (1, 2, 3, 1))]:
            result = runtime.encode_result({'image': torch.zeros((1, 2, 3, 3)),
                                             'mask': torch.zeros((1, 2, 3))})
            result[kind]['shape'] = list(shape)
            with self.assertRaises(ValueError):
                runtime.decode_result(result)


class SchedulingBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_interrupt_during_result_wait_cleans_up_promptly(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        wait_cancelled = asyncio.Event()
        dispatches, cleanups = [], []
        async def dispatch(worker, index):
            dispatches.append(index)
            return {'index': index}
        async def wait_next(_inflight):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                wait_cancelled.set()
        async def cleanup(worker, handle):
            cleanups.append((worker, handle['index']))
        task = asyncio.create_task(runtime.dynamic_schedule(
            3, ['one'], dispatch, wait_next, cleanup, cancelled.is_set))
        await entered.wait()
        cancelled.set()
        with self.assertRaises(runtime.ImageListCancelledError):
            await asyncio.wait_for(task, timeout=1)
        self.assertTrue(wait_cancelled.is_set())
        self.assertEqual(dispatches, [0])
        self.assertEqual(cleanups, [('one', 0)])

    async def test_already_interrupted_does_not_submit_any_work(self):
        async def unused(*args):
            self.fail('pre-cancelled job must not dispatch or wait')
        with self.assertRaises(runtime.ImageListCancelledError):
            await runtime.dynamic_schedule(2, ['one'], unused, unused, unused, lambda: True)

    async def test_task_cancel_during_submit_waits_for_owned_handle(self):
        entered, release = asyncio.Event(), asyncio.Event()
        cleanups = []
        async def dispatch(worker, index):
            entered.set()
            await release.wait()
            return {'index': index, 'prompt_id': 'owned'}
        async def unused(*args):
            self.fail('cancelled dispatch must not enter result wait')
        async def cleanup(worker, handle):
            cleanups.append(handle['prompt_id'])
        task = asyncio.create_task(runtime.dynamic_schedule(2, ['one'], dispatch, unused, cleanup))
        await entered.wait()
        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(cleanups, ['owned'])

    def test_output_nodes_inside_region_are_rejected(self):
        prompt = graph_two_sequential_regions()
        output_class = type('OutputOperation', (), {'OUTPUT_NODE': True})
        registry = types.SimpleNamespace(NODE_CLASS_MAPPINGS={'FirstFrameOperation': output_class})
        with patch.dict(sys.modules, {'nodes': registry}):
            with self.assertRaisesRegex(ValueError, 'must be placed after'):
                runtime.validate_image_list_regions(prompt)

    def test_shared_scalars_are_validated_before_dispatch(self):
        for value in (float('nan'), object(), torch.zeros(2, dtype=torch.int32)):
            with self.assertRaises(TypeError):
                runtime.normalize_image_list_inputs(torch.zeros((1, 2, 2, 3)), values={'value_1': [value]})

    def test_three_sequential_regions_can_share_worker_loader(self):
        prompt = graph_two_sequential_regions()
        prompt.pop('8')
        prompt.update({
            '8': start_node(['7', 0]),
            '9': {'class_type': 'ThirdOperation', 'inputs': {'image': ['8', 0]}},
            '10': {'class_type': runtime.END_CLASS,
                   'inputs': {'images': ['9', 0], 'context': ['8', runtime.CONTEXT_OUT]}},
            '11': {'class_type': 'GenericModelLoader', 'inputs': {}},
        })
        for node_id in ['3', '6', '9']:
            prompt[node_id]['inputs']['model'] = ['11', 0]
        # Shared dependencies are verified against the registered output schema.
        loader = type('GenericModelLoader', (), {'RETURN_TYPES': ('MODEL',)})
        registry = types.SimpleNamespace(NODE_CLASS_MAPPINGS={'GenericModelLoader': loader})
        with patch.dict(sys.modules, {'nodes': registry}):
            regions = runtime.validate_image_list_regions(prompt)
        self.assertEqual(len(regions), 3)
        for region in regions:
            self.assertIn('11', region['node_ids'])
