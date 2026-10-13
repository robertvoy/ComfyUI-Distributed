"""Exercise production per-item orchestration through controlled provider seams."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import torch

from test_image_list import graph_one_region

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, data=None, status=200):
        self.data, self.status = data, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self.data

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f'HTTP {self.status}')


class RuntimeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        name = '_image_list_boundaries'
        self.modules = {}
        for suffix in ('', '.image_list', '.api', '.api.orchestration', '.utils'):
            module = types.ModuleType(name + suffix)
            module.__path__ = []
            self.modules[name + suffix] = module
        self.patch_modules = patch.dict(sys.modules, self.modules)
        self.patch_modules.start()
        self.addCleanup(self.patch_modules.stop)
        spec = importlib.util.spec_from_file_location(name + '.image_list.runtime', ROOT / 'image_list/runtime.py')
        self.rt = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.rt
        spec.loader.exec_module(self.rt)
        self.worker = {'id': 'worker', 'host': 'worker', 'type': 'remote', 'port': 8188}
        self.config = {'workers': [self.worker], 'settings': {}}
        self.prompts, self.uploads, self.deletions, self.cancel_requests = [], [], [], []
        self.reject = False
        self.interrupted = False
        self.completed = True
        self.queue = {'queue_running': [], 'queue_pending': []}
        self.callback = Mock(side_effect=lambda worker, **_kwargs:
                             'http://127.0.0.1:8188' if worker['type'] == 'local' else 'http://public-master')
        self.sync = AsyncMock(side_effect=self.sync_media)
        self.dispatch = AsyncMock(side_effect=self.dispatch_prompt)
        self.logs = []
        dependencies = {
            '.api.orchestration.dispatch': types.SimpleNamespace(
                preflight_image_list_workers=AsyncMock(return_value={'worker': True}),
                dispatch_worker_prompt=self.dispatch),
            '.api.orchestration.prompt_transform': types.SimpleNamespace(
                _resolve_direct_workers_for_image_list=lambda config, ids: config['workers']),
            '.api.orchestration.media_sync': types.SimpleNamespace(
                sync_worker_media=self.sync, fetch_worker_path_separator=AsyncMock(return_value='/'),
                convert_paths_for_platform=lambda prompt, separator: prompt),
            '.utils.config': types.SimpleNamespace(load_config=lambda: self.config,
                get_worker_timeout_seconds=lambda: 60,
                get_image_list_job_timeout_seconds=lambda config=None: 0),
            '.utils.network': types.SimpleNamespace(build_master_url=lambda **kwargs: 'http://public-master',
                build_master_callback_url=self.callback,
                build_worker_url=lambda worker, path='': f"http://{worker['id']}{path}",
                get_client_session=AsyncMock(return_value=self)),
            '.utils.logging': types.SimpleNamespace(log=self.logs.append),
        }
        for suffix, module in dependencies.items():
            sys.modules[name + suffix] = module
        self.config_module = dependencies['.utils.config']
        # Other legacy test modules replace aiohttp during collection. This
        # controlled provider fixture needs only its explicit timeout/error seam.
        sys.modules['aiohttp'] = types.SimpleNamespace(
            ClientTimeout=lambda **kwargs: kwargs, ClientError=ConnectionError)
        comfy = types.ModuleType('comfy')
        comfy.model_management = types.SimpleNamespace(processing_interrupted=lambda: self.interrupted)
        sys.modules['comfy'] = comfy
        sys.modules['comfy.model_management'] = comfy.model_management

    async def sync_media(self, worker, prompt):
        prompt['7']['inputs']['image'] = 'shared/ref.png'

    async def dispatch_prompt(self, worker, prompt, meta, **kwargs):
        if self.reject:
            raise RuntimeError('prompt rejected')
        self.prompts.append((json.loads(json.dumps(prompt)), meta))
        inputs = prompt['5']['inputs']
        token = inputs['worker_result_token']
        item = self.rt.consume_worker_item(token)
        self.rt.set_master_result(self.context, inputs['worker_item_index'], token, {'image': item['image']})
        return f'prompt-{len(self.prompts)}'

    def post(self, url, *, json=None, data=None, **kwargs):
        if data is not None:
            json = __import__('json').loads(data)
        if url.endswith('/distributed/image-list/item'):
            self.uploads.append(json['token'])
            self.rt.store_worker_item(json['token'], json['item'])
        else:
            self.cancel_requests.append((url, json))
        return Response({})

    def delete(self, url, **kwargs):
        token = url.rsplit('/', 1)[-1]
        self.deletions.append(token)
        self.rt.delete_worker_item(token)
        return Response({})

    def get(self, url, **kwargs):
        if url.endswith('/queue'):
            return Response(self.queue)
        pid = url.rsplit('/', 1)[-1]
        return Response({pid: {'status': {'completed': True, 'status_str': 'success'}}}
                        if self.completed else {})

    def create(self, count=2):
        prompt = graph_one_region()
        prompt['7'] = {'class_type': 'LoadImage', 'inputs': {'image': 'shared/ref.png [output]'}}
        prompt['4']['inputs']['reference'] = ['7', 0]
        self.metadata = {'nodes': [{'id': 42}], 'extra': {'label': 'generic fixture'}}
        self.context = self.rt.create_master_job('3', '5', prompt,
            [{'image': torch.zeros((1, 2, 2, 3)), 'index': i} for i in range(count)],
            ['worker'], 'http://public-master', self.metadata)
        return self.context

    async def test_each_assignment_reuses_media_paths_with_fresh_tokens_and_metadata(self):
        context = self.create()
        await self.rt.run_image_list_job(context)
        self.assertEqual(self.sync.await_count, 1)
        self.assertEqual([p['7']['inputs']['image'] for p, _ in self.prompts],
                         ['shared/ref.png', 'shared/ref.png'])
        tokens = [p['5']['inputs']['worker_result_token'] for p, _ in self.prompts]
        self.assertEqual(len(set(tokens)), 2)
        self.assertEqual([p['5']['inputs']['worker_item_index'] for p, _ in self.prompts], [0, 1])
        self.assertEqual([meta for _, meta in self.prompts], [self.metadata, self.metadata])

    async def test_annotated_media_roots_survive_windows_prep_and_repeated_items(self):
        import tempfile

        context = self.create(2)
        state = self.rt.get_master_job(context)
        state.original_prompt['8'] = {'class_type': 'LoadImage', 'inputs': {'image': 'shared\\ref.png [temp]'}}
        state.original_prompt['4']['inputs']['second_reference'] = ['8', 0]
        name = '_image_list_boundaries.api.orchestration.media_sync'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'api/orchestration/media_sync.py')
        assert spec is not None and spec.loader is not None
        media = importlib.util.module_from_spec(spec)
        sys.modules['_image_list_boundaries.utils.logging'].debug_log = self.logs.append
        sys.modules['_image_list_boundaries.utils.trace_logger'] = types.SimpleNamespace(
            trace_debug=lambda *args: None, trace_info=lambda *args: None)
        spec.loader.exec_module(media)
        media.fetch_worker_path_separator = AsyncMock(return_value='\\')
        converted = Mock(wraps=media.convert_paths_for_platform)
        media.convert_paths_for_platform = converted
        uploads = []

        async def upload(worker, filename, content, digest, mime, **kwargs):
            uploads.append((filename, content))
            return True, f'worker-input/{len(uploads)}.png'

        media._upload_media_to_worker = AsyncMock(side_effect=upload)
        media.sync_worker_media = AsyncMock(wraps=media.sync_worker_media)
        with tempfile.TemporaryDirectory(prefix='image-list-media-') as directory:
            root = Path(directory)
            for kind in ('input', 'output', 'temp'):
                path = root / kind / 'shared/ref.png'
                path.parent.mkdir(parents=True)
                path.write_bytes(kind.encode())

            def resolve_annotated(filename):
                for kind in ('output', 'temp', 'input'):
                    suffix = f' [{kind}]'
                    if filename.endswith(suffix):
                        return str(root / kind / filename[:-len(suffix)])
                return str(root / 'input' / filename)

            with patch.dict(sys.modules, {name: media, 'folder_paths': types.SimpleNamespace(
                    get_annotated_filepath=resolve_annotated)}):
                await self.rt.run_image_list_job(context)
        self.assertEqual(uploads, [('shared/ref.png [output]', b'output'),
                                   ('shared/ref.png [temp]', b'temp')])
        media.sync_worker_media.assert_awaited_once()
        media.fetch_worker_path_separator.assert_awaited_once()
        converted.assert_called_once()
        self.assertEqual(converted.call_args.args[1], '\\')
        for prompt, _ in self.prompts:
            self.assertEqual(prompt['7']['inputs']['image'], 'worker-input/1.png')
            self.assertEqual(prompt['8']['inputs']['image'], 'worker-input/2.png')
        self.assertEqual(state.original_prompt['7']['inputs']['image'], 'shared/ref.png [output]')

    async def test_local_worker_callback_does_not_use_public_master_address(self):
        self.worker['type'] = 'local'
        await self.rt.run_image_list_job(self.create(1))
        self.assertEqual(self.prompts[0][0]['5']['inputs']['master_callback_url'], 'http://127.0.0.1:8188')
        self.callback.assert_called()

    async def test_remote_worker_uses_topology_callback_helper(self):
        await self.rt.run_image_list_job(self.create(1))
        self.assertEqual(self.prompts[0][0]['5']['inputs']['master_callback_url'], 'http://public-master')
        self.callback.assert_called()

    def test_metadata_is_forwarded_into_master_start_prompt(self):
        from test_prompt_transform import pt
        prompt = graph_one_region()
        metadata = {'nodes': [{'id': 1}]}
        index = pt.PromptIndex(prompt)
        transformed = pt.apply_participant_overrides(
            prompt, 'master', ['worker'], {'3': 'run'}, 'http://public-master', False,
            index, workflow_meta=metadata)
        self.assertEqual(transformed['3']['inputs']['image_list_workflow_meta'], metadata)

    async def test_aggregate_input_limit_fails_before_upload(self):
        self.rt.MAX_REQUEST_BYTES = 64
        with self.assertRaisesRegex(ValueError, 'complete.*request'):
            await self.rt.run_image_list_job(self.create(1))
        self.assertFalse(self.uploads)
        self.assertFalse(self.rt._WORKER_ITEMS)

    def test_tensor_and_combined_request_limits_preserve_dtype(self):
        item = {'image': torch.zeros((1, 4, 4, 3)), 'index': 0}
        encoded = self.rt.encode_item(item)
        payload = {'context_id': 'example', 'token': 't' * 48, 'item': encoded}
        single = self.rt.encode_request_payload(payload)
        self.assertEqual(json.loads(single)['item']['image']['dtype'], 'float32')
        self.rt.MAX_REQUEST_BYTES = len(single)
        self.assertEqual(self.rt.encode_request_payload(payload), single)
        item['values'] = {'value_1': item['image']}
        item['mask'] = torch.zeros((1, 4, 4))
        payload['item'] = self.rt.encode_item(item)
        with self.assertRaisesRegex(ValueError, 'complete.*request'):
            self.rt.encode_request_payload(payload)
        # Size is checked before allocating/copying a 4K tensor onto the CPU.
        with self.assertRaisesRegex(ValueError, 'transport limit'):
            self.rt.encode_value(torch.empty((1, 2160, 3840, 3), device='meta'), 'IMAGE')

    async def test_rejected_submission_reclaims_uploaded_token_without_prompt_id(self):
        self.reject = True
        with self.assertRaisesRegex(RuntimeError, 'prompt rejected'):
            await self.rt.run_image_list_job(self.create(1))
        self.assertEqual(self.deletions, self.uploads)
        self.assertFalse(self.rt._WORKER_ITEMS)
        self.assertFalse(self.rt._MASTER_JOBS)
        self.assertFalse(self.cancel_requests)

    async def test_user_interrupt_during_pending_submission_is_bounded(self):
        entered = asyncio.Event()
        async def never_ack(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        self.dispatch.side_effect = never_ack
        self.rt.SUBMISSION_CANCEL_GRACE_SECONDS = 0.02
        task = asyncio.create_task(self.rt.run_image_list_job(self.create(2)))
        await asyncio.wait_for(entered.wait(), 1)
        self.interrupted = True
        with self.assertRaises(self.rt.ImageListCancelledError):
            await asyncio.wait_for(task, 0.8)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(self.deletions, self.uploads)
        self.assertFalse(self.rt._MASTER_JOBS)
        self.assertTrue(any('ambiguous' in line for line in self.logs), self.logs)

    async def test_owned_running_prompt_survives_cold_load_and_old_hour_cap(self):
        self.completed = False
        context = self.create(2)
        clock = [0.0]
        calls = [0]
        original_get = self.get
        def get(url, **kwargs):
            if '/history/' in url:
                calls[0] += 1
                clock[0] += 3700
                self.completed = calls[0] >= 3
            if url.endswith('/queue'):
                return Response({'queue_running': [[0, 'prompt-1']] if calls[0] == 2 else [],
                                 'queue_pending': [[0, 'prompt-1']] if calls[0] == 1 else []})
            return original_get(url, **kwargs)
        self.rt.time = types.SimpleNamespace(monotonic=lambda: clock[0])
        with patch.object(self, 'get', side_effect=get):
            result = await self.rt.run_image_list_job(context)
        self.assertEqual(len(result[0]), 2)
        self.assertGreaterEqual(clock[0], 7200)
        self.assertFalse(self.cancel_requests)

    async def test_unrelated_busy_prompt_does_not_refresh_owned_liveness(self):
        self.completed = False
        clock = [0.0]
        original_get = self.get
        def get(url, **kwargs):
            if '/history/' in url:
                clock[0] += 70
            if url.endswith('/queue'):
                return Response({'queue_running': [[0, 'not-our-prompt']], 'queue_pending': []})
            return original_get(url, **kwargs)
        self.rt.time = types.SimpleNamespace(monotonic=lambda: clock[0])
        with patch.object(self, 'get', side_effect=get):
            with self.assertRaisesRegex(TimeoutError, 'liveness'):
                await self.rt.run_image_list_job(self.create(1))
        self.assertEqual(self.cancel_requests, [
            ('http://worker/queue', {'delete': ['prompt-1']}),
            ('http://worker/interrupt', {'prompt_id': 'prompt-1'})])

    async def test_failed_queue_cleanup_still_interrupts_and_reclaims_staging(self):
        self.completed = False
        self.config_module.get_image_list_job_timeout_seconds = lambda config=None: 0.03
        original_post = self.post
        def post(url, **kwargs):
            result = original_post(url, **kwargs)
            return Response(status=500) if url.endswith('/queue') else result
        with patch.object(self, 'post', side_effect=post):
            with self.assertRaisesRegex(TimeoutError, 'total deadline'):
                await self.rt.run_image_list_job(self.create(2))
        self.assertEqual(self.cancel_requests, [
            ('http://worker/queue', {'delete': ['prompt-1']}),
            ('http://worker/interrupt', {'prompt_id': 'prompt-1'})])
        self.assertEqual(self.deletions, self.uploads)
        self.assertTrue(any('cleanup unconfirmed' in line for line in self.logs), self.logs)
        self.assertEqual(self.dispatch.await_count, 1)

    async def test_configured_total_deadline_cleans_owned_work(self):
        self.completed = False
        self.config_module.get_image_list_job_timeout_seconds = lambda config=None: 0.03
        self.queue = {'queue_running': [[0, 'prompt-1']], 'queue_pending': []}
        with self.assertRaisesRegex(TimeoutError, 'total deadline'):
            await asyncio.wait_for(self.rt.run_image_list_job(self.create(2)), 1)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(self.deletions, self.uploads)
        self.assertFalse(self.rt._MASTER_JOBS)
