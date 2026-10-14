"""End policy reaches cloud exports and initial worker compatibility preflight."""
import asyncio
import copy
import unittest

import pytest
import torch

import test_image_list_serverless as cloud
import test_image_list_master_selection as selection

FORMATS = ('exact', 'legacy_png', 'png', 'ffv1', 'jpeg', 'webp', 'h264', 'h265', 'av1')


@pytest.mark.parametrize('fmt', FORMATS)
def test_cloud_returns_use_selected_image_format_but_inputs_remain_exact(fmt):
    async def check():
        prompt = cloud.source()
        prompt['ns/end']['inputs'].update(transfer_format=fmt, transfer_quality=73)
        original = copy.deepcopy(prompt)
        value = cloud.item(dtype=torch.float64)
        job = await cloud.prepare(prompt, value)
        assert prompt == original
        for binding in job['manifest']:
            controls = job['workflow'][binding['node_id']]['inputs']
            if binding['kind'] == 'IMAGE' and fmt != 'exact':
                assert controls['transfer_format'] == fmt
                assert controls['transfer_quality'] == 73
            else:
                assert 'transfer_format' not in controls and 'transfer_quality' not in controls
        for artifact in job['media']['artifacts']:
            assert artifact['descriptor']['encoding'] == 'tensor-le-v1'
        inputs = await cloud.decoded_inputs(job)
        for ref in job['media']['inputs']:
            assert torch.equal(inputs[ref['artifact_id']], value[ref['input_name']])
        cloud.handler._validate_request({k: job[k] for k in ('workflow', 'images', 'media')})
    asyncio.run(check())


@pytest.mark.parametrize('fmt,quality', [('JPEG',90),('jpeg',True),('jpeg',101),('jpeg','90')])
def test_cloud_rejects_invalid_policy(fmt, quality):
    prompt = cloud.source()
    prompt['ns/end']['inputs'].update(transfer_format=fmt, transfer_quality=quality)
    with pytest.raises(ValueError):
        asyncio.run(cloud.prepare(prompt))


class TransferPreflightTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = selection.MasterSelectionTests.asyncSetUp
    queue = selection.MasterSelectionTests.queue

    async def test_queue_requests_image_codec_capability(self):
        self.prompt['3']['inputs'].update(transfer_format='jpeg', transfer_quality=73)
        await self.queue()
        self.assertEqual(self.preflight.call_args.kwargs['required_transfer_formats'], ['jpeg'])

    async def test_invalid_policy_fails_before_worker_probe_or_queue(self):
        self.prompt['3']['inputs'].update(transfer_format='bad-codec')
        with self.assertRaises(ValueError):
            await self.queue()
        self.q.select_active_workers.assert_not_awaited()
        self.preflight.assert_not_awaited()
        self.q.queue_prompt_payload.assert_not_awaited()

    async def test_missing_capability_cannot_silently_fall_back_to_master(self):
        self.prompt['3']['inputs'].update(transfer_format='jpeg')
        self.preflight.side_effect = RuntimeError('upgrade worker for image return encoding')
        with self.assertRaisesRegex(RuntimeError, 'upgrade'):
            await self.queue()
        self.q.queue_prompt_payload.assert_not_awaited()
