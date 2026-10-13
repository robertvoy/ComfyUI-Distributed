"""Typed, lossless list collection for serverless media participants.

The master compiler supplies expected_media before pruning delegate inputs.
The provider manager owns remote deadlines and delivers one atomic native-media
message per participant. This node never serializes, concatenates or resamples.
"""
import asyncio
import json
import time

import comfy.model_management
from comfy_api.v0_0_2 import io
import server

from ..serverless_bridge.contract import MAX_ITEMS, SOCKET_KINDS, identifier
from ..utils.async_helpers import run_async_in_server_loop

POLL_SECONDS = 0.1
START_GRACE_SECONDS = 2.0


def _control(value, name, kind):
    # Only execution controls are scalars wrapped by ComfyUI's list mapper.
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f'{name} must be a singleton control')
        value = value[0]
    if type(value) is not kind:
        raise ValueError(f'Invalid {name} control')
    return value


def _names(value, name):
    try:
        result = json.loads(_control(value, name, str))
    except (ValueError, TypeError):
        raise ValueError(f'{name} must be a JSON string list') from None
    if (not isinstance(result, list) or any(not isinstance(item, str) for item in result)
            or len(result) != len(set(result))):
        raise ValueError(f'{name} must contain unique strings')
    return result


def _validate_media(media, expected):
    if not isinstance(media, dict) or set(media) != set(expected):
        raise ValueError('Media sockets do not match expected_media')
    for socket, items in media.items():
        if (socket not in SOCKET_KINDS or not isinstance(items, list)
                or not 1 <= len(items) <= MAX_ITEMS or any(item is None for item in items)):
            raise ValueError('Connected media sockets require nonempty bounded item lists')


def _prepare(media, controls):
    worker = _control(controls['is_worker'], 'is_worker', bool)
    delegate = _control(controls['delegate_only'], 'delegate_only', bool)
    passthrough = _control(controls['pass_through'], 'pass_through', bool)
    _control(controls['load_balance'], 'load_balance', bool)  # Routing is master-owned.
    if worker:
        raise ValueError('DistributedMediaCollector does not support direct-worker execution')
    participants = _names(controls['enabled_worker_ids'], 'enabled_worker_ids')
    for participant in participants:
        identifier(participant)
        if not participant.startswith('serverless:') or not participant[len('serverless:'):]:
            raise ValueError('DistributedMediaCollector requires serverless participant IDs')
    expected = _names(controls['expected_media'], 'expected_media')
    if any(socket not in SOCKET_KINDS for socket in expected):
        raise ValueError('Unknown expected_media socket')
    local = {socket: items for socket, items in media.items() if items is not None}
    _validate_media(local, local)
    expected = expected or list(local)
    if not expected:
        raise ValueError('At least one connected media socket is required')
    if not delegate or passthrough:
        _validate_media(local, expected)
    if delegate and not passthrough and not participants:
        raise ValueError('Delegate-only collection requires serverless participants')
    return local, participants, expected, delegate, passthrough


def _outputs(media):
    return tuple(media.get(socket, []) for socket in SOCKET_KINDS)


def _consume(message, participants, expected, received):
    if not isinstance(message, dict):
        raise RuntimeError('Invalid serverless media delivery')
    participant = message.get('worker_id')
    if not isinstance(participant, str) or participant not in participants:
        raise RuntimeError('Unexpected serverless participant')
    if participant in received:
        raise RuntimeError('Duplicate serverless participant delivery')
    if message.get('is_last') is not True:
        raise RuntimeError('Serverless media delivery must be atomic and final')
    if 'serverless_error' in message:
        if (set(message) != {'worker_id', 'serverless_error', 'is_last'}
                or not isinstance(message['serverless_error'], str) or not message['serverless_error']):
            raise RuntimeError('Invalid serverless error delivery')
        # Do not echo provider-controlled text, credentials or signed URLs.
        raise RuntimeError(f'Serverless participant {participant} failed to produce media')
    if set(message) != {'worker_id', 'media', 'is_last'}:
        raise RuntimeError('Invalid serverless media delivery fields')
    _validate_media(message['media'], expected)
    received[participant] = message['media']


async def _collect(job, media, controls):
    ps = server.PromptServer.instance
    queue = None
    try:
        # Reuse the master's precreated queue, including early completions.
        async with ps.distributed_jobs_lock:
            queue = ps.distributed_pending_jobs.setdefault(job, asyncio.Queue())
        local, participants, expected, delegate, passthrough = _prepare(media, controls)
        if passthrough or not participants:
            return _outputs(local)
        received = {}
        started = time.monotonic()
        while len(received) < len(participants):
            comfy.model_management.throw_exception_if_processing_interrupted()
            async with ps.distributed_jobs_lock:
                if ps.distributed_pending_jobs.get(job) is not queue:
                    raise RuntimeError('Serverless collector queue was removed or replaced')
            try:
                message = await asyncio.wait_for(queue.get(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                comfy.model_management.throw_exception_if_processing_interrupted()
                # wait_for cancellation may race an atomic manager delivery.
                if not queue.empty():
                    continue
                manager = getattr(ps, 'distributed_serverless', None)
                for participant in participants:
                    if participant in received:
                        continue
                    if manager is not None and manager.is_pending(job, participant):
                        continue  # The manager enforces its remote absolute deadline.
                    if manager is not None and manager.owns(job, participant):
                        raise RuntimeError('Serverless participant stopped without delivering media')
                    # The master queues its prompt before manager.start journals slots.
                    if time.monotonic() - started >= START_GRACE_SECONDS:
                        raise RuntimeError('Serverless manager did not start the expected participant')
                continue
            _consume(message, participants, expected, received)

        comfy.model_management.throw_exception_if_processing_interrupted()
        # Drain and close atomically with the manager's delivery lock. A queued
        # duplicate after the final expected message must not yield success.
        async with ps.distributed_jobs_lock:
            if ps.distributed_pending_jobs.get(job) is not queue:
                raise RuntimeError('Serverless collector queue was removed or replaced')
            while not queue.empty():
                _consume(queue.get_nowait(), participants, expected, received)
            del ps.distributed_pending_jobs[job]
        ordered = [] if delegate else [local]
        ordered.extend(received[participant] for participant in participants)
        return tuple([item for source in ordered for item in source.get(socket, [])]
                     for socket in SOCKET_KINDS)
    except BaseException:
        # Remove only our queue before cancellation can trigger final deliveries.
        async with ps.distributed_jobs_lock:
            if queue is not None and ps.distributed_pending_jobs.get(job) is queue:
                del ps.distributed_pending_jobs[job]
        manager = getattr(ps, 'distributed_serverless', None)
        if manager is not None:
            try:
                await manager.cancel(collector_job_id=job)
            except BaseException:
                pass  # Preserve the original failure/interruption, not cleanup's.
        raise
    finally:
        async with ps.distributed_jobs_lock:
            if queue is not None and ps.distributed_pending_jobs.get(job) is queue:
                del ps.distributed_pending_jobs[job]


async def _bridge_result(coro):
    # The legacy sync bridge catches Exception, not asyncio.CancelledError.
    # Carry BaseException as data so cancellation is re-raised on the caller.
    try:
        return await coro, None
    except BaseException as error:
        return None, error


class DistributedMediaCollector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        types = (io.Image, io.Audio, io.Mask, io.String, io.Video)
        return io.Schema(
            node_id='DistributedMediaCollector', display_name='Distributed Media Collector',
            category='utils',
            inputs=[io.Boolean.Input('load_balance', default=False),
                    # Native non-widget inputs expose forceInput via extra_dict.
                    *[kind.Input(socket, optional=True,
                                 **({'force_input': True} if socket == 'texts'
                                    else {'extra_dict': {'forceInput': True}}))
                      for socket, kind in zip(SOCKET_KINDS, types)]],
            outputs=[kind.Output(display_name=socket, is_output_list=True)
                     for socket, kind in zip(SOCKET_KINDS, types)],
            is_input_list=True, accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, images=None, audio=None, masks=None, texts=None, video=None,
                load_balance=False, multi_job_id='', is_worker=False,
                enabled_worker_ids='[]', delegate_only=False, pass_through=False,
                expected_media='[]'):
        media = dict(images=images, audio=audio, masks=masks, texts=texts, video=video)
        controls = dict(load_balance=load_balance, is_worker=is_worker,
                        enabled_worker_ids=enabled_worker_ids, delegate_only=delegate_only,
                        pass_through=pass_through, expected_media=expected_media)
        job = _control(multi_job_id, 'multi_job_id', str)
        if not job:
            local, participants, _, _, passthrough = _prepare(media, controls)
            if participants and not passthrough:
                raise ValueError('Serverless collection requires multi_job_id')
            return io.NodeOutput(*_outputs(local))
        identifier(job)
        result, error = run_async_in_server_loop(_bridge_result(_collect(job, media, controls)))
        if error is not None:
            raise error
        return io.NodeOutput(*result)
