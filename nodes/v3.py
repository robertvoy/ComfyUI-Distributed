"""Native V3 schemas with the existing execution algorithms kept intact.

Runtime objects are private, per-execution helpers, not registered V1 nodes.
This avoids sharing mutable instance state through V3's sanitized class clones.
"""
import comfy.samplers
from comfy_api.v0_0_2 import io

from . import utilities as _utilities
from .collector import DistributedCollectorNode as _CollectorRuntime
from ..utils import transfer_codec as _transfer_codec
from .distributed_upscale import UltimateSDUpscaleDistributed as _UpscaleRuntime


class DistributedSeed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='DistributedSeed', display_name='Distributed Seed', category='utils',
            inputs=[io.Int.Input('seed', default=1125899906842, min=0,
                                 max=1125899906842624, force_input=False)],
            outputs=[io.Int.Output(display_name='seed')],
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, seed, is_worker=False, worker_id=''):
        return io.NodeOutput(*_utilities.DistributedSeed().distribute(seed, is_worker, worker_id))


class DistributedValue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='DistributedValue', display_name='Distributed Value', category='utils',
            inputs=[io.String.Input('default_value', default=''),
                    io.String.Input('worker_values', default='{}')],
            outputs=[io.AnyType.Output(display_name='value')],
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, default_value, worker_values='{}', is_worker=False, worker_id=''):
        return io.NodeOutput(*_utilities.DistributedValue().distribute(
            default_value, worker_values, is_worker, worker_id))


class DistributedModelName(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='DistributedModelName', display_name='Distributed Model Name', category='utils',
            inputs=[io.String.Input('text', default='')],
            outputs=[io.AnyType.Output(display_name='output')],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo], is_output_node=True,
        )

    @classmethod
    def execute(cls, text):
        result = _utilities.DistributedModelName().log_input(
            text, unique_id=cls.hidden.unique_id, extra_pnginfo=cls.hidden.extra_pnginfo)
        return io.NodeOutput(*result['result'], ui=result['ui'])


class ImageBatchDivider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='ImageBatchDivider', display_name='Image Batch Divider', category='image',
            inputs=[io.Image.Input('images'),
                    io.Int.Input('divide_by', default=2, min=1, max=10, step=1,
                                 display_mode=io.NumberDisplay.number,
                                 tooltip='Number of parts to divide the batch into')],
            # The existing frontend still displays only divide_by sockets.
            outputs=[io.Image.Output(display_name=f'batch_{index + 1}') for index in range(10)],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, divide_by):
        return io.NodeOutput(*_utilities.ImageBatchDivider().divide_batch(images, divide_by))


class AudioBatchDivider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='AudioBatchDivider', display_name='Audio Segment Divider', category='audio',
            inputs=[io.Audio.Input('audio'),
                    io.Int.Input('divide_by', default=2, min=1, max=10, step=1,
                                 display_mode=io.NumberDisplay.number,
                                 tooltip='Number of sequential time segments to create')],
            outputs=[io.Audio.Output(display_name=f'audio_{index + 1}') for index in range(10)],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, divide_by):
        return io.NodeOutput(*_utilities.AudioBatchDivider().divide_audio(audio, divide_by))


class DistributedEmptyImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='DistributedEmptyImage', display_name='Distributed Empty Image', category='image',
            inputs=[io.Int.Input('height', default=64, min=1, max=4096, step=1),
                    io.Int.Input('width', default=64, min=1, max=4096, step=1),
                    io.Int.Input('channels', default=3, min=1, max=4, step=1)],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, height, width, channels):
        return io.NodeOutput(*_utilities.DistributedEmptyImage().create(height, width, channels))


class DistributedCollector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='DistributedCollector', display_name='Distributed Collector', category='image',
            inputs=[io.Boolean.Input('load_balance', default=False,
                                     tooltip='Run this workflow on one least-busy participant (master included when participating).'),
                    io.Image.Input('images', optional=True), io.Audio.Input('audio', optional=True),
                    io.Combo.Input('transfer_format', options=list(_transfer_codec.TRANSFER_FORMATS),
                                   default=_transfer_codec.DEFAULT_TRANSFER_FORMAT, optional=True,
                                   tooltip=_transfer_codec.TRANSFER_FORMAT_TOOLTIP),
                    io.Int.Input('transfer_quality', default=_transfer_codec.DEFAULT_TRANSFER_QUALITY,
                                 min=1, max=100, step=1, optional=True,
                                 tooltip=_transfer_codec.TRANSFER_QUALITY_TOOLTIP)],
            outputs=[io.Image.Output(display_name='images'), io.Audio.Output(display_name='audio')],
            is_input_list=True, accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, images=None, load_balance=False, audio=None, multi_job_id='',
                is_worker=False, master_url='', enabled_worker_ids='[]', worker_batch_size=1,
                worker_id='', pass_through=False, delegate_only=False,
                transfer_format=_transfer_codec.DEFAULT_TRANSFER_FORMAT,
                transfer_quality=_transfer_codec.DEFAULT_TRANSFER_QUALITY):
        return io.NodeOutput(*_CollectorRuntime().run(
            images=images, load_balance=load_balance, audio=audio, multi_job_id=multi_job_id,
            is_worker=is_worker, master_url=master_url, enabled_worker_ids=enabled_worker_ids,
            worker_batch_size=worker_batch_size, worker_id=worker_id,
            pass_through=pass_through, delegate_only=delegate_only,
            transfer_format=transfer_format, transfer_quality=transfer_quality))


class UltimateSDUpscaleDistributed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='UltimateSDUpscaleDistributed',
            display_name='Ultimate SD Upscale Distributed (No Upscale)', category='image/upscaling',
            inputs=[
                io.Image.Input('upscaled_image'), io.Model.Input('model'),
                io.Conditioning.Input('positive'), io.Conditioning.Input('negative'), io.Vae.Input('vae'),
                io.Int.Input('seed', default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input('steps', default=20, min=1, max=10000),
                io.Float.Input('cfg', default=8.0, min=0.0, max=100.0),
                io.Combo.Input('sampler_name', options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input('scheduler', options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input('denoise', default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input('tile_width', default=512, min=64, max=2048, step=8),
                io.Int.Input('tile_height', default=512, min=64, max=2048, step=8),
                io.Int.Input('padding', default=32, min=0, max=256, step=8),
                io.Int.Input('mask_blur', default=8, min=0, max=256),
                io.Boolean.Input('force_uniform_tiles', default=True),
                io.Boolean.Input('tiled_decode', default=False),
            ], outputs=[io.Image.Output()], accept_all_inputs=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _UpscaleRuntime.IS_CHANGED(**kwargs)

    @classmethod
    def execute(cls, upscaled_image, model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise, tile_width, tile_height, padding,
                mask_blur, force_uniform_tiles, tiled_decode, multi_job_id='', is_worker=False,
                master_url='', enabled_worker_ids='[]', worker_id='', tile_indices='', dynamic_threshold=8):
        return io.NodeOutput(*_UpscaleRuntime().run(
            upscaled_image, model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler, denoise, tile_width, tile_height, padding,
            mask_blur, force_uniform_tiles, tiled_decode, multi_job_id, is_worker,
            master_url, enabled_worker_ids, worker_id, tile_indices, dynamic_threshold))


NODES = [DistributedCollector, DistributedSeed, DistributedModelName, DistributedValue,
         ImageBatchDivider, AudioBatchDivider, DistributedEmptyImage, UltimateSDUpscaleDistributed]
