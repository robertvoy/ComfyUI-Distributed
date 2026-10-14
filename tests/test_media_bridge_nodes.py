"""Real ComfyUI V3 bridge mapping, file ownership and exact CPU round trips."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_real_media_bridge_nodes():
    core = os.environ.get("COMFYUI_SOURCE_ROOT")
    if core is None:
        pytest.skip("Set COMFYUI_SOURCE_ROOT for CPU media bridge acceptance")
    assert core and (Path(core) / "comfy_api/latest/_input_impl/video_types.py").is_file(), (
        "Configured COMFYUI_SOURCE_ROOT must provide native video classes"
    )
    probe = r'''
import asyncio, importlib.util, io, json, os, sys, tempfile
from fractions import Fraction
import av
import numpy as np
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, sys.argv[1])
import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.disable_assets = True
def forbidden(*args, **kwargs):
    raise AssertionError('Bridge tests must not query or initialize a GPU')
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
for name in ('init', '_lazy_init', 'current_device'):
    setattr(torch.cuda, name, forbidden)
import execution
import folder_paths
from comfy_api.latest._input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList
from comfy_api.latest._util import VideoComponents
for name in ('is_available', 'device_count'):
    setattr(torch.cuda, name, forbidden)
spec = importlib.util.spec_from_file_location('typed_bridge_probe', Path(sys.argv[2])/'serverless_bridge/__init__.py', submodule_search_locations=[str(Path(sys.argv[2])/'serverless_bridge')])
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)
from typed_bridge_probe.nodes import MediaImportV1, MediaExportV1, export_values, import_value
from typed_bridge_probe.contract import SOCKET_KINDS
from typed_bridge_probe.video_codec import encode_video

def tiny_video():
    buffer = io.BytesIO()
    with av.open(buffer, 'w', format='matroska') as container:
        stream = container.add_stream('ffv1', rate=Fraction(24, 1))
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for level in (16, 64, 128):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), level, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    return buffer.getvalue()

def exact_tensor(actual, expected):
    assert actual.device.type == 'cpu' and actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

def exact_video(actual, expected, wire):
    assert type(actual) is type(expected)
    assert encode_video(actual) == wire
    if type(actual) is VideoFromComponents:
        a = vars(actual)['_VideoFromComponents__components']
        e = vars(expected)['_VideoFromComponents__components']
        assert type(a) is VideoComponents and type(a.frame_rate) is Fraction
        assert a.frame_rate == e.frame_rate and a.metadata == e.metadata
        assert a.metadata is not e.metadata
        for name in ('images', 'alpha'): exact_tensor(getattr(a, name), getattr(e, name))
        exact_tensor(a.audio['waveform'], e.audio['waveform'])
        assert a.audio['sample_rate'] == e.audio['sample_rate']
        assert actual.get_bit_depth() == 10 and actual.get_color_space() == 'HDR PQ'
    else:
        values = actual.videos if type(actual) is VideoFromList else [actual]
        for value in values:
            buffer = vars(value)['_VideoFromFile__file']
            assert type(buffer) is io.BytesIO and not buffer.closed
            buffer.seek(0)
            with av.open(buffer) as container:
                assert len(list(container.decode(video=0))) == 3
        if type(actual) is VideoFromList:
            exact_tensor(actual.complete_audio['waveform'], expected.complete_audio['waveform'])

async def check():
    extension = await bridge.comfy_entrypoint()
    assert await extension.get_node_list() == [MediaImportV1, MediaExportV1]
    imp = MediaImportV1.GET_NODE_INFO_V1()
    exp = MediaExportV1.GET_NODE_INFO_V1()
    assert imp['output'] == ['IMAGE','AUDIO','MASK','STRING','VIDEO','*'], imp
    assert exp['output_node'] and MediaExportV1.define_schema().is_input_list, exp
    with tempfile.TemporaryDirectory(prefix='typed-media-', dir=os.environ.get('TMPDIR')) as scratch:
        base = Path(scratch)
        roots = {key: base/key for key in ('input','output','temp')}
        for path in roots.values(): path.mkdir()
        def annotated(name):
            for key, root in roots.items():
                suffix = f' [{key}]'
                if name.endswith(suffix): return str(root/name[:-len(suffix)])
            return str(roots['input']/name)
        with patch.object(folder_paths,'get_input_directory',lambda: str(roots['input'])), patch.object(folder_paths,'get_output_directory',lambda: str(roots['output'])), patch.object(folder_paths,'get_temp_directory',lambda: str(roots['temp'])), patch.object(folder_paths,'get_annotated_filepath',annotated):
            examples = {'images': torch.rand(2,2,3,4,dtype=torch.float64),
                        'masks': torch.tensor([[[0.123456789,0.987654321]]],dtype=torch.float64),
                        'audio': {'waveform':torch.randn(1,6,31,dtype=torch.float64),'sample_rate':44100},
                        'texts': '  café\n🎥\r\n'}
            media = tiny_video()
            sources = [roots['input']/f'source-{n}.mkv' for n in range(3)]
            for path in sources: path.write_bytes(media)
            images = torch.arange(36, dtype=torch.float64, device='cpu').reshape(2,2,3,3) / 997
            images[0,0,0,0] = -0.0
            component_video = VideoFromComponents(VideoComponents(
                images=images, frame_rate=Fraction(30000,1001),
                alpha=torch.full((2,2,3,1),0.123456789,dtype=torch.float64,device='cpu'),
                audio=examples['audio'], metadata={'title':'雪', 'nested':[True,None,-4,0.125]}),
                bit_depth=10,color_space='HDR PQ')
            videos = [component_video,
                      VideoFromFile(str(sources[0]),start_time=-0.0,duration=0,crop=(0,0,2,2)),
                      VideoFromList([VideoFromFile(str(sources[1]),start_time=-1.25,duration=-2.5,crop=(-3,5,7,-9)),
                                     VideoFromFile(str(sources[2]),start_time=0,duration=0)],
                                    complete_audio=examples['audio'])]
            for socket,value in [*examples.items(), *[('video',v) for v in videos]]:
                kind = SOCKET_KINDS[socket]
                expected_wire = encode_video(value) if kind == 'VIDEO' else None
                binding = dict(binding_id='b-'+socket,collector_id='collector',job_id='job',participant_id='worker',node_id='export-'+socket,socket=socket,kind=kind)
                controls = {'request_id':'request','binding_json':json.dumps(binding)}
                data, missing, v3 = execution.get_input_data(controls,MediaExportV1,'export-'+socket)
                assert not missing, missing
                data[socket] = [value]
                results = await execution._async_map_node_over_list('probe','export-'+socket,MediaExportV1,data,MediaExportV1.FUNCTION,v3_data=v3)
                results = await execution.resolve_map_node_over_list_results(results)
                raw = results[0].ui
                exported = raw['media_exports'][0]
                assert exported['binding']['count'] == 1
                artifact = exported['artifacts'][0]
                transport = artifact.pop('transport')
                filename = transport['subfolder']+'/'+transport['filename']+' [temp]'
                decoded = import_value(artifact,filename)
                if kind in ('IMAGE','MASK'):
                    assert decoded.dtype == value.dtype and torch.equal(decoded,value)
                elif kind == 'AUDIO':
                    assert decoded['sample_rate'] == value['sample_rate']
                    assert torch.equal(decoded['waveform'],value['waveform'])
                elif kind == 'VIDEO': exact_video(decoded,value,expected_wire)
                else: assert decoded == value
                mapped, missing, hidden = execution.get_input_data({'artifact_json':json.dumps(artifact),'filename':filename},MediaImportV1,'import')
                got = await execution._async_map_node_over_list('probe','import',MediaImportV1,mapped,MediaImportV1.FUNCTION,v3_data=hidden)
                got = await execution.resolve_map_node_over_list_results(got)
                assert len(got[0].result) == 6
                assert all(x is None for n,x in enumerate(got[0].result) if n != list(SOCKET_KINDS).index(socket))
                bad = dict(artifact,sha256='0'*64)
                try: import_value(bad,filename)
                except ValueError: pass
                else: raise AssertionError('checksum accepted')
                try: import_value(artifact,str(base/'outside.bin'))
                except ValueError: pass
                else: raise AssertionError('outside path accepted')
                if kind == 'VIDEO':
                    mapped_video = got[0].result[list(SOCKET_KINDS).index(socket)]
                    # Remove both the exported artifact and the original input
                    # containers before checking independently owned BytesIOs.
                    Path(annotated(filename)).unlink()
                    originals = value.videos if type(value) is VideoFromList else [value]
                    for original in originals:
                        if type(original) is VideoFromFile:
                            Path(vars(original)['_VideoFromFile__file']).unlink()
                    exact_video(decoded,value,expected_wire)
                    exact_video(mapped_video,value,expected_wire)
                    if type(value) is VideoFromFile:
                        assert tuple(decoded.get_components().images.shape) == (3,2,2,3)
            text_binding = dict(binding_id='empty',collector_id='collector',job_id='job',participant_id='worker',node_id='export-empty',socket='texts',kind='STRING')
            out = export_values('request',text_binding,[''])
            assert out['artifacts'][0]['byte_length'] == 0
            before = set(roots['temp'].rglob('*.bin'))
            try: export_values('request',text_binding,['first',None])
            except ValueError: pass
            else: raise AssertionError('invalid text accepted')
            assert set(roots['temp'].rglob('*.bin')) == before
    print('MEDIA_BRIDGE_CPU_OK')
asyncio.run(check())
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("COMFYUI_SOURCE_ROOT", None)
    scratch = Path(os.environ.get("TMPDIR", str(Path.home() / ".hermes/cache/scratch")))
    scratch.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               TMPDIR=str(scratch))
    result = subprocess.run([sys.executable, "-c", probe, str(Path(core).resolve()), str(ROOT)],
                            capture_output=True, text=True, timeout=90, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MEDIA_BRIDGE_CPU_OK" in result.stdout
