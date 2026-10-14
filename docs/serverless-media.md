# Serverless images and audio

**Distributed Collector** (`DistributedCollector`) collects `IMAGE` and/or `AUDIO`
from direct workers and serverless pools. There is no separate
`DistributedMediaCollector`; replace that removed node with `DistributedCollector`
for image/audio workflows. The Collector has no MASK, STRING, or native VIDEO
output sockets.

`comfy-media-v1` is an **explicit endpoint profile**, not an automatically detected
capability. It returns typed image/audio items to the Collector. The standalone
bridge's generic contract also supports `MASK`, `STRING`, and native ComfyUI
`VIDEO`, but that does not add those outputs to the Collector. Omitting
`media_profile` still selects the image-only `legacy-image-v1` contract.
See [legacy serverless setup](serverless-workers.md) for credentials and recovery.

## Use the Collector with a typed endpoint

1. Prepare an endpoint with the [standalone media bridge](../serverless_bridge/README.md)
   and the workflow's models/custom nodes. The endpoint needs only the bridge,
   **not the full ComfyUI-Distributed extension**.
2. In the serverless worker settings, set **Endpoint media profile** to
   **Typed media v1 (requires endpoint bridge)** (`media_profile: "comfy-media-v1"`).
   Start with **Images per run = 1**. This existing setting counts remote jobs,
   not the number of typed items each job returns.
3. Use **Distributed Collector** (`DistributedCollector`), connect `images`,
   `audio`, or both, and connect the corresponding outputs downstream. Image-only
   and audio-only workflows use the same node.
4. Queue through Distributed. **Delegate Only** excludes local upstream
   generation; normal collection includes the master's local values.

**Test Saved Endpoint** checks provider authentication/health, not bridge
installation, model availability, or typed execution. Selecting a profile does
not install or upgrade anything on the endpoint.

The Collector accepts optional `images` and `audio` inputs, including ComfyUI list
inputs, and returns two ordinary outputs—not list outputs:

| Socket / output index | Native type | Aggregation |
| --- | --- | --- |
| `images` / 0 | `IMAGE` | Concatenate BHWC tensors along the batch dimension |
| `audio` / 1 | `AUDIO` | Concatenate waveforms along the sample dimension |

For each connected socket, order is **master first, then prepared participant
order, then that participant's item order**—never completion order. Delegate-only
mode omits the master. Image dimensions/channels and audio batch/channel dimensions
must be compatible for concatenation. Use a common audio sample rate; the Collector
does not resample or align streams. Audio-only collection returns `None` for images;
when no audio is supplied it returns a silent placeholder, not an empty list.

The typed wire contract carries 1–64 items per connected socket. The entire remote
response is validated and decoded before any of its results are delivered;
missing, duplicate, mismatched, or corrupt typed results fail collection. Validated
wire lists are then combined into the Collector's ordinary outputs.

Direct workers, `legacy-image-v1` pools and `comfy-media-v1` pools can share an
image-only Collector run. Audio requires direct workers or `comfy-media-v1` pools.
If any Collector requests `exact`, a selected `legacy-image-v1` pool is rejected
before dispatch instead of silently returning 8-bit images.

## Image return format and quality

The Collector's saved `transfer_format` default remains **`legacy_png`**, with
`transfer_quality` **90**. For `comfy-media-v1`, the compiler maps `legacy_png` to
**`png`**, so normal serverless returns use compressed 8-bit PNG. `exact` is opt-in.
These controls affect **remote IMAGE returns only**, not master-local images,
uploaded inputs or AUDIO. The bridge's generic MASK, STRING and native VIDEO
codecs are also unaffected.

- `exact`: preserve supported floating-point dtype and values, with no image
  quantization. Available for direct workers and `comfy-media-v1` pools; it is
  never downgraded to an 8-bit format. An older master that rejects exact transfer
  causes an error, not a legacy retry. Update both sides before selecting it.
- `legacy_png`, `png`, `ffv1`: lossless **8-bit image pixels**, not exact float
  tensors. For typed serverless returns, image values are clipped to [0,1] and
  quantized; `legacy_png` is sent as `png`. Direct workers retain the original
  JSON/base64 framing for `legacy_png`.
- `jpeg`, `webp`: lossy still-image compression.
- `h264`, `h265` (**experimental**), `av1`: lossy video encoding of each IMAGE
  tensor's frame batch. This is an image-return encoding, not a native VIDEO output.
- Quality 1–100 affects lossy formats only, using the existing Collector's
  quality-to-CRF mapping for video. Dimensions, frame order and channel count
  are preserved for typed serverless returns; alpha is carried separately where
  needed, not silently dropped.

Both master and endpoint must have this bridge revision for compressed image
returns. Pillow handles still formats and CPU/software PyAV encoders handle video;
unavailable codecs fail clearly rather than silently changing the selection.
Compression can reduce result egress and inline response size, but does not relax
encoded/decoded byte budgets or the input request limit. Compressed IMAGE batches
are capped at 240 frames and 8192 pixels per dimension, with a conservative
256 MiB aggregate decode budget (including codec scratch and alpha). Direct
workers have different limits and non-exact compatibility fallbacks; see
[result transfer](compressed-result-transfer.md). The bridge's internal
`MediaExportV1` default is still `exact` for generic callers that omit controls;
the Collector compiler explicitly supplies `png` for the normal saved default.

## Inputs and fidelity

Automatic local-file transfer supports stock `LoadImage`, plus explicit stock
`LoadAudio` (`audio` filename) and `LoadVideo` (`file` filename) adapters. Audio and
video loaders require a literal filename, no extra controls, and output slot 0.
Files must be under ComfyUI's input/output/temp roots. Custom file loaders are not
supported; detected unsupported file references fail rather than silently
becoming endpoint-local paths. Models and custom nodes are never transferred.

- `LoadImage` uses the existing bounded file/base64 image staging path.
- `LoadAudio` is decoded on the master using native PCM semantics with bounded
  accumulation, then transferred as a waveform and sample rate. It does not
  resample; integer PCM conversion follows ComfyUI's `f32_pcm` behavior.
- `LoadVideo` transfers opaque file bytes as native `VideoFromFile`, without
  decoding frames or transcoding the container.
- Typed tensor transport is `tensor-le-v1`: contiguous little-endian float16,
  bfloat16, float32, or float64 bytes with exact dtype/shape/layout metadata.
  Only dense, finite tensors with positive dimensions are supported. IMAGE has
  1, 3, or 4 channels. Bridge export explicitly detaches/stages tensors to CPU;
  the pure codec itself rejects non-CPU storage. Remote decoding returns
  independently owned CPU values. STRING transport is exact UTF-8 (`utf8-v1`).

### Bridge-level native VIDEO support

The generic bridge can import/export native VIDEO, and the typed input adapter
can stage `LoadVideo` for upstream processing. This is not a Collector VIDEO
output or a generic video converter.

The adapter supports the pinned native classes **exactly**, not arbitrary
subclasses or objects exposing `save_to()` / `get_components()`:

- `VideoFromComponents`: images, optional matching BHW or BHW1 alpha, optional
  audio, positive rational frame rate, bounded JSON metadata, positive integer
  bit depth, and `sRGB`, `HDR`, or `HDR PQ` color space are preserved.
- `VideoFromFile`: regular local files within allowed media roots or an exact
  `BytesIO`; opaque bytes, finite signed start/duration, and optional four-integer
  crop are preserved without normalization.
- `VideoFromList`: 1–64 flat exact `VideoFromFile` segments, plus optional
  `complete_audio`, in order.

Unknown classes, incompatible native state, nested/component segment lists,
unsupported tensors, invalid metadata, and oversized payloads are rejected.
There is no fallback transcode, frame extraction, color conversion, or container
repair/validation. A downstream native consumer still needs to read the original
container. Decoded file-backed VIDEO values own independent `BytesIO` storage;
component tensors also own their storage, so staging-file cleanup cannot
invalidate a returned VIDEO.

## Size and operational limits

- The typed contract caps each artifact at **256 MiB**, aggregate artifact bytes
  at **512 MiB**, and each artifact list at **128** entries. These are validation
  ceilings, not a promise that inline requests can carry that much data.
- The built-in master packs inputs inline and enforces **10,000,000 bytes for the
  complete serialized Runpod `/run` request**, including workflow, policy and
  base64. Legacy image reads also retain their 10,000,000-byte aggregate guard.
  There is **no built-in master-side large-input uploader** in PR2.
- The endpoint defaults to inline results and caps its result JSON at
  **10,000,000 bytes**. Larger results require an operator-supplied publisher
  returning expiring public HTTPS artifact URLs. See the bridge guide for
  bounded download/upload responsibilities and retention.
- The generic five-kind typed envelope is a serverless bridge contract. Direct
  workers support Collector images/audio, including opt-in exact IMAGE transport,
  not generic MASK/STRING/VIDEO Collector outputs. The typed compiler rejects
  unsupported distributed/batch nodes and collector-to-collector dependencies.
  Downstream processing stays on the master. [Image-list scheduling](image-list-queue.md)
  is a separate execution mode.

## Timeout is not proof that Comfy stopped

Provider cancellation, a handler timeout, or cancellation of its Python task
**does not establish that the local Comfy prompt stopped**. The bridge never
posts `/interrupt`: the pinned core route checks a queue snapshot and then
interrupts globally, so even a prompt ID does not make it race-free.

If submission/termination or callback teardown cannot be confirmed, the bridge
raises `MediaExecutionUncertain` and retains owned input/exporter files—even
after terminal Comfy history if a downloader/publisher ignores cancellation.
Callback teardown has a separate 10-second grace; late callback exit does not
clean retained files. Callbacks must be cooperative async code; synchronous
event-loop blocking cannot be hard-timed-out.

The endpoint operator must **quarantine the owning worker/environment, reconcile
the local prompt and readers or terminate that environment, and establish that
they have stopped before cleanup or worker reuse**. Do not automatically retry
the request or turn every handler exception into a safely reusable terminal
provider job. Master-side provider status/reconciliation alone is not proof
that a surviving local Comfy process or callback has stopped.

These docs describe the implementation, not a test result or live Runpod/provider
or GPU end-to-end validation.
