# Typed serverless media (PR2)

`comfy-media-v1` adds typed, lossless list results for `IMAGE`, `AUDIO`, `MASK`,
`STRING`, and native ComfyUI `VIDEO`. It is an **explicit endpoint profile**, not
an automatically detected capability. Omitting `media_profile` still selects
`legacy-image-v1`; existing image collectors and handlers keep their old contract.
See [legacy serverless setup](serverless-workers.md) for credentials and recovery.

## Use a typed collector

1. Prepare an endpoint with the [standalone media bridge](../serverless_bridge/README.md)
   and the workflow's models/custom nodes. The endpoint needs only the bridge,
   **not the full ComfyUI-Distributed extension**.
2. In the serverless worker settings, set **Endpoint media profile** to
   **Typed media v1 (requires endpoint bridge)** (`media_profile: "comfy-media-v1"`).
   Start with **Images per run = 1**. This existing setting counts remote jobs,
   not the number of typed items each job returns.
3. Use **Distributed Media Collector** (`DistributedMediaCollector`), connect at
   least one typed input, and connect its outputs to the appropriate downstream
   nodes. For literal text, connect a `PrimitiveString` node to `texts`.
4. Queue through Distributed. **Delegate Only** excludes local upstream
   generation; normal collection includes the master's local values.

**Test Saved Endpoint** checks provider authentication/health, not bridge
installation, model availability, or typed execution. Selecting a profile does
not install or upgrade anything on the endpoint.

The collector has five optional inputs and five **list outputs**, in this order:

| Socket / output index | Native type | Each list item |
| --- | --- | --- |
| `images` / 0 | `IMAGE` | One BHWC tensor, including its batch dimension |
| `audio` / 1 | `AUDIO` | `{ "waveform": BCT_tensor, "sample_rate": positive_int }` |
| `masks` / 2 | `MASK` | One BHW tensor |
| `texts` / 3 | `STRING` | Exact text, including whitespace or an empty string |
| `video` / 4 | `VIDEO` | One supported native video object |

For each connected socket, order is **master first, then the prepared participant
order, then that participant's item order**—never completion order. Delegate-only
mode omits the master. An unconnected socket returns `[]`. A connected remote
socket must return 1–64 items; an empty string is an item, an empty list is not.
Lists are not concatenated into one tensor, resampled, resized, or aligned across
sockets. Different items may have different valid shapes or audio sample rates.
The entire remote response is validated and decoded before any of its results
are delivered; missing, duplicate, mismatched, or corrupt results fail collection.

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

### Native VIDEO is not a generic video converter

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
- Remote typed transport is **serverless only**. Direct-worker typed transport
  is not implemented. Do not mix a typed collector with legacy image collectors,
  unsupported distributed/batch nodes, or collector-to-collector dependencies.
  Downstream processing stays on the master. Image-list scheduling is separate
  PR3 work, not part of this contract.

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

These docs describe the implementation and its offline tests; they do not claim
live Runpod/provider or GPU end-to-end validation.
