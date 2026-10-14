# Standalone `comfy-media-v1` endpoint bridge

This directory is independently installable. It registers two internal Comfy V3
nodes, `MediaImportV1` and `MediaExportV1`, and provides the asynchronous HTTP
executor `execute_media_request`. It does **not** start ComfyUI, provision an
endpoint, install models, or provide a provider worker/supervisor. The endpoint
needs this bridge, not the full ComfyUI-Distributed extension.

For collector setup, list ordering, supported native VIDEO states, and PR2
limitations, see [typed serverless media](../docs/serverless-media.md).

## Install in the endpoint image

1. Copy this complete directory to `<ComfyUI>/custom_nodes/serverless_bridge`.
   Use a compatible ComfyUI with `comfy_api.v0_0_2` and the native VIDEO classes
   used by [video_codec.py](video_codec.py). These exact-class/private-state
   adapters are tied to the supported core implementation, not arbitrary VIDEO
   plugins or future core layouts.
2. Restart ComfyUI so `comfy_entrypoint()` registers the two nodes. Keep the
   workflow's own models and custom nodes installed separately.
3. Make `<ComfyUI>/custom_nodes` importable by the handler process (for example,
   via its `PYTHONPATH`). The handler uses `aiohttp`; the nodes/codecs use
   ComfyUI's PyTorch/native media dependencies. The handler module itself does
   not import ComfyUI, PyTorch, Runpod, or Distributed bootstrap code.
4. Give the handler filesystem access to **the same input directory** used by
   that ComfyUI instance, plus its trusted HTTP URL. These are operator settings,
   never request-controlled paths/URLs. Staging/cleanup requires POSIX directory
   descriptor and no-follow file operations.

Installing the nodes alone is insufficient: the endpoint must route media-v1
inputs through this executor and implement the uncertainty policy below. A
legacy image handler cannot interpret the typed envelope.

## Callable integration

```python
from serverless_bridge.handler import execute_media_request


async def execute_endpoint_input(
    input_data, *, comfy_url, input_dir, temp_dir=None,
    download=None, publish=None,
):
    # Invoke only under a supervisor implementing the uncertainty policy below.
    # Exceptions intentionally propagate; this is NOT a provider worker loop.
    return await execute_media_request(
        input_data,
        comfy_url=comfy_url,
        input_dir=input_dir,
        timeout_seconds=1800,
        session=None,
        download=download,
        publish=publish,
        temp_dir=temp_dir,
    )
```

Pass the provider job's **`input` object**, not the whole job. Return the resulting
`{"media": ...}` object as provider output; Runpod exposes it as `output.media`.
Do not add another `output` wrapper inside it. `timeout_seconds` must be finite
and positive. If supplied, `session` is a caller-owned `aiohttp.ClientSession`
for local Comfy HTTP and remains open; otherwise the executor creates/closes its
own session. Input dictionaries are copied before graph rewriting.

**This callable is not a turnkey Runpod handler.** An outer SDK that catches every
exception and immediately reuses the worker is unsafe. The supervisor must
recognize `serverless_bridge.handler.MediaExecutionUncertain`, stop intake, and
quarantine/reconcile or terminate the owning environment before reuse. Merely
returning `{"error": ...}` or marking a provider job `FAILED` is not that policy.

## Request and response

A minimal text request builder (no files or model required by this graph):

```python
import json
import uuid


def make_text_request(text="hello"):
    request_id = uuid.uuid4().hex
    binding = {
        "binding_id": "text-binding",
        "collector_id": "collector-1",
        "job_id": "collector-job-1",
        "participant_id": "serverless:example:0",
        "node_id": "export-1",
        "socket": "texts",
        "kind": "STRING",
    }
    return {
        "workflow": {
            "text-1": {"class_type": "PrimitiveString", "inputs": {"value": text}},
            "export-1": {
                "class_type": "MediaExportV1",
                "inputs": {
                    "request_id": request_id,
                    "binding_json": json.dumps(binding),
                    "texts": ["text-1", 0],
                },
            },
        },
        "images": [],
        "media": {
            "version": 1,
            "profile": "comfy-media-v1",
            "request_id": request_id,
            "artifacts": [],
            "inputs": [],
            "outputs": [binding],
        },
    }
```

`workflow` is an API graph, not an editor workflow. Top-level input fields are
exactly `workflow`, `media`, and optional `images`. `media` has exactly the six
fields shown above. `media.outputs` declares one exporter per connected socket;
its binding must match that node's `binding_json` and request ID exactly. Socket
names/types are `images/IMAGE`, `audio/AUDIO`, `masks/MASK`, `texts/STRING`, and
`video/VIDEO`. Use unique request IDs. Identifiers are 1–200 characters from
`[A-Za-z0-9_.:/-]`; request scopes additionally reject empty, `.` or `..` slash
components. The executor validates the graph and submits **once**, with exactly
the declared exporter IDs as `partial_execution_targets`. Unrelated graph roots,
cycles, undeclared exporters, or caller-supplied `MediaImportV1` nodes fail.

For typed inputs, `media.artifacts` contains records with exactly:

| Field | Value |
| --- | --- |
| `id`, `kind` | Unique artifact ID and one of the five native types |
| `descriptor` | Kind-specific codec metadata; use the shared codec, not pickle |
| `byte_length`, `sha256` | Exact payload byte count and lowercase SHA-256 hex |
| `transport` | `{"type": "inline", "data_base64": "..."}` or the URL transport below |

Each `media.inputs` entry is
`{"node_id": "consumer", "input_name": "socket", "artifact_id": "artifact-id"}`.
It replaces an existing consumer input with a generated import link; every
artifact must be used, and destinations cannot repeat. The executor verifies
bytes/checksums, stages owned files, inserts imports, then prunes disconnected
loader ancestors. Optional legacy `images` entries are `{ "name": "input.png",
"image": "<base64>" }`, must match stock `LoadImage` references, and are rewritten
to owned staging paths. They are not an arbitrary filename-upload interface.

The returned `media` object has exactly `version`, `profile`, `request_id`,
`bindings`, `artifacts`, and `items`:

- `bindings`: every declared binding, unchanged, plus `count` (1–64).
- `artifacts`: the same artifact schema above with inline or URL transport.
- `items`: `{ "binding_id": "...", "item_index": 0, "artifact_id": "..." }`
  records covering every index exactly once, ordered by `item_index` when decoded.

All identity, completeness, type, size, and checksum checks must pass; do not
consume a partial result. The internal history transport `comfy-file` is local
only and must not be returned to the master. The executor reads validated
`media_exports` history records and bounded `/view` payloads before constructing
the external envelope. [contract.py](contract.py) is the schema authority.

## Optional image return encoding

IMAGE `MediaExportV1` nodes may also contain literal `transfer_format` and
`transfer_quality` controls. Defaults are `exact` and `90`; old exact requests
omit the controls and retain their original representation. Other media exports
must not carry image controls. The master compiler supplies them from the typed
Collector's widgets; callers should not alter binding identity metadata.

Formats: `exact`, `legacy_png`, `png`, `ffv1`, `jpeg`, `webp`, `h264`, `h265`, `av1`.
`exact` preserves float tensor bits. The image/video codecs quantize RGB/grayscale
to 8-bit; PNG/FFV1 are lossless at that precision, while the other codecs are
lossy. Quality is an integer 1–100 and affects only lossy formats. IMAGE batch
order, shape and channel count are retained, and alpha is not discarded.
Native VIDEO/MASK/AUDIO/STRING still use their original exact codecs.

Copy/update the complete bridge directory on both sides, including
`image_transfer.py` and `image_codec.py`. Pillow and the selected CPU/software
PyAV codec must be available; missing encoders fail without automatic fallback.
Compressed IMAGE artifacts use `image-transfer-v1`; decode through the shared
`codec.decode_payload`, which enforces payload and decoded-shape limits.
See [image transfer choices](../docs/serverless-media.md#image-return-format-and-quality).

## Transport budgets and ownership

- Contract maxima: 256 MiB per artifact, 512 MiB aggregate artifact bytes,
  128 artifacts, 64 items per binding. Request input totals include decoded
  legacy-image bytes. Decoders validate tensor/VIDEO sizes and layouts before
  allocating native storage; these are payload limits, **not peak-RSS guarantees**.
- The built-in master currently emits inline inputs only. Its complete Runpod
  request JSON is capped at **10,000,000 bytes**, including base64, graph and
  policy. The larger contract limits do not bypass this cap; there is no
  built-in large-input/object-storage uploader.
- The executor's result JSON is capped at **10,000,000 bytes**. Inline results
  are the default. When needed, `publish` is called for selected larger
  artifacts to fit that budget; supplying it does not force all results to URLs.
  Without a publisher, an oversized response fails instead of being truncated.
- Optional **async** `download(url, *, max_bytes)` returns `bytes`. The executor
  supplies the declared artifact length and checks exact length and SHA-256.
  The callback must enforce public HTTPS, DNS/IP restrictions, redirect checks,
  and bounded streaming **before allocation**; syntactic contract validation
  does not provide SSRF protection. No downloader is installed by default.
- Optional **async** `publish(metadata, payload)` uploads the exact `bytes` and
  returns exactly `{"type": "url", "url": "https://...", "expires_at": "..."}`.
  `metadata` is the artifact record without `transport`. Expiry must be a future,
  timezone-aware ISO timestamp; URLs use HTTPS port 443, no userinfo/fragment,
  and at most 8192 characters. Keep objects available long enough for collection.
  The callback/operator owns upload credentials, object expiry and cleanup,
  including orphan uploads after failure. There is no bridge storage backend.
- The master uses a separate bounded public-HTTPS downloader without provider
  credentials. Never send provider credentials to artifact hosts or log signed
  URLs. Endpoint callbacks must enforce equivalent protections themselves.

Callbacks must cooperate with async cancellation, finish/abort their own I/O,
and not detach readers/uploads that outlive the callback. Synchronous code that
blocks the event loop cannot be hard-timed-out by this executor.

## Cancellation, quarantine, and file retention

`timeout_seconds` covers staging, submission, polling, downloads and publication.
On failure/cancellation, there is a separate **10-second callback-drain grace**
and, when needed, a **10-second prompt-reconciliation grace**. These are recovery
allowances, not evidence that execution stopped. Ambiguous `/prompt` submission
is never retried. An identified pending prompt may be deleted from `/queue`,
with exact read-back; running prompts are never sent `/interrupt`. The pinned
core route has a snapshot/global-interrupt race even when given a prompt ID.

`MediaExecutionUncertain` means prompt acceptance/termination **or callback
teardown** is unconfirmed. Even successful terminal Comfy history is insufficient
if an async downloader/publisher ignores cancellation. The bridge retains input
and exporter files in that case; late callback exit does not automatically
remove retained files. Before accepting another job on that environment:

1. Quarantine it and stop new intake; do not automatically resubmit the request.
2. Reconcile the local prompt and all readers, or terminate the owning execution
   environment when their termination cannot be established.
3. Only then clean retained files and allow reuse/replacement. A provider terminal
   job status alone cannot certify a surviving local Comfy process or callback.

Staged inputs use random exclusive files under
`input_dir/distributed-media-v1/<sha256(request_id)>/`. Cleanup removes only files
created by that invocation, after confirmed terminal state (or before submission)
and reader teardown; it never recursively deletes a shared namespace.

`temp_dir=None` leaves exporter retention to the Comfy temp-directory/environment
lifecycle. Optionally pass the **trusted local Comfy temp root** to enable
best-effort cleanup of validated, invocation-owned exporter UUID `.bin` files
under the corresponding hashed namespace, only after terminal history and
collection/publisher teardown. Symlinked, hard-linked, replaced, missing, or
unidentifiable files are not claimed; directories and all unidentified leftovers
remain lifecycle-owned. This is not a temp-directory sweep. Independently decoded
VIDEO objects do not depend on either staging or exporter files.

Verification scope: source and offline tests. No live provider/GPU validation is
claimed, and this directory is not a tested turnkey provider deployment.
