# Dynamic image-list queue

Process a list of images across the master, direct local or remote ComfyUI workers, and compatible Runpod serverless pools without assigning fixed chunks to each participant.

The master evaluates the input list once. By default, the master and each selected direct worker receive one image job at a time; a serverless pool contributes bounded concurrent slots. All slots share the same queue: when a participant finishes, it gets the next unassigned image. Faster participants can therefore process more images. Each source frame is assigned once, and the master restores input order before running downstream nodes.

The existing **Orchestrator Only** setting (`settings.master_delegate_only`) excludes the master from per-image processing. The master still creates the input list and runs downstream nodes. API callers can override the setting for a run with `delegate_master`. There is no second image-list-specific toggle.

## Wiring

```text
Image list or IMAGE batch
          |
Distributed Image List Start -------- context ---------+
          | image                                     |
          v                                           |
    Per-image processing                              |
          | IMAGE                                     |
          v                                           v
                 Distributed Image List End
                            | image_list
                            v
                     Preview / Save / next stage
```

1. Install the same feature-capable version of Distributed on the master and every selected direct worker. Direct workers on an older image-list protocol fail preflight before any work is sent. Restart those instances and refresh the frontend after installation. Serverless endpoints instead need the compatible media bridge and supervision described below.
2. Enable Distributed and select direct workers and/or **Typed media v1** serverless pools. The master also participates unless **Orchestrator Only** is enabled. With the master participating, an empty worker selection runs locally through the same image-list scheduler; orchestrator-only mode requires at least one eligible direct worker or compatible serverless pool. Serverless-only selection is supported. The master must be reachable from direct remote workers at its configured callback address; serverless jobs do not use this callback.
3. Feed your image list or tensor batch into **Distributed Image List Start → images**.
4. Connect its `image` output through the per-image processing nodes to **Distributed Image List End → images**.
5. Connect Start's **context** directly to End's **context**. This is a dedicated `DISTRIBUTED_IMAGE_LIST_CONTEXT` socket, not editable text.
6. Put preview/save nodes **after End**, then queue normally. API callers should submit through `/distributed/queue`, not directly through `/prompt`.

ComfyUI represents these as ordinary `IMAGE` sockets with list semantics, not a separate `IMAGE_LIST` socket type. Outside the region there is a list; for each assigned item, Start provides one frame and its aligned values. End returns ordered lists, not a concatenated tensor batch, so differently sized images can be preserved.

## Inputs and outputs

Start accepts:

- `images`: one image, an IMAGE batch, or a ComfyUI image list. A batch is split into individual frames; each frame becomes one job.
- Optional `mask`.
- Optional `value_1`, `value_2`, … inputs. Only `value_1` is shown at first; connecting it adds `value_2`, and so on, up to `value_10`. Each accepts an IMAGE, MASK, STRING, INT, FLOAT or BOOLEAN.

`mask` and every value follow one rule:

- **1 value** is shared by every image.
- **One value per image** pairs them in order.
- **Any other count** is an error. Values are never padded or repeated silently.

Only `images` is split into frames. A batch connected to a value input stays a single value, so a 2-image batch can travel with each item (for example, two reference images per job). To pass a different batch to each image, connect a list with one batch per image.

Start outputs, in order: `image`, `mask`, `item_index` (zero-based), `context`, then one `value_N` output per connected `value_N` input, with the same type. Unused value outputs are hidden. Seeds and strengths are ordinary values: connect a single number to share it, a list for one per image, or derive it from `item_index`.

Values that cannot be transported exactly (models, latents, conditioning and other objects) are rejected. Load models inside the region from a loader connected directly to the processing nodes, as described below.

End accepts `images`, optional `masks` / `texts`, and `context`. It returns `image_list`, `masks`, and `texts` in original item order. Each job must produce exactly one image, at most one mask, and at most one text value. Unconnected optional outputs contain `None` entries.

Models are loaded and cached on each processing participant, not serialized between machines. Connect loader outputs directly to the processing nodes. The master needs the region's model files when participating, as does each selected worker. The master still validates the prompt: it needs the relevant node definitions and any model-name/file availability required by their validation (including combo lists). This is not a model-free-master guarantee.

Per-image data must pass through Start; do not bypass it with upstream images, latents or generated batches connected directly into the processing region. Shared dependencies across Start are restricted to registered model outputs or scalar constants, not arbitrary image-producing leaf nodes.

End's optional `texts` is a connection-only socket. For workflows saved before this socket became connection-only, recreate End and reconnect its wires if an old text widget or literal `texts: ""` remains in the compiled prompt. Arbitrary text literals are not accepted at End.

File-based dependencies keep ComfyUI's `[output]` / `[temp]` root annotations and are read on the master. Direct-worker copies use separate content-addressed paths under `input/distributed-media/`; each direct worker's rewritten prompt is reused across items without reusing item tokens. Cloud requests use the PR2 bounded file-input adapters instead.

## Serverless pools and bounded capacity

Select `media_profile: "comfy-media-v1"` for each cloud pool. The endpoint must
include the [standalone PR2 media bridge](../serverless_bridge/README.md), route
typed requests through its executor, and provide the required supervisor. Merely
installing the bridge nodes or choosing the profile is not sufficient. The
endpoint needs the workflow's models and custom nodes, but not the full
Distributed extension. A selected `legacy-image-v1` pool fails before the master
prompt is queued; it is not silently skipped in favor of local execution.
Provider health/authentication probing does not verify bridge/model installation.

- Each pool is selected once by its configured pool ID. **Images per run
  (`jobs_per_run`) does not multiply input images or set image-list parallelism.**
  An assigned cloud job processes exactly one source item and its paired values.
- Cloud slots are bounded by the existing **16 outstanding jobs per pool** cap,
  available manager capacity (**64 pending jobs** across pools), reservations,
  and the source item count. Unresolved submissions and unconfirmed cancellations
  still consume capacity. Selecting more items does not enqueue all of them as
  provider jobs at once. Endpoint-side worker limits can reduce actual concurrency.
- Cloud slots, direct workers and the optional master pull from the same queue;
  no region is cloned once per Images-per-run setting. Results are restored to
  source order even when participants finish out of order.
- Each cloud request uses the standard media-v1 envelope and existing durable
  serverless job lifecycle. The private item graph replaces Start/End with typed
  inputs and explicit exporter roots, not independent all-output execution.
  IMAGE/MASK tensor data and text are preserved by the typed codecs. There is no
  image-list-specific endpoint protocol or new visible node socket.
- Cloud `value_N` tensors must have native IMAGE `[B,H,W,C]` or MASK `[B,H,W]`
  shapes, not arbitrary rank-one/rank-two tensor payloads. Typed tensors must be
  finite; connected masks must match their image grid. Unsupported values fail
  rather than being resized, flattened or converted to lossy images.
- Inputs are inline and the **complete serialized Runpod request is capped at
  10,000,000 bytes**, including workflow and base64. Default inline result JSON
  has the same cap. The PR2 contract's larger artifact ceilings do not remove
  these inline limits; larger outputs need an operator-provided artifact publisher.
  There is no built-in large-input uploader. See [typed media limits](serverless-media.md#size-and-operational-limits).
- Missing/duplicate/mismatched results or provider failures fail the region.
  Cancellation closes only this region's owned cloud jobs and collector queues,
  including jobs whose dispatch was cancelled before returning a handle. It does
  not clear unrelated pool work. No failed or ambiguous job is retried automatically.
- A provider cancellation or timeout is **not proof that endpoint ComfyUI stopped**.
  The PR2 supervisor must quarantine/reconcile or terminate an uncertain endpoint
  before reuse. Preserve its retained files until execution and readers are known
  to have stopped; see the [bridge uncertainty policy](../serverless_bridge/README.md).

## Scope and failure behavior

- This version supports master, direct workers and `comfy-media-v1` serverless pools in one image-list region. Direct capability/idle HTTP probes are never sent to serverless pools; pool probing uses the existing serverless health/authentication path.
- Worker eligibility is determined before execution and checked again before dispatch. Direct workers must be idle in orchestrator-only mode; master-participating mode can use reachable compatible peers without requiring an empty queue. Transiently unreachable workers are excluded; healthy candidates can continue. Malformed capability/queue responses and incompatible versions fail explicitly. Workers joining later are not added to an already-running region.
- Omitted worker IDs select enabled direct workers and compatible serverless pools. Explicit IDs select the named configured workers/pools, including disabled ones; an empty list stays empty and unknown IDs fail. Pool IDs are retained in Start's hidden selection metadata, not expanded participant IDs. Local direct workers use the master's loopback callback; remote direct workers use its configured public address.
- Only the enclosed processing is distributed. List creation and downstream saving run on the master in both modes. With participation enabled, the master is a normal scheduler participant and can run alone when no workers are selected or reachable. **Orchestrator-only mode never silently falls back to master processing.** Failed items are not reassigned automatically.
- Local items execute while the enclosing master prompt waits; they are not queued behind that prompt on the master's `/prompt` queue. The enclosed branch is not also evaluated as a full list by the outer prompt.
- Start/End regions may be chained sequentially and share model-loader dependencies. Nested/overlapping per-image regions and data escaping a region without End are rejected. Collector and distributed-upscaler nodes cannot be mixed into the same prompt in this first version.
- Queue preparation works on an execution copy, leaving the saved canvas unchanged. Unused per-item preview/save branches are pruned by dependency reachability, not node names. Required computational utilities marked `OUTPUT_NODE` stay in the graph; explicit `partial_execution_targets` prevent them from also becoming independent outer-master roots. End and legitimate post-End or unrelated outputs are retained. Bypass/rejoin branches still fail boundary validation rather than being silently removed.
- Every selected direct worker must advertise `execution_targets: true` as well as the existing image-list protocol. Older workers fail preflight before dispatch. Both HTTP and WebSocket item submissions target End only; the participating master uses its existing End-only local executor. Cloud jobs use explicit media exporter roots. Submit through `/distributed/queue`: an ordinary all-output `/prompt` request does not receive this relaxation.
- **Target selection is not a sandbox.** A required utility still runs its own save, preview or other side effects when it executes for an item. Keep intentional final saving after End.
- The entire input list is materialized before scheduling; this is not streaming input. The limit is 4,096 frames. Direct-worker transport preserves floating-point tensor dtype and values rather than encoding to lossy images. Each direct-worker tensor is limited to 50 MiB, and the complete JSON/base64 HTTP request to 101 MiB; image/mask/value combinations can hit the aggregate request limit first. Serverless inline limits are smaller, as described above. In-process master items use tensors directly and are not subject to those HTTP limits.
- Results are accepted once per item using assignment tokens. Missing, duplicate, malformed and wrong-index results fail rather than silently returning a partial list. No failed or ambiguous submission is retried automatically.
- Worker errors fail the region. For direct workers, the worker timeout measures loss of evidence for the owned prompt ID, not maximum render duration. Pending/running evidence refreshes liveness; unrelated busy jobs and failed polls do not. Healthy cold loads and long renders can continue. Serverless jobs instead use their configured serverless timeout and durable manager lifecycle; they are excluded from direct-history/liveness polling.
- User interruption is checked during submission and result waiting. Cleanup targets only this region's acknowledged prompt IDs and staged input tokens. Submission may wait briefly for an acknowledgement before cancellation; remote transport and cleanup have finite time limits. Local cleanup stops at node/progress boundaries and waits for its execution thread to finish before the parent prompt can resume. A synchronous node that does not cooperate with cancellation can delay this beyond the configured deadline. Lost acknowledgements are ambiguous, never retried automatically. Unreachable cleanup is logged as unconfirmed and may require checking the worker manually; cancellation does not guarantee remote termination through a broken network.
- Image-list ordering/context state is in memory. Restarting the master does not resume an interrupted region. Cloud job IDs and uncertainty state remain in the existing durable serverless ledger for reconciliation, not automatic list resumption. Abandoned master contexts and expired direct-worker staging are actively swept on the server loop; active/queued master work is protected from age-only expiry. End also removes its context on success, failure or interruption. This releases feature-owned references, not ComfyUI's separate execution cache.

### Time and size limits

`settings.image_list_job_timeout_seconds` defaults to `0` (no arbitrary total-run cap). A positive finite number opts into an overall deadline with owned-work cleanup; negative, non-finite or malformed settings fail explicitly. There is no fixed one-hour region deadline. The setting is separate from the existing worker liveness timeout.

Uploads, media preparation and submission acknowledgements are independently bounded. WebSocket acknowledgement waiting includes connect/send/receive and is not extended by heartbeats or unrelated messages. A user interrupt allows up to one second for a late acknowledgement; cleanup requests use five-second bounds each.

**4K float32 RGB outputs cannot be returned by direct workers under these transport limits.** A 3840×2160 RGB float32 tensor exceeds 50 MiB before base64 encoding. Mixed master/worker 1080p-to-4K upscaling therefore cannot use this configuration; master-only execution does not use the HTTP transport. Dtypes/values are never silently reduced to fit. Complete encoded input requests are checked before upload, including image/mask/value combinations; output limits are checked when the worker has computed the result. Reverse proxies may impose lower HTTP limits.

Use trusted workers and trusted workflows, as with the existing ComfyUI API. Capability tokens protect assignment/result matching; they are not a replacement for network access control.

## Verification

The generic acceptance helper starts two isolated loopback workers using real ComfyUI HTTP handlers, `PromptQueue` records and CPU `PromptExecutor` instances. Production queue preparation, list nodes, graph extraction, HTTP item/result routes, dynamic scheduling, float transport, ordering and cancellation are exercised. The harness checks orchestrator-only, mixed master/worker and master-only execution, including exact-once required output utilities, discarded previews, unchanged source graphs, End-only worker roots, post-End capture, source-once materialization and cached local model-loader dependencies. WebSocket target propagation is covered by focused route/dispatch tests. No managed ComfyUI service or model is started.

Offline cloud integration tests additionally cover mixed-role and cloud-only
scheduling, out-of-order image/mask/text pairing, one job per source item regardless
of Images per run, profile/selection rejection before queueing, exact cleanup on
setup/dispatch/result failures, and the real CPU ComfyUI `processing_interrupted`
signal while awaiting a mocked cloud result. Provider calls are mocked: this is
not live Runpod, endpoint-supervisor or GPU-workflow acceptance.

```bash
COMFYUI_SOURCE_ROOT=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python -m pytest tests -q
bash scripts/test-web.sh
```

Without `COMFYUI_SOURCE_ROOT`, real-framework tests are explicitly skipped. This coverage is not a substitute for checking your model workflow and connectivity on the actual GPU worker machines.
