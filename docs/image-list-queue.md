# Dynamic image-list queue

Process a list of images across **direct local or remote ComfyUI workers** without assigning fixed chunks to each worker.

The master evaluates the input list once. Each selected idle worker receives one image job; when it returns a result, it gets the next unassigned image. Faster workers can therefore process more images. The master restores input order before running downstream nodes.

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

1. Install the same feature-capable version of Distributed on the master and every selected worker. Restart those instances and refresh the frontend after installation.
2. Enable Distributed and select at least one reachable, idle **direct** worker in the panel. The master must be reachable from remote workers at its configured callback address.
3. Feed your image list or tensor batch into **Distributed Image List Start → images**.
4. Connect its `image` output through the per-image processing nodes to **Distributed Image List End → images**.
5. Connect Start's **context** directly to End's **context**. This is a dedicated `DISTRIBUTED_IMAGE_LIST_CONTEXT` socket, not editable text.
6. Put preview/save nodes **after End**, then queue normally. API callers should submit through `/distributed/queue`, not directly through `/prompt`.

ComfyUI represents these as ordinary `IMAGE` sockets with list semantics, not a separate `IMAGE_LIST` socket type. Outside the region there is a list; on a worker, Start provides one frame and its aligned values. End returns ordered lists, not a concatenated tensor batch, so differently sized images can be preserved.

## Inputs and outputs

Start accepts:

- `images`: one image, an IMAGE batch, or a ComfyUI image list. A batch is split into individual frames.
- Optional `reference_image`, `mask`, `text`, `strength`, and `seed`.
- A `per_item` / `shared` mode for each optional input. `per_item` requires exactly one value per image; `shared` requires exactly one value, reused for all images. Inputs are never silently padded by repeating the last element.

Reference images and masks default to `per_item`. Text, strength and seed default to `shared`; strength defaults to `1.0`, seed to `0`.

Start outputs, in order: `image`, `reference_image`, `mask`, `text`, `strength`, `seed`, `item_index` (zero-based), `context`. Connect only optional values that your processing region actually provides and uses. Seeds are tied to input values, not worker assignments; use a per-item seed list for deterministic variation.

End accepts `images`, optional `masks` / `texts`, and `context`. It returns `image_list`, `masks`, and `texts` in original item order. Each job must produce exactly one image, at most one mask, and at most one text value. Unconnected optional outputs contain `None` entries.

Models are **loaded and cached on workers**, not serialized from the master. Connect loader outputs directly to the processing nodes. Each worker needs the region's node packs and model files. Per-image data must pass through Start; do not bypass it with a second image list connected directly into the processing region.

## Scope and failure behavior

- This version supports direct workers only, not Runpod serverless pools. Unsupported pools are excluded without contacting the provider for this job.
- Worker eligibility is determined before execution and checked again before dispatch. Busy workers are excluded. Workers joining later are not added to an already-running region.
- Only the enclosed processing is distributed. List creation and downstream saving run on the master, even when master-delegate-only is enabled. There is no master fallback when workers are unavailable.
- Start/End regions may be chained sequentially and share model-loader dependencies. Nested/overlapping per-image regions and data escaping a region without End are rejected. Collector and distributed-upscaler nodes cannot be mixed into the same prompt in this first version.
- Output nodes inside a region are rejected because ComfyUI would execute them independently on the master. Avoid other side-effecting processing nodes as well; save after End.
- The entire input list is materialized before scheduling; this is not streaming input. The limit is 4,096 frames. Transport preserves floating-point tensor dtype and values rather than encoding to lossy images. Each tensor is limited to 50 MiB, and the complete JSON/base64 HTTP request to 101 MiB; image/reference/mask combinations can hit the aggregate request limit first.
- Results are accepted once per item using assignment tokens. Missing, duplicate, malformed and wrong-index results fail rather than silently returning a partial list. No failed or ambiguous submission is retried automatically.
- Worker errors and the configured worker timeout fail the region. User interruption stops further scheduling and issues targeted queue deletion/interruption for this region's acknowledged prompts. An in-progress submission is allowed to resolve so its prompt ID can be cleaned up. Unreachable workers or a lost submission acknowledgement can require checking that worker manually; cancellation is not a guarantee of remote termination through a broken network.
- State is in memory. Restarting the master does not resume an interrupted region. A region also has a one-hour overall deadline.

Use trusted workers and trusted workflows, as with the existing ComfyUI API. Capability tokens protect assignment/result matching; they are not a replacement for network access control.

## Verification

The generic acceptance helper starts two isolated loopback worker test servers and uses actual CPU ComfyUI `PromptExecutor` instances. Production queue preparation, list nodes, graph extraction, HTTP item/result routes, dynamic scheduling, float transport, ordering and cancellation are exercised. The lightweight queue/history facade is test code; no managed ComfyUI service or model is started.

```bash
COMFYUI_SOURCE_ROOT=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python -m pytest tests -q
bash scripts/test-web.sh
```

Without `COMFYUI_SOURCE_ROOT`, real-framework tests are explicitly skipped. This coverage is not a substitute for checking your model workflow and connectivity on the actual GPU worker machines.
