# Runpod serverless pools (preview)

A **Serverless (Runpod)** worker is a queue-based endpoint/pool, not a permanent
ComfyUI machine. Runpod chooses the actual GPU workers. Distributed submits one
normal ComfyUI API workflow per logical slot and collects the returned images.
The endpoint workers **do not need ComfyUI-Distributed installed**.

## Setup

1. Add a worker in the Distributed panel and select **Serverless (Runpod)**.
2. Enter its name, Runpod **endpoint ID** (not a URL), and API key.
3. Start with **Jobs per run = 1**, **Concurrent jobs = 1**, and **Timeout = 1800**.
4. Save, then choose **Test Saved Endpoint**. This checks authentication and
   endpoint health only; it does not start GPU inference or prove model/node compatibility.
5. Enable the pool when you want queued Distributed workflows to use it.
6. Use an image **Distributed Collector** and queue through Distributed as usual.
   Enable **Delegate Only** if the master should collect without generating locally.

Keys are write-only. Leave the key field blank to retain a saved key, or select
**Clear saved API key** to remove it (disable the pool first). A saved key is
required before enabling a pool. Use HTTPS or a trusted local connection when
entering keys. Like ComfyUI itself, the panel must not be exposed to untrusted
users: anyone able to queue arbitrary workflows can run code and spend endpoint credits.

## Endpoint contract

The endpoint must accept this generic input shape; named application modes are
not used:

```json
{
  "input": {
    "workflow": {
      "1": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0}},
      "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "example"}}
    },
    "images": []
  }
}
```

For image inputs, `images` contains `{ "name": "input.png", "image": "<base64>" }`
entries and the handler must rewrite `LoadImage` references to the staged files.
The adapter transfers local input images; it does not copy models or install nodes.
All referenced models and custom nodes must already exist in the endpoint image.

On completion, Runpod's `output.images` must contain artifact records such as:

```json
{
  "filename": "example_00001_.png",
  "subfolder": "",
  "artifact_type": "output",
  "artifact_path": "example_00001_.png",
  "type": "base64",
  "data": "<base64>"
}
```

Image data can alternatively use `type: "s3_url"`, `"gcs_url"`, or `"url"`
with an HTTPS URL in `data`/`url`. Signed artifact URLs are downloaded without
Runpod credentials; private-network destinations and unsafe redirects are refused.
The handler must preserve injected `SaveImage.filename_prefix` values so results
can be matched to the right collector and slot. Duplicate basenames in different
subfolders remain different artifact identities.

A Runpod job marked `COMPLETED` is still rejected if `output.error` or
`output.errors` is nonempty, required collector outputs are missing, or image
artifacts are invalid. Failures are surfaced rather than silently returning an
incomplete serverless batch.

## Scheduling and cancellation

- **Jobs per run** adds logical collector participants. Each receives its own
  Distributed Seed/Value overrides. It is not a physical GPU count.
- **Concurrent jobs** bounds this master's outstanding submissions per pool.
  Other applications using the same endpoint are unaffected.
- **Load balance** selects one participant, so it does not fan out every pool slot.
- **Timeout** includes waiting for a local slot, Runpod queue/cold-start time,
  execution, and output retrieval. Provider policy limits are submitted as well.
- A zero-worker endpoint can be healthy: Runpod scales from zero on submission.
- Interrupting a collector or using the panel cancellation action requests
  cancellation only for jobs submitted by this master. It never purges the endpoint.
- A network failure during submission is **not retried automatically**: the job
  may already be billable. Check the Runpod job list before queuing again.

## Persistence and recovery

Credentials live outside `gpu_config.json` and Git in the master ComfyUI user
folder under `distributed/serverless/credentials.json`, with restrictive POSIX
permissions. They are omitted from public configuration, workflow exports, and
browser configuration state. The first version requires a POSIX (Linux/macOS)
master for serverless secret storage; existing non-serverless workers remain
available on Windows.

`distributed/serverless/jobs.json` records provider job IDs and sanitized states,
not image data, workflow payloads, or API keys. A restarted master cannot resume
an interrupted collector. It does **not** automatically resubmit jobs. Retained
IDs are available through `/distributed/serverless/status/<worker_id>`; the pool's
cancel action can request cancellation of known pre-restart jobs. If no job ID
was received, check Runpod manually. Cancellation is best-effort and may race
with completion; `CANCEL_REQUESTED` is not a provider-confirmed cancellation.

## First-version limits

- Image collectors only. Audio/video transfer and audio collectors are rejected.
- Coordinated `UltimateSDUpscaleDistributed` workloads and unsupported Distributed
  node types are rejected before submission.
- Downstream post-collector processing remains local; the remote graph contains
  only the upstream compute required for image saves.
- Inputs must be supported images under ComfyUI's input/output/temp roots.
- Raw image input reads are guarded at 10,000,000 bytes total, and the complete
  serialized `/run` JSON request body (workflow, base64 images, and policy) must
  also fit within 10,000,000 bytes. Base64 adds about 33%, so the usable raw-image
  budget is lower and depends on workflow size. Large-media/object-storage upload
  support is not included.
- No remote GPU discovery, endpoint provisioning, model synchronization, automatic
  scaling changes, or inbound master callback/tunnel is needed.

This feature is verified with offline provider/transport tests. A successful
health test is not a paid end-to-end generation test; test your deployed handler
and model inventory with one slot before increasing concurrency.
