# Runpod serverless workers (preview)

A **Serverless (Runpod)** worker is a queue-based endpoint, not a permanent
ComfyUI machine. Runpod chooses the actual GPU workers. Distributed submits one
normal ComfyUI API workflow per requested image and collects the returned images.
The endpoint workers **do not need ComfyUI-Distributed installed**.

## Setup

1. Add a worker in the Distributed panel and select **Serverless (Runpod)**.
2. Enter its name, Runpod **endpoint ID** (not a URL; a pasted `https://api.runpod.ai/v2/<id>/…`
   URL is trimmed to the ID), and API key.
3. Start with **Images per run = 1** and **Timeout = 1800**.
4. Save, then choose **Test Saved Endpoint**. This checks authentication and
   endpoint health only; it does not start GPU inference or prove model/node compatibility.
5. Enable the worker when you want queued Distributed workflows to use it.
6. Use an image **Distributed Collector** and queue through Distributed as usual.
   Enable **Delegate Only** if the master should collect without generating locally.

Keys are write-only. Leave the key field blank to retain a saved key, or select
**Clear saved API key** to remove it. The checkbox is unavailable while the
worker is enabled, so turn the worker off first. A saved key is required before
enabling a worker. Use HTTPS or a trusted local connection when
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

- **Images per run** (1–16) is how many separate Runpod jobs this worker submits
  for each queued run. Each returns one image and receives its own Distributed
  Seed/Value overrides. All of them are submitted at once.
- How many of those jobs actually run in parallel is decided by the Runpod
  endpoint's **max workers**, not by Distributed. Jobs beyond that wait in
  Runpod's queue. To run every image at the same time, keep Images per run at or
  below the endpoint's max workers.
- Distributed also keeps a fixed limit of 16 outstanding jobs per serverless
  worker (and 64 across all of them). Retained jobs after a restart, ambiguous
  submissions, and unconfirmed cancellations keep counting against it until
  reconciled. Other applications using the same endpoint are unaffected.
- Older configs may contain a `concurrency` field from the previous separate
  setting. It is still accepted when loading or saving, but ignored and dropped.
- **Load balance** selects one participant, so it sends one job instead of
  Images per run.
- **Timeout** includes Runpod queue/cold-start time, execution, and output
  retrieval. Provider policy limits are submitted as well.
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
IDs are available through `/distributed/serverless/status/<worker_id>`; the worker's
cancel action can request cancellation of known pre-restart jobs. If no job ID
was received, check Runpod manually. Cancellation is best-effort and may race
with completion; `CANCEL_REQUESTED` is not a provider-confirmed cancellation.

Unresolved records protect the worker's endpoint, API key, worker type,
and deletion even when no local task remains. Disabling the worker is still allowed;
it does not cancel existing jobs. On restart, never-submitted `WAITING` slots are
closed locally, while interrupted `SUBMITTING` records without IDs become
`UNKNOWN` and remain protected.

### Reconcile or acknowledge recovery

Recovery uses the same trusted-master API as the panel; it never resubmits a job.

1. Read `GET /distributed/serverless/status/<worker_id>`. The `jobs` list includes
   every unresolved record with its local `record_id`, provider `job_id` (if
   received), and state. These records remain visible if endpoint health fails.
2. Request cancellation from the panel, or send
   `POST /distributed/serverless/reconcile` with the body below to **read** the
   retained provider IDs. Only a confirmed terminal state releases protection.
   Network errors, expired IDs, or ambiguous responses do not.

   ```json
   {"worker_id": "YOUR_WORKER_ID"}
   ```

3. If no ID was returned, or Runpod no longer retains its status, inspect Runpod
   manually first. **Only after verifying that the job is no longer running or
   queued**, send `POST /distributed/serverless/resolve` for that one record:

   ```json
   {"worker_id": "YOUR_WORKER_ID", "record_id": "LOCAL_RECORD_ID", "provider_checked": true}
   ```

   This records `RESOLVED`: an explicit operator acknowledgment, **not** a
   provider-confirmed cancellation. It cannot resolve a live local task and does
   not stop remote compute. It releases the record's reservation and may let
   already-waiting jobs proceed. Never automate this acknowledgment as a retry
   mechanism; an unverified job can still be billable.

Worker setting changes become available once all its local tasks and unresolved records
are finished. No credentials or ledger files need to be edited by hand.

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
and model inventory with one image per run before increasing it.
