"""Master-owned asynchronous Runpod jobs, independent of worker callbacks."""
import asyncio
import time
import uuid
from pathlib import Path

from .runpod_client import RunpodClient, RunpodError, AmbiguousSubmission
from .serverless_ledger import JobLedger, TERMINAL


async def decode_results(output, manifest):
    from .serverless_media import collect_images
    import torch
    images = await collect_images(output, manifest)
    return {job_id: [torch.from_numpy(image).unsqueeze(0) for image in batch] for job_id, batch in images.items()}


def state_directory():
    import folder_paths
    return Path(folder_paths.get_user_directory()) / "distributed" / "serverless"


def get_secret_store():
    from ...utils.serverless_config import SecretStore
    return SecretStore(state_directory() / "credentials.json")


def get_manager():
    import server
    from ...utils.network import get_client_session
    ps = server.PromptServer.instance
    manager = getattr(ps, "distributed_serverless", None)
    if manager is None:
        manager = ServerlessManager(ps, get_secret_store(), state_directory() / "jobs.json", get_client_session)
        ps.distributed_serverless = manager
    return manager


class ServerlessManager:
    def __init__(self, prompt_server, secrets, ledger_path, session_factory):
        self.server = prompt_server
        self.secrets = secrets
        self.ledger = JobLedger(ledger_path)
        self.session_factory = session_factory
        self.tasks = {}
        self.runs = {}
        self.pool_limits = {}
        self.health_cache = {}
        self.health_locks = {}
        self.poll_seconds = 2.0

    async def _client(self, worker):
        pool_id = worker.get("pool_id", worker["id"])
        key = self.secrets.get(pool_id)
        return RunpodClient(await self.session_factory(), worker["endpoint_id"], key)

    async def health(self, worker, *, fresh=False):
        pool_id = worker.get("pool_id", worker["id"])
        cache_key = (pool_id, worker["endpoint_id"])
        lock = self.health_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self.health_cache.get(cache_key)
            if cached and not fresh and time.monotonic() - cached[0] < 10:
                return cached[1]
            client = await self._client(worker)
            data = await client.health()
            jobs = data.get("jobs", {})
            if not isinstance(jobs, dict) or not isinstance(data.get("workers"), dict):
                raise RunpodError("Runpod returned an invalid endpoint health record")
            try:
                pending = max(0, int(jobs.get("inQueue", 0))) + max(0, int(jobs.get("inProgress", 0)))
            except (ValueError, TypeError):
                raise RunpodError("Runpod returned invalid queue counts") from None
            result = {"ok": True, "queueRemaining": pending, "message": "Queued / processing" if pending else "Ready (scales from zero)"}
            self.health_cache[cache_key] = (time.monotonic(), result)
            return result

    def _semaphore(self, worker):
        pool_id = worker.get("pool_id", worker["id"])
        limit = worker["concurrency"]
        current = self.pool_limits.get(pool_id)
        if current is None:
            current = (limit, asyncio.Semaphore(limit))
            self.pool_limits[pool_id] = current
        elif current[0] != limit:
            if any(not task.done() and self.runs[rid]["worker"].get("pool_id", self.runs[rid]["worker"]["id"]) == pool_id for rid, task in self.tasks.items()):
                raise ValueError("Wait for this serverless pool's current jobs before changing concurrency")
            current = (limit, asyncio.Semaphore(limit))
            self.pool_limits[pool_id] = current
        return current[1]

    def ensure_capacity(self, count):
        if sum(not task.done() for task in self.tasks.values()) + count > 64:
            raise ValueError("Distributed already has too many pending serverless jobs (limit 64)")

    async def start(self, execution_id, prompt_id, prepared_jobs):
        self.ensure_capacity(len(prepared_jobs))
        new_runs = []
        # Journal every local slot before starting any billed submission.
        for prepared in prepared_jobs:
            worker = prepared["worker"]
            semaphore = self._semaphore(worker)
            record_id = uuid.uuid4().hex
            now = time.time()
            self.ledger.update(record_id, execution_id=execution_id, prompt_id=prompt_id,
                               worker_id=worker.get("pool_id", worker["id"]), participant_id=worker["id"],
                               endpoint_id=worker["endpoint_id"], job_id=None, status="WAITING", created_at=now, updated_at=now)
            run = dict(prepared, execution_id=execution_id, prompt_id=prompt_id, record_id=record_id,
                       provider_job_id=None, client=None, deadline=time.monotonic() + worker["timeout_seconds"])
            self.runs[record_id] = run
            new_runs.append((record_id, run, semaphore))
        for record_id, run, semaphore in new_runs:
            self.tasks[record_id] = asyncio.create_task(self._run(run, semaphore), name=f"distributed-serverless-{record_id}")
        # Bound in-memory completed tracking as well as the on-disk journal.
        done = [rid for rid, task in self.tasks.items() if task.done()]
        for rid in done[:-200]:
            self.tasks.pop(rid, None)
            self.runs.pop(rid, None)

    def is_pending(self, collector_job_id, participant_id):
        return any(not task.done() and run["worker"]["id"] == participant_id and
                   any(m["job_id"] == collector_job_id for m in run["manifest"])
                   for rid, task in self.tasks.items() for run in [self.runs[rid]])

    def owns(self, collector_job_id, participant_id):
        return any(run["worker"]["id"] == participant_id and
                   any(m["job_id"] == collector_job_id for m in run["manifest"]) for run in self.runs.values())

    async def _deliver(self, run, images=None, error=None):
        async with self.server.distributed_jobs_lock:
            for binding in run["manifest"]:
                queue = self.server.distributed_pending_jobs.get(binding["job_id"])
                if queue is None:
                    continue  # A cancelled/finished collector must never be resurrected.
                if error:
                    queue.put_nowait({"worker_id": run["worker"]["id"], "serverless_error": error, "is_last": True})
                else:
                    batch = images[binding["job_id"]]
                    for index, tensor in enumerate(batch):
                        queue.put_nowait({"worker_id": run["worker"]["id"], "image_index": index,
                                          "tensor": tensor, "is_last": index == len(batch) - 1})

    async def _cancel_provider(self, run):
        if run["provider_job_id"] and run["client"]:
            try:
                result = await run["client"].cancel(run["provider_job_id"])
                return isinstance(result, dict) and result.get("status") == "CANCELLED"
            except Exception:
                return False
        return False

    def _check_master(self, run):
        queue = getattr(self.server, "prompt_queue", None)
        if queue is None:
            return
        history = queue.get_history(run["prompt_id"])
        record = history.get(run["prompt_id"], {})
        if record.get("status", {}).get("status_str") == "error":
            raise RunpodError("Master workflow failed; serverless work stopped")
        if not record and hasattr(queue, "get_current_queue"):
            running, pending = queue.get_current_queue()
            if not any(str(item[1]) == run["prompt_id"] for item in [*running, *pending]):
                raise RunpodError("Master workflow was removed from the queue; serverless work stopped")

    async def _run(self, run, semaphore):
        rid = run["record_id"]
        submitted = False
        provider_terminal = False
        try:
            async with asyncio.timeout(max(0, run["deadline"] - time.monotonic())):
                async with semaphore:
                    self._check_master(run)
                    client = await self._client(run["worker"])
                    run["client"] = client
                    self.ledger.update(rid, status="SUBMITTING", updated_at=time.time())
                    submitted = True
                    job_id = await client.submit(run["workflow"], run["images"], run["worker"]["timeout_seconds"])
                    run["provider_job_id"] = job_id
                    # Persist immediately: all subsequent actions use this exact ID.
                    self.ledger.update(rid, job_id=job_id, status="IN_QUEUE", updated_at=time.time())
                    while True:
                        self._check_master(run)
                        response = await client.status(job_id)
                        status = response.get("status")
                        if status not in {"IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                            raise RunpodError("Runpod returned an unknown job state")
                        provider_terminal = status in TERMINAL
                        if status == "COMPLETED":
                            output = response.get("output")
                            if not isinstance(output, dict) or output.get("error") or output.get("errors"):
                                raise RunpodError("Serverless workflow reported an output error; inspect the provider job")
                            self.ledger.update(rid, status="DOWNLOADING", updated_at=time.time())
                            images = await decode_results(output, run["manifest"])
                            await self._deliver(run, images=images)
                            self.ledger.update(rid, status="COMPLETED", updated_at=time.time())
                            return
                        self.ledger.update(rid, status=status, updated_at=time.time())
                        if status in TERMINAL:
                            raise RunpodError(f"Serverless job ended with {status}")
                        await asyncio.sleep(self.poll_seconds)
        except AmbiguousSubmission as exc:
            self.ledger.update(rid, status="UNKNOWN", error=str(exc), updated_at=time.time())
            await self._deliver(run, error=str(exc))
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            confirmed = await self._cancel_provider(run)
            if submitted and not run["provider_job_id"]:
                status = "UNKNOWN"
                error = "Submission interrupted before its job ID was received; check Runpod before retrying"
            elif isinstance(exc, asyncio.TimeoutError):
                status = "TIMED_OUT" if not run["provider_job_id"] or confirmed else "CANCEL_REQUESTED"
                error = "Serverless job exceeded its queue/execution deadline"
            else:
                status = "CANCELLED" if not run["provider_job_id"] or confirmed else "CANCEL_REQUESTED"
                error = "Serverless job cancelled" if confirmed or not submitted else "Serverless cancellation requested; provider confirmation is pending"
            self.ledger.update(rid, status=status, error=error, updated_at=time.time())
            await self._deliver(run, error=error)
        except Exception as exc:
            # If polling/decoding failed, stop only our own provider job, never the pool.
            confirmed = await self._cancel_provider(run) if not provider_terminal else True
            error = str(exc) if isinstance(exc, RunpodError) else "Serverless processing failed; check the workflow and endpoint contract"
            status = "CANCEL_REQUESTED" if run["provider_job_id"] and not confirmed else "FAILED"
            self.ledger.update(rid, status=status, error=error, updated_at=time.time())
            await self._deliver(run, error=error)
        finally:
            # Completed tracking retains IDs/bindings only, not potentially large
            # base64 inputs, workflow metadata, or a client holding credentials.
            run.pop("workflow", None)
            run.pop("images", None)
            run["client"] = None

    async def wait_for_execution(self, execution_id):
        tasks = [task for rid, task in self.tasks.items() if self.runs[rid]["execution_id"] == execution_id]
        await asyncio.gather(*tasks, return_exceptions=False)

    async def cancel(self, worker_id=None, execution_id=None, collector_job_id=None):
        targets = []
        for rid, task in self.tasks.items():
            run = self.runs[rid]
            if worker_id and run["worker"].get("pool_id", run["worker"]["id"]) != worker_id:
                continue
            if execution_id and run["execution_id"] != execution_id:
                continue
            if collector_job_id and not any(m["job_id"] == collector_job_id for m in run["manifest"]):
                continue
            if not task.done():
                task.cancel()
                targets.append(task)
        if targets:
            await asyncio.gather(*targets, return_exceptions=True)
        # A restarted master must still be able to cancel IDs it journalled.
        # No workflow payloads are persisted, so never resume or resubmit them.
        recovered = 0
        for rec in self.ledger.all():
            task = self.tasks.get(rec["record_id"])
            if (task is not None and not task.done()) or rec.get("status") in TERMINAL:
                continue
            if worker_id and rec.get("worker_id") != worker_id:
                continue
            if execution_id and rec.get("execution_id") != execution_id:
                continue
            if collector_job_id or not rec.get("job_id"):
                continue
            try:
                client = await self._client({"id": rec["worker_id"], "endpoint_id": rec["endpoint_id"]})
                result = await client.cancel(rec["job_id"])
                status = "CANCELLED" if result.get("status") == "CANCELLED" else "CANCEL_REQUESTED"
            except Exception:
                status = "CANCEL_REQUESTED"
            self.ledger.update(rec["record_id"], status=status, updated_at=time.time())
            recovered += 1
        return len(targets) + recovered

    def public_jobs(self, worker_id):
        return [{key: rec.get(key) for key in ("job_id", "status", "worker_id", "execution_id", "error")}
                for rec in reversed(self.ledger.all()) if rec.get("worker_id") == worker_id][:20]
