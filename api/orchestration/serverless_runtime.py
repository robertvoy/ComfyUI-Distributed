"""Master-owned asynchronous Runpod jobs, independent of worker callbacks."""
import asyncio
import time
import uuid
from pathlib import Path

from .runpod_client import RunpodClient, RunpodError, AmbiguousSubmission
from .serverless_ledger import JobLedger
from .serverless_state import (
    PROVIDER_STATES, PROVIDER_TERMINAL, belongs_to_pool, is_unresolved,
    occupies_capacity, terminal_provider_status,
)


# Fixed per-worker cap on this master's outstanding Runpod jobs; equals the
# Images per run maximum (MAX_JOBS_PER_RUN in utils/serverless_config.py), so
# every image of a run is submitted at once. Deliberately not configurable or
# derived from jobs_per_run: unresolved jobs from earlier runs (ambiguous
# submissions, unconfirmed cancels) still count against it, and a constant
# never has to change while such jobs exist.
PER_WORKER_JOB_LIMIT = 16


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
        self.capacity_changed = asyncio.Condition()
        self.recovery_lock = asyncio.Lock()
        self._recover_local_records()

    def _recover_local_records(self):
        # WAITING is committed before admission; it cannot have reached Runpod.
        # SUBMITTING is an intent record, so a lost ID must remain reserved.
        for record in self.ledger.all():
            if record.get("status") == "WAITING" and not record.get("job_id"):
                self.ledger.update(record["record_id"], status="CANCELLED", updated_at=time.time())
            elif record.get("status") == "SUBMITTING" and not record.get("job_id"):
                self.ledger.update(record["record_id"], status="UNKNOWN", updated_at=time.time())

    def _task_is_active(self, record_id):
        task = self.tasks.get(record_id)
        return task is not None and not task.done()

    def pool_is_active(self, worker_id):
        """Protect both live local work and unresolved pre-restart provider work."""
        for rid, run in self.runs.items():
            pool_id = run["worker"].get("pool_id", run["worker"]["id"])
            if str(pool_id) == str(worker_id) and self._task_is_active(rid):
                return True
        return any(belongs_to_pool(record, worker_id) and is_unresolved(record)
                   for record in self.ledger.all())

    async def _reserve_provider_capacity(self, run):
        pool_id = run["worker"].get("pool_id", run["worker"]["id"])
        async with self.capacity_changed:
            while True:
                self._check_master(run)
                outstanding = sum(belongs_to_pool(record, pool_id) and occupies_capacity(record)
                                  for record in self.ledger.all())
                if outstanding < PER_WORKER_JOB_LIMIT:
                    # Claim atomically before any await that could submit a billed job.
                    self.ledger.update(run["record_id"], status="SUBMITTING", updated_at=time.time())
                    return
                await self.capacity_changed.wait()

    async def _notify_capacity(self):
        async with self.capacity_changed:
            self.capacity_changed.notify_all()

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
        # The limit is a constant, so it never has to be swapped while this
        # worker still has unresolved jobs.
        pool_id = worker.get("pool_id", worker["id"])
        semaphore = self.pool_limits.get(pool_id)
        if semaphore is None:
            semaphore = asyncio.Semaphore(PER_WORKER_JOB_LIMIT)
            self.pool_limits[pool_id] = semaphore
        return semaphore

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
                return terminal_provider_status(result)
            except Exception:
                return None
        return None

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
                    await self._reserve_provider_capacity(run)
                    client = await self._client(run["worker"])
                    run["client"] = client
                    submitted = True
                    job_id = await client.submit(run["workflow"], run["images"], run["worker"]["timeout_seconds"])
                    run["provider_job_id"] = job_id
                    # Persist immediately: all subsequent actions use this exact ID.
                    self.ledger.update(rid, job_id=job_id, status="IN_QUEUE", updated_at=time.time())
                    while True:
                        self._check_master(run)
                        response = await client.status(job_id)
                        status = response.get("status")
                        if not isinstance(status, str) or status not in PROVIDER_STATES:
                            raise RunpodError("Runpod returned an unknown job state")
                        provider_terminal = status in PROVIDER_TERMINAL
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
                        if provider_terminal:
                            raise RunpodError(f"Serverless job ended with {status}")
                        await asyncio.sleep(self.poll_seconds)
        except AmbiguousSubmission as exc:
            self.ledger.update(rid, status="UNKNOWN", error=str(exc), updated_at=time.time())
            await self._deliver(run, error=str(exc))
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            terminal = "FAILED" if provider_terminal else await self._cancel_provider(run)
            if submitted and not run["provider_job_id"]:
                status = "UNKNOWN"
                error = "Submission interrupted before its job ID was received; check Runpod before retrying"
            elif isinstance(exc, asyncio.TimeoutError):
                status = terminal or ("CANCEL_REQUESTED" if run["provider_job_id"] else "TIMED_OUT")
                error = "Serverless job exceeded its queue/execution deadline"
            else:
                status = terminal or ("CANCEL_REQUESTED" if run["provider_job_id"] else "CANCELLED")
                error = "Serverless work stopped" if terminal or not submitted else "Serverless cancellation requested; provider confirmation is pending"
            self.ledger.update(rid, status=status, error=error, updated_at=time.time())
            await self._deliver(run, error=error)
        except Exception as exc:
            # If polling/decoding failed, stop only our own provider job, never the pool.
            terminal = "FAILED" if provider_terminal else await self._cancel_provider(run)
            error = str(exc) if isinstance(exc, RunpodError) else "Serverless processing failed; check the workflow and endpoint contract"
            status = terminal or ("CANCEL_REQUESTED" if run["provider_job_id"] else "FAILED")
            self.ledger.update(rid, status=status, error=error, updated_at=time.time())
            await self._deliver(run, error=error)
        finally:
            # Completed tracking retains IDs/bindings only, not potentially large
            # base64 inputs, workflow metadata, or a client holding credentials.
            run.pop("workflow", None)
            run.pop("images", None)
            run["client"] = None
            # A released local semaphore alone cannot release a durable reservation.
            await self._notify_capacity()

    async def wait_for_execution(self, execution_id):
        tasks = [task for rid, task in self.tasks.items() if self.runs[rid]["execution_id"] == execution_id]
        await asyncio.gather(*tasks, return_exceptions=False)

    def _detached_records(self, worker_id=None, execution_id=None):
        return [record for record in self.ledger.all()
                if is_unresolved(record) and not self._task_is_active(record["record_id"])
                and (worker_id is None or belongs_to_pool(record, worker_id))
                and (execution_id is None or record.get("execution_id") == execution_id)]

    async def _recover_provider_record(self, record, *, cancel=False):
        """Reconcile an exact saved ID, never submit or resume a workflow."""
        if not record.get("job_id"):
            return False
        status = None
        try:
            client = await self._client({"id": record["worker_id"], "endpoint_id": record["endpoint_id"]})
            response = await (client.cancel(record["job_id"]) if cancel else client.status(record["job_id"]))
            status = terminal_provider_status(response)
        except Exception:
            pass  # Unreachable/expired is not proof of provider termination.
        if status is not None or cancel:
            self.ledger.update(record["record_id"], status=status or "CANCEL_REQUESTED", updated_at=time.time())
        return status is not None

    async def reconcile(self, worker_id=None):
        """Read terminal states for detached jobs and release only proven capacity."""
        resolved = 0
        async with self.recovery_lock:
            for record in self._detached_records(worker_id):
                resolved += await self._recover_provider_record(record)
        await self._notify_capacity()
        return resolved

    async def acknowledge_recovery(self, worker_id, record_id, *, provider_checked=False):
        """Explicit operator escape hatch for lost IDs/expired provider records.

        This records a human's verification, not a cancellation or provider
        status. It is never called automatically or allowed for live tasks.
        """
        if provider_checked is not True:
            raise ValueError("Check Runpod and confirm that this job is no longer running before resolving it")
        async with self.recovery_lock:
            record = next((r for r in self._detached_records(worker_id) if r["record_id"] == record_id), None)
            if record is None:
                raise ValueError("Unresolved inactive job not found for this worker")
            self.ledger.update(record_id, status="RESOLVED", updated_at=time.time(),
                               error="Recovery explicitly acknowledged after operator checked Runpod; not provider-confirmed")
        await self._notify_capacity()

    async def cancel(self, worker_id=None, execution_id=None, collector_job_id=None):
        targets = {}
        for rid, task in self.tasks.items():
            run = self.runs[rid]
            if worker_id and run["worker"].get("pool_id", run["worker"]["id"]) != worker_id:
                continue
            if execution_id and run["execution_id"] != execution_id:
                continue
            if collector_job_id and not any(m["job_id"] == collector_job_id for m in run["manifest"]):
                continue
            if not task.done():
                # Repeated requests must not interrupt the first cancellation's cleanup.
                if not task.cancelling():
                    task.cancel()
                targets[rid] = task
        if targets:
            await asyncio.gather(*targets.values(), return_exceptions=True)
        recovered = 0
        async with self.recovery_lock:
            for record in self._detached_records(worker_id, execution_id):
                rid = record["record_id"]
                # A task cancelled before its first step never enters _run's handlers.
                # A failed start() may also leave a journalled but unstarted local slot.
                if record.get("status") == "WAITING" and not record.get("job_id"):
                    run = self.runs.get(rid)
                    if run is not None and (collector_job_id is None or any(m["job_id"] == collector_job_id for m in run["manifest"])):
                        self.ledger.update(rid, status="CANCELLED", updated_at=time.time())
                        run.pop("workflow", None)
                        run.pop("images", None)
                        await self._deliver(run, error="Serverless job cancelled before submission")
                    continue
                # Do not send two cancellation requests for the same live job in
                # one action. Its unresolved reservation survives for later recovery.
                if rid in targets or collector_job_id or not record.get("job_id"):
                    continue
                await self._recover_provider_record(record, cancel=True)
                recovered += 1
        await self._notify_capacity()
        return len(targets) + recovered

    def public_jobs(self, worker_id):
        records = [record for record in reversed(self.ledger.all()) if belongs_to_pool(record, worker_id)]
        # Never hide unresolved IDs behind the bounded completed history.
        visible = [record for record in records if is_unresolved(record)]
        visible.extend(record for record in records if not is_unresolved(record))
        unresolved_count = sum(is_unresolved(record) for record in records)
        return [{key: record.get(key) for key in ("record_id", "job_id", "status", "worker_id", "execution_id", "error")}
                for record in visible[:max(20, unresolved_count)]]
