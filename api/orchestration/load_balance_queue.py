"""Hand load-balanced runs to the next participant that frees up.

A Collector workflow queued with ``load_balance`` enabled while every
participant is busy stays in the master's own ComfyUI prompt queue, so it is
still visible and cancellable there, and nothing is sent to a worker yet. One
dispatcher task then watches the candidate workers; as soon as one reports an
empty queue, the oldest waiting run is taken out of the master queue,
dispatched to that worker, and re-queued on the master as a delegate prompt
that only collects the result. When the master participates, its executor
takes whichever waiting run it reaches first in the normal way: the run's
prompt is the full workflow, and the Collector claims it for the master.

A delegate prompt blocks ComfyUI's single executor while it waits for its
worker, which would stall every full workflow queued behind it on a
participating master. So on a participating master the delegate prompt is
queued only once the worker's final result has arrived (or the worker is
seen idle or unreachable without delivering, so the Collector can report
it); an orchestrator-only master has nothing else to run and queues it at
once.

The claim is atomic through ComfyUI's queue mutex: a run can be removed from
the queue by the dispatcher or popped by the master executor, never both.
"""
import asyncio
import time

import server

from ...utils.logging import debug_log, log
from ...utils.network import build_worker_url, probe_worker
from .dispatch import _extract_queue_remaining
from .dispatch import _select_idle_round_robin

try:
    from ...utils.trace_logger import trace_debug
except ImportError:  # pragma: no cover - optional runtime helper
    def trace_debug(*_args, **_kwargs):
        return None

POLL_INTERVAL_SECONDS = 1.0
# A worker that was just handed a run is treated as busy for this long even if
# a probe races ahead of the worker enqueueing the prompt.
RECENT_DISPATCH_TTL_SECONDS = 5.0
ASSIGNMENT_WAIT_SLICE_SECONDS = 0.5
# A delegate prompt that could not be re-queued after its worker was
# dispatched is retried with this backoff; the remote job stays tracked.
REQUEUE_RETRY_MIN_SECONDS = 1.0
REQUEUE_RETRY_MAX_SECONDS = 30.0
# Consecutive failed probes before a dispatched worker counts as gone.
WORKER_UNREACHABLE_PROBES = 3

MASTER = "master"


def _worker_key(worker):
    return str(worker.get("id"))


async def probe_worker_queues(workers, probe_concurrency=8, probe_timeout=3.0):
    """Return ``[{"worker", "queue_remaining"}]`` for every worker that answered."""
    semaphore = asyncio.Semaphore(max(1, int(probe_concurrency or 1)))

    async def _probe(worker):
        async with semaphore:
            if worker.get("type") == "serverless":
                from .serverless_runtime import get_manager
                status = await get_manager().health(worker)
                return {"worker": worker, "queue_remaining": status["queueRemaining"]}
            payload = await probe_worker(build_worker_url(worker), timeout=probe_timeout)
            if payload is None:
                return None
            return {"worker": worker, "queue_remaining": _extract_queue_remaining(payload)}

    statuses = await asyncio.gather(*[_probe(worker) for worker in workers])
    return [status for status in statuses if status is not None]


async def select_idle_worker(workers, trace_execution_id=None, probe_concurrency=8, probe_timeout=3.0):
    """Pick one idle participant, round-robin; ``(None, probed)`` when all are busy.

    ``probed`` is how many participants answered the probe, so a caller can
    tell "everyone is busy" from "nobody answered".
    """
    if not workers:
        return None, 0
    statuses = await probe_worker_queues(workers, probe_concurrency, probe_timeout)
    idle = [status for status in statuses if status["queue_remaining"] == 0]
    if not idle:
        if trace_execution_id:
            trace_debug(trace_execution_id, f"No idle participant among {len(statuses)} probed; run will wait for the next free one.")
        return None, len(statuses)
    selected = _select_idle_round_robin(idle)["worker"]
    if trace_execution_id:
        trace_debug(trace_execution_id, f"Idle participant selected: {selected.get('name')} ({selected.get('id')})")
    return selected, len(statuses)


class PendingRun:
    """One load-balanced prompt waiting in the master queue for a free participant."""

    def __init__(self, prompt_id, job_ids, candidates, master_may_run,
                 dispatch_to_worker, requeue_as_delegate, trace_execution_id=None):
        self.prompt_id = str(prompt_id)
        self.job_ids = {str(job_id) for job_id in job_ids}
        self.candidates = list(candidates)
        self.master_may_run = bool(master_may_run)
        # async (worker) -> None: prepare and send the worker prompt.
        self.dispatch_to_worker = dispatch_to_worker
        # async (worker) -> None: queue the delegate master prompt under prompt_id.
        self.requeue_as_delegate = requeue_as_delegate
        self.trace_execution_id = trace_execution_id
        self.assigned = None  # None | MASTER | worker dict
        self.claiming = False
        # False until the master prompt is actually in the ComfyUI queue; the
        # dispatcher never sweeps or claims a run before that.
        self.queued = False
        # A worker runs the job but the master's delegate prompt is not queued
        # yet; the dispatcher keeps retrying so the result is still collected.
        self.requeue_pending = False
        self.requeue_failures = 0
        self.requeue_retry_at = 0.0
        # Participating master: hold the delegate prompt back until the worker
        # delivered, so the master executor is not parked on a waiting collector.
        self.wait_for_result = False
        self.delivered_jobs = set()
        self.dispatched_at = 0.0
        self.missed_probes = 0
        self.finished_jobs = set()
        self.created_at = time.monotonic()
        self._event = asyncio.Event()

    def _assign(self, participant):
        self.assigned = participant
        self._event.set()

    @property
    def assigned_id(self):
        if self.assigned is None:
            return None
        if self.assigned == MASTER:
            return MASTER
        return _worker_key(self.assigned)

    def accepts(self, worker):
        key = _worker_key(worker)
        return any(_worker_key(candidate) == key for candidate in self.candidates)


class LoadBalanceQueue:
    """Dispatcher that pulls waiting runs for whichever participant frees up next."""

    def __init__(self, prompt_queue_getter):
        self._prompt_queue_getter = prompt_queue_getter
        self._records = {}  # prompt_id -> PendingRun, insertion ordered
        self._by_job = {}  # job_id -> PendingRun
        self._recent_dispatch = {}  # worker id -> monotonic time
        self._wake = asyncio.Event()
        self._cycle_lock = asyncio.Lock()
        self._task = None
        self.poll_interval = POLL_INTERVAL_SECONDS

    # ----- registration -------------------------------------------------
    def register(self, record):
        """Track a run before its master prompt is queued (see ``mark_queued``)."""
        self._records[record.prompt_id] = record
        for job_id in record.job_ids:
            self._by_job[job_id] = record
        return record

    def mark_queued(self, record):
        """The master prompt is now in the ComfyUI queue: start dispatching it."""
        record.queued = True
        self._ensure_task()
        self.wake()

    def register_dispatched(self, record, worker):
        """Track a run that was sent straight to an idle worker.

        Used on a participating master: the delegate prompt is queued later,
        when the worker delivers, instead of parking the master executor now.
        """
        self.register(record)
        record.queued = True
        self._hold_delegate(record, worker)
        self._ensure_task()
        self.wake()

    def result_delivered(self, job_id):
        """A worker posted the final result for ``job_id``."""
        record = self._by_job.get(str(job_id))
        if record is None:
            return False
        record.delivered_jobs.add(str(job_id))
        self.wake()
        return True

    def _hold_delegate(self, record, worker):
        record._assign(worker)
        record.requeue_pending = True
        record.wait_for_result = True
        record.dispatched_at = time.monotonic()
        record.missed_probes = 0
        self.mark_worker_busy(_worker_key(worker))

    def lookup_job(self, job_id):
        return self._by_job.get(str(job_id))

    def pending_runs(self):
        return [record for record in self._records.values() if record.assigned is None]

    def __len__(self):
        return len(self._records)

    def mark_worker_busy(self, worker_id):
        self._recent_dispatch[str(worker_id)] = time.monotonic()

    def wake(self):
        self._wake.set()

    def _remove(self, record):
        self._records.pop(record.prompt_id, None)
        for job_id in record.job_ids:
            if self._by_job.get(job_id) is record:
                self._by_job.pop(job_id, None)

    # ----- collector side ------------------------------------------------
    async def wait_for_assignment(self, record, interrupt_check=None):
        """Return the participant that runs ``record`` (``MASTER`` or a worker dict).

        Called by the master Collector. A run whose prompt is already executing
        on a participating master is claimed for the master right away; an
        orchestrator-only master waits here until the dispatcher assigns a
        worker.
        """
        # A Collector only runs once the prompt reached the queue, whatever
        # the orchestrator managed to report in between.
        record.queued = True
        if record.assigned is None and record.master_may_run and not record.claiming:
            record._assign(MASTER)
            debug_log(f"Load balance: master took run {record.prompt_id}")
        while record.assigned is None:
            if interrupt_check is not None:
                interrupt_check()
            self._ensure_task()
            self.wake()
            try:
                await asyncio.wait_for(record._event.wait(), timeout=ASSIGNMENT_WAIT_SLICE_SECONDS)
            except asyncio.TimeoutError:
                continue
        return record.assigned

    def finish_job(self, record, job_id, failed=False):
        record.finished_jobs.add(str(job_id))
        if failed or record.finished_jobs >= record.job_ids:
            self._remove(record)

    # ----- queue inspection ---------------------------------------------
    def _prompt_queue(self):
        return self._prompt_queue_getter()

    def _is_running(self, prompt_id):
        prompt_queue = self._prompt_queue()
        with prompt_queue.mutex:
            return any(item[1] == prompt_id for item in prompt_queue.currently_running.values())

    def _is_queued(self, prompt_id):
        prompt_queue = self._prompt_queue()
        with prompt_queue.mutex:
            return any(item[1] == prompt_id for item in prompt_queue.queue)

    def _take_from_queue(self, prompt_id):
        """Remove the prompt from the master queue; return the item or None."""
        captured = []

        def _match(item):
            if item[1] == prompt_id:
                captured.append(item)
                return True
            return False

        if self._prompt_queue().delete_queue_item(_match):
            return captured[0]
        return None

    def _restore(self, item):
        self._prompt_queue().put(item)

    # ----- dispatcher loop ----------------------------------------------
    def _ensure_task(self):
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._loop())

    async def _loop(self):
        try:
            while self._records:
                self._wake.clear()
                try:
                    await self.run_cycle()
                except Exception as exc:  # keep the dispatcher alive
                    log(f"[Distributed] Load-balance dispatcher error: {exc}")
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._task = None

    def sweep(self):
        """Drop runs that left the master queue without being collected."""
        for record in list(self._records.values()):
            if record.claiming or not record.queued or record.requeue_pending:
                continue
            if self._is_queued(record.prompt_id):
                continue
            if self._is_running(record.prompt_id):
                if record.assigned is None and record.master_may_run:
                    record._assign(MASTER)
                    debug_log(f"Load balance: master is running run {record.prompt_id}")
                continue
            debug_log(f"Load balance: run {record.prompt_id} left the master queue; dropping it")
            self._remove(record)

    def _recently_dispatched(self, worker):
        stamp = self._recent_dispatch.get(_worker_key(worker))
        if stamp is None:
            return False
        if time.monotonic() - stamp > RECENT_DISPATCH_TTL_SECONDS:
            self._recent_dispatch.pop(_worker_key(worker), None)
            return False
        return True

    async def run_cycle(self, probe_concurrency=8, probe_timeout=3.0):
        """One dispatcher pass: sweep, probe candidates, hand runs to idle workers."""
        async with self._cycle_lock:
            return await self._run_cycle(probe_concurrency, probe_timeout)

    async def _run_cycle(self, probe_concurrency, probe_timeout):
        self.sweep()
        await self._retry_requeues()
        pending = [record for record in self._records.values()
                   if record.queued and record.assigned is None and not record.claiming]
        if not pending:
            return []
        candidates = {}
        for record in pending:
            for worker in record.candidates:
                candidates.setdefault(_worker_key(worker), worker)
        probe_targets = [worker for worker in candidates.values() if not self._recently_dispatched(worker)]
        if not probe_targets:
            return []
        statuses = await probe_worker_queues(probe_targets, probe_concurrency, probe_timeout)
        idle_workers = [status["worker"] for status in statuses if status["queue_remaining"] == 0]
        claims = []
        for worker in idle_workers:
            record = next((record for record in pending
                           if record.assigned is None and not record.claiming and record.accepts(worker)), None)
            if record is None:
                continue
            record.claiming = True
            self.mark_worker_busy(_worker_key(worker))
            claims.append((record, worker))
        if claims:
            await asyncio.gather(*[self._claim(record, worker) for record, worker in claims])
        return claims

    async def _requeue(self, record, worker):
        try:
            await record.requeue_as_delegate(worker)
        except Exception as exc:
            record.requeue_failures += 1
            delay = min(REQUEUE_RETRY_MIN_SECONDS * (2 ** (record.requeue_failures - 1)), REQUEUE_RETRY_MAX_SECONDS)
            record.requeue_retry_at = time.monotonic() + delay
            log(
                f"[Distributed] Load balance: could not re-queue run {record.prompt_id} on the master after "
                f"dispatch to {worker.get('name')} ({worker.get('id')}): {exc}. Retrying in {delay:.0f}s; "
                "the worker's result stays tracked."
            )
            return False
        record.requeue_pending = False
        record.requeue_failures = 0
        return True

    async def _worker_done_without_result(self, record):
        """True when the dispatched worker is idle or gone and never delivered."""
        if time.monotonic() - record.dispatched_at < RECENT_DISPATCH_TTL_SECONDS:
            return False
        statuses = await probe_worker_queues([record.assigned])
        if not statuses:
            record.missed_probes += 1
            return record.missed_probes >= WORKER_UNREACHABLE_PROBES
        record.missed_probes = 0
        return statuses[0]["queue_remaining"] == 0

    async def _retry_requeues(self):
        now = time.monotonic()
        for record in list(self._records.values()):
            if not record.requeue_pending:
                continue
            if record.assigned is None or record.assigned == MASTER:
                record.requeue_pending = False
                continue
            if record.wait_for_result and not record.delivered_jobs >= record.job_ids:
                if not await self._worker_done_without_result(record):
                    continue
                log(
                    f"[Distributed] Load balance: worker {record.assigned_id} is idle or unreachable without "
                    f"delivering run {record.prompt_id[:8]}; queueing its collector so the run can report."
                )
                record.wait_for_result = False
            if record.requeue_retry_at > now:
                continue
            if await self._requeue(record, record.assigned):
                debug_log(f"Load balance: run {record.prompt_id} is on the master queue to collect its result")

    async def _claim(self, record, worker):
        worker_label = f"{worker.get('name')} ({worker.get('id')})"
        try:
            item = self._take_from_queue(record.prompt_id)
            if item is None:
                if not self._is_running(record.prompt_id):
                    debug_log(f"Load balance: run {record.prompt_id} is gone; nothing to dispatch")
                    self._remove(record)
                    return False
                if record.master_may_run:
                    record._assign(MASTER)
                    return False
                # Orchestrator-only master: its Collector is already waiting for us.
            try:
                await record.dispatch_to_worker(worker)
            except Exception as exc:
                log(f"[Distributed] Load balance: dispatch to {worker_label} failed ({exc}); run {record.prompt_id} keeps waiting")
                if item is not None:
                    self._restore(item)
                return False
            if item is not None and record.master_may_run:
                # Keep the master executor free for full workflows: the
                # delegate prompt is queued once the worker delivers.
                self._hold_delegate(record, worker)
            else:
                if item is not None and not await self._requeue(record, worker):
                    # The worker is running the job; keep the run tracked and
                    # keep trying to put its collector back on the master queue.
                    record.requeue_pending = True
                record._assign(worker)
                self.mark_worker_busy(_worker_key(worker))
            trace_debug(record.trace_execution_id, f"Load balance: run {record.prompt_id} dispatched to free worker {worker_label}")
            log(f"[Distributed] Load balance: {worker_label} freed up; running queued prompt {record.prompt_id[:8]} there")
            return True
        finally:
            record.claiming = False


def get_load_balance_queue(server_instance=None):
    """Process-wide dispatcher stored on the prompt server."""
    ps = server_instance or server.PromptServer.instance
    queue = getattr(ps, "distributed_load_balance_queue", None)
    if queue is None:
        queue = LoadBalanceQueue(lambda: ps.prompt_queue)
        ps.distributed_load_balance_queue = queue
    return queue
