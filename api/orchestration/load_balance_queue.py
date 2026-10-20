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

Two lifecycle rules keep a run from executing twice or vanishing:

* A submission whose acceptance is unknown (the transport raised
  ``AmbiguousDispatchError``) is never replayed. The run is held as
  *unresolved* until the worker delivers, or until the worker's own queue
  and history are checked for this run's job ids: found there means it was
  accepted and the result is collected; confirmed absent on consecutive
  checks means it never arrived and the original prompt goes back. Aggregate
  queue depth is never used for this, since an idle worker may have run the
  job and lost the delivery, and a busy worker may be on something else.
* While a run is out of the master queue for a hand-off, cancellation
  (queue clear, queue delete, interrupting its Collector) is recorded on the
  run itself. Work that was not yet submitted is dropped along with its
  result queues; work a worker already accepted stays tracked until its
  result arrives or the worker is seen done, and is then discarded instead
  of reappearing.
"""
import asyncio
import threading
import time
from collections import OrderedDict

import server

from ...utils.logging import debug_log, log
from ...utils.network import build_worker_url, probe_worker
from .dispatch import _extract_queue_remaining
from .dispatch import _select_idle_round_robin

try:
    from .dispatch import AmbiguousDispatchError
except ImportError:  # pragma: no cover - older transport without the typed error
    class AmbiguousDispatchError(RuntimeError):
        """Raised when a worker may or may not have accepted a prompt."""

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
# An unresolved submission is restored only after this many consecutive
# worker queue+history checks found no trace of its job ids.
ABSENT_CONFIRMATIONS = 2
# How far back the worker's history is searched for an unresolved run.
HISTORY_LOOKBACK_ITEMS = 200
# Job ids whose result queues were discarded; a late worker must not recreate them.
CANCELLED_JOB_MEMORY = 512

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


def _graph_has_job(graph, job_ids):
    if not isinstance(graph, dict):
        return False
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and str(inputs.get("multi_job_id", "")) in job_ids:
            return True
    return False


async def fetch_worker_job_state(worker, job_ids, timeout=5.0):
    """Ask the worker itself whether a specific run is there.

    Returns ``"running"`` when a prompt carrying one of ``job_ids`` is in the
    worker's running or pending queue, ``"finished"`` when it is in the
    worker's recent history, ``"absent"`` when the worker answered both
    questions and has no trace of it, and ``None`` when the worker could not
    be asked. Direct workers only; serverless pools never reach this path.
    """
    import aiohttp
    from ...utils.network import get_client_session

    job_ids = {str(job_id) for job_id in job_ids}
    base_url = build_worker_url(worker)
    session = await get_client_session()
    try:
        async with session.get(f"{base_url}/queue", timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                return None
            queue_info = await response.json()
        if not isinstance(queue_info, dict):
            return None
        for key in ("queue_running", "queue_pending"):
            for item in queue_info.get(key) or []:
                if isinstance(item, (list, tuple)) and len(item) > 2 and _graph_has_job(item[2], job_ids):
                    return "running"
        async with session.get(
            f"{base_url}/history",
            params={"max_items": str(HISTORY_LOOKBACK_ITEMS)},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                return None
            history = await response.json()
        if not isinstance(history, dict):
            return None
        for entry in history.values():
            item = entry.get("prompt") if isinstance(entry, dict) else None
            if isinstance(item, (list, tuple)) and len(item) > 2 and _graph_has_job(item[2], job_ids):
                return "finished"
        return "absent"
    except Exception as exc:
        debug_log(f"Load balance: could not check worker {_worker_key(worker)} for a run: {exc}")
        return None


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
                 prepare_for_worker, send_to_worker, requeue_as_delegate, trace_execution_id=None):
        self.prompt_id = str(prompt_id)
        self.job_ids = {str(job_id) for job_id in job_ids}
        self.candidates = list(candidates)
        self.master_may_run = bool(master_may_run)
        # async (worker) -> payload: build the worker prompt (media sync included).
        self.prepare_for_worker = prepare_for_worker
        # async (worker, payload) -> None: submit it; may raise AmbiguousDispatchError.
        self.send_to_worker = send_to_worker
        # async (worker) -> None: queue the delegate master prompt under prompt_id.
        self.requeue_as_delegate = requeue_as_delegate
        self.trace_execution_id = trace_execution_id
        self.assigned = None  # None | MASTER | worker dict
        self.claiming = False
        # False until the master prompt is actually in the ComfyUI queue; the
        # dispatcher never sweeps or claims a run before that.
        self.queued = False
        # The user cleared or deleted this run (or interrupted its Collector)
        # while it was out of the master queue. Nothing new is submitted or
        # re-queued for it; accepted remote work is discarded when it ends.
        self.cancelled = False
        # Submission whose acceptance is unknown: worker dict, the master
        # queue item to restore if it turns out rejected, and when it happened.
        self.unresolved = None
        self.unresolved_item = None
        self.unresolved_since = 0.0
        self.absent_checks = 0
        # No master prompt exists for this run yet (sent straight to a worker
        # by a participating master): its collector still has to be queued.
        self.delegate_needed = False
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

    @property
    def delivered(self):
        return self.delivered_jobs >= self.job_ids

    @property
    def on_worker(self):
        """A worker accepted (or may have accepted) this run."""
        return self.unresolved is not None or (self.assigned is not None and self.assigned != MASTER)

    @property
    def in_flight(self):
        """Out of the master queue for a hand-off: cancellation must be recorded here."""
        return self.claiming or self.on_worker

    def accepts(self, worker):
        key = _worker_key(worker)
        return any(_worker_key(candidate) == key for candidate in self.candidates)


class LoadBalanceQueue:
    """Dispatcher that pulls waiting runs for whichever participant frees up next."""

    def __init__(self, prompt_queue_getter, pending_jobs_getter=None):
        self._prompt_queue_getter = prompt_queue_getter
        self._pending_jobs_getter = pending_jobs_getter or (lambda: None)
        self._records = {}  # prompt_id -> PendingRun, insertion ordered
        self._by_job = {}  # job_id -> PendingRun
        self._recent_dispatch = {}  # worker id -> monotonic time
        self._cancelled_jobs = OrderedDict()
        self._wake = asyncio.Event()
        self._cycle_lock = asyncio.Lock()
        self._task = None
        self._hooks_installed = False
        self._delete_unhooked = None
        self._hook_depth = 0
        self._hook_lock = threading.RLock()
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
        record.delegate_needed = True
        self._hold_delegate(record, worker)
        self._ensure_task()
        self.wake()

    def mark_unresolved(self, record, worker, item=None):
        """The submission to ``worker`` may or may not have been accepted.

        The run is neither replayed nor dropped until the worker delivers, is
        seen busy, or is seen idle without delivering.
        """
        record.assigned = None
        record.requeue_pending = False
        record.wait_for_result = False
        record.unresolved = worker
        record.unresolved_item = item
        record.unresolved_since = time.monotonic()
        record.missed_probes = 0
        record.absent_checks = 0
        self.mark_worker_busy(_worker_key(worker))
        self._ensure_task()
        self.wake()

    def discard_run(self, prompt_id):
        """Forget a run that never made it (and its result queues)."""
        record = self._records.get(str(prompt_id))
        if record is not None:
            self._discard(record)

    def result_delivered(self, job_id):
        """A worker posted the final result for ``job_id``."""
        record = self._by_job.get(str(job_id))
        if record is None:
            return False
        record.delivered_jobs.add(str(job_id))
        self.wake()
        return True

    def is_cancelled_job(self, job_id):
        """True when this job's result queue was discarded; it must not be recreated."""
        return str(job_id) in self._cancelled_jobs

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

    def _discard(self, record):
        """Drop the run and the result queues nobody will consume any more."""
        self._remove(record)
        pending_jobs = self._pending_jobs_getter()
        for job_id in record.job_ids:
            if isinstance(pending_jobs, dict):
                pending_jobs.pop(job_id, None)
            self._cancelled_jobs[job_id] = True
            self._cancelled_jobs.move_to_end(job_id)
        while len(self._cancelled_jobs) > CANCELLED_JOB_MEMORY:
            self._cancelled_jobs.popitem(last=False)

    # ----- cancellation ---------------------------------------------------
    def install_cancellation_hooks(self):
        """See queue clears and deletes that happen while a run is in flight.

        ComfyUI's queue only knows about items it holds; a run that was taken
        out for a hand-off would otherwise be invisible to "Clear queue" and
        come back when the hand-off completes.
        """
        if self._hooks_installed:
            return
        prompt_queue = self._prompt_queue()
        original_wipe = prompt_queue.wipe_queue
        original_delete = prompt_queue.delete_queue_item
        self._delete_unhooked = original_delete
        dispatcher = self

        def wipe_queue():
            with dispatcher._hook_lock:
                nested = dispatcher._hook_depth > 0
            original_wipe()
            if not nested:
                # ComfyUI's delete_queue_item wipes internally when the last
                # item goes; only a real clear cancels in-flight runs.
                dispatcher.cancel_matching(lambda _item: True)

        def delete_queue_item(function):
            with dispatcher._hook_lock:
                dispatcher._hook_depth += 1
            try:
                removed = original_delete(function)
            finally:
                with dispatcher._hook_lock:
                    dispatcher._hook_depth -= 1
            dispatcher.cancel_matching(function)
            return removed

        prompt_queue.wipe_queue = wipe_queue
        prompt_queue.delete_queue_item = delete_queue_item
        self._hooks_installed = True

    def cancel_matching(self, predicate):
        """Cancel in-flight runs whose queue item ``predicate`` would have deleted."""
        cancelled = []
        for record in list(self._records.values()):
            if record.cancelled or not record.in_flight:
                continue
            item = record.unresolved_item or (0, record.prompt_id, {}, {}, [])
            try:
                matched = bool(predicate(item))
            except Exception:
                matched = False
            if matched:
                self._cancel(record)
                cancelled.append(record)
        if cancelled:
            self.wake()
        return cancelled

    def _cancel(self, record):
        record.cancelled = True
        record.requeue_pending = False
        record.wait_for_result = False
        if not record.in_flight:
            self._discard(record)
            return
        log(
            f"[Distributed] Load balance: run {record.prompt_id[:8]} was cancelled while being handed to a worker; "
            "nothing more is submitted for it and its remote result will be discarded."
        )

    async def _finalize_cancelled(self):
        """Discard cancelled runs once their remote work is known to be over."""
        for record in list(self._records.values()):
            if not record.cancelled or record.claiming or record.unresolved is not None:
                continue
            if record.assigned is None or record.assigned == MASTER:
                self._discard(record)
                continue
            if record.delivered or await self._worker_done_without_result(record):
                debug_log(f"Load balance: cancelled run {record.prompt_id} finished on {record.assigned_id}; discarded")
                self._discard(record)

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
        if record.assigned is None and record.master_may_run and not record.in_flight:
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
        """The master Collector finished (or gave up on) one of the run's jobs.

        On failure or interruption the run is cancelled: a hand-off still in
        flight is stopped before submission, and work a worker already
        accepted is discarded when it ends, so no result queue is left behind.
        """
        if failed:
            if record.prompt_id in self._records:
                self._cancel(record)
            return
        record.finished_jobs.add(str(job_id))
        if record.finished_jobs >= record.job_ids:
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

        delete = self._delete_unhooked or self._prompt_queue().delete_queue_item
        # ComfyUI's delete wipes internally when the last item goes; that
        # internal wipe must not read as the user clearing the queue.
        with self._hook_lock:
            self._hook_depth += 1
        try:
            removed = delete(_match)
        finally:
            with self._hook_lock:
                self._hook_depth -= 1
        if removed:
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
            if not record.queued or record.in_flight or record.cancelled:
                continue
            if self._is_queued(record.prompt_id):
                continue
            if self._is_running(record.prompt_id):
                if record.assigned is None and record.master_may_run:
                    record._assign(MASTER)
                    debug_log(f"Load balance: master is running run {record.prompt_id}")
                continue
            debug_log(f"Load balance: run {record.prompt_id} left the master queue; dropping it")
            self._discard(record)

    def _recently_dispatched(self, worker):
        stamp = self._recent_dispatch.get(_worker_key(worker))
        if stamp is None:
            return False
        if time.monotonic() - stamp > RECENT_DISPATCH_TTL_SECONDS:
            self._recent_dispatch.pop(_worker_key(worker), None)
            return False
        return True

    async def run_cycle(self, probe_concurrency=8, probe_timeout=3.0):
        """One dispatcher pass: sweep, resolve, retry, then hand runs to idle workers."""
        async with self._cycle_lock:
            return await self._run_cycle(probe_concurrency, probe_timeout)

    async def _run_cycle(self, probe_concurrency, probe_timeout):
        self.sweep()
        await self._resolve_unresolved()
        await self._finalize_cancelled()
        await self._retry_requeues()
        pending = [record for record in self._records.values()
                   if record.queued and record.assigned is None and not record.in_flight and not record.cancelled]
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
                           if record.assigned is None and not record.in_flight and not record.cancelled
                           and record.accepts(worker)), None)
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
            if record.cancelled or getattr(exc, "cancelled_insertion", False):
                # Cancelled while the delegate prompt was being validated: the
                # insertion was refused on purpose, nothing to retry.
                debug_log(f"Load balance: delegate for run {record.prompt_id} not queued, run was cancelled")
                return False
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
        record.delegate_needed = False
        return True

    async def _probe_one(self, record, worker):
        """Return 'unreachable', 'idle' or 'busy' for the worker running ``record``."""
        statuses = await probe_worker_queues([worker])
        if not statuses:
            record.missed_probes += 1
            return "unreachable"
        record.missed_probes = 0
        return "idle" if statuses[0]["queue_remaining"] == 0 else "busy"

    async def _worker_done_without_result(self, record):
        """True when the worker no longer has this run and never delivered.

        Asks the worker about the run itself: still queued or running means
        wait; finished (delivery lost) or no trace of it means the collector
        should be queued so the run can report; unreachable counts up.
        """
        if time.monotonic() - record.dispatched_at < RECENT_DISPATCH_TTL_SECONDS:
            return False
        state = await fetch_worker_job_state(record.assigned, record.job_ids)
        if state is None:
            record.missed_probes += 1
            return record.missed_probes >= WORKER_UNREACHABLE_PROBES
        record.missed_probes = 0
        return state != "running"

    async def _resolve_unresolved(self):
        """Settle submissions whose acceptance was unknown; never replay them.

        The worker is asked about this run specifically (its job ids in the
        worker's queue or history). Aggregate idle/busy state is not evidence:
        an idle worker may have run the job and lost the delivery, a busy one
        may be on unrelated work.
        """
        for record in list(self._records.values()):
            worker = record.unresolved
            if worker is None:
                continue
            if record.delivered:
                self._accept_unresolved(record, "delivered its result")
                continue
            if time.monotonic() - record.unresolved_since < RECENT_DISPATCH_TTL_SECONDS:
                continue
            state = await fetch_worker_job_state(worker, record.job_ids)
            if state == "running":
                self._accept_unresolved(record, "has it queued or running")
            elif state == "finished":
                self._accept_unresolved(record, "ran it (result not received yet)")
            elif state == "absent":
                record.absent_checks += 1
                record.missed_probes = 0
                if record.absent_checks >= ABSENT_CONFIRMATIONS:
                    self._reject_unresolved(record)
            else:
                record.absent_checks = 0
                record.missed_probes += 1
                if record.missed_probes in (1, WORKER_UNREACHABLE_PROBES):
                    log(
                        f"[Distributed] Load balance: worker {_worker_key(worker)} cannot be asked about run "
                        f"{record.prompt_id[:8]}; it stays on hold until its acceptance can be confirmed."
                    )

    def _accept_unresolved(self, record, how):
        worker, item = record.unresolved, record.unresolved_item
        record.unresolved = None
        record.unresolved_item = None
        log(f"[Distributed] Load balance: worker {_worker_key(worker)} {how}; run {record.prompt_id[:8]} is confirmed there")
        if record.cancelled:
            record._assign(worker)
            record.dispatched_at = time.monotonic()
            return
        if record.delegate_needed or (item is not None and record.master_may_run):
            # Participating master: its collector is queued once the worker
            # delivers (an eager run never had a master prompt at all).
            self._hold_delegate(record, worker)
        elif item is not None:
            # Orchestrator-only: queue the collector now.
            record._assign(worker)
            record.dispatched_at = time.monotonic()
            record.requeue_pending = True
            record.requeue_retry_at = 0.0
        else:
            # The collector is already executing on the master and waiting.
            record._assign(worker)
            record.dispatched_at = time.monotonic()

    def _reject_unresolved(self, record):
        worker, item = record.unresolved, record.unresolved_item
        record.unresolved = None
        record.unresolved_item = None
        if record.cancelled:
            self._discard(record)
            return
        if item is None:
            # Nothing to put back: the run only ever existed on that worker.
            log(f"[Distributed] Load balance: worker {_worker_key(worker)} never took run {record.prompt_id[:8]}; dropping it")
            self._discard(record)
            return
        self._restore(item)
        log(f"[Distributed] Load balance: worker {_worker_key(worker)} never took run {record.prompt_id[:8]}; back in the master queue")

    async def _retry_requeues(self):
        now = time.monotonic()
        for record in list(self._records.values()):
            if not record.requeue_pending or record.cancelled:
                continue
            if record.assigned is None or record.assigned == MASTER:
                record.requeue_pending = False
                continue
            if record.wait_for_result and not record.delivered:
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
                    self._discard(record)
                    return False
                if record.master_may_run:
                    record._assign(MASTER)
                    return False
                # Orchestrator-only master: its Collector is already waiting for us.
            if record.cancelled:
                self._discard(record)
                return False
            try:
                payload = await record.prepare_for_worker(worker)
            except Exception as exc:
                log(f"[Distributed] Load balance: could not prepare run {record.prompt_id[:8]} for {worker_label} ({exc}); it keeps waiting")
                if record.cancelled:
                    self._discard(record)
                elif item is not None:
                    self._restore(item)
                return False
            if record.cancelled:
                # Cleared while preparing: nothing was submitted, so nothing comes back.
                debug_log(f"Load balance: run {record.prompt_id} cancelled before submission to {worker_label}")
                self._discard(record)
                return False
            try:
                await record.send_to_worker(worker, payload)
            except AmbiguousDispatchError as exc:
                log(
                    f"[Distributed] Load balance: {worker_label} may have accepted run {record.prompt_id[:8]} "
                    f"({exc}); holding it until the worker confirms, it is not re-sent anywhere."
                )
                self.mark_unresolved(record, worker, item)
                return False
            except Exception as exc:
                log(f"[Distributed] Load balance: dispatch to {worker_label} failed ({exc}); run {record.prompt_id[:8]} keeps waiting")
                if record.cancelled:
                    self._discard(record)
                elif item is not None:
                    self._restore(item)
                return False
            if record.cancelled:
                # Accepted, then cleared: keep ownership until the worker is done.
                record._assign(worker)
                record.dispatched_at = time.monotonic()
                self.mark_worker_busy(_worker_key(worker))
                return False
            if item is not None and record.master_may_run:
                # Keep the master executor free for full workflows: the
                # delegate prompt is queued once the worker delivers.
                self._hold_delegate(record, worker)
            else:
                if item is not None and not await self._requeue(record, worker) and not record.cancelled:
                    # The worker is running the job; keep the run tracked and
                    # keep trying to put its collector back on the master queue.
                    record.requeue_pending = True
                record._assign(worker)
                record.dispatched_at = time.monotonic()
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
        queue = LoadBalanceQueue(lambda: ps.prompt_queue, lambda: getattr(ps, "distributed_pending_jobs", None))
        ps.distributed_load_balance_queue = queue
    if not queue._hooks_installed and getattr(ps, "prompt_queue", None) is not None:
        try:
            queue.install_cancellation_hooks()
        except Exception as exc:  # pragma: no cover - defensive against unusual queue objects
            debug_log(f"Load balance: cancellation hooks not installed: {exc}")
    return queue
