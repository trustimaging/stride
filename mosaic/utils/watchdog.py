
import asyncio
import mosaic


__all__ = ['Watchdog', 'WatchdogCancelled']


class WatchdogCancelled(Exception):
    """Raised when a Watchdog cancels its protected coroutine."""


class _WatchdogLogger:
    """
    ``mosaic.logger()`` with ``'Watchdog: '`` prefixed to every message.

    Uses ``__getattr__`` so a single 4-line class covers debug/info/perf/
    warning/error without listing them explicitly.
    """

    def __getattr__(self, name):
        def call(msg, uid=None):
            getattr(mosaic.logger(), name)('Watchdog: ' + msg, uid=uid)
        return call


logger = _WatchdogLogger()


class Watchdog:
    """
    Run coroutines under a worker-drop watchdog with retry/accept policy.

    Three knobs: ``drop_threshold`` (fraction of gradients we'll accept
    losing per attempt; ``None`` disables protection), ``desired_workers``
    (target pool to wait for between attempts), ``min_workers`` (floor
    below which retry raises).

    Parameters
    ----------
    runtime : Runtime
        Runtime whose worker pool to monitor.
    drop_threshold : float or None
        Loss tolerance during dispatch. ``None`` disables protection.
    min_workers : int, optional
        Hard floor. Must be >= 1. Defaults to 1.
    desired_workers : int, optional
        Target for rollback re-waits. Defaults to ``min_workers``.
    max_attempts : int, optional
        Cap on retry attempts. Defaults to 5.

    """

    def __init__(self, runtime, drop_threshold, min_workers=1,
                 desired_workers=None, max_attempts=5):
        if min_workers < 1:
            raise ValueError('min_workers must be >= 1')
        if desired_workers is None:
            desired_workers = min_workers
        if desired_workers < min_workers:
            raise ValueError(
                f'desired_workers ({desired_workers}) must be >= '
                f'min_workers ({min_workers})'
            )
        self.runtime = runtime
        self.drop_threshold = drop_threshold
        self.min_workers = min_workers
        self.desired_workers = desired_workers
        self.max_attempts = max_attempts

    @staticmethod
    async def wait_for_workers(runtime, desired_workers, min_workers=None,
                               timeout=300.0, heartbeat=30.0):
        """
        Wait up to ``timeout`` for ``desired_workers`` workers to appear
        and their nodes to report ready. On timeout, proceed if
        ``num_workers >= min_workers`` (default ``desired_workers``),
        else raise ``RuntimeError``. Static — callable without a
        Watchdog instance.

        Parameters
        ----------
        runtime : Runtime
        desired_workers : int
            Target count; return as soon as reached.
        min_workers : int, optional
            Hard floor. Defaults to ``desired_workers`` (strict).
        timeout : float, optional
            Total seconds across both phases. Defaults to 300.
        heartbeat : float, optional
            Progress log interval. Defaults to 30.

        Raises
        ------
        RuntimeError
            Pool below ``min_workers`` after ``timeout``.

        """
        if min_workers is None:
            min_workers = desired_workers

        present_uids = set(w.uid for w in runtime.workers)
        if runtime.num_workers >= desired_workers:
            logger.debug(
                f'already have {runtime.num_workers}/{desired_workers} workers'
                f' — skipping wait (present: {sorted(present_uids)})'
            )
            return

        logger.debug(
            f'waiting for workers — desired {desired_workers} '
            f'min {min_workers}, have {runtime.num_workers} '
            f'(present: {sorted(present_uids)})'
        )

        tic = asyncio.get_event_loop().time()
        end_time = tic + timeout

        await Watchdog._wait_for_worker_count(runtime, desired_workers,
                                              end_time, heartbeat)
        await Watchdog._wait_for_nodes_ready(runtime, end_time)

        elapsed = asyncio.get_event_loop().time() - tic

        if runtime.num_workers < min_workers:
            raise RuntimeError(
                f'pool has {runtime.num_workers} workers after {elapsed:.0f}s, '
                f'need at least {min_workers} (desired {desired_workers})'
            )

        if runtime.num_workers < desired_workers:
            logger.warning(
                f'proceeding with {runtime.num_workers}/{desired_workers}'
                f' workers after {elapsed:.0f}s (min {min_workers})'
            )
        else:
            logger.debug(
                f'wait done after {elapsed:.0f}s — {runtime.num_workers}'
                f' workers (desired {desired_workers})'
            )

    async def broadcast(self, args, label='broadcast'):
        """
        Publish each arg to all workers under drop protection. On drop,
        cancel and retry on the surviving pool.

        Parameters
        ----------
        args : tuple or list
            Each element becomes one published TesseraProxy.
        label : str, optional
            Log-line tag. Defaults to ``'broadcast'``.

        Returns
        -------
        list of TesseraProxy

        Raises
        ------
        RuntimeError
            All ``max_attempts`` attempts were cancelled.

        """
        async def _broadcast():
            proxies = [self.runtime.put(each, publish=False) for each in args]
            return await asyncio.gather(*proxies)

        if self.drop_threshold is None:
            return await _broadcast()

        for attempt in range(self.max_attempts):
            try:
                return await self._guarded(_broadcast(), threshold=0.0)
            except WatchdogCancelled:
                logger.perf(
                    f'{label} cancelled (attempt {attempt + 1}/'
                    f'{self.max_attempts}), pool={self.runtime.num_workers}'
                    f' workers'
                )
                await self._wait_if_below_min(label)
        raise RuntimeError(
            f'{label} exhausted {self.max_attempts} retry attempts'
        )

    async def dispatch(self, make_coro, get_completion, on_rollback,
                       label='dispatch'):
        """
        Run ``make_coro()`` under drop protection with partial-accept.
        Cancels mid-flight if drop fraction > ``drop_threshold``; accepts
        the result when ``get_completion() >= 1 - drop_threshold``,
        otherwise rolls back and retries.

        Parameters
        ----------
        make_coro : callable
            Zero-arg dispatch-coroutine factory.
        get_completion : callable
            Returns completion fraction in ``[0, 1]``.
        on_rollback : callable
            Sync or async, run before each retry (bump attempt, clear
            partial uploads).
        label : str
            Log-line tag.

        Returns
        -------
        tuple
            ``('full', None)`` or ``('partial', completion)``.

        Raises
        ------
        RuntimeError
            All ``max_attempts`` attempts fell below threshold.

        """
        if self.drop_threshold is None:
            await make_coro()
            return ('full', None)

        accept_threshold = 1.0 - self.drop_threshold

        for attempt in range(self.max_attempts):
            try:
                await self._guarded(make_coro(),
                                    threshold=self.drop_threshold)
                completion = get_completion()
                if completion >= 1.0:
                    return ('full', None)
            except WatchdogCancelled:
                # clear async_for's re-entrancy guard for new attempt
                self.runtime._inside_async_for = False
                completion = get_completion()
            except Exception as exc:
                self.runtime._inside_async_for = False
                logger.warning(
                    f'{label} attempt {attempt + 1}/{self.max_attempts}'
                    f' failed ({type(exc).__name__}: {exc})'
                )
                completion = get_completion()

            if completion >= accept_threshold:
                logger.perf(
                    f'{label} accepting completion {completion * 100:.0f}%'
                    f' (>= {accept_threshold * 100:.0f}% required)'
                )
                if completion < 1.0:
                    return ('partial', completion)
                return ('full', None)

            logger.perf(
                f'{label} completion {completion * 100:.0f}% < '
                f'{accept_threshold * 100:.0f}% required, rolling back '
                f'(attempt {attempt + 1}/{self.max_attempts})'
            )
            result = on_rollback()
            if asyncio.iscoroutine(result):
                await result
            await self._wait_if_below_min(label)

        raise RuntimeError(
            f'{label} exhausted {self.max_attempts} retries'
        )

    async def _guarded(self, coro, threshold):
        """
        Run *coro* under a one-shot watchdog at *threshold*. Raises
        ``WatchdogCancelled`` on watchdog-initiated cancel; external
        ``CancelledError`` propagates unchanged.

        Parameters
        ----------
        coro : coroutine
        threshold : float
            Drop fraction (lost / initial) at which to cancel.

        Returns
        -------
        Whatever *coro* returns.

        """
        task = asyncio.ensure_future(coro)
        cleanup, fired = self._start_watchdog(task, threshold)
        try:
            try:
                return await task
            except (asyncio.CancelledError, Exception) as exc:
                if fired():
                    raise WatchdogCancelled() from exc
                raise
        finally:
            cleanup()

    def _start_watchdog(self, target_task, threshold):
        """
        Arm a watchdog against *target_task*, baselining the current
        worker UIDs so replacements with new UIDs don't mask a drop.

        Returns
        -------
        (cleanup, fired) : tuple of callables
            ``cleanup()`` cancels the watchdog and deregisters its event-
            bus callback. ``fired()`` returns True if it cancelled the
            target.

        """
        initial_uids = set(w.uid for w in self.runtime.workers)
        logger.debug(
            f'arming — initial_uids={sorted(initial_uids)} '
            f'threshold={threshold:.2f}'
        )

        state = {'fired': False}
        event = asyncio.Event()
        cb = event.set
        self.runtime._on_worker_count_changed.append(cb)
        watchdog_task = asyncio.ensure_future(
            self._cancel_on_drops(initial_uids, threshold,
                                  target_task, event, state)
        )

        def cleanup():
            watchdog_task.cancel()
            try:
                self.runtime._on_worker_count_changed.remove(cb)
            except ValueError:
                pass

        return cleanup, (lambda: state['fired'])

    async def _cancel_on_drops(self, initial_uids, threshold,
                               target_task, event, state):
        """
        Watchdog body. Wakes on ``_on_worker_count_changed``; if drop
        fraction > *threshold* after a 2 s debounce, sets
        ``state['fired']`` and cancels *target_task*.
        """
        n = len(initial_uids)
        logger.debug(
            f'drop-monitor started — initial_uids={sorted(initial_uids)} '
            f'threshold={threshold:.2f}'
        )

        while not target_task.done():
            await event.wait()
            event.clear()

            current_uids = set(w.uid for w in self.runtime.workers)
            lost = initial_uids - current_uids
            fraction = len(lost) / n if n > 0 else 0.0
            logger.debug(
                f'drop-monitor check — lost={sorted(lost)} '
                f'fraction={fraction:.2f} threshold={threshold:.2f}'
            )

            if n > 0 and fraction > threshold:
                await asyncio.sleep(2)
                current_uids = set(w.uid for w in self.runtime.workers)
                lost = initial_uids - current_uids
                fraction = len(lost) / n if n > 0 else 0.0
                logger.warning(
                    f'drop threshold exceeded ({fraction:.2f} > '
                    f'{threshold:.2f}, lost={sorted(lost)}) — cancelling'
                )
                state['fired'] = True
                target_task.cancel()
                return

        logger.debug('drop-monitor: target task already done, exiting')

    async def _wait_if_below_min(self, label):
        """Block until pool >= ``desired_workers`` (or raise below ``min_workers``)."""
        if self.runtime.num_workers >= self.desired_workers:
            return
        logger.perf(
            f'{label} pool below desired ({self.runtime.num_workers} < '
            f'{self.desired_workers}, min {self.min_workers}), '
            f'waiting for replacements'
        )
        await Watchdog.wait_for_workers(
            self.runtime,
            desired_workers=self.desired_workers,
            min_workers=self.min_workers,
        )

    @staticmethod
    async def _wait_for_worker_count(runtime, target, end_time, heartbeat):
        """
        Block until ``num_workers >= target`` or ``end_time`` is reached.
        Event-driven via ``runtime._on_worker_count_changed``; logs a
        heartbeat every ``heartbeat`` seconds. Never raises.
        """
        event = asyncio.Event()
        cb = event.set
        runtime._on_worker_count_changed.append(cb)
        present_uids = set(w.uid for w in runtime.workers)
        try:
            while runtime.num_workers < target:
                remaining = end_time - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning(
                        f'count phase timed out — proceeding with '
                        f'{runtime.num_workers}/{target} workers '
                        f'(present: {sorted(w.uid for w in runtime.workers)})'
                    )
                    return
                try:
                    await asyncio.wait_for(event.wait(),
                                           timeout=min(heartbeat, remaining))
                    event.clear()
                    current_uids = set(w.uid for w in runtime.workers)
                    logger.debug(
                        f'pool changed — {runtime.num_workers}/{target} '
                        f'workers (joined: '
                        f'{sorted(current_uids - present_uids)}, left: '
                        f'{sorted(present_uids - current_uids)})'
                    )
                    present_uids = current_uids
                except asyncio.TimeoutError:
                    logger.debug(
                        f'still waiting — {runtime.num_workers}/{target}'
                        f' workers present'
                    )
        finally:
            try:
                runtime._on_worker_count_changed.remove(cb)
            except ValueError:
                pass

    @staticmethod
    async def _wait_for_nodes_ready(runtime, end_time):
        """
        Poll ``monitor.check_node_status`` until every worker's node
        reports ready or ``end_time`` passes. No-op without a monitor;
        never raises (timeout / RPC failure log and return).
        """
        monitor = runtime.get_monitor()
        if monitor is None:
            return

        while True:
            worker_uids = [w.uid for w in runtime.workers]
            try:
                status = await monitor.check_node_status(
                    worker_uids=worker_uids, reply=True
                )
            except Exception:
                logger.warning(
                    'check_node_status RPC failed, proceeding'
                )
                return
            missing = sorted(nid for nid, ready in status.items() if not ready)
            if not missing:
                logger.debug(f'all {len(status)} node(s) confirmed ready')
                return
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    f'node readiness timed out — still missing: {missing}'
                )
                return
            logger.debug(f'waiting for node(s): {missing}')
            await asyncio.sleep(min(1.0, remaining))
