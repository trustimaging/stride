
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

    Three knobs:

    - ``desired_workers`` — ideal pool size. Rollback waits up to
      ``timeout`` for this, then falls back to ``min_workers``.
    - ``min_workers`` — pool floor below which we cannot retry. Raises
      ``RuntimeError`` if not met after timeout.
    - ``drop_threshold`` — fraction of gradients we'll accept losing.
      Used by ``dispatch`` to decide both when to cancel mid-flight and
      whether to accept the partial result. ``run`` ignores this value
      (broadcasts cannot tolerate partial fan-out — they cancel on any
      drop).

    ``drop_threshold=None`` disables protection entirely — every call
    short-circuits to a plain await (local/HPC mode, unchanged behaviour).

    Parameters
    ----------
    runtime : Runtime
        The mosaic runtime whose worker pool to monitor.
    drop_threshold : float or None
        Fraction of gradients we'll accept losing during dispatch. Set
        ``None`` to disable protection.
    min_workers : int, optional
        Absolute floor below which we wait for replacements before
        retrying. Must be >= 1 (cannot run on empty pool). Defaults to 1.
    desired_workers : int, optional
        Ideal pool size for rollback re-waits. Defaults to
        ``min_workers`` (strict — no over-waiting).
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
                'desired_workers (%d) must be >= min_workers (%d)'
                % (desired_workers, min_workers)
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
        Wait for ``runtime`` to have ``desired_workers`` ready, blocking
        up to ``timeout`` seconds for them to appear.

        Two phases run in sequence:

        1. Count — event-driven wait until ``num_workers >= desired_workers``.
        2. Readiness — poll ``monitor.check_node_status`` until every
           worker's node reports ready.

        On timeout:

        - If ``num_workers >= min_workers``, log a warning and proceed
          with what's available.
        - If below ``min_workers``, raise ``RuntimeError``.

        Heartbeat progress is logged every ``heartbeat`` seconds.

        Static — callable without a Watchdog instance
        (``Watchdog.wait_for_workers(runtime, desired_workers=N)``) so
        setup scripts don't need to construct a Watchdog just to wait.

        Parameters
        ----------
        runtime : Runtime
            Mosaic runtime whose worker pool to inspect.
        desired_workers : int
            Ideal worker count. Return as soon as this is reached.
        min_workers : int, optional
            Hard floor. If below this after timeout, raise ``RuntimeError``.
            Defaults to ``desired_workers`` (strict).
        timeout : float, optional
            Maximum seconds to wait across both phases. Defaults to 300.
        heartbeat : float, optional
            Interval in seconds between progress log lines. Defaults to 30.

        Raises
        ------
        RuntimeError
            Pool remained below ``min_workers`` after ``timeout``.

        """
        if min_workers is None:
            min_workers = desired_workers

        present_uids = set(w.uid for w in runtime.workers)
        if runtime.num_workers >= desired_workers:
            logger.debug(
                'already have %d/%d workers — skipping wait (present: %s)'
                % (runtime.num_workers, desired_workers, sorted(present_uids))
            )
            return

        logger.debug(
            'waiting for workers — desired %d min %d, have %d (present: %s)'
            % (desired_workers, min_workers, runtime.num_workers,
               sorted(present_uids))
        )

        tic = asyncio.get_event_loop().time()
        end_time = tic + timeout

        await Watchdog._wait_for_worker_count(runtime, desired_workers,
                                              end_time, heartbeat)
        await Watchdog._wait_for_nodes_ready(runtime, end_time)

        elapsed = asyncio.get_event_loop().time() - tic

        if runtime.num_workers < min_workers:
            raise RuntimeError(
                'pool has %d workers after %.0fs, need at least %d '
                '(desired %d)'
                % (runtime.num_workers, elapsed, min_workers, desired_workers)
            )

        if runtime.num_workers < desired_workers:
            logger.warning(
                'proceeding with %d/%d workers after %.0fs (min %d)'
                % (runtime.num_workers, desired_workers, elapsed, min_workers)
            )
        else:
            logger.debug(
                'wait done after %.0fs — %d workers (desired %d)'
                % (elapsed, runtime.num_workers, desired_workers)
            )

    async def broadcast(self, args, label='broadcast'):
        """
        Broadcast positional args to all workers via
        ``runtime.put(publish=True)``, protected against worker drops.

        Publishes each arg to every worker and gathers acks. On any
        worker drop mid-broadcast the watchdog cancels the gather and
        retries on the surviving pool (waiting first if pool <
        ``min_workers``). Raises ``RuntimeError`` after ``max_attempts``.

        Parameters
        ----------
        args : tuple or list
            Positional args to broadcast. Each element becomes one
            published TesseraProxy.
        label : str, optional
            Short tag for log lines. Defaults to ``'broadcast'``.

        Returns
        -------
        list
            Published TesseraProxies, one per arg.

        Raises
        ------
        RuntimeError
            All ``max_attempts`` attempts were cancelled by the watchdog.

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
                    '%s cancelled (attempt %d/%d), pool=%d workers'
                    % (label, attempt + 1, self.max_attempts,
                       self.runtime.num_workers)
                )
                await self._wait_if_below_min(label)
        raise RuntimeError(
            '%s exhausted %d retry attempts' % (label, self.max_attempts)
        )

    async def dispatch(self, make_coro, get_completion, on_rollback,
                       label='dispatch'):
        """
        Run ``make_coro()`` for gradient dispatch with partial-accept.

        Watchdog cancels mid-flight if drop fraction exceeds
        ``drop_threshold``. Whether cancelled or terminated naturally,
        ``get_completion()`` is checked: completion >=
        ``1 - drop_threshold`` → accept the result. Otherwise rollback
        and retry on the surviving pool (waiting first if below
        ``min_workers``).

        Parameters
        ----------
        make_coro : callable
            Zero-arg dispatch-coroutine factory.
        get_completion : callable
            Zero-arg callable returning completion fraction in
            ``[0.0, 1.0]``.
        on_rollback : callable
            Sync or async callable run before retrying to invalidate
            partial state (bump tasks.json attempt, clear stale uploads).
        label : str
            Short tag for log lines.

        Returns
        -------
        tuple
            ``('full', None)`` if every unit completed.
            ``('partial', completion)`` if accepted partial.

        Raises
        ------
        RuntimeError
            All ``max_attempts`` attempts produced sub-threshold completion.

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
                # `runtime.async_for` sets a re-entrancy guard flag when
                # dispatch begins and clears it on natural completion.
                # A watchdog-cancel skips the natural clear, so the next
                # attempt's async_for would trip the guard's assertion.
                self.runtime._inside_async_for = False
                completion = get_completion()
            except Exception as exc:
                # Any other failure inside the dispatch (e.g. a stale-state race
                # between a cancelled attempt and this retry) is treated as
                # retry-worthy — log it, clear the async_for guard, and fall
                # through to the accept-or-retry decision.
                self.runtime._inside_async_for = False
                logger.warning(
                    '%s attempt %d/%d failed (%s: %s)'
                    % (label, attempt + 1, self.max_attempts,
                       type(exc).__name__, exc)
                )
                completion = get_completion()

            if completion >= accept_threshold:
                logger.perf(
                    '%s accepting completion %.0f%% (>= %.0f%% required)'
                    % (label, completion * 100, accept_threshold * 100)
                )
                if completion < 1.0:
                    return ('partial', completion)
                return ('full', None)

            logger.perf(
                '%s completion %.0f%% < %.0f%% required, '
                'rolling back (attempt %d/%d)'
                % (label, completion * 100, accept_threshold * 100,
                   attempt + 1, self.max_attempts)
            )
            result = on_rollback()
            if asyncio.iscoroutine(result):
                await result
            await self._wait_if_below_min(label)

        raise RuntimeError(
            '%s exhausted %d retries' % (label, self.max_attempts)
        )

    async def _guarded(self, coro, threshold):
        """
        Run *coro* under a one-shot watchdog at the given *threshold*.

        Raises ``WatchdogCancelled`` if the watchdog cancels the
        coroutine. External ``CancelledError`` propagates unchanged.

        Parameters
        ----------
        coro : coroutine
            Single-use coroutine to run under protection.
        threshold : float
            Drop fraction (lost / initial) above which the watchdog
            cancels *coro*.

        Returns
        -------
        object
            Whatever *coro* returns on completion.

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
        Start a worker-drop watchdog on *target_task*.

        Captures the current set of worker UIDs as the baseline so that
        a replacement worker joining with a new UID doesn't mask a drop.

        Parameters
        ----------
        target_task : asyncio.Task
            The protected task that the watchdog may cancel.
        threshold : float
            Drop fraction above which the watchdog cancels *target_task*.

        Returns
        -------
        tuple
            ``(cleanup, fired)`` — ``cleanup`` is a no-arg callable that
            cancels the watchdog and deregisters its event-bus
            subscription; ``fired`` is a no-arg callable returning True
            if the watchdog cancelled *target_task*.

        """
        initial_uids = set(w.uid for w in self.runtime.workers)
        logger.debug(
            'arming — initial_uids=%s threshold=%.2f'
            % (sorted(initial_uids), threshold)
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
        Watchdog body. Cancels *target_task* if drop fraction > threshold.

        Wakes whenever ``runtime._on_worker_count_changed`` fires.
        Debounces for 2 seconds after a threshold exceedance so the full
        disconnect cascade can complete before we make the cancel
        decision.

        Parameters
        ----------
        initial_uids : set of str
            Worker UIDs alive when *target_task* started.
        threshold : float
            Drop fraction (lost / initial) above which *target_task* is
            cancelled.
        target_task : asyncio.Task
            The protected task to cancel.
        event : asyncio.Event
            Wake-up event fired by the worker-count event bus.
        state : dict
            Shared state dict; ``state['fired']`` is flipped True just
            before cancellation so callers can distinguish watchdog-cancel
            from external cancel.

        Returns
        -------

        """
        n = len(initial_uids)
        logger.debug(
            'drop-monitor started — initial_uids=%s threshold=%.2f'
            % (sorted(initial_uids), threshold)
        )

        while not target_task.done():
            await event.wait()
            event.clear()

            current_uids = set(w.uid for w in self.runtime.workers)
            lost = initial_uids - current_uids
            fraction = len(lost) / n if n > 0 else 0.0
            logger.debug(
                'drop-monitor check — lost=%s fraction=%.2f threshold=%.2f'
                % (sorted(lost), fraction, threshold)
            )

            if n > 0 and fraction > threshold:
                await asyncio.sleep(2)
                current_uids = set(w.uid for w in self.runtime.workers)
                lost = initial_uids - current_uids
                fraction = len(lost) / n if n > 0 else 0.0
                logger.warning(
                    'drop threshold exceeded (%.2f > %.2f, lost=%s) — cancelling'
                    % (fraction, threshold, sorted(lost))
                )
                state['fired'] = True
                target_task.cancel()
                return

        logger.debug('drop-monitor: target task already done, exiting')

    async def _wait_if_below_min(self, label):
        """
        Block until pool >= ``min_workers`` (no-op if already at or above).

        Parameters
        ----------
        label : str
            Short tag for log lines.

        Returns
        -------

        """
        if self.runtime.num_workers >= self.desired_workers:
            return
        logger.perf(
            '%s pool below desired (%d < %d, min %d), waiting for replacements'
            % (label, self.runtime.num_workers, self.desired_workers,
               self.min_workers)
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

        Event-driven via ``runtime._on_worker_count_changed``. Logs a
        heartbeat every ``heartbeat`` seconds while waiting. Always
        returns; never raises (a timeout logs a warning and proceeds).

        Parameters
        ----------
        runtime : Runtime
            Mosaic runtime whose worker pool to inspect.
        target : int
            Worker-count threshold to wait for.
        end_time : float
            Absolute event-loop timestamp past which we stop waiting.
        heartbeat : float
            Interval in seconds between progress log lines.

        Returns
        -------

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
                        'count phase timed out — proceeding with %d/%d workers '
                        '(present: %s)'
                        % (runtime.num_workers, target,
                           sorted(w.uid for w in runtime.workers))
                    )
                    return
                try:
                    await asyncio.wait_for(event.wait(),
                                           timeout=min(heartbeat, remaining))
                    event.clear()
                    current_uids = set(w.uid for w in runtime.workers)
                    logger.debug(
                        'pool changed — %d/%d workers (joined: %s, left: %s)'
                        % (runtime.num_workers, target,
                           sorted(current_uids - present_uids),
                           sorted(present_uids - current_uids))
                    )
                    present_uids = current_uids
                except asyncio.TimeoutError:
                    logger.debug(
                        'still waiting — %d/%d workers present'
                        % (runtime.num_workers, target)
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
        reports ready or ``end_time`` is reached.

        No-op when no monitor is configured. Always returns; never
        raises (a timeout, RPC failure, or missing monitor all log and
        proceed).

        Parameters
        ----------
        runtime : Runtime
            Mosaic runtime whose worker pool to inspect.
        end_time : float
            Absolute event-loop timestamp past which we stop waiting.

        Returns
        -------

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
                logger.debug(
                    'all %d node(s) confirmed ready' % len(status)
                )
                return
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    'node readiness timed out — still missing: %s' % missing
                )
                return
            logger.debug('waiting for node(s): %s' % missing)
            await asyncio.sleep(min(1.0, remaining))
