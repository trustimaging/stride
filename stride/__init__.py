

import os
import signal
import pickle
import asyncio
import warnings
import numpy as np
from pytools import prefork
import multiprocess as multiprocessing


# pre-fork before importing anything else
if multiprocessing.get_start_method() == 'fork':
    prefork_fork = prefork._fork_server


    def _fork_server(sock):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prefork_fork(sock)


    prefork._fork_server = _fork_server
    try:
        prefork.enable_prefork()
    except pickle.UnpicklingError:
        pass


    def _close_prefork_atsignal(signum, frame):
        try:
            prefork.forker._quit()
        except (AttributeError, BrokenPipeError, ConnectionResetError, pickle.UnpicklingError):
            pass

        os._exit(-1)


    signal.signal(signal.SIGINT, _close_prefork_atsignal)
    signal.signal(signal.SIGTERM, _close_prefork_atsignal)


import mosaic
from mosaic.utils import gpu_count

from .core import *
from .problem import *
from .physics import *
from .optimisation import *
from .utils.operators import *
from .utils.artifacts import ArtifactConfig


async def forward(problem, pde, *args, **kwargs):
    """
    Use a ``problem`` forward using a given ``pde``. The given ``args`` and ``kwargs``
    will be passed on to the PDE.

    Parameters
    ----------
    problem : Problem
        Problem to run the PDE on.
    pde : Operator
        PDE operator to run for each shot in the problem.
    dump : bool, optional
        Whether or not to wave to disk the result of the forward run, defaults to True.
    deallocate : bool, optional
        Whether or not to deallocate the resulting traces after running forward, defaults to False.
    shot_ids : list, optional
        List of specific shots to run, defaults to all remaining shots.
    safe : bool, optional
        Whether to discard workers that fail during execution.
    args : optional
        Extra positional arguments for the PDE.
    kwargs : optional
        Extra keyword arguments for the PDE.

    Returns
    -------

    """
    logger = mosaic.logger()
    runtime = mosaic.runtime()

    dump = kwargs.pop('dump', True)
    shot_ids = kwargs.pop('shot_ids', None)
    deallocate = kwargs.pop('deallocate', False)
    safe = kwargs.pop('safe', True)

    art_warehouse = mosaic.get_artifact_warehouse()
    if art_warehouse is not None:
        art_warehouse.ensure_bucket()

    # Skip the disk-load/early-return optimisation when an artifact warehouse
    # is configured: each workflow run uses a fresh MinIO prefix (keyed by
    # run ID), so local disk data from a previous run with the same exp_name
    # must not cause the forward pass to be skipped — the MinIO objects would
    # never be written and load_artifacts() would fail with NoSuchKey.
    if dump is True and art_warehouse is None:
        try:
            problem.acquisitions.load(path=problem.output_folder,
                                      project_name=problem.name, version=0)
        except OSError:
            pass

    if shot_ids is None:
        shot_ids = problem.acquisitions.remaining_shot_ids
        if not len(shot_ids):
            logger.warning('No need to run forward, observed already exists')
            return

    if not isinstance(shot_ids, list):
        shot_ids = [shot_ids]
    num_shots = len(shot_ids)
    submitted_shots = []

    published_args = [runtime.put(each, publish=True) for each in args]
    published_args = await asyncio.gather(*published_args)

    platform = kwargs.get('platform', 'cpu')
    using_gpu = 'nvidia' in platform or 'gpu' in platform
    if using_gpu:
        devices = kwargs.pop('devices', None)
        num_gpus = gpu_count() if devices is None else len(devices)
        devices = list(range(num_gpus)) if devices is None else devices

    @runtime.async_for(shot_ids, safe=safe)
    async def loop(worker, shot_id):
        _kwargs = kwargs.copy()

        num_submitted = len(submitted_shots)

        logger.perf('\n')
        logger.perf('Giving shot %d to %s (%d out of %d)'
                    % (shot_id, worker.uid,
                       num_submitted + 1, num_shots))

        sub_problem = problem.sub_problem(shot_id)
        submitted_shots.append(shot_id)
        wavelets = sub_problem.shot.wavelets

        if using_gpu:
            deviceid = devices[worker.indices[1] % num_gpus]
            if platform in ['nvidia-acc', 'nvidia-cuda']:
                devito_args = _kwargs.get('devito_args', {}).copy()
                devito_args['deviceid'] = deviceid
                _kwargs['devito_args'] = devito_args
            elif platform == 'gpu':
                _kwargs['deviceid'] = deviceid
            else:
                raise ValueError('Unknown platform %s' % platform)

        # run PDE
        traces = await pde(wavelets, *published_args,
                           problem=sub_problem,
                           runtime=worker, **_kwargs).result()

        logger.perf('Shot %d retrieved' % sub_problem.shot_id)

        # save data
        shot = problem.acquisitions.get(shot_id)
        try:
            shot.observed.data[:] = traces.data
        except ValueError as err:
            logger.warning('Shot %d data assignment error: %s' % (shot_id, str(err)))

        if np.any(np.isnan(shot.observed.data)) or np.any(np.isinf(shot.observed.data)):
            raise ValueError('Nan or inf detected in shot %d' % shot_id)

        if art_warehouse is not None:
            key_obs = '%s/%d/observed.npy' % (art_warehouse.shot_prefix, shot_id)
            key_wav = '%s/%d/wavelets.npy' % (art_warehouse.shot_prefix, shot_id)
            art_warehouse.push_remote(key_obs, shot.observed.data)
            art_warehouse.push_remote(key_wav, shot.wavelets.data)
            shot.observed = ArtifactTraces(
                name=shot.observed.name,
                transducer_ids=shot.observed.transducer_ids,
                grid=shot.observed.grid,
                artifact_key=key_obs,
            )
            logger.perf('Uploaded observed traces and wavelets for shot %d to artifact store'
                        % shot_id)

        if dump is True:
            shot.append_observed(path=problem.output_folder,
                                 project_name=problem.name)

            logger.perf('Appended traces for shot %d to observed file' % sub_problem.shot_id)

        if deallocate is True:
            shot.observed.deallocate()

    await loop


async def _wait_for_workers(runtime, target, timeout=300.0, heartbeat=30.0):
    """Wait (up to *timeout* seconds) until *runtime* has at least *target* workers.

    Called after a worker-drop retry so that replacement pods have time to
    phone home before the new ``async_for`` worker queue is built.  If
    replacements never arrive within the timeout we proceed with whoever is
    available and let the iteration succeed on a smaller crew.

    Logs progress every *heartbeat* seconds so the operator can see the wait
    is active and which UIDs have joined.
    """
    logger = mosaic.logger()

    present_uids = set(w.uid for w in runtime.workers)
    if runtime.num_workers >= target:
        logger.info('WAIT-FOR-WORKERS: already have %d/%d workers — skipping wait '
                    '(present: %s)' % (runtime.num_workers, target, sorted(present_uids)))
        return

    logger.info('WAIT-FOR-WORKERS: start — need %d workers, have %d (present: %s)'
                % (target, runtime.num_workers, sorted(present_uids)))

    event = asyncio.Event()
    cb = event.set
    runtime._on_worker_count_changed.append(cb)
    tic = asyncio.get_event_loop().time()
    try:
        end_time = tic + timeout
        while runtime.num_workers < target:
            remaining = end_time - asyncio.get_event_loop().time()
            if remaining <= 0:
                elapsed = asyncio.get_event_loop().time() - tic
                logger.warning(
                    'WAIT-FOR-WORKERS: timed out after %.0fs — '
                    'proceeding with %d/%d workers (present: %s)'
                    % (elapsed, runtime.num_workers, target,
                       sorted(w.uid for w in runtime.workers)))
                break
            try:
                await asyncio.wait_for(event.wait(), timeout=min(heartbeat, remaining))
                # Event fired — a worker joined or left.
                event.clear()
                current_uids = set(w.uid for w in runtime.workers)
                elapsed = asyncio.get_event_loop().time() - tic
                logger.info(
                    'WAIT-FOR-WORKERS: pool changed at +%.0fs — '
                    '%d/%d workers (present: %s, joined: %s, left: %s)'
                    % (elapsed, runtime.num_workers, target,
                       sorted(current_uids),
                       sorted(current_uids - present_uids),
                       sorted(present_uids - current_uids)))
                present_uids = current_uids
            except asyncio.TimeoutError:
                # Heartbeat tick — log progress so operators can see we are still waiting.
                elapsed = asyncio.get_event_loop().time() - tic
                logger.info(
                    'WAIT-FOR-WORKERS: still waiting at +%.0fs — '
                    '%d/%d workers present (present: %s)'
                    % (elapsed, runtime.num_workers, target,
                       sorted(w.uid for w in runtime.workers)))
    finally:
        try:
            runtime._on_worker_count_changed.remove(cb)
        except ValueError:
            pass

    # Phase 2: confirm each worker's node has completed init (NODE-CONNECTED).
    monitor = runtime.get_monitor()
    if monitor is not None:
        node_wait_tic = asyncio.get_event_loop().time()
        node_timeout = max(0, tic + timeout - node_wait_tic)
        node_end = node_wait_tic + node_timeout
        while True:
            worker_uids = [w.uid for w in runtime.workers]
            try:
                status = await monitor.check_node_status(
                    worker_uids=worker_uids, reply=True)
            except Exception:
                logger.warning('WAIT-FOR-WORKERS: check_node_status RPC failed, proceeding')
                break
            missing = [nid for nid, ready in status.items() if not ready]
            if not missing:
                logger.info('WAIT-FOR-WORKERS: all %d node(s) confirmed ready'
                            % len(status))
                break
            remaining = node_end - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    'WAIT-FOR-WORKERS: node readiness timed out — '
                    'still missing: %s' % sorted(missing))
                break
            logger.info('WAIT-FOR-WORKERS: waiting for node(s): %s' % sorted(missing))
            await asyncio.sleep(min(1.0, remaining))

    elapsed = asyncio.get_event_loop().time() - tic
    logger.info('WAIT-FOR-WORKERS: done after %.0fs — proceeding with %d workers '
                '(target was %d, present: %s)'
                % (elapsed, runtime.num_workers, target,
                   sorted(w.uid for w in runtime.workers)))


async def _watch_workers(runtime, initial_uids, threshold, task, event):
    """Cancel *task* if the fraction of *dropped* original workers exceeds *threshold*.

    Tracks which specific worker UIDs were alive at the start of the attempt.
    A replacement worker joining with a new UID does not mask a drop.

    Purely event-driven: wakes whenever runtime._on_worker_count_changed fires
    (i.e. any worker joins or leaves via remove_proxy_from_uid).

    Debounces for 2 seconds after a drop to let the full node disconnect
    cascade complete before cancelling.
    """
    logger = mosaic.logger()
    n = len(initial_uids)
    logger.debug('_watch_workers started — initial_uids=%s threshold=%.2f' % (initial_uids, threshold))
    while not task.done():
        await event.wait()
        event.clear()
        current_uids = set(w.uid for w in runtime.workers)
        lost = initial_uids - current_uids
        fraction = len(lost) / n if n > 0 else 0.0
        logger.debug('_watch_workers check — lost=%s fraction=%.2f threshold=%.2f' % (lost, fraction, threshold))
        if n > 0 and fraction > threshold:
            await asyncio.sleep(2)
            current_uids = set(w.uid for w in runtime.workers)
            lost = initial_uids - current_uids
            fraction = len(lost) / n if n > 0 else 0.0
            logger.warning('_watch_workers: drop threshold exceeded (%.2f > %.2f, lost=%s) '
                           '— cancelling' % (fraction, threshold, sorted(lost)))
            task.cancel()
            return
    logger.debug('_watch_workers: loop_task already done, exiting')


def _start_worker_monitor(runtime, drop_threshold, loop_task):
    """Start a worker-drop monitor for *loop_task*.

    Captures the current set of worker UIDs as the baseline so that a
    replacement worker joining with a new UID does not mask a drop.

    Returns (monitor_task, cleanup) where *cleanup()* must be called in a
    finally block to deregister the callback regardless of outcome.
    Returns (None, no-op) when monitoring is disabled.
    """
    if drop_threshold is None:
        return None, lambda: None

    logger = mosaic.logger()
    initial_uids = set(w.uid for w in runtime.workers)
    logger.debug('_start_worker_monitor: initial_uids=%s threshold=%.2f'
                 % (initial_uids, drop_threshold))

    event = asyncio.Event()
    cb = event.set
    runtime._on_worker_count_changed.append(cb)
    monitor_task = asyncio.ensure_future(
        _watch_workers(runtime, initial_uids, drop_threshold, loop_task, event))

    def cleanup():
        monitor_task.cancel()
        try:
            runtime._on_worker_count_changed.remove(cb)
        except ValueError:
            pass

    return monitor_task, cleanup


async def adjoint(problem, pde, loss, optimisation_loop, optimiser, *args, **kwargs):
    """
    Use a ``problem`` in adjoint mode using a given ``pde``. The given ``args`` and ``kwargs``
    will be passed on to the PDE.

    Parameters
    ----------
    problem : Problem
        Problem to run the optimisation on.
    pde : Operator
        PDE operator to run for each shot in the problem.
    loss : Operator
        Loss function operator.
    optimisation_loop : OptimisationLoop
        Optimisation loop.
    optimiser : LocalOptimiser
        Local optimiser associated with the variable for which we are inverting.
    num_iters : int, optional
        Number of iterations to run the inversion for.
    select_shots : dict, optional
        Rules for selecting available shots per iteration, defaults to taking all shots. For
        details on this see :func:`~stride.problem.acquisitions.Acquisitions.select_shot_ids`.
    lazy_loading : bool, optional
        Whether to load shot data every iteration to save memory.
    dump : bool, optional
        Whether or not to save to disk the updated variable after every iteration.
    f_min : float, optional
        Min. frequency to filter wavelets and data with. If not given, no high-pass filter
        is applied.
    f_max : float, optional
        Max. frequency to filter wavelets and data with. If not given, no low-pass filter
        is applied.
    safe : bool, optional
        Whether to discard workers that fail during execution.
    args : optional
        Extra positional arguments for the operators.
    kwargs : optional
        Extra keyword arguments for the operators.

    Returns
    -------

    """
    logger = mosaic.logger()
    runtime = mosaic.runtime()

    block = optimisation_loop.current_block
    num_iters = kwargs.pop('num_iters', 1)
    select_shots = kwargs.pop('select_shots', {})
    drop_threshold = kwargs.pop('drop_threshold', None)

    lazy_loading = kwargs.pop('lazy_loading', False)
    dump = kwargs.pop('dump', True)
    safe = kwargs.pop('safe', True)

    art_warehouse = mosaic.get_artifact_warehouse()

    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)

    filter_traces = kwargs.pop('filter_traces', True)
    filter_wavelets = kwargs.pop('filter_wavelets', filter_traces)

    filter_wavelets_relaxation = kwargs.pop('filter_wavelets_relaxation', 0.75)
    filter_traces_relaxation = kwargs.pop('filter_traces_relaxation',
                                          0.75 if filter_wavelets else 1.00)

    process_wavelets = ProcessWavelets.remote(f_min=f_min, f_max=f_max,
                                              filter_traces=filter_wavelets,
                                              filter_relaxation=filter_wavelets_relaxation,
                                              len=runtime.num_workers, **kwargs)
    process_observed = ProcessObserved.remote(f_min=f_min, f_max=f_max,
                                              filter_traces=filter_wavelets,
                                              filter_relaxation=filter_wavelets_relaxation,
                                              len=runtime.num_workers, **kwargs)
    process_wavelets_observed = ProcessWaveletsObserved.remote(f_min=f_min, f_max=f_max,
                                                               filter_traces=filter_wavelets,
                                                               len=runtime.num_workers, **kwargs)
    process_traces = ProcessTraces.remote(f_min=f_min, f_max=f_max,
                                          filter_traces=filter_traces,
                                          filter_relaxation=filter_traces_relaxation,
                                          len=runtime.num_workers, **kwargs)

    step_size = kwargs.pop('step_size', optimiser.step_size)
    keep_residual = isinstance(step_size, LineSearch)

    platform = kwargs.get('platform', 'cpu')
    using_gpu = 'nvidia' in platform or 'gpu' in platform
    if using_gpu:
        devices = kwargs.pop('devices', None)
        num_gpus = gpu_count() if devices is None else len(devices)
        devices = list(range(num_gpus)) if devices is None else devices

    problem.acquisitions.reset_selection()

    if optimiser.reset_block:
        optimiser.reset()

    desired_worker_count = runtime.num_workers

    for iteration in block.iterations(num_iters):
        optimiser.clear_grad()

        if optimiser.reset_iteration:
            optimiser.reset()

        await _wait_for_workers(runtime, desired_worker_count)

        published_args = [runtime.put(each, publish=False) for each in args]
        published_args = await asyncio.gather(*published_args)

        logger.perf('Starting iteration %d (out of %d), '
                    'block %d (out of %d)' %
                    (iteration.id+1, block.num_iterations, block.id+1,
                     optimisation_loop.num_blocks))

        if dump and optimisation_loop.restart and not optimisation_loop.started:
            if iteration.abs_id > 0:
                # reload the latest version of the optimiser variable
                try:
                    optimiser.load(path=problem.output_folder,
                                   project_name=problem.name,
                                   version=iteration.abs_id)

                    logger.perf('\n')
                    logger.perf('Loaded optimiser variable for restart, version %d' % iteration.abs_id)
                except OSError:
                    raise OSError('Optimisation loop cannot be restarted,'
                                  'variable version or optimiser version %d cannot be found.' %
                                  iteration.abs_id)

                # ensure previously used shots are not repeated
                problem.acquisitions.filter_shot_ids(optimisation_loop, **select_shots)

        shot_ids = problem.acquisitions.select_shot_ids(**select_shots)
        num_shots = len(shot_ids)

        if art_warehouse is not None:
            art_warehouse.set_iteration(iteration.abs_id)

        if lazy_loading:
            problem.acquisitions.load(shot_ids=shot_ids, lazy_loading=False, fast=True)

        attempt = 0
        if art_warehouse is not None:
            art_warehouse.write_shot_list(iteration.abs_id, shot_ids, attempt=attempt)

        initial_worker_count = runtime.num_workers

        while True:
            @runtime.async_for(shot_ids, safe=safe)
            async def loop(worker, shot_id):
                _kwargs = kwargs.copy()
                if art_warehouse is not None:
                    _kwargs['_abs_iteration'] = iteration.abs_id
                    _kwargs['_shot_id'] = shot_id

                logger.perf('\n')
                logger.perf('Giving shot %d to %s (%d out of %d)'
                            % (shot_id, worker.uid,
                               iteration.num_submitted + 1, num_shots))

                sub_problem = problem.sub_problem(shot_id)
                iteration.add_submitted(sub_problem.shot)
                wavelets = sub_problem.shot.wavelets
                observed = sub_problem.shot.observed

                if wavelets is None:
                    raise RuntimeError('Shot %d has no wavelet data' % shot_id)

                if observed is None:
                    raise RuntimeError('Shot %d has no observed data' % shot_id)

                if using_gpu:
                    deviceid = devices[worker.indices[1] % num_gpus]
                    if platform in ['nvidia-acc', 'nvidia-cuda']:
                        devito_args = _kwargs.get('devito_args', {}).copy()
                        devito_args['deviceid'] = deviceid
                        _kwargs['devito_args'] = devito_args
                    elif platform == 'gpu':
                        _kwargs['deviceid'] = deviceid
                    else:
                        raise ValueError('Unknown platform %s' % platform)

                # pre-process wavelets and observed traces
                wavelets = process_wavelets(wavelets,
                                            iteration=iteration, problem=sub_problem,
                                            runtime=worker, **_kwargs)
                observed = process_observed(observed,
                                            iteration=iteration, problem=sub_problem,
                                            runtime=worker, **_kwargs)
                processed = process_wavelets_observed(wavelets, observed,
                                                      iteration=iteration, problem=sub_problem,
                                                      runtime=worker, **_kwargs)
                wavelets = processed.outputs[0]
                observed = processed.outputs[1]

                # run PDE
                modelled = pde(wavelets, *published_args,
                               iteration=iteration, problem=sub_problem,
                               runtime=worker, **_kwargs)

                # post-process modelled and observed traces
                scale_to = sub_problem.shot.observed.copy(compressed=False) \
                    if sub_problem.shot.observed.compressed else sub_problem.shot.observed
                traces = process_traces(modelled, observed,
                                        scale_to=scale_to,
                                        iteration=iteration, problem=sub_problem,
                                        runtime=worker, **_kwargs)
                modelled = traces.outputs[0]
                observed = traces.outputs[1]

                # calculate loss
                fun = loss(modelled, observed,
                           keep_residual=keep_residual,
                           iteration=iteration, problem=sub_problem,
                           runtime=worker, **_kwargs)

                # run adjoint
                fun_value = await fun.remote.adjoint(**_kwargs).result()

                iteration.add_loss(fun_value)
                logger.perf('Functional value for shot %d: %s' % (shot_id, fun_value))

                iteration.add_completed(sub_problem.shot)
                logger.perf('Retrieved gradient for shot %d (%d out of %d)'
                            % (sub_problem.shot_id,
                               iteration.num_completed, num_shots))

            async def _run_loop():
                await loop

            loop_task = asyncio.ensure_future(_run_loop())
            monitor_task, cleanup_monitor = _start_worker_monitor(
                runtime,
                drop_threshold if art_warehouse is not None else None,
                loop_task)

            loop_ok = False
            try:
                await loop_task
                logger.info('FAULT-TOLERANCE: loop_task completed normally')
                loop_ok = True
            except asyncio.CancelledError:
                runtime._inside_async_for = False
                logger.info('FAULT-TOLERANCE: loop_task cancelled')
            except Exception as exc:
                runtime._inside_async_for = False
                logger.warning('FAULT-TOLERANCE: loop_task failed (%s: %s)' % (type(exc).__name__, exc))
            finally:
                cleanup_monitor()

            if loop_ok and iteration.num_completed >= num_shots:
                break

            drop_fraction = 1.0 - (iteration.num_completed / num_shots) if num_shots > 0 else 0.0

            if loop_ok and drop_threshold is not None and drop_fraction <= drop_threshold:
                logger.info('FAULT-TOLERANCE: %d/%d shots succeeded (%.0f%% loss within %.0f%% threshold), '
                            'continuing with partial result'
                            % (iteration.num_completed, num_shots,
                               drop_fraction * 100, drop_threshold * 100))
                if art_warehouse is not None:
                    completed_ids = list(iteration.curr_run.completed_shots)
                    art_warehouse.write_shot_list(iteration.abs_id, completed_ids, attempt=attempt)
                break

            if loop_ok:
                logger.info('FAULT-TOLERANCE: loop completed but only %d/%d shots succeeded'
                            % (iteration.num_completed, num_shots))

            if drop_threshold is not None and art_warehouse is not None:
                attempt += 1
                logger.info(
                    'ADJOINT-RETRY: iteration %d attempt %d — restarting. '
                    'Workers: %d/%d %s'
                    % (iteration.abs_id, attempt,
                       runtime.num_workers, initial_worker_count,
                       sorted(w.uid for w in runtime.workers)))

                reset_rpcs = []
                for worker in runtime.workers:
                    try:
                        reset_rpcs.append(
                            worker.cancel_and_reset_tasks(reply=True))
                    except Exception:
                        pass
                if reset_rpcs:
                    await asyncio.gather(*reset_rpcs, return_exceptions=True)

                iteration.clear()
                optimiser.clear_grad()
                art_warehouse.clear_iteration_gradients(iteration.abs_id)
                art_warehouse.write_shot_list(iteration.abs_id, shot_ids, attempt=attempt)
                if runtime.num_workers < 1:
                    await _wait_for_workers(runtime, 1)
                continue

            break

        async def step_loop():
            iteration.next_run()

            published_args = [runtime.put(each, publish=True) for each in args]
            published_args = await asyncio.gather(*published_args)

            @runtime.async_for(shot_ids, safe=safe)
            async def loop(worker, shot_id):
                _kwargs = kwargs.copy()

                logger.perf('\n')
                logger.perf('Giving shot %d to %s (%d out of %d)'
                            % (shot_id, worker.uid,
                               iteration.num_submitted + 1, num_shots))

                sub_problem = problem.sub_problem(shot_id)
                iteration.add_submitted(sub_problem.shot)
                wavelets = sub_problem.shot.wavelets
                observed = sub_problem.shot.observed

                if wavelets is None:
                    raise RuntimeError('Shot %d has no wavelet data' % shot_id)

                if observed is None:
                    raise RuntimeError('Shot %d has no observed data' % shot_id)

                if using_gpu:
                    deviceid = devices[worker.indices[1] % num_gpus]
                    if platform in ['nvidia-acc', 'nvidia-cuda']:
                        devito_args = _kwargs.get('devito_args', {}).copy()
                        devito_args['deviceid'] = deviceid
                        _kwargs['devito_args'] = devito_args
                    elif platform == 'gpu':
                        _kwargs['deviceid'] = deviceid
                    else:
                        raise ValueError('Unknown platform %s' % platform)

                # pre-process wavelets and observed traces
                wavelets = process_wavelets(wavelets,
                                            iteration=iteration, problem=sub_problem,
                                            runtime=worker, **_kwargs)
                observed = process_observed(observed,
                                            iteration=iteration, problem=sub_problem,
                                            runtime=worker, **_kwargs)
                processed = process_wavelets_observed(wavelets, observed,
                                                      iteration=iteration, problem=sub_problem,
                                                      runtime=worker, **_kwargs)
                wavelets = processed.outputs[0]
                observed = processed.outputs[1]

                # run PDE
                modelled = pde(wavelets, *published_args,
                               iteration=iteration, problem=sub_problem,
                               runtime=worker, **_kwargs)

                # post-process modelled and observed traces
                scale_to = sub_problem.shot.observed.copy(compressed=False) \
                    if sub_problem.shot.observed.compressed else sub_problem.shot.observed
                traces = process_traces(modelled, observed,
                                        scale_to=scale_to,
                                        iteration=iteration, problem=sub_problem,
                                        runtime=worker, **_kwargs)
                modelled = traces.outputs[0]
                observed = traces.outputs[1]

                # calculate loss
                fun = await loss(modelled, observed,
                                 keep_residual=keep_residual,
                                 iteration=iteration, problem=sub_problem,
                                 runtime=worker, **_kwargs).result()

                # clear up
                # await pde.deallocate_wavefield(deallocate=True, runtime=worker, **_kwargs)
                fun.clear_graph()

                iteration.add_loss(fun)
                logger.perf('Functional value for shot %d: %s' % (shot_id, fun))

                iteration.add_completed(sub_problem.shot)
                logger.perf('Retrieved test step for shot %d (%d out of %d)'
                            % (sub_problem.shot_id,
                               iteration.num_completed, num_shots))

            await loop

        await optimiser.step(iteration=iteration, problem=problem,
                             grad=None,
                             f_min=f_min, f_max=f_max,
                             filter_relaxation=min(filter_wavelets_relaxation, filter_traces_relaxation),
                             step_loop=step_loop,
                             **kwargs)

        if dump:
            optimiser.dump(path=problem.output_folder,
                           project_name=problem.name,
                           version=iteration.abs_id+1)

        if iteration.prev_run is not None:
            prev_loss = iteration.prev_run.total_loss
            prev_loss = ' [previous loss %e]' % prev_loss
        else:
            prev_loss = ''

        logger.perf('Done iteration %d (out of %d), '
                    'block %d (out of %d) - Total loss %e%s' %
                    (iteration.id+1, block.num_iterations, block.id+1,
                     optimisation_loop.num_blocks, iteration.total_loss, prev_loss))
        logger.perf('====================================================================')

        if lazy_loading:
            problem.acquisitions.deallocate(shot_ids=shot_ids)

        iteration.clear_run()
