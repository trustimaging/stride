

import os
import signal
import pickle
import asyncio
import warnings
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

    if dump is True:
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
    using_gpu = platform in ['nvidia-acc', 'gpu']
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
                       num_submitted, num_shots))

        sub_problem = problem.sub_problem(shot_id)
        submitted_shots.append(shot_id)
        wavelets = sub_problem.shot.wavelets

        if using_gpu:
            deviceid = devices[worker.indices[1] % num_gpus]
            if platform == 'nvidia-acc':
                devito_args = _kwargs.get('devito_args', {})
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
        shot.observed.data[:] = traces.data
        if np.any(np.isnan(shot.observed.data)) or np.any(np.isinf(shot.observed.data)):
            raise ValueError('Nan or inf detected in shot %d' % shot_id)

        if dump is True:
            shot.append_observed(path=problem.output_folder,
                                 project_name=problem.name)

            logger.perf('Appended traces for shot %d to observed file' % sub_problem.shot_id)

        if deallocate is True:
            shot.observed.deallocate()

    await loop


async def adjoint(problem, pde, loss, optimisation_loop, optimiser, *args, **kwargs):
    """
    Use a ``problem`` forward using a given ``pde``. The given ``args`` and ``kwargs``
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

    restart = kwargs.pop('restart', None)
    restart_id = kwargs.pop('restart_id', -1)

    dump = kwargs.pop('dump', True)
    safe = kwargs.pop('safe', True)

    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)

    filter_traces = kwargs.pop('filter_traces', True)
    filter_wavelets = kwargs.pop('filter_wavelets', filter_traces)

    fw3d_mode = kwargs.get('fw3d_mode', False)
    filter_wavelets_relaxation = kwargs.pop('filter_wavelets_relaxation',
                                            0.75 if not fw3d_mode else 0.725)
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
    using_gpu = platform in ['nvidia-acc', 'gpu']
    if using_gpu:
        devices = kwargs.pop('devices', None)
        num_gpus = gpu_count() if devices is None else len(devices)
        devices = list(range(num_gpus)) if devices is None else devices

    problem.acquisitions.reset_selection()

    if optimiser.reset_block:
        optimiser.reset()

    for iteration in block.iterations(num_iters, restart=restart, restart_id=restart_id):
        optimiser.clear_grad()

        if optimiser.reset_iteration:
            optimiser.reset()

        published_args = [runtime.put(each, publish=True) for each in args]
        published_args = await asyncio.gather(*published_args)

        logger.perf('Starting iteration %d (out of %d), '
                    'block %d (out of %d)' %
                    (iteration.id, block.num_iterations, block.id,
                     optimisation_loop.num_blocks))

        if dump and block.restart and not optimisation_loop.started:
            if iteration.abs_id > 0:
                try:
                    optimiser.load(path=problem.output_folder,
                                   project_name=problem.name,
                                   version=iteration.abs_id)
                except OSError:
                    raise OSError('Optimisation loop cannot be restarted,'
                                  'variable version or optimiser version %d cannot be found.' %
                                  iteration.abs_id)

        shot_ids = problem.acquisitions.select_shot_ids(**select_shots)
        num_shots = len(shot_ids)

        @runtime.async_for(shot_ids, safe=safe)
        async def loop(worker, shot_id):
            _kwargs = kwargs.copy()

            logger.perf('\n')
            logger.perf('Giving shot %d to %s (%d out of %d)'
                        % (shot_id, worker.uid,
                           iteration.num_submitted, num_shots))

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
                if platform == 'nvidia-acc':
                    devito_args = _kwargs.get('devito_args', {})
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
            await wavelets.init_future
            observed = process_observed(observed,
                                        iteration=iteration, problem=sub_problem,
                                        runtime=worker, **_kwargs)
            await observed.init_future
            processed = process_wavelets_observed(wavelets, observed,
                                                  iteration=iteration, problem=sub_problem,
                                                  runtime=worker, **_kwargs)
            await processed.init_future
            wavelets = processed.outputs[0]
            observed = processed.outputs[1]

            # run PDE
            modelled = pde(wavelets, *published_args,
                           iteration=iteration, problem=sub_problem,
                           runtime=worker, **_kwargs)
            await modelled.init_future

            # post-process modelled and observed traces
            traces = process_traces(modelled, observed,
                                    scale_to=sub_problem.shot.observed,
                                    iteration=iteration, problem=sub_problem,
                                    runtime=worker, **_kwargs)
            await traces.init_future
            modelled = traces.outputs[0]
            observed = traces.outputs[1]

            # calculate loss
            fun = await loss(modelled, observed,
                             keep_residual=keep_residual,
                             iteration=iteration, problem=sub_problem,
                             runtime=worker, **_kwargs).result()

            iteration.add_loss(fun)
            logger.perf('Functional value for shot %d: %s' % (shot_id, fun))

            # run adjoint
            await fun.adjoint(**_kwargs)
            iteration.add_completed(sub_problem.shot)

            logger.perf('Retrieved gradient for shot %d (%d out of %d)'
                        % (sub_problem.shot_id,
                           iteration.num_completed, num_shots))

        await loop

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
                               iteration.num_submitted, num_shots))

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
                    if platform == 'nvidia-acc':
                        devito_args = _kwargs.get('devito_args', {})
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
                await wavelets.init_future
                observed = process_observed(observed,
                                            iteration=iteration, problem=sub_problem,
                                            runtime=worker, **_kwargs)
                await observed.init_future
                processed = process_wavelets_observed(wavelets, observed,
                                                      iteration=iteration, problem=sub_problem,
                                                      runtime=worker, **_kwargs)
                await processed.init_future
                wavelets = processed.outputs[0]
                observed = processed.outputs[1]

                # run PDE
                modelled = pde(wavelets, *published_args,
                               iteration=iteration, problem=sub_problem,
                               runtime=worker, **_kwargs)
                await modelled.init_future

                # post-process modelled and observed traces
                traces = process_traces(modelled, observed,
                                        scale_to=sub_problem.shot.observed,
                                        iteration=iteration, problem=sub_problem,
                                        runtime=worker, **_kwargs)
                await traces.init_future
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
                iteration.add_completed(sub_problem.shot)
                logger.perf('Functional value for shot %d: %s' % (shot_id, fun))

                logger.perf('Retrieved test step for shot %d (%d out of %d)'
                            % (sub_problem.shot_id,
                               iteration.num_completed, num_shots))

            await loop

        await optimiser.step(iteration=iteration, problem=problem,
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
                    (iteration.id, block.num_iterations, block.id,
                     optimisation_loop.num_blocks, iteration.total_loss, prev_loss))
        logger.perf('====================================================================')

        iteration.clear_run()
