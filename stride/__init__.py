

import os
import signal
import asyncio
import warnings
from pytools import prefork


# pre-fork before importing anything else
prefork_fork = prefork._fork_server


def _fork_server(sock):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prefork_fork(sock)


prefork._fork_server = _fork_server
prefork.enable_prefork()


def _close_prefork_atsignal(signum, frame):
    try:
        prefork.forker._quit()
    except (AttributeError, BrokenPipeError):
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
    safe = kwargs.pop('safe', False)

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

    published_args = [runtime.put(each, publish=True) for each in args]
    published_args = await asyncio.gather(*published_args)

    using_gpu = kwargs.get('platform', 'cpu') == 'nvidia-acc'
    if using_gpu:
        devices = kwargs.pop('devices', None)
        num_gpus = gpu_count() if devices is None else len(devices)
        devices = list(range(num_gpus)) if devices is None else devices

    @runtime.async_for(shot_ids, safe=safe)
    async def loop(worker, shot_id):
        logger.info('\n')
        logger.info('Giving shot %d to %s' % (shot_id, worker.uid))

        sub_problem = problem.sub_problem(shot_id)
        wavelets = sub_problem.shot.wavelets

        if using_gpu:
            devito_args = kwargs.get('devito_args', {})
            devito_args['deviceid'] = devices[worker.indices[1] % num_gpus]
            kwargs['devito_args'] = devito_args

        traces = await pde(wavelets, *published_args,
                           problem=sub_problem,
                           runtime=worker, **kwargs).result()

        logger.info('Shot %d retrieved' % sub_problem.shot_id)

        shot = problem.acquisitions.get(shot_id)
        shot.observed.data[:] = traces.data

        if dump is True:
            shot.append_observed(path=problem.output_folder,
                                 project_name=problem.name)

            logger.info('Appended traces for shot %d to observed file' % sub_problem.shot_id)

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
    safe = kwargs.pop('safe', False)

    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)
    process_wavelets = ProcessWavelets.remote(f_min=f_min, f_max=f_max,
                                              len=runtime.num_workers)
    process_traces = ProcessTraces.remote(f_min=f_min, f_max=f_max,
                                          len=runtime.num_workers)

    using_gpu = kwargs.get('platform', 'cpu') == 'nvidia-acc'
    if using_gpu:
        devices = kwargs.pop('devices', None)
        num_gpus = gpu_count() if devices is None else len(devices)
        devices = list(range(num_gpus)) if devices is None else devices

    for iteration in block.iterations(num_iters, restart=restart, restart_id=restart_id):
        optimiser.clear_grad()

        published_args = [runtime.put(each, publish=True) for each in args]
        published_args = await asyncio.gather(*published_args)

        logger.info('Starting iteration %d (out of %d), '
                    'block %d (out of %d)' %
                    (iteration.id, block.num_iterations, block.id,
                     optimisation_loop.num_blocks))

        if dump and block.restart and not optimisation_loop.started:
            if iteration.abs_id-1 >= 0:
                try:
                    optimiser.variable.load(path=problem.output_folder,
                                            project_name=problem.name,
                                            version=iteration.abs_id-1)
                except OSError:
                    raise OSError('Optimisation loop cannot be restarted,'
                                  'variable version %d cannot be found.' %
                                  iteration.abs_id-1)

        shot_ids = problem.acquisitions.select_shot_ids(**select_shots)

        @runtime.async_for(shot_ids, safe=safe)
        async def loop(worker, shot_id):
            logger.info('\n')
            logger.info('Giving shot %d to %s' % (shot_id, worker.uid))

            sub_problem = problem.sub_problem(shot_id)
            wavelets = sub_problem.shot.wavelets
            observed = sub_problem.shot.observed

            if wavelets is None:
                raise RuntimeError('Shot %d has no wavelet data' % shot_id)

            if observed is None:
                raise RuntimeError('Shot %d has no observed data' % shot_id)

            if using_gpu:
                devito_args = kwargs.get('devito_args', {})
                devito_args['deviceid'] = devices[worker.indices[1] % num_gpus]
                kwargs['devito_args'] = devito_args

            wavelets = process_wavelets(wavelets, runtime=worker, **kwargs)
            await wavelets.init_future
            modelled = pde(wavelets, *published_args, problem=sub_problem, runtime=worker, **kwargs)
            await modelled.init_future

            traces = process_traces(modelled, observed, runtime=worker, **kwargs)
            await traces.init_future
            fun = await loss(traces.outputs[0], traces.outputs[1],
                             problem=sub_problem, runtime=worker, **kwargs).result()

            iteration.add_fun(fun)
            logger.info('Functional value for shot %d: %s' % (shot_id, fun))

            await fun.adjoint(**kwargs)

            logger.info('Retrieved gradient for shot %d' % sub_problem.shot_id)

        await loop

        await optimiser.step()

        if dump:
            optimiser.variable.dump(path=problem.output_folder,
                                    project_name=problem.name,
                                    version=iteration.abs_id)

        logger.info('Done iteration %d (out of %d), '
                    'block %d (out of %d) - Total loss %e' %
                    (iteration.id, block.num_iterations, block.id,
                     optimisation_loop.num_blocks, iteration.fun_value))
        logger.info('====================================================================')
