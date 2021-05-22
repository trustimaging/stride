

__version__ = '1.1'


import mosaic

from .problem import *
from .physics import *
from .optimisation import *


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
    args : optional
        Extra positional arguments for the PDE.
    kwargs : optional
        Extra keyword arguments for the PDE.

    Returns
    -------

    """
    runtime = mosaic.runtime()

    dump = kwargs.pop('dump', True)
    shot_ids = kwargs.pop('shot_ids', None)
    deallocate = kwargs.pop('deallocate', False)

    if dump is True:
        try:
            problem.acquisitions.load(path=problem.output_folder,
                                      project_name=problem.name, version=0)
        except OSError:
            pass

    if shot_ids is None:
        shot_ids = problem.acquisitions.remaining_shot_ids
        if not len(shot_ids):
            runtime.logger.warning('No need to run forward, observed already exists')
            return

    if not isinstance(shot_ids, list):
        shot_ids = [shot_ids]

    @runtime.async_for(shot_ids)
    async def loop(worker, shot_id):
        runtime.logger.info('\n')
        runtime.logger.info('Giving shot %d to %s' % (shot_id, worker.uid))

        sub_problem = problem.sub_problem(shot_id)
        wavelets = sub_problem.shot.wavelets
        traces = await pde(wavelets, *args,
                           problem=sub_problem,
                           runtime=worker, **kwargs).result()

        runtime.logger.info('Shot %d retrieved' % sub_problem.shot_id)

        shot = problem.acquisitions.get(shot_id)
        shot.observed.data[:] = traces.data

        if dump is True:
            shot.append_observed(path=problem.output_folder,
                                 project_name=problem.name)

            runtime.logger.info('Appended traces for shot %d to observed file' % sub_problem.shot_id)

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
    args : optional
        Extra positional arguments for the operators.
    kwargs : optional
        Extra keyword arguments for the operators.

    Returns
    -------

    """
    runtime = mosaic.runtime()

    block = optimisation_loop.current_block
    num_iters = kwargs.pop('num_iters', 1)
    select_shots = kwargs.pop('select_shots', {})

    dump = kwargs.pop('dump', True)

    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)
    process_wavelets = ProcessWavelets.remote(f_min=f_min, f_max=f_max,
                                              len=runtime.num_workers)
    process_traces = ProcessTraces.remote(f_min=f_min, f_max=f_max,
                                          len=runtime.num_workers)

    for iteration in block.iterations(num_iters):
        runtime.logger.info('Starting iteration %d (out of %d), '
                            'block %d (out of %d)' %
                            (iteration.id, block.num_iterations, block.id,
                             optimisation_loop.num_blocks))

        shot_ids = problem.acquisitions.select_shot_ids(**select_shots)

        optimiser.clear_grad()

        @runtime.async_for(shot_ids)
        async def loop(worker, shot_id):
            runtime.logger.info('\n')
            runtime.logger.info('Giving shot %d to %s' % (shot_id, worker.uid))

            sub_problem = problem.sub_problem(shot_id)
            wavelets = sub_problem.shot.wavelets
            observed = sub_problem.shot.observed

            wavelets = process_wavelets(wavelets, runtime=worker, **kwargs)
            modelled = pde(wavelets, *args, problem=sub_problem, runtime=worker, **kwargs)

            traces = process_traces(modelled, observed, runtime=worker, **kwargs)
            fun = await loss(traces.outputs[0], traces.outputs[1],
                             problem=sub_problem, runtime=worker, **kwargs).result()

            iteration.add_fun(fun)
            runtime.logger.info('Functional value for shot %d: %s' % (shot_id, fun))

            await fun.adjoint(**kwargs)

            runtime.logger.info('Retrieved gradient for shot %d' % sub_problem.shot_id)

        await loop

        await optimiser.step()

        if dump:
            optimiser.variable.dump(path=problem.output_folder, project_name=problem.name)

        runtime.logger.info('Done iteration %d (out of %d), '
                            'block %d (out of %d) - Total loss %e' %
                            (iteration.id, block.num_iterations, block.id,
                             optimisation_loop.num_blocks, iteration.fun_value))
        runtime.logger.info('====================================================================')
