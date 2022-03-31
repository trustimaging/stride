
"""
This example inverts experimental data acquired for a tissue-mimicking
phantom in the laboratory.

The phantom being imaged has a 2D shape, which has been extruded along the
z axis to avoid off-plane effects when imaging it.

The phantom was imaged using two P4-1 ultrasound transducer arrays, attached to
two rotary systems allowing independent rotation of the two arrays. Each array
contains 96 transducer elements.

The geometry, acquisition sequence, and data have been pre-saved in Stride
format for easy access, and can be loaded easily through the problem. The
experimental data has been band-pass filtered to fit the relevant frequency
range.

Data to run the script can be downloaded from <TBC>.

This script can be used to reproduce results presented in https://arxiv.org/abs/2110.03345.

"""


from stride import *


async def main(runtime):
    # Create problem and load it
    problem = Problem(name='BB',
                      input_folder='./data')
    problem.load()

    # Mute early water arrivals
    time = problem.time
    cutoff = int(35e-6 / time.step)
    for shot in problem.acquisitions.shots:
        shot.observed.data[:, :cutoff] = 0.

    # Create medium
    vp = ScalarField.parameter(name='vp', grid=problem.grid, needs_grad=True)
    vp.fill(1481.)

    problem.medium.add(vp)

    # Plot
    problem.plot()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Create loss
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Create optimiser
    step_size = 5.
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400., max=1700.)

    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model)

    # Run optimisation
    logger = runtime.logger
    optimisation_loop = OptimisationLoop(problem=problem)

    f_max = 0.70e6

    num_shots = 10
    num_blocks = 4
    num_iters = 38

    num_gpus = 2
    platform = 'nvidia-acc'  # or 'cpu'

    for block in optimisation_loop.blocks(num_blocks):
        process_wavelets = ProcessWavelets.remote(len=runtime.num_workers)
        process_traces = ProcessTraces.remote(f_max=f_max,
                                              len=runtime.num_workers)
        problem.acquisitions._shot_selection = []

        for iteration in block.iterations(num_iters):
            vp.clear_grad()
            await vp.push(publish=True)

            logger.info('Starting iteration %d (out of %d), '
                        'block %d (out of %d)' %
                        (iteration.id, block.num_iterations, block.id,
                         optimisation_loop.num_blocks))

            shot_ids = problem.acquisitions.select_shot_ids(num=num_shots, randomly=True)

            @runtime.async_for(shot_ids)
            async def loop(worker, shot_id):
                logger.info('\n')
                logger.info('Giving shot %d to %s' % (shot_id, worker.uid))

                sub_problem = problem.sub_problem(shot_id)
                wavelets = sub_problem.shot.wavelets
                observed = sub_problem.shot.observed

                if 'nvidia' in platform:
                    # distribute across different GPUs
                    device_id = worker.indices[1] % num_gpus
                    devito_args = dict(deviceid=device_id)
                else:
                    devito_args = dict()

                wavelets = process_wavelets(wavelets, runtime=worker)
                modelled = pde(wavelets, vp,
                               problem=sub_problem,
                               kernel='OT4',
                               platform=platform,
                               devito_args=devito_args,
                               runtime=worker)

                traces = process_traces(modelled, observed, runtime=worker)

                fun = await loss(traces.outputs[0], traces.outputs[1],
                                 problem=sub_problem, runtime=worker).result()

                iteration.add_fun(fun)
                logger.info('Functional value for shot %d: %s' % (shot_id, fun))

                await fun.adjoint()

                logger.info('Retrieved gradient for shot %d' % sub_problem.shot_id)

            await loop

            await vp.pull()
            await optimiser.step()

            optimiser.variable.dump(path=problem.output_folder,
                                    project_name=problem.name,
                                    version=iteration.abs_id)

            logger.info('Done iteration %d (out of %d), '
                        'block %d (out of %d) - Total loss %e' %
                        (iteration.id, block.num_iterations, block.id,
                         optimisation_loop.num_blocks, iteration.fun_value))
            logger.info('====================================================================')

    vp.plot()


if __name__ == '__main__':
    mosaic.run(main)
