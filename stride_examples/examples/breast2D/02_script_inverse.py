
from stride import *


async def main(runtime):
    # Create the grid
    shape = (356, 385)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)

    start = 0.
    step = 0.08e-6
    num = 2500

    time = Time(start=start,
                step=step,
                num=num)

    # Create problem
    problem = Problem(name='anastasio2D',
                      space=space, time=time)

    # Create medium
    vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
    vp.fill(1500.)

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_locations = 128
    problem.geometry.default('elliptical', num_locations)

    # Create acquisitions
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Plot
    problem.plot()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Create loss
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    # Create optimiser
    step_size = 10
    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1400., max=1700.)

    optimiser = GradientDescent(vp, step_size=step_size,
                                process_grad=process_grad,
                                process_model=process_model)

    # Run optimisation
    optimisation_loop = OptimisationLoop()

    max_freqs = [0.3e6, 0.4e6, 0.5e6, 0.6e6]

    num_blocks = len(max_freqs)
    num_iters = 8

    for block, freq in optimisation_loop.blocks(num_blocks, max_freqs):
        await adjoint(problem, pde, loss,
                      optimisation_loop, optimiser, vp,
                      num_iters=num_iters,
                      select_shots=dict(num=16, randomly=True),
                      f_min=0.05e6, f_max=freq)

        vp.plot()

if __name__ == '__main__':
    mosaic.run(main)
