
import numpy as np

from stride.problem import *
from stride import *


async def main(runtime):
    # Create the grid
    shape = (356, 385, 160)
    extra = (50, 50, 50)
    absorbing = (40, 40, 40)
    spacing = (0.5e-3, 0.5e-3, 0.5e-3)

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
    problem = Problem(name='anastasio3D',
                      space=space, time=time)

    # Create medium
    vp = ScalarField.parameter(name='vp',
                               grid=problem.grid, needs_grad=True)
    vp.fill(1500.)

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    radius = ((space.limit[0] - 30e-3) / 2,
              (space.limit[1] - 05e-3) / 2,
              (space.limit[2] - 05e-3))
    centre = (space.limit[0] / 2,
              space.limit[1] / 2,
              space.limit[2])

    num_locations = 1024

    problem.geometry.default('ellipsoidal', num_locations, radius, centre,
                             theta=np.pi, threshold=0.5)

    # Create acquisitions
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Plot
    problem.plot()

    # Create the PDE
    pde = physics.IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Create loss
    loss = optimisation.L2DistanceLoss.remote(len=runtime.num_workers)

    # Create optimiser
    step_size = 10
    process_grad = optimisation.ProcessGlobalGradient()
    process_model = optimisation.ProcessModelIteration(min=1400., max=1700.)

    optimiser = optimisation.GradientDescent(vp, step_size=step_size,
                                             process_grad=process_grad,
                                             process_model=process_model)

    # Run optimisation
    optimisation_loop = optimisation.OptimisationLoop()

    max_freqs = [0.3e6, 0.4e6, 0.5e6, 0.6e6]

    num_blocks = 4
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
