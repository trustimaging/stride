
from stride.problem import *
from stride import *
from stride.utils import fetch, wavelets


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
    vp = ScalarField(name='vp', grid=problem.grid)
    fetch('anastasio2D', dest='data/anastasio2D-TrueModel.h5')
    vp.load('data/anastasio2D-TrueModel.h5')

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_locations = 128
    problem.geometry.default('elliptical', num_locations)

    # Create acquisitions
    problem.acquisitions.default()

    # Create wavelets
    f_centre = 0.50e6
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    # Plot
    problem.plot()

    # Create the PDE
    pde = physics.IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Run
    await forward(problem, pde, vp)


if __name__ == '__main__':
    mosaic.run(main)
