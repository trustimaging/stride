
"""
So far, we have been running our Stride examples from a Jupyter notebook. However,
we might also want to run our programs using a traditional Python script, e.g.
when running on an HPC cluster.

To do this, we will need to write our script within an asynchronous function. This
will allow us to keep using the asynchronous syntax that we have been using so far:

    async def main(runtime):
        <our asynchronous code>

    if __name__ == '__main__':
        mosaic.run(main)

In the next example, we will reproduce the forward problem of our last imaging example
using this script format, and we will save the resulting time traces to disk in order
to be able to use them later on.

To run the script within the Mosaic runtime, we will need to use the command:

    mrun -nw 4 python 07_script_running.py

To know more about what parameters can be passed to the mrun command, check out:

    mrun --help

"""

from stride import *
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    space = Space(shape=(356, 385), extra=(50, 50), absorbing=(40, 40), spacing=0.5e-3)
    time = Time(0.0e-6, 0.08e-6, 2500)

    grid = Grid(space, time)

    # Create problem
    problem = Problem(name='breast2D', space=space, time=time)

    # Create transducers
    # the default option will create a single point transducer
    problem.transducers.default()

    # Create geometry
    # a default elliptical geometry will be generated in this case
    num_locations = 120
    problem.geometry.default('elliptical', num_locations)

    # Populate acquisitions with default shots
    problem.acquisitions.default()

    # Create wavelets
    f_centre = 0.50e6
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles, time.num, time.step)

    # Create medium
    vp_true = ScalarField(name='vp', grid=grid)
    vp_true.load('../examples/breast2D/data/anastasio2D-TrueModel.h5')

    problem.medium.add(vp_true)

    # Plot all components of the problem
    problem.plot()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)

    # Run default forward workflow
    await forward(problem, pde, vp_true, dump=True)

    # Plot the result
    problem.acquisitions.plot()


if __name__ == '__main__':
    mosaic.run(main)
