
import mosaic

from stride.problem_definition import *
from stride.optimisation import *
from stride.utils import geometries
from stride import plotting


async def main(runtime):
    # Create the grid
    shape = (500, 370)
    extra = (100, 100)
    absorbing = (90, 90)
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
    problem = Problem(name='alpha2D',
                      space=space, time=time)

    # Create medium
    medium = Medium(problem=problem)
    problem.medium = medium

    vp = ScalarField('vp', grid=problem.grid)
    vp.load('data/alpha2D-StartingModel.h5')

    medium.add(vp)

    # Create transducers
    transducers = Transducers(problem=problem)
    problem.transducers = transducers

    transducers.default()

    # Create geometry
    num_transducers = 120

    geometry = Geometry(transducers=transducers, problem=problem)
    problem.geometry = geometry

    radius = ((problem.space.limit[0] - 15.e-3) / 2,
              (problem.space.limit[1] - 13.e-3) / 2)
    centre = (problem.space.limit[0] / 2,
              problem.space.limit[1] / 2)

    coordinates = geometries.elliptical(num_transducers, radius, centre)

    for index in range(num_transducers):
        geometry.add(index, transducers.get(0), coordinates[index, :])

    # Create acquisitions
    acquisitions = Acquisitions(geometry=geometry, problem=problem)
    problem.acquisitions = acquisitions

    acquisitions.load(path=problem.output_folder,
                      project_name=problem.name, version=0)

    # Plot
    problem.plot()

    # Create optimisation variable
    optimisation = Optimisation()

    optim_vp = Vp('vp', grid=problem.grid)
    optim_vp.extended_data[:] = vp.extended_data[:]

    # Create optimiser
    step_size = 200
    optimiser = GradientDescent(optim_vp, step=step_size)

    optimisation.add(optim_vp, optimiser)

    # Run optimisation
    max_freqs = [0.1e6, 0.2e6, 0.3e6, 0.4e6]

    for freq, block in zip(max_freqs, optimisation.blocks(4)):
        block.config(num_iterations=10,
                     f_max=freq, f_min=0.05e6,
                     min=1450., max=3000.,
                     select_shots={'num': 12, 'randomly': True})

        await optimisation.run(block, problem)

    axis = optim_vp.plot()
    plotting.show(axis)

if __name__ == '__main__':
    mosaic.run(main)
