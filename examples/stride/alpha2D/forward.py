
import mosaic

from stride.problem_definition import *
from stride.utils import wavelets, geometries


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
    vp.load('data/alpha2D-TrueModel.h5')

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

    acquisitions.default()

    # Create wavelets
    f_centre = 0.20e6
    n_cycles = 2

    for shot in acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    # Plot
    problem.plot()

    # Run
    await problem.forward()


if __name__ == '__main__':
    mosaic.run(main)
