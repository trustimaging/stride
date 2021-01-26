
import mosaic

from stride.problem_definition import Problem, ScalarField, Space, Time
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    shape = (430, 496, 345)
    extra = (100, 100, 100)
    absorbing = (90, 90, 90)
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
    problem = Problem(name='alpha3D',
                      space=space, time=time)

    # Create medium
    vp = ScalarField('vp', grid=problem.grid)
    vp.load('data/alpha3D-TrueModel.h5')

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    radius = ((space.limit[0] - 30.0e-3) / 2,
              (space.limit[1] - 15.0e-3) / 2,
              (space.limit[2]) / 2)
    centre = (space.limit[0] / 2,
              space.limit[1] / 2 + 2.0e-3,
              space.limit[2] / 2 - 2.5e-3)

    num_locations = 1024

    problem.geometry.default('ellipsoidal', num_locations, radius, centre,
                             theta=-0.1, threshold=0.3)

    # Create acquisitions
    problem.acquisitions.default()

    # Create wavelets
    f_centre = 0.20e6
    n_cycles = 2

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                       time.num, time.step)

    # Plot
    problem.plot()

    # Run
    await problem.forward()


if __name__ == '__main__':
    mosaic.run(main)
