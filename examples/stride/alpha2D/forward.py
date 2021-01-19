
import mosaic

from stride.problem_definition import Problem, ScalarField, Space, Time
from stride.utils import wavelets


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
    vp = ScalarField('vp', grid=problem.grid)
    vp.load('data/alpha2D-TrueModel.h5')

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_transducers = 120
    problem.geometry.default('elliptical', num_transducers)

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
