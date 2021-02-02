
import numpy as np

import mosaic

from stride.problem_definition import Problem, ScalarField, Space, Time
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    shape = (200, 200)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)

    start = 0.
    step = 0.08e-6
    num = 2000

    time = Time(start=start,
                step=step,
                num=num)

    # Create problem
    problem = Problem(name='example',
                      space=space, time=time)

    # Create medium
    vp = ScalarField('vp', grid=problem.grid)
    vp.fill(1600.)

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    problem.geometry.add(0, problem.transducers.get(0),
                         np.array([space.limit[0]/3, space.limit[1]/2]))

    problem.geometry.add(1, problem.transducers.get(0),
                         np.array([2*space.limit[0]/3, space.limit[1]/2]))

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

    # Run
    shot = problem.acquisitions.get(0)

    await problem.forward(deallocate=False, dump=False)

    import matplotlib.pyplot as plt

    plt.plot(shot.observed.data[1]/np.max(shot.observed.data[1]), c='r')
    plt.plot(shot.wavelets.data[0]/np.max(shot.wavelets.data[0]), c='k')

    plt.show()


if __name__ == '__main__':
    mosaic.run(main)
