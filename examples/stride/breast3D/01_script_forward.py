
import numpy as np

import mosaic

from stride import *
from stride.utils import fetch, wavelets


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
    vp = ScalarField('vp', grid=problem.grid)
    fetch('anastasio3D',
          dest='data/anastasio3D-TrueModel.h5')
    vp.load('data/anastasio3D-TrueModel.h5')

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
    await problem.forward()


if __name__ == '__main__':
    mosaic.run(main)
