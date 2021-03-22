
import numpy as np
import matplotlib.pyplot as plt

import mosaic

from stride import *
from utils import analytical_3d
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    shape = (300, 300, 300)
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
    problem = Problem(name='test3D',
                      space=space, time=time)

    # Create medium
    vp = ScalarField('vp', grid=problem.grid)
    vp.fill(1500.)

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_locations = 1000
    problem.geometry.default('ellipsoidal', num_locations, threshold=0)

    # Create acquisitions
    source = problem.geometry.locations[0]
    receivers = problem.geometry.locations

    shot = Shot(source.id,
                sources=[source], receivers=receivers,
                geometry=problem.geometry, problem=problem)

    problem.acquisitions.add(shot)

    # Create wavelets
    f_centre = 0.50e6
    n_cycles = 3

    shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                   time.num, time.step)

    # Run
    await problem.forward(dump=False, drp=True, kernel='OT4')

    # Check consistency with analytical solution
    data_stride = shot.observed.data.copy()
    data_analytic = analytical_3d(space, time, shot, 1500.)

    data_stride /= np.max(np.abs(data_stride))
    data_analytic /= np.max(np.abs(data_analytic))

    shot.observed.data[:] = data_stride
    _, axis = shot.observed.plot(plot=False, colour='r', skip=5)

    shot.observed.data[:] = data_analytic
    _, axis = shot.observed.plot(plot=False, colour='k', axis=axis, skip=5)

    error = np.sqrt(np.sum((data_stride - data_analytic)**2)/data_analytic.shape[0])
    runtime.logger.info('Error: %f' % error)

    plt.show()


if __name__ == '__main__':
    mosaic.run(main)
