
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import mosaic

from stride import *
from utils import analytical_2d
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    shape = (500, 500)
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
    problem = Problem(name='test2D',
                      space=space, time=time)

    # Create medium
    vp = ScalarField('vp', grid=problem.grid)
    vp.fill(1500.)

    rho = ScalarField('rho', grid=problem.grid)
    rho.fill(1000.)

    problem.medium.add(vp)
    problem.medium.add(rho)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_locations = 120
    problem.geometry.default('elliptical', num_locations)

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

    # Set up test cases
    cases = {
        'OT2': {'drp': False, 'kernel': 'OT2', 'colour': 'r', 'line_style': '--'},
        'OT2-DRP': {'drp': True, 'kernel': 'OT2', 'colour': 'g', 'line_style': '--'},
        'OT2-DRP-hicks': {'drp': True, 'kernel': 'OT2', 'interpolation_type': 'hicks', 'colour': 'b', 'line_style': '--'},
        'OT2-DRP-PML': {'drp': True, 'kernel': 'OT2', 'boundary_type': 'complex_frequency_shift_PML_2', 'colour': 'y', 'line_style': '--'},
        'OT4': {'drp': False, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-DRP': {'drp': True, 'kernel': 'OT4', 'colour': 'g', 'line_style': '-.'},
        'OT4-DRP-hicks': {'drp': True, 'kernel': 'OT4', 'interpolation_type': 'hicks', 'colour': 'b', 'line_style': '-.'},
    }

    # Run
    data_analytic = analytical_2d(space, time, shot, 1500.)
    data_analytic /= np.max(np.abs(data_analytic))

    shot.observed.data[:] = data_analytic
    _, axis = shot.observed.plot(plot=False, colour='k', skip=5)

    results = {}
    legends = {}
    for case, config in cases.items():
        runtime.logger.info('\n')
        runtime.logger.info('===== Running %s' % case)

        shot.observed.deallocate()

        await problem.forward(**config, dump=False)

        # Check consistency with analytical solution
        data_stride = shot.observed.data.copy()
        data_stride /= np.max(np.abs(data_stride))
        error = np.sqrt(np.sum((data_stride - data_analytic)**2)/data_analytic.shape[0])

        # Show results
        results[case] = error

        shot.observed.data[:] = data_stride
        _, axis = shot.observed.plot(plot=False, axis=axis, skip=5,
                                     colour=config['colour'], line_style=config['line_style'])
        legends[case] = lines.Line2D([0, 1], [1, 0], color=config['colour'], linestyle=config['line_style'])

    runtime.logger.info('\n')
    runtime.logger.info('Error results:')
    for case, error in results.items():
        runtime.logger.info('\t* %s : %f' % (case, error))

    plt.legend(legends.values(), legends.keys(), loc='lower right')
    plt.show()


if __name__ == '__main__':
    mosaic.run(main)
