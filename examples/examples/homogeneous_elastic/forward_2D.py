
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from stride import *
from stride.utils import wavelets

import IPython.terminal.debugger as ipdb

async def main(runtime):
    # Create the grid
    extent = (120e-3, 70e-3) # [m]
    spacing = (0.5e-3, 0.5e-3)  # [m]

    shape = tuple([int(each_extent / each_spacing) + 1 for each_extent, each_spacing in zip(extent, spacing)])  # [grid_points]
    extra = (50, 50) # boundary [grid_points]
    absorbing = (40, 40) # absorbing boundary [grid_points]

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)
    print('Space complete')

    start = 0.  # [s], start = 0.
    step = 7.855e-8  # [s], step = 0.08e-6
    end = 1.e-04/2 # [s]
    # num = int(end/step)+1 # [time_points], num = 2500

    time = Time(start=start,
                step=step,
                stop=end)
    print('Time complete')

    # Create problem
    problem = Problem(name='test2D_elastic',
                      space=space, time=time)
    print('Problem complete')

    # Create medium
    vp = ScalarField(name='vp', grid=problem.grid)
    vp.fill(1500.)  # [m / s]

    vs = ScalarField(name='vs', grid=problem.grid)
    vs.fill(1000.)  # [m / s]

    rho = ScalarField(name='rho', grid=problem.grid)
    rho.fill(1000.) # [g / cm^3]

    problem.medium.add(vp)
    problem.medium.add(vs)
    problem.medium.add(rho)
    print('Media complete')

    # Create transducers
    problem.transducers.default()  # This generates a single transducer, that's a point source and recevier

    # Create geometry
    problem.geometry.add(id = 0, transducer = problem.transducers.get(0), coordinates=np.array(extent) / 2)  # [m]
    print('Geometry complete')

    # Create acquisitions
    source = problem.geometry.locations[0]
    receivers = problem.geometry.locations

    shot = Shot(source.id,
                sources=[source], receivers=receivers,
                geometry=problem.geometry, problem=problem)

    problem.acquisitions.add(shot)

    # Create wavelets
    f_centre = 0.30e6 # [MHz]
    n_cycles = 3

    shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                   time.num, time.step)
    print('Shots complete')

    # Create the PDE
    pde = IsoElasticDevito(space=space, time=time)

    # Set up test cases
    cases = {
        'default': {'colour': 'r', 'line_style': '--'},
    }

    # Run
    results = {}
    legends = {}
    for case, config in cases.items():
        runtime.logger.info('\n')
        runtime.logger.info('===== Running %s' % case)

        shot.observed.deallocate()
        sub_problem = problem.sub_problem(shot.id)
        shot_wavelets = sub_problem.shot.wavelets

        # ipdb.set_trace()
        pde.clear_operators()
        print('Operators cleared')
        traces = await pde(shot_wavelets, vp, vs, rho, problem=sub_problem, **config)

        # Check consistency with analytical solution
        data_stride = traces.data.copy()
        data_stride /= np.max(np.abs(data_stride))

        shot.observed.data[:] = data_stride
        _, axis = shot.observed.plot(plot=False, skip=5,
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
