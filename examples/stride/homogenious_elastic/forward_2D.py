
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from stride import *
from stride.utils import wavelets


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

    start = 0.  # [s], start = 0.
    step = 7.855e-8  # [s], step = 0.08e-6
    end = 1.e-04 # [s]
    num = int(end/step)+1 # [time_points], num = 2500

    time = Time(start=start,
                step=step,
                num=num)

    # Create problem
    problem = Problem(name='test2D_elastic',
                      space=space, time=time)

    # Create medium
    density = 1000  # ??[g / cm^3]
    vs = 1000  # [m / s]
    vp = 1500  # [m / s]

    lam = ScalarField(name='lam', grid=problem.grid)
    lam.fill(density * (vp ** 2 - 2. * vs ** 2))

    mu = ScalarField(name='mu', grid=problem.grid)
    mu.fill(density * vs ** 2)

    byn = ScalarField(name='byn', grid=problem.grid)
    byn.fill(1 / density)

    problem.medium.add(lam)
    problem.medium.add(mu)
    problem.medium.add(byn)

    # Create transducers
    problem.transducers.default()  ## should I change this?

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
    f_centre = 0.30e6 # [MHz]
    n_cycles = 3

    shot.wavelets.data[0, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                   time.num, time.step)

    # Create the PDE
    pde = IsoElasticDevito.remote(space=space, time=time)

    # Set up test cases
    cases = {
        'OT2': {'kernel': 'OT2', 'colour': 'r', 'line_style': '--'},
        'OT2-hicks': {'kernel': 'OT2', 'interpolation_type': 'hicks', 'colour': 'b', 'line_style': '--'},
        'OT2-PML': {'kernel': 'OT2', 'boundary_type': 'complex_frequency_shift_PML_2', 'colour': 'y', 'line_style': '--'},
        'OT4': {'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-hicks': {'kernel': 'OT4', 'interpolation_type': 'hicks', 'colour': 'b', 'line_style': '-.'},
        'OT4-rho': {'rho': rho, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-alpha-0': {'alpha': alpha, 'attenuation_power': 0, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-alpha-2': {'alpha': alpha, 'attenuation_power': 2, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-rho-alpha-0': {'rho': rho, 'alpha': alpha, 'attenuation_power': 0, 'kernel': 'OT4', 'colour': 'r',
                            'line_style': '-.'},
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
        sub_problem = problem.sub_problem(shot.id)
        shot_wavelets = sub_problem.shot.wavelets

        await pde.clear_operators()
        traces = await pde(shot_wavelets, vp, problem=sub_problem, **config).result()

        # Check consistency with analytical solution
        data_stride = traces.data.copy()
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
