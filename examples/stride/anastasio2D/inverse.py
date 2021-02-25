
import mosaic

from stride import *


async def main(runtime):
    # Create the grid
    shape = (356, 385)
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
    problem = Problem(name='anastasio2D',
                      space=space, time=time)

    # Create medium
    vp = ScalarField('vp', grid=problem.grid)
    vp.fill(1500.)

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_locations = 128
    problem.geometry.default('elliptical', num_locations)

    # Create acquisitions
    problem.acquisitions.load(path=problem.output_folder,
                              project_name=problem.name, version=0)

    # Plot
    problem.plot()

    # Create optimisation variable
    optimisation = Optimisation()

    optim_vp = Vp('vp', grid=problem.grid)
    optim_vp.extended_data[:] = vp.extended_data[:]

    # Create optimiser
    step_size = 10
    optimiser = GradientDescent(optim_vp, step=step_size)

    optimisation.add(optim_vp, optimiser)

    # Run optimisation
    max_freqs = [0.3e6, 0.4e6, 0.5e6, 0.6e6]

    for freq, block in zip(max_freqs, optimisation.blocks(4)):
        block.config(num_iterations=8,
                     f_min=0.05e6, f_max=freq,
                     min=1400., max=1700.,
                     select_shots={'num': 16, 'randomly': True})

        await optimisation.run(block, problem)

    optim_vp.plot()

if __name__ == '__main__':
    mosaic.run(main)
