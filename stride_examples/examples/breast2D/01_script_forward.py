
from stride import *
from stride_private import *
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    shape = (500, 370)
    extra = (10, 10)
    absorbing = (7, 7)
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
    vp = ScalarField(name='vp', grid=problem.grid)
    vp.load('../alpha2D/data/alpha2D-TrueModel.h5')

    problem.medium.add(vp)

    # Create transducers
    problem.transducers.default()

    # Create geometry
    num_locations = 128
    problem.geometry.default('elliptical', num_locations)

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

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, dtype=np.float32, len=runtime.num_workers)

    # Run
    await forward(problem, pde, vp, dump=False, deallocate=False,
                  scale=False, shot_ids=[0], save_wavefield=True, save_undersampling=4, kernel='OT4',
                  interpolation_type='hicks', drp=True)
    _, ax = problem.acquisitions.shots[0].observed.plot(colour='k', plot=False)
    obs = problem.acquisitions.shots[0].observed.data.copy()
    problem.acquisitions.shots[0].observed.deallocate()

    # Create the PDE
    pde = IsoAcousticDevito.remote(grid=problem.grid, dtype=np.float16, len=runtime.num_workers)

    # Run
    await forward(problem, pde, vp, dump=False, deallocate=False,
                  scale=True, shot_ids=[0], save_wavefield=True, save_undersampling=4, kernel='OT4',
                  interpolation_type='hicks', drp=True)
    problem.acquisitions.shots[0].observed.plot(colour='r', axis=ax)
    dat = problem.acquisitions.shots[0].observed.data.copy()

    print('before', np.min(obs), np.max(obs))
    print('after', np.min(dat), np.max(dat))
    print('rel', np.min(obs)/np.min(dat), np.max(obs)/np.max(dat))

    print('Error:', np.linalg.norm(dat - obs) / num_locations)


if __name__ == '__main__':
    mosaic.run(main)
