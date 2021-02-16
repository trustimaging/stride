
import numpy as np

import mosaic

from stride.problem_definition import Problem, ScalarField, Space, Time
from stride.utils import wavelets


async def main(runtime):
    # Create the grid
    shape = (200, 200)
    extra = (50, 50)
    absorbing = (45, 45)
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
    vp.fill(1500.)

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

    wavelet = shot.wavelets.data[0]
    observed = shot.observed.data[1]

    import matplotlib.pyplot as plt

    # from stride.utils import filters
    # f_min = 0.15e6 * time.step
    # observed = filters.highpass_filter_fir(observed, f_min)

    plt.figure()
    plt.plot(wavelet/np.max(wavelet), c='k')
    plt.plot(observed/np.max(observed), c='r')

    plt.figure()

    if not time.num % 2:
        num_freqs = (time.num + 1) // 2
    else:
        num_freqs = time.num // 2 + 1

    wavelets_fft = np.fft.fft(wavelet, axis=-1)[:num_freqs]
    observed_fft = np.fft.fft(observed, axis=-1)[:num_freqs]
    freqs = np.fft.fftfreq(time.num, time.step)[:num_freqs]

    wavelets_fft = np.abs(wavelets_fft)
    observed_fft = np.abs(observed_fft)
    wavelets_fft = 20 * np.log10(wavelets_fft / np.max(wavelets_fft))
    observed_fft = 20 * np.log10(observed_fft / np.max(observed_fft))

    plt.plot(freqs, observed_fft, c='r')
    plt.plot(freqs, wavelets_fft, c='k')

    plt.show()


if __name__ == '__main__':
    mosaic.run(main)
