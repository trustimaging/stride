
import numpy as np


def analytical_2d(space, time, shot, sos):
    source = shot.sources[0]
    receivers = shot.receivers
    wavelet = shot.wavelets.data[0, :]
    wavelet = np.gradient(wavelet, time.step)

    k = 1 / (2*np.pi*sos**2)

    analytic_traces = np.zeros((len(receivers), time.num))
    for receiver in receivers:
        src_location = source.coordinates
        rec_location = receiver.coordinates

        r = np.sqrt(np.sum((src_location - rec_location) ** 2))
        tau = int(np.round(r / (sos * time.step)))

        signal = np.zeros_like(wavelet)
        if tau >= 0 and time.num - tau > 0:
            h = np.zeros((time.num,))
            h[tau] = k
            h[tau+1:] = k * 1 / (np.sqrt(time.grid[tau+1:]**2 - r**2/sos**2))

            signal = np.convolve(h, wavelet*time.step, mode='full')[:time.num]

        analytic_traces[receiver.id, :] = signal

    return analytic_traces


def analytical_3d(space, time, shot, sos):
    source = shot.sources[0]
    receivers = shot.receivers
    wavelet = shot.wavelets.data[0, :]
    wavelet = np.gradient(wavelet, time.step)

    k = 1 / (4*np.pi*sos**2)

    analytic_traces = np.zeros((len(receivers), time.num))
    for receiver in receivers:
        src_location = source.coordinates
        rec_location = receiver.coordinates

        r = np.sqrt(np.sum((src_location - rec_location) ** 2))
        tau = int(np.round(r / (sos * time.step)))

        signal = np.zeros_like(wavelet)
        if tau >= 0 and time.num - tau > 0:
            if r > 0:
                k_ = k * 1/r
            else:
                k_ = k * 2.66/space.spacing[0]

            h = np.zeros((time.num,))
            h[tau] = k_

            signal = np.convolve(h, wavelet*time.step, mode='full')[:time.num]

        analytic_traces[receiver.id, :] = signal

    return analytic_traces
