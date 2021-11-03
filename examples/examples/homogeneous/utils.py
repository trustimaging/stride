
import numpy as np


def analytical_2d(space, time, shot, sos):
    source = shot.sources[0]
    receivers = shot.receivers
    wavelet = shot.wavelets.data[0, :]

    analytic_traces = np.zeros((len(receivers), time.num))
    for receiver in receivers:
        src_location = source.coordinates
        rec_location = receiver.coordinates

        r = np.sqrt(np.sum((src_location - rec_location) ** 2))
        extended_time = np.linspace(-time.stop + time.step, time.stop - time.step, 2 * time.num - 1, endpoint=True)
        h = np.heaviside(extended_time - r / sos, 0)
        den = extended_time ** 2 - (r / sos) ** 2 + 1e-31
        den[den < 0] = 1e-31

        analytic_traces[receiver.id, :] = np.convolve(h / den, wavelet, mode='valid')

    return analytic_traces


def analytical_3d(space, time, shot, sos):
    source = shot.sources[0]
    receivers = shot.receivers
    wavelet = shot.wavelets.data[0, :]

    analytic_traces = np.zeros((len(receivers), time.num))
    for receiver in receivers:
        src_location = source.coordinates
        rec_location = receiver.coordinates

        r = np.sqrt(np.sum((src_location - rec_location) ** 2))
        tau = int(np.round(r / (sos * time.step)))

        signal = np.zeros_like(wavelet)
        if time.num - tau > 0:
            signal[tau:] = wavelet[:time.num - tau]

        analytic_traces[receiver.id, :] = signal

    return analytic_traces
