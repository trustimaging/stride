
import numpy as np


__all__ = ['magnitude_spectrum', 'bandwidth']


def magnitude_spectrum(signal, dt, db=True):
    """
    Calculate magnitude spectrum of a signal.

    Parameters
    ----------
    signal : ndarray
        Signal or array of signals.
    dt : float
        Discretisation step for the time axis.
    db : bool, optional
        Whether to calculate the spectrum in decibels, defaults to True.

    Returns
    -------
    1-dimensional array
        Frequencies of the spectrum
    ndarray
        Magnitude spectrum of the signal or signals.

    """
    num = signal.shape[-1]

    if not num % 2:
        num_freqs = num // 2
    else:
        num_freqs = (num + 1) // 2

    signal_fft = np.fft.fft(signal, axis=-1).take(range(num_freqs), axis=-1)
    freqs = np.fft.fftfreq(num, dt)[:num_freqs]

    signal_fft = np.abs(signal_fft)

    if db is True:
        signal_fft = 20 * np.log10(signal_fft / np.max(signal_fft))

    return freqs, signal_fft


def bandwidth(signal, dt, cutoff=-10):
    """
    Calculate the bandwidth of a signal at a given dB level.

    Parameters
    ----------
    signal : ndarray
        Signal or array of signals.
    dt : float
        Discretisation step for the time axis.
    cutoff : float
        dB level to calculate bandwidth.

    Returns
    -------
    1-dimensional array
        Frequencies of the spectrum
    ndarray
        Magnitude spectrum of the signal or signals.

    """
    freqs, signal_fft = magnitude_spectrum(signal, dt, db=True)

    if len(signal_fft.shape) > 1:
        signal_fft = np.mean(signal_fft, axis=0)

    num_freqs = signal_fft.shape[-1]

    f_min = 0
    for f in range(num_freqs):
        if signal_fft[f] > cutoff:
            f_min = freqs[f]
            break

    f_max = num_freqs
    for f in reversed(range(num_freqs)):
        if signal_fft[f] > cutoff:
            f_max = freqs[f]
            break

    return f_min, f_max
