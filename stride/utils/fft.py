
import numpy as np


__all__ = ['magnitude_spectrum', 'bandwidth']


def magnitude_spectrum(signal, dt, db=True):
    r"""
    Calculate magnitude spectrum of a signal.

    Uses an FFT to decompose the signal into frequency components. Only the non-negative
    frequency components are returned.

    This function uses the `forward` normalisation of the FFT. See numpy FFT
    documentation for more information:
    https://numpy.org/devdocs/reference/routines.fft.html#implementation-details

    When the input signal is composed only of waves with frequencies that correspond to
    periods that are integer multiples of dt, then the magnitude terms returned will
    exactly match the amplitude of the complex wave decomposition. Note that for
    frequencies other than 0 or the Nyquist frequency, this amplitude is equal to half
    of the amplitude of a sinusoidal wave because :math:`\cos(2\pi f) =
    \frac{1}{2}(e^{i2\pi f} + e^{-i2\pi f})`.

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

    Examples
    --------
    The following code extracts the amplitude of a sinusoidal waveform at a given
    target frequency (which is neither 0 nor the Nyquist frequency). Note that in order
    for the amplitudes to be exactly correct, the frequency needs exist within the returned
    freqs (that is, the period corresponds to an integer multiple of dt).

    >>> freqs, signal_magnitude = fft.magnitude_spectrum(y, dt, db=False)
    >>> assert target_frequency in freqs
    >>> target_freq_idx = np.argmin(np.abs(freqs - target_frequency))
    >>> waveform_amplitude = 2 * signal_magnitude[..., target_freq_idx]
    """
    num = signal.shape[-1]
    freqs = np.fft.rfftfreq(num, dt)
    signal_fft = np.fft.rfft(signal, axis=-1, norm="forward")
    signal_magnitude = np.abs(signal_fft)

    if db is True:
        signal_magnitude = 20 * np.log10(
            (signal_magnitude + 1e-31) / (np.max(signal_magnitude) + 1e-31)
        )

    return freqs, signal_magnitude


def phase_spectrum(signal, dt):
    """
    Calculate phase spectrum of a signal.

    Uses an FFT to decompose the signal into frequency components. Only the non-negative
    frequency components are returned.

    Parameters
    ----------
    signal : ndarray
        Signal or array of signals.
    dt : float
        Discretisation step for the time axis.

    Returns
    -------
    1-dimensional array
        Frequencies of the spectrum
    ndarray
        Phase spectrum of the signal or signals.

    """
    num = signal.shape[-1]
    freqs = np.fft.rfftfreq(num, dt)
    signal_fft = np.fft.rfft(signal, axis=-1)
    signal_phase = np.angle(signal_fft)

    return freqs, signal_phase


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
    float
        Min frequency in the BW.
    float
        Centre frequency in the BW.
    float
        Max frequency in the BW.

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

    f_centre = freqs[np.argmax(signal_fft)]

    f_max = num_freqs
    for f in reversed(range(num_freqs)):
        if signal_fft[f] > cutoff:
            f_max = freqs[f]
            break

    return f_min, f_centre, f_max
