
import numpy as np
import scipy.signal


def tone_burst(centre_freq, n_cycles, n_samples, dt, envelope='gaussian', offset=0):
    """
    Generate a tone burst wavelet.

    Parameters
    ----------
    centre_freq : float
        Centre frequency of the signal.
    n_cycles : float
        Number of cycles for the signal.
    n_samples : int
        Length of the wavelet.
    dt : float
        Discretisation step for the time axis.
    envelope : str, optional
        Type of envelope to be applied to the signal, ``gaussian`` (default) or ``rectangular``.
    offset : int, optional
        Offset in timesteps to the start of the wavelet, defaults to 0.

    Returns
    -------
    1-dimensional array
        Generated wavelet.

    """
    tone_length = n_cycles / centre_freq

    time_array = np.linspace(0, tone_length, int(tone_length//dt + 1), endpoint=False)
    signal = np.sin(2 * np.pi * centre_freq * time_array)
    n_tone = signal.shape[0]

    if envelope == 'gaussian':
        limit = 3
        window_x = np.linspace(-limit, limit, n_tone)
        window = np.exp(-window_x ** 2 / 2)
    elif envelope == 'rectangular':
        window = np.ones((signal.shape[0],))
    else:
        raise Exception('Envelope type not implemented')

    signal = np.multiply(signal, window)

    window = scipy.signal.get_window(('tukey', 0.05), n_tone, False)
    signal = np.multiply(signal, window)

    signal = np.pad(signal, ((offset, n_samples - offset - n_tone),), mode='constant', constant_values=0.)

    return signal


def ricker(centre_freq, n_samples, dt, offset=0):
    """
    Generate a ricker wavelet.

    Parameters
    ----------
    centre_freq : float
        Centre frequency of the signal.
    n_samples : int
        Length of the wavelet.
    dt : float
        Discretisation step for the time axis.
    offset : int, optional
        Offset in timesteps to the start of the wavelet, defaults to 0.

    Returns
    -------
    1-dimensional array
        Generated wavelet.

    """
    tone_length = 3 * np.sqrt(6) / (centre_freq * np.pi)
    time_array = np.linspace(-tone_length/2, tone_length/2, int(tone_length//dt + 1), endpoint=False)

    signal = (1 - 2 * np.pi**2 * centre_freq**2 * time_array**2) * np.exp(-np.pi**2 * centre_freq**2 * time_array**2)
    n_tone = signal.shape[0]

    window = scipy.signal.get_window(('tukey', 0.05), n_tone, False)
    signal = np.multiply(signal, window)

    signal = np.pad(signal, ((offset, n_samples - offset - n_tone),), mode='constant', constant_values=0.)

    return signal


def continuous_wave(centre_freq, n_samples, dt, ramp_length=4, phase=0):
    """
    Generate a continuous wave.

    Parameters
    ----------
    centre_freq : float
        Centre frequency of the signal.
    n_samples : int
        Length of the wavelet.
    dt : float
        Discretisation step for the time axis.
    ramp_length : int, optional
        Length of the up-ramp used to reduce start-up
        transients in periods, defaults to 4.
    phase : float, optional
        Phase shift on the wave, in [rad]. Defaults to 0.

    Returns
    -------
    1-dimensional array
        Generated wavelet.

    """
    time_array = np.linspace(0, dt * (n_samples - 1), n_samples, endpoint=True)
    signal = np.sin(2 * np.pi * centre_freq * time_array + phase)

    if ramp_length > 0:
        period = 1 / centre_freq

        ramp_length_points = int(np.round(ramp_length * period / dt))
        ramp_axis = np.linspace(0, np.pi, ramp_length_points, endpoint=True)

        ramp = (-np.cos(ramp_axis) + 1) * 0.5

        if len(signal) > len(ramp):
            signal[:ramp_length_points] *= ramp

    return signal
