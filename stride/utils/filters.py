
import numpy as np
import scipy.signal


__all__ = ['bandpass_filter_butterworth', 'lowpass_filter_butterworth', 'highpass_filter_butterworth',
           'bandpass_filter_fir', 'lowpass_filter_fir', 'highpass_filter_fir']


# TODO Implement more efficient threaded filters


def bandpass_filter_butterworth(data, f_min, f_max, padding=0, order=8, axis=-1):
    """
    Apply a zero-phase Butterworth bandpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_min : float
        Minimum frequency of the filter, dimensionless
    f_max : float
        Maximum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    order : int, optional
        Order of the filter, defaults to 8.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    2-dimensional array
        Data after filtering, with shape (number_of_traces, number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5
    f_max = f_max / 0.5

    if padding > 0:
        data = np.pad(data, ((0, 0), (padding, padding)), mode='constant', constant_values=0.)
    sos = scipy.signal.butter(order, [f_min, f_max], analog=False, btype='band', output='sos')

    return scipy.signal.sosfiltfilt(sos, data, axis=axis)


def lowpass_filter_butterworth(data, f_max, padding=0, order=8, axis=-1):
    """
    Apply a zero-phase Butterworth lowpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_max : float
        Maximum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    order : int, optional
        Order of the filter, defaults to 8.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    2-dimensional array
        Data after filtering, with shape (number_of_traces, number_of_timesteps+2*padding)

    """
    f_max = f_max / 0.5

    if padding > 0:
        data = np.pad(data, ((0, 0), (padding, padding)), mode='constant', constant_values=0.)
    sos = scipy.signal.butter(order, f_max, analog=False, btype='lowpass', output='sos')

    return scipy.signal.sosfiltfilt(sos, data, axis=axis)


def highpass_filter_butterworth(data, f_min, padding=0, order=8, axis=-1):
    """
    Apply a zero-phase Butterworth highpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_min : float
        Minimum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    order : int, optional
        Order of the filter, defaults to 8.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    2-dimensional array
        Data after filtering, with shape (number_of_traces, number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5

    if padding > 0:
        data = np.pad(data, ((0, 0), (padding, padding)), mode='constant', constant_values=0.)
    sos = scipy.signal.butter(order, f_min, analog=False, btype='highpass', output='sos')

    return scipy.signal.sosfiltfilt(sos, data, axis=axis)


def bandpass_filter_fir(data, f_min, f_max, padding=0, attenuation=50, axis=-1):
    """
    Apply a zero-phase FIR bandpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_min : float
        Minimum frequency of the filter, dimensionless
    f_max : float
        Minimum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    attenuation : float, optional
        Attenuation of the reject band in dB, defaults to 50.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    2-dimensional array
        Data after filtering, with shape (number_of_traces, number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5
    f_max = f_max / 0.5

    if padding > 0:
        data = np.pad(data, ((0, 0), (padding, padding)), mode='constant', constant_values=0.)

    transition_width = 0.050
    order, beta = scipy.signal.kaiserord(attenuation, transition_width)
    order = order // 2 * 2 + 1

    filt = scipy.signal.firwin(order, [f_min, f_max], pass_zero='bandpass', window=('kaiser', beta), scale=True)

    return scipy.signal.filtfilt(filt, 1., data, axis=axis)


def lowpass_filter_fir(data, f_max, padding=0, attenuation=50, axis=-1):
    """
    Apply a zero-phase FIR lowpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_max : float
        Maximum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    attenuation : float, optional
        Attenuation of the reject band in dB, defaults to 50.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    2-dimensional array
        Data after filtering, with shape (number_of_traces, number_of_timesteps+2*padding)

    """
    f_max = f_max / 0.5

    if padding > 0:
        data = np.pad(data, ((0, 0), (padding, padding)), mode='constant', constant_values=0.)

    transition_width = 0.050
    order, beta = scipy.signal.kaiserord(attenuation, transition_width)
    order = order // 2 * 2 + 1

    filt = scipy.signal.firwin(order, f_max, pass_zero='lowpass', window=('kaiser', beta), scale=True)

    return scipy.signal.filtfilt(filt, 1., data, axis=axis)


def highpass_filter_fir(data, f_min, padding=0, attenuation=50, axis=-1):
    """
    Apply a zero-phase FIR highpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_min : float
        Minimum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    attenuation : float, optional
        Attenuation of the reject band in dB, defaults to 50.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    2-dimensional array
        Data after filtering, with shape (number_of_traces, number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5

    if padding > 0:
        data = np.pad(data, ((0, 0), (padding, padding)), mode='constant', constant_values=0.)

    transition_width = 0.050
    order, beta = scipy.signal.kaiserord(attenuation, transition_width)
    order = order // 2 * 2 + 1

    filt = scipy.signal.firwin(order, f_min, pass_zero='highpass', window=('kaiser', beta), scale=True)

    return scipy.signal.filtfilt(filt, 1., data, axis=axis)
