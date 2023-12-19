
import numpy as np
import scipy.signal
import scipy.ndimage


__all__ = ['bandpass_filter_butterworth', 'lowpass_filter_butterworth', 'highpass_filter_butterworth',
           'bandpass_filter_fir', 'lowpass_filter_fir', 'highpass_filter_fir', 'lowpass_filter_cos']


# TODO Implement more efficient threaded filters


def bandpass_filter_butterworth(data, f_min, f_max, padding=0, order=8,
                                zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a Butterworth bandpass filter using cascaded second-order sections.

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
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5
    f_max = f_max / 0.5

    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    sos = scipy.signal.butter(order, [f_min, f_max], analog=False, btype='band', output='sos')

    if zero_phase:
        method = scipy.signal.sosfiltfilt
    else:
        method = scipy.signal.sosfilt

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(sos, data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def lowpass_filter_butterworth(data, f_max, padding=0, order=8,
                               zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a Butterworth lowpass filter using cascaded second-order sections.

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
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    f_max = f_max / 0.5

    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    sos = scipy.signal.butter(order, f_max, analog=False, btype='lowpass', output='sos')

    if zero_phase:
        method = scipy.signal.sosfiltfilt
    else:
        method = scipy.signal.sosfilt

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(sos, data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def highpass_filter_butterworth(data, f_min, padding=0, order=8,
                                zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a Butterworth highpass filter using cascaded second-order sections.

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
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5

    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    sos = scipy.signal.butter(order, f_min, analog=False, btype='highpass', output='sos')

    if zero_phase:
        method = scipy.signal.sosfiltfilt
    else:
        method = scipy.signal.sosfilt

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(sos, data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def lowpass_filter_hann(data, order, freq_max, padding=0,
                        zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a Hann lowpass filter using cascaded second-order sections.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    order : int
        Order of the filter.
    freq_max : float
        Max frequency of Hann filter.
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, not implemented yet
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    win = scipy.signal.firwin(order, freq_max, pass_zero='lowpass', window='hann', scale=True)

    if zero_phase:
        method = scipy.signal.filtfilt
    else:
        method = scipy.signal.lfilter

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(win, 1., data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def bandpass_filter_fir(data, f_min, f_max, padding=0, attenuation=30,
                        zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a FIR bandpass filter  designed using a kaiser window.

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
        Attenuation of the reject band in dB, defaults to 30.
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    f_min = f_min / 0.5
    f_max = f_max / 0.5

    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    transition_width = 0.050
    order, beta = scipy.signal.kaiserord(attenuation, transition_width)
    order = order // 2 * 2 + 1

    filt = scipy.signal.firwin(order, [f_min, f_max], pass_zero='bandpass', window=('kaiser', beta), scale=True)

    if zero_phase:
        method = scipy.signal.filtfilt
    else:
        method = scipy.signal.lfilter

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(filt, 1., data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def lowpass_filter_fir(data, f_max, padding=0, attenuation=30,
                       zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a FIR lowpass filter  designed using a kaiser window.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_max : float
        Maximum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    attenuation : float, optional
        Attenuation of the reject band in dB, defaults to 30.
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    f_max = f_max / 0.5

    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    transition_width = 0.050
    order, beta = scipy.signal.kaiserord(attenuation, transition_width)
    order = order // 2 * 2 + 1

    filt = scipy.signal.firwin(order, f_max, pass_zero='lowpass', window=('kaiser', beta), scale=True)

    if zero_phase:
        method = scipy.signal.filtfilt
    else:
        method = scipy.signal.lfilter

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(filt, 1., data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def highpass_filter_fir(data, f_min, padding=0, attenuation=30,
                        zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a FIR highpass filter designed using a kaiser window.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_min : float
        Minimum frequency of the filter, dimensionless
    padding : int, optional
        Padding to apply before AND after the traces to compensate for the filtering, defaults to 0.
    attenuation : float, optional
        Attenuation of the reject band in dB, defaults to 30.
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps)

    """
    f_min = f_min / 0.5

    if padding > 0:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (padding, padding)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    transition_width = 0.050
    order, beta = scipy.signal.kaiserord(attenuation, transition_width)
    order = order // 2 * 2 + 1

    filt = scipy.signal.firwin(order, f_min, pass_zero='highpass', window=('kaiser', beta), scale=True)

    if zero_phase:
        method = scipy.signal.filtfilt
    else:
        method = scipy.signal.lfilter

    if adjoint:
        data = np.flip(data, axis=axis)

    filtered = method(filt, 1., data, axis=axis)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered


def _make_filter_cos(filter_length):
    table = np.zeros((filter_length,))

    q = 0.
    for i in range(1, filter_length+1):
        table[i-1] = 1. - np.cos(2*np.pi * i / (filter_length + 1))
        q += table[i-1]

    table /= q

    return table


def lowpass_filter_cos(data, f_max, order=1,
                       zero_phase=True, adjoint=False, axis=-1, **kwargs):
    """
    Apply a cosine lowpass filter.

    Parameters
    ----------
    data : 2-dimensional array
        Data to apply the filter to, with shape (number_of_traces, number_of_timesteps)
    f_max : float
        Maximum frequency of the filter, dimensionless
    order : int, optional
        Order of the filter, defaults to 2.
    zero_phase : bool, optional
        Whether the filter should be zero phase, defaults to True.
    adjoint : bool, optional
        Whether to run the adjoint of the filter, defaults to False.
    axis : int, optional
        Axis on which to perform the filtering, defaults to -1

    Returns
    -------
    n-dimensional array
        Data after filtering, with shape (..., number_of_timesteps+2*padding)

    """
    f_max = f_max / 0.5

    period = int(1 / f_max)
    filter_length = 2*period + 2

    table = _make_filter_cos(filter_length)

    if adjoint:
        data = np.flip(data, axis=axis)

    if not zero_phase:
        pad = [(0, 0)] * data.ndim
        pad[axis] = (period, 0)
        data = np.pad(data, pad, mode='constant', constant_values=0.)

    filtered = data
    for _ in range(order):
        filtered = scipy.ndimage.convolve1d(filtered, table, mode='constant', axis=axis)

    if not zero_phase:
        filtered = filtered.take(range(0, filtered.shape[-1]-period), axis=-1)

    if adjoint:
        filtered = np.flip(filtered, axis=axis)

    return filtered
