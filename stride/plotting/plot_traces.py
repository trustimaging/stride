
import os
import numpy as np


__all__ = ['plot_trace', 'plot_magnitude_spectrum', 'plot_gather']


def plot_trace(*args, axis=None, colour='black', line_style='solid', title=None, **kwargs):
    """
    Utility function to plot individual traces using matplotlib.

    Parameters
    ----------
    args : arrays
        Optional time grid and signal to be plotted.
    axis : matplotlib figure, optional
        Figure in which to make the plotting, defaults to new empty figure.
    colour : str, optional
        Colour to apply to the points, defaults to red.
    line_style : str, optional
        Line style to be used.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    matplotlib figure
        Generated matplotlib figure

    """
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    default_kwargs = dict(c=colour, linestyle=line_style)
    default_kwargs.update(kwargs)

    im = axis.plot(*args, **default_kwargs)

    if title is not None:
        axis.set_title(title)

    return axis


def plot_magnitude_spectrum(sampling_rate, *args, skip=1, time_range=None, norm=True,
                            colour='black', line_style='solid', title=None, axis=None,
                            per_trace=False, **kwargs):
    """
    Plot the magnitude spectrum of wavelet traces in dB scale.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the wavelet signal in Hz.
    args : arrays
        Optional trace ID grid, frequency grid, and signal data.
    skip : int, optional
        Traces to skip, defaults to 1.
    time_range : tuple, optional
        Time range to plot, defaults to all.
    norm : bool, optional
        Normalize the whole gather, defaults to True.
    colour : str, optional
        Colour for the traces, defaults to black.
    line_style : str, optional
        Line style for the plot, defaults to solid.
    title : str, optional
        Plot title, defaults to None.
    per_trace : bool, optional
        If True, normalize each trace individually.
    axis : matplotlib axis, optional
        Axis on which to plot, defaults to a new figure.

    Returns
    -------
    matplotlib axis
        Axis with the plotted data.
    """
    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None, None

    # Set up axis
    if axis is None:
        figure, axis = plt.subplots(1, 1)

    # Extract arguments
    if len(args) > 2:
        trace_ids, freq_grid, signal_data = args
    elif len(args) > 1:
        trace_ids = None
        freq_grid, signal_data = args
    else:
        trace_ids = None
        signal_data = args[0]

    signal_data = signal_data[::skip]
    if trace_ids is not None:
        trace_ids = trace_ids[::skip]

    # Apply time range if specified
    if time_range:
        start, end = time_range
        start_idx, end_idx = int(start * sampling_rate), int(end * sampling_rate)
        signal_data = signal_data[:, start_idx:end_idx]

    # Normalize signal data if required
    if norm:
        signal_data = signal_data / np.max(np.abs(signal_data), axis=-1, keepdims=True)

    # Perform FFT
    fft_values = np.fft.fft(signal_data, axis=-1)
    frequencies = np.fft.fftfreq(signal_data.shape[-1], 1 / sampling_rate) / 1e3  # in kHz
    magnitude_spectrum_db = 20 * np.log10(np.abs(fft_values) + 1e-10)

    # Keep positive frequencies only
    positive_freqs = frequencies[frequencies >= 0]
    magnitude_spectrum_db = magnitude_spectrum_db[:, frequencies >= 0]

    # Normalize magnitude spectra if per_trace mode
    if per_trace:
        magnitude_spectrum_db = (magnitude_spectrum_db - np.min(magnitude_spectrum_db, axis=-1, keepdims=True)) / \
                                 (np.max(magnitude_spectrum_db, axis=-1, keepdims=True)
                                    - np.min(magnitude_spectrum_db, axis=-1, keepdims=True)) - 1
        offset_step = 1
        offset_magnitude_db = magnitude_spectrum_db + np.arange(magnitude_spectrum_db.shape[0]).reshape(-1, 1) * offset_step

        # Plot individual traces with vertical gray lines
        axis.plot(offset_magnitude_db.T, positive_freqs, colour, linestyle=line_style, **kwargs)
        for i in range(magnitude_spectrum_db.shape[0]):
            axis.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)
        axis.set_ylabel('Frequency (Hz)')
        axis.set_xlabel('Trace ID')
        axis.set_xticks(np.arange(magnitude_spectrum_db.shape[0]))
        axis.set_xticklabels([str(tid) for tid in trace_ids] if trace_ids is not None
            else np.arange(magnitude_spectrum_db.shape[0]))
    else:
        axis.plot(positive_freqs, magnitude_spectrum_db.T, colour, linestyle=line_style, **kwargs)
        axis.set_xlabel('Frequency (Hz)')
        axis.set_ylabel('Magnitude (dB)')

    # Set title if provided
    if title:
        axis.set_title(title)

    return axis


def plot_gather(*args, skip=1, time_range=None, norm=True, norm_trace=True,
                colour='black', line_style='solid', title=None, axis=None, **kwargs):
    """
    Utility function to plot gather using matplotlib.

    Parameters
    ----------
    args : arrays
        Optional trace ID grid, optional time grid and signal to be plotted.
    skip : int, optional
        Traces to skip, defaults to 1.
    time_range : tuple, optional
        Range of time to plot, defaults to all time.
    norm : bool, optional
        Whether or not to normalise the gather, defaults to True.
    norm_trace : bool, optional
        Whether or not to normalise trace by trace, defaults to True.
    axis : matplotlib figure, optional
        Figure in which to make the plotting, defaults to new empty figure.
    colour : str, optional
        Colour to apply to the points, defaults to red.
    line_style : str, optional
        Line style to be used.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    matplotlib figure
        Generated matplotlib figure

    """

    try:
        if not os.environ.get('DISPLAY', None):
            raise ModuleNotFoundError
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None, None

    if len(args) > 2:
        trace_axis = args[0]
        time_axis = args[1]
        signal = args[2]

    elif len(args) > 1:
        trace_axis = None
        time_axis = args[0]
        signal = args[1]

    else:
        trace_axis = None
        time_axis = None
        signal = args[0]

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    if time_range is None:
        time_range = (0, signal.shape[-1])

    if norm is True:
        signal = signal / (np.max(np.abs(signal))+1e-30)

    num_traces = signal.shape[0]

    if norm_trace is True:
        signal = signal / (np.max(np.abs(signal), axis=-1).reshape((num_traces, 1))+1e-30)

    signal_under = signal[0:num_traces:skip, time_range[0]:time_range[1]]
    num_under_traces = signal_under.shape[0]

    shift = np.arange(0, num_under_traces) * 2.00
    shift = np.reshape(shift, (shift.shape[0], 1))

    signal_shifted = np.transpose(signal_under + shift)

    if time_axis is None:
        time_axis = np.linspace(0, time_range[1]-time_range[0]-1, time_range[1]-time_range[0], endpoint=False)

    time_axis = np.broadcast_to(np.reshape(time_axis, (time_axis.shape[0], 1)), signal_shifted.shape)

    default_kwargs = dict(c=colour, linestyle=line_style)
    default_kwargs.update(kwargs)

    print('gather shape ', signal_shifted.shape)
    print('time axis shape ', time_axis.shape)
    axis.plot(signal_shifted, time_axis, **default_kwargs)
    axis.set_ylim(time_axis[-1, 0], time_axis[0, 0])

    axis.set_xlabel('trace')
    axis.set_ylabel('time')

    if trace_axis is None:
        trace_axis = np.linspace(0, num_traces-1, num_under_traces, endpoint=True)
    else:
        trace_axis = trace_axis[::skip]

    trace_axis = [str(each) for each in trace_axis]

    axis.set_xticks(shift.flatten()[::2])
    axis.set_xticklabels(trace_axis[::2], rotation='vertical')

    if title is not None:
        axis.set_title(title)

    return shift, axis
