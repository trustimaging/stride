
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

def plot_magnitude_spectrum(sampling_rate, *args, skip=1, time_range=None, norm=True, norm_trace=True,
                            colour='black', line_style='solid', title=None, axis=None, **kwargs):
    """
    Utility function to plot the magnitude spectrum of wavelet traces in dB scale using matplotlib.

    Parameters
    ----------
    sampling_rate : float
        The sampling rate of the wavelet signal in Hz.
    args : arrays
        Optional trace ID grid, optional time grid, and signal data to be plotted.
    skip : int, optional
        Traces to skip, defaults to 1.
    time_range : tuple, optional
        Range of time to plot, defaults to all time.
    norm : bool, optional
        Whether or not to normalize the gather, defaults to True.
    norm_trace : bool, optional
        Whether or not to normalize trace by trace, defaults to True.
    axis : matplotlib figure, optional
        Figure in which to make the plotting, defaults to new empty figure.
    colour : str, optional
        Colour to apply to the points, defaults to black.
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

    if len(args) > 2:
        trace_ids = args[0]
        time_grid = args[1]
        signal_data = args[2]
    elif len(args) > 1:
        trace_ids = None
        time_grid = args[0]
        signal_data = args[1]
    else:
        trace_ids = None
        time_grid = None
        signal_data = args[0]

    # Normalize the data if specified
    if norm:
        signal_data = signal_data / (np.max(np.abs(signal_data)) + 1e-30)

    if norm_trace:
        signal_data = signal_data / (np.max(np.abs(signal_data), axis=-1).reshape((signal_data.shape[0], 1)) + 1e-30)

    # Apply time range if specified
    if time_range is not None:
        start, end = time_range
        start_index = int(start * sampling_rate)
        end_index = int(end * sampling_rate)
        signal_data = signal_data[:, start_index:end_index]

    # Iterate over each trace and compute its magnitude spectrum
    for i in range(0, signal_data.shape[0], skip):
        trace = signal_data[i]

        # Compute the Fourier Transform of the wavelet signal
        fft_values = np.fft.fft(trace)
        frequencies = np.fft.fftfreq(len(trace), 1 / sampling_rate)

        # Convert frequencies to kHz
        frequencies_khz = frequencies / 1e3

        # Calculate the magnitude spectrum
        magnitude_spectrum = np.abs(fft_values)

        # Normalize the magnitude spectrum to have a peak of 0 dB
        if np.max(magnitude_spectrum) > 0:
            magnitude_spectrum_normalized = magnitude_spectrum / np.max(magnitude_spectrum)
        else:
            magnitude_spectrum_normalized = magnitude_spectrum

        # Convert to dB scale
        magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum_normalized + 1e-10)  # Add small value to avoid log(0)

        # Filter out negative frequencies
        positive_frequencies = frequencies_khz[frequencies_khz >= 0]
        positive_magnitude_db = magnitude_spectrum_db[:len(positive_frequencies)]

        # Plot the magnitude spectrum in dB scale
        default_kwargs = dict(c=colour, linestyle=line_style)
        if trace_ids is not None:
            default_kwargs['label'] = f'Trace ID: {trace_ids[i]}'
        default_kwargs.update(kwargs)

        axis.plot(positive_frequencies, positive_magnitude_db, **default_kwargs)

    if title is not None:
        axis.set_title(title)

    axis.set_xlabel('Frequency (kHz)')
    axis.set_ylabel('Magnitude (dB)')

    if trace_ids is not None:
        axis.legend()

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
