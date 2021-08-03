
import os
import numpy as np


try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    import matplotlib.pyplot as plt

    ENABLED_2D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_2D_PLOTTING = False


__all__ = ['plot_trace', 'plot_gather']


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
    if not ENABLED_2D_PLOTTING:
        return None

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    default_kwargs = dict(c=colour, linestyle=line_style)
    default_kwargs.update(kwargs)

    im = axis.plot(*args, **default_kwargs)

    if title is not None:
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

    if not ENABLED_2D_PLOTTING:
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
        signal = signal / (np.max(np.abs(signal))+1e-31)

    num_traces = signal.shape[0]

    if norm_trace is True:
        signal = signal / (np.max(np.abs(signal), axis=-1).reshape((num_traces, 1))+1e-31)

    signal_under = signal[0:num_traces:skip, time_range[0]:time_range[1]]
    num_under_traces = signal_under.shape[0]

    shift = np.arange(0, num_under_traces) * 1.10
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
        trace_axis = np.linspace(0, num_traces-1, num_traces, endpoint=False)

    trace_axis = [str(each) for each in trace_axis]

    axis.set_xticks(shift[::2])
    axis.set_xticklabels(trace_axis[::2])

    if title is not None:
        axis.set_title(title)

    return shift, axis
