
from stride.utils import filters

from ....core import Operator


class FilterTraces(Operator):
    """
    Filter a set of time traces.

    Parameters
    ----------
    f_min : float, optional
        Lower value for the frequency filter, defaults to None (no lower filtering).
    f_max : float, optional
        Upper value for the frequency filter, defaults to None (no upper filtering).
    filter_type : str, optional
        Type of filter to apply, from ``butterworth`` (default for band pass and high pass),
        ``fir``, or ``cos`` (default for low pass).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f_min = kwargs.pop('f_min', None)
        self.f_max = kwargs.pop('f_max', None)

        self.filter_type = kwargs.pop('filter_type', None)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        filtered = []
        for each in traces:
            filtered.append(self._apply(each, **kwargs))

        if len(traces) > 1:
            return tuple(filtered)

        else:
            return filtered[0]

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        filtered = []
        for each in d_traces:
            filtered.append(self._apply(each, adjoint=True, **kwargs))

        self._num_traces = None

        if len(d_traces) > 1:
            return tuple(filtered)

        else:
            return filtered[0]

    def _apply(self, traces, **kwargs):
        time = traces.time

        f_min = kwargs.pop('f_min', self.f_min)
        f_max = kwargs.pop('f_max', self.f_max)

        f_min_dim_less = f_min*time.step if f_min is not None else 0
        f_max_dim_less = f_max*time.step if f_max is not None else 0

        out_traces = traces.alike(name='filtered_%s' % traces.name)

        if f_min is None and f_max is not None:
            pass_type = 'lowpass'
            args = (f_max_dim_less,)
        elif f_min is not None and f_max is None:
            pass_type = 'highpass'
            args = (f_min_dim_less,)
        elif f_min is not None and f_max is not None:
            pass_type = 'bandpass'
            args = (f_min_dim_less, f_max_dim_less)
        else:
            out_traces.extended_data[:] = traces.extended_data
            return out_traces

        default_filter_type = 'cos' if f_min is None else 'butterworth'
        filter_type = kwargs.pop('filter_type', self.filter_type or default_filter_type)

        method_name = '%s_filter_%s' % (pass_type, filter_type)
        method = getattr(filters, method_name)

        filtered = method(traces.extended_data, *args, zero_phase=False, **kwargs)

        out_traces.extended_data[:] = filtered

        return out_traces
