
from .utils import name_from_op_name
from ....core import Operator


class ShiftTraces(Operator):
    """
    Shift a set of time traces.

    Parameters
    ----------
    f_max : float, optional
        Upper value for the frequency filter, defaults to None (no upper filtering).
    filter_relaxation : float, optional
        Relaxation factor for the filter in range (0, 1], defaults to 1 (no dilation).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f_max = kwargs.pop('f_max', None)
        self.relaxation = kwargs.pop('filter_relaxation', 1.0)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        shifted = []
        for each in traces:
            shifted.append(self._apply(each, **kwargs))

        if len(traces) > 1:
            return tuple(shifted)

        else:
            return shifted[0]

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        if len(d_traces) > 1:
            return tuple(d_traces)

        else:
            return d_traces[0]

    def _apply(self, traces, **kwargs):
        time = traces.time

        f_max = kwargs.pop('f_max', self.f_max)
        relaxation = kwargs.pop('filter_relaxation', self.relaxation)

        if f_max is None:
            return traces

        f_max_dim_less = 1/relaxation*f_max*time.step if f_max is not None else 0
        period = int(1 / f_max_dim_less)
        shift = period//4

        out_data = traces.extended_data.copy()
        out_traces = traces.alike(name=name_from_op_name(self, traces))

        if shift > 0:
            out_data[:, :-shift] = out_data[:, shift:]

        out_traces.extended_data[:] = out_data

        return out_traces
