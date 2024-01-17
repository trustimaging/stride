
import numpy as np

from .utils import name_from_op_name
from ....core import Operator
from ....utils import filters


class MuteTraces(Operator):
    """
    Mute traces with respect to each other.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.f_max = kwargs.pop('f_max', None)
        self.filter_relaxation = kwargs.pop('filter_relaxation', 1)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        modelled = [t for t in traces if 'modelled' in t.name]
        observed = [t for t in traces if 'observed' in t.name]

        if not len(observed) or not len(modelled):
            return traces

        observed = observed[0]
        modelled = modelled[0]

        mask = np.ones(observed.extended_shape)
        mask[np.abs(observed.extended_data) == 0.] = 0.

        dt = modelled.time.step
        f_max = kwargs.pop('f_max', self.f_max)
        filter_relaxation = kwargs.pop('filter_relaxation', self.filter_relaxation)
        if f_max is not None:
            f_max = f_max * dt / filter_relaxation
        else:
            f_max = 0.5

        mask = filters.lowpass_filter_cos(mask, f_max=f_max, zero_phase=True, axis=-1)

        out_traces = []
        for trace in traces:
            if 'modelled' in trace.name:
                out_trace = trace.alike(name=name_from_op_name(self, trace))
                out_trace.extended_data[:] = mask * trace.extended_data
            else:
                out_trace = trace.copy()
            out_traces.append(out_trace)

        return tuple(out_traces)

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        if len(d_traces) == 1:
            d_traces = d_traces[0]

        return d_traces
