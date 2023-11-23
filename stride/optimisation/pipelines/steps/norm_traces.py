
import numpy as np

from ....core import Operator


class Norm(Operator):
    """
    Normalised a series of time traces.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        normed = tuple([self._apply(each, **kwargs) for each in traces])

        if len(normed) == 1:
            normed = normed[0]

        return normed

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        if len(d_traces) == 1:
            d_traces = d_traces[0]

        return d_traces

    def _norm(self, traces, **kwargs):
        norm_value = 0.

        for index in range(traces.extended_shape[0]):
            norm_value += np.sum(traces.extended_data[index] ** 2)

        return np.sqrt(norm_value / traces.extended_shape[0])

    def _apply(self, traces, **kwargs):
        pass


class NormPerShot(Norm):
    """
    Normalise a series of time traces to the norm value of the set.

    Parameters
    ----------

    """

    def _apply(self, traces, **kwargs):
        norm_value = self._norm(traces, **kwargs) + 1e-31

        out_traces = traces.alike(name='normed_%s' % traces.name)
        out_traces.extended_data[:] = traces.extended_data / norm_value

        return out_traces


class NormPerTrace(Norm):
    """
    Normalise a series of time traces individually.

    Parameters
    ----------

    """

    def _apply(self, traces, **kwargs):
        out_traces = traces.alike(name='normed_%s' % traces.name)

        for index in range(traces.extended_shape[0]):
            norm_value = np.sqrt(np.sum(traces.extended_data[index]**2)) + 1e-31
            out_traces.extended_data[index] = traces.extended_data[index] / norm_value

        return out_traces
