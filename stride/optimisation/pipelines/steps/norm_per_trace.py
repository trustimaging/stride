
import numpy as np

from ....core import Operator


class NormPerTrace(Operator):
    """
    Normalised a series of time traces individually.

    Parameters
    ----------
    amplitude : bool, optional
        Whether to keep the true amplitude of the modelled after
        normalisation.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.amplitude = kwargs.pop('amplitude', False)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        amplitude = kwargs.pop('amplitude', self.amplitude)

        if amplitude:
            scaling = self._norm(traces[0], **kwargs)
        else:
            scaling = 1.

        normed = tuple([self._apply(each, scaling=scaling, **kwargs) for each in traces])

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

        norm_value = np.sqrt(norm_value / traces.extended_shape[0]) + 1e-31

        return norm_value

    def _apply(self, traces, scaling, **kwargs):
        out_traces = traces.alike(name='normed_%s' % traces.name)

        for index in range(traces.extended_shape[0]):
            norm_value = np.sqrt(np.sum(traces.extended_data[index]**2)) + 1e-31
            out_traces.extended_data[index] = scaling * traces.extended_data[index] / norm_value

        return out_traces
