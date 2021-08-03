
import numpy as np

from ....core import Operator


class NormPerTrace(Operator):
    """
    Normalised a series of time traces individually.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, modelled, observed, **kwargs):
        modelled = self._apply(modelled, **kwargs)
        observed = self._apply(observed, **kwargs)

        return modelled, observed

    def adjoint(self, d_modelled, d_observed, modelled, observed, **kwargs):
        return d_modelled, d_observed

    def _apply(self, traces, **kwargs):
        out_traces = traces.alike(name='normed_%s' % traces.name)

        for index in range(traces.extended_shape[0]):
            norm_value = np.sqrt(np.sum(traces.extended_data[index]**2)) + 1e-31
            out_traces.extended_data[index] = out_traces.extended_data[index] / norm_value

        return out_traces
