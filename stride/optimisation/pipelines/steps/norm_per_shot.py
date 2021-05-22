tuple,
import numpy as np

from ....core import Operator


class NormPerShot(Operator):
    """
    Normalised a series of time traces to the maximum value of the set.

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
        norm_value = 0.

        for index in range(traces.extended_shape[0]):
            norm_value += np.sum(traces.extended_data[index]**2)

        norm_value = np.sqrt(norm_value/traces.extended_shape[0]) + 1e-31
        traces.extended_data[:] /= norm_value

        return traces
