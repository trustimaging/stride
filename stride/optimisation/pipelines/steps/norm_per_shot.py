
import numpy as np

from ....core import Operator


class NormPerShot(Operator):
    """
    Normalised a series of time traces to the maximum value of the set.

    Parameters
    ----------
    amplitude : bool, optional
        Whether to keep the true amplitude of the modelled after
        normalisation.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.amplitude = kwargs.pop('amplitude', False)

    def forward(self, modelled, observed, **kwargs):
        amplitude = kwargs.pop('amplitude', self.amplitude)

        if amplitude:
            scaling = self._norm(modelled, **kwargs)
        else:
            scaling = 1.

        modelled = self._apply(modelled, scaling=scaling, **kwargs)
        observed = self._apply(observed, scaling=scaling, **kwargs)

        return modelled, observed

    def adjoint(self, d_modelled, d_observed, modelled, observed, **kwargs):
        return d_modelled, d_observed

    def _norm(self, traces, **kwargs):
        norm_value = 0.

        for index in range(traces.extended_shape[0]):
            norm_value += np.sum(traces.extended_data[index] ** 2)

        norm_value = np.sqrt(norm_value / traces.extended_shape[0]) + 1e-31

        return norm_value

    def _apply(self, traces, scaling, **kwargs):
        norm_value = self._norm(traces, **kwargs)

        out_traces = traces.alike(name='normed_%s' % traces.name)
        out_traces.extended_data[:] = scaling * traces.extended_data / norm_value

        return out_traces
