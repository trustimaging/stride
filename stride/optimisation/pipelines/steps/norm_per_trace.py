
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

    def _apply(self, traces, scaling, **kwargs):
        out_traces = traces.alike(name='normed_%s' % traces.name)

        for index in range(traces.extended_shape[0]):
            norm_value = np.sqrt(np.sum(traces.extended_data[index]**2)) + 1e-31
            out_traces.extended_data[index] = scaling * traces.extended_data[index] / norm_value

        return out_traces
