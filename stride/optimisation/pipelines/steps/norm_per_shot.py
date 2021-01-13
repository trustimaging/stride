
import numpy as np

from ..pipeline import PipelineStep


class Step(PipelineStep):

    def __init__(self, **kwargs):
        pass

    def apply(self, modelled, observed, **kwargs):
        modelled = self._apply(modelled, **kwargs)
        observed = self._apply(observed, **kwargs)

        return modelled, observed

    def _apply(self, traces, **kwargs):
        norm_value = 0.

        for index in range(traces.extended_shape[0]):
            norm_value += np.sum(traces.extended_data[index]**2)

        norm_value = np.sqrt(norm_value/traces.extended_shape[0]) + 1e-31
        traces.extended_data[:] /= norm_value

        return traces
