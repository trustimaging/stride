
import numpy as np

from ....core import Operator


class MuteTraces(Operator):
    """
    Mute traces with respect to each other.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, modelled, observed, **kwargs):

        out_traces = modelled.alike(name='muted_%s' % modelled.name)

        data = modelled.extended_data.copy()
        data[np.abs(observed.extended_data) < 1e-31] = 0.

        out_traces.extended_data[:] = data

        return out_traces, observed

    def adjoint(self, d_modelled, d_observed, modelled, observed, **kwargs):
        return d_modelled, d_observed
