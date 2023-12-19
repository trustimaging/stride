
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

        self.taper_factor = kwargs.pop('taper_factor', 0.5)

    def forward(self, modelled, observed, **kwargs):

        out_traces = modelled.alike(name=name_from_op_name(self, modelled))

        data = modelled.extended_data.copy()

        mask = np.ones(observed.extended_shape)
        mask[np.abs(observed.extended_data) == 0.] = 0.

        taper_factor = kwargs.pop('taper_factor', self.taper_factor)
        mask = filters.lowpass_filter_cos(mask, f_max=taper_factor, zero_phase=True, axis=-1)

        data *= mask
        out_traces.extended_data[:] = data

        return out_traces, observed.copy()

    def adjoint(self, d_modelled, d_observed, modelled, observed, **kwargs):
        return d_modelled, d_observed
