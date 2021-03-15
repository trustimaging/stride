
import numpy as np

from ..pipeline import PipelineStep


class Clip(PipelineStep):
    """
    Clip data between two extreme values.

    Parameters
    ----------
    min : float, optional
        Lower value for the clipping, defaults to None (no lower clipping).
    max : float, optional
        Upper value for the clipping, defaults to None (no upper clipping).

    """

    def __init__(self, **kwargs):
        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)

    def apply(self, field, **kwargs):
        if self.min is not None or self.max is not None:
            field.extended_data[:] = np.clip(field.extended_data,
                                             self.min, self.max)

        return field
