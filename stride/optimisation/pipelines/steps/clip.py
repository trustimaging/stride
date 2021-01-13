
import numpy as np

from ..pipeline import PipelineStep


class Step(PipelineStep):

    def __init__(self, **kwargs):
        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)

    def apply(self, field, **kwargs):
        if self.min is not None or self.max is not None:
            field.extended_data[:] = np.clip(field.extended_data,
                                             self.min, self.max)

        return field
