
import numpy as np

from ..pipeline import PipelineStep


class Step(PipelineStep):

    def __init__(self, **kwargs):
        self.norm_value = None

    def apply(self, field, **kwargs):
        self.norm_value = np.max(np.abs(field.extended_data)) + 1e-31
        field.extended_data[:] /= self.norm_value

        return field
