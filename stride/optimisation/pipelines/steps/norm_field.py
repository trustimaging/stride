
import numpy as np

from ..pipeline import PipelineStep


class NormField(PipelineStep):
    """
    Normalise a StructuredData object between -1 and +1.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        self.norm_value = None

    def apply(self, field, **kwargs):
        self.norm_value = np.max(np.abs(field.extended_data)) + 1e-31
        field.extended_data[:] /= self.norm_value

        return field
