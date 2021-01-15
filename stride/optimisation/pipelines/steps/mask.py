
import numpy as np

from ..pipeline import PipelineStep


class Step(PipelineStep):
    """
    Mask a StructuredData object to remove values outside inner domain.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        pass

    def apply(self, field, **kwargs):
        mask = np.zeros(field.extended_shape)
        mask[field.inner] = 1

        field *= mask

        return field
