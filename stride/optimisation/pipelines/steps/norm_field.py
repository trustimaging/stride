
import numpy as np

from ....core import Operator


class NormField(Operator):
    """
    Normalise a StructuredData object between -1 and +1.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.norm_value = None

    def forward(self, field, **kwargs):
        self.norm_value = np.max(np.abs(field.extended_data)) + 1e-31
        field.extended_data[:] /= self.norm_value

        return field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
