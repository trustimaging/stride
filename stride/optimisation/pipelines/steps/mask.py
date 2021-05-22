
import numpy as np

from ....core import Operator


class Mask(Operator):
    """
    Mask a StructuredData object to remove values outside inner domain.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, field, **kwargs):
        mask = np.zeros(field.extended_shape)
        mask[field.inner] = 1

        field *= mask

        return field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
