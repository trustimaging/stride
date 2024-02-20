
import numpy as np

from .utils import name_from_op_name
from ....core import Operator


class MaskField(Operator):
    """
    Mask a StructuredData object to remove values outside inner domain.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mask = kwargs.pop('mask', None)

    def forward(self, field, **kwargs):
        mask = kwargs.pop('mask', self._mask)
        if mask is None:
            mask = np.zeros(field.extended_shape)
            mask[field.inner] = 1

        out_field = field.alike(name=name_from_op_name(self, field))
        out_field.extended_data[:] = field.extended_data
        out_field *= mask

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
