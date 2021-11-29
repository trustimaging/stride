
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

        self._mask = kwargs.pop('mask', None)

    def forward(self, field, **kwargs):
        if self._mask is None:
            self._mask = np.zeros(field.extended_shape)
            self._mask[field.inner] = 1
        mask = self._mask

        out_field = field.alike(name='masked_%s' % field.name)
        out_field.extended_data[:] = field.extended_data
        out_field *= mask

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
