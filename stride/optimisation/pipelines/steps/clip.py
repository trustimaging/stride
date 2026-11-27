
import numpy as np

from .utils import name_from_op_name
from ....core import Operator


class Clip(Operator):
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
        super().__init__(**kwargs)

        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)

    def forward(self, field, **kwargs):
        out_field = field.alike(name=name_from_op_name(self, field))
        out_field.extended_data[:] = field.extended_data

        if self.min is not None or self.max is not None:
            out_field.extended_data[:] = np.clip(field.extended_data,
                                                 self.min, self.max)

        print('Clip max {} min {}'.format(self.max, self.min))

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
