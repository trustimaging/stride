
import numpy as np

from .utils import name_from_op_name
from ....core import Operator


def _rampoff_mask(shape, ramp_size):
    mask = np.ones(shape, dtype=np.float32)

    for dim_i in range(len(shape)):
        if 2*ramp_size > shape[dim_i]:
            continue
        for index in range(ramp_size):
            pos = np.abs((ramp_size - index - 1) / float(ramp_size - 1))
            val = 1 - np.cos(np.pi / 2 * (1 - pos))

            # : slices
            all_ind = [slice(index, s - index + 1) for s in shape]

            # Left slice
            all_ind[dim_i] = index
            mask[tuple(all_ind)] = val

            # : slices
            all_ind = [slice(index, s - index + 1) for s in shape]

            # right slice
            all_ind[dim_i] = -index
            mask[tuple(all_ind)] = val

    return mask


class MaskField(Operator):
    """
    Mask a StructuredData object to remove values outside inner domain.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_rampoff = kwargs.pop('mask_rampoff', 10)
        self._mask = kwargs.pop('mask', None)

    def forward(self, field, **kwargs):
        mask = kwargs.pop('mask', None)
        mask_rampoff = kwargs.pop('mask_rampoff', self.mask_rampoff)
        mask = self._mask if mask is None else mask
        if mask is None or np.any([m != f for m, f in zip(mask.shape, field.extended_shape)]):
            mask = np.zeros(field.extended_shape, dtype=np.float32)
            mask[field.inner] = 1
            mask *= _rampoff_mask(mask.shape, mask_rampoff)
            self._mask = mask

        out_field = field.alike(name=name_from_op_name(self, field))
        out_field.extended_data[:] = field.extended_data
        out_field.extended_data[:] *= mask

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
