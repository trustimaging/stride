
import numpy as np
import scipy.ndimage

from .utils import name_from_op_name
from ....core import Operator


class SmoothField(Operator):
    """
    Apply Gaussian smoothing to a StructuredData object.

    Parameters
    ----------
    smooth_sigma : float, optional
        Standard deviation of the Gaussian kernel, defaults to 0.25 (25% of a cell).

    """

    def __init__(self, **kwargs):
        super().__init__(*kwargs)

        self.sigma = kwargs.pop('smooth_sigma', 0.25)

    def forward(self, field, **kwargs):
        space = field.space
        dim = space.dim if space is not None else field.ndim

        sigma = kwargs.pop('smooth_sigma', self.sigma)
        if not np.iterable(sigma):
            sigma = (sigma,) * dim

        if all(s <= 0 for s in sigma):
            return field

        axes_offset = field.ndim - dim
        axes = tuple(a + axes_offset for a in range(dim))

        out_field = field.alike(name=name_from_op_name(self, field))
        output = scipy.ndimage.gaussian_filter(field.extended_data,
                                               sigma=sigma, axes=axes, mode='nearest')
        out_field.extended_data[:] = output

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
