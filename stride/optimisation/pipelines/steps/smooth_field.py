
import scipy.ndimage

from ....core import Operator


class SmoothField(Operator):
    """
    Apply Gaussian smoothing to a StructuredData object.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the Gaussian kernel, defaults to 0.25 (25% of a grid point).

    """

    def __init__(self, **kwargs):
        super().__init__(*kwargs)

        self.sigma = kwargs.pop('sigma', 0.25)

    def forward(self, field, **kwargs):
        out_field = field.alike(name='smoothed_%s' % field.name)

        out_field.extended_data[:] = scipy.ndimage.gaussian_filter(field.extended_data,
                                                                   sigma=self.sigma, mode='nearest')

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
