
import scipy.ndimage

from ..pipeline import PipelineStep


class SmoothField(PipelineStep):
    """
    Apply Gaussian smoothing to a StructuredData object.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the Gaussian kernel, defaults to 0.25 (25% of a grid point).

    """

    def __init__(self, **kwargs):
        self.sigma = kwargs.pop('sigma', 0.25)

    def apply(self, field, **kwargs):
        field.extended_data[:] = scipy.ndimage.gaussian_filter(field.extended_data,
                                                               sigma=self.sigma, mode='nearest')

        return field
