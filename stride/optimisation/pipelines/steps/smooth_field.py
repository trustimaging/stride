
import scipy.ndimage

from ..pipeline import PipelineStep


class Step(PipelineStep):

    def __init__(self, **kwargs):
        self.sigma = kwargs.pop('sigma', 0.25)

    def apply(self, field, **kwargs):
        field.extended_data[:] = scipy.ndimage.gaussian_filter(field.extended_data,
                                                               sigma=self.sigma, mode='nearest')

        return field
