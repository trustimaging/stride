
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

        self.global_norm = kwargs.pop('global_norm', False)
        self.norm_value = None

    def forward(self, field, **kwargs):
        global_norm = kwargs.pop('global_norm', self.global_norm)

        if self.norm_value is None or not global_norm:
            self.norm_value = np.max(np.abs(field.extended_data)) + 1e-31

        out_field = field.alike(name='normed_%s' % field.name)
        out_field.extended_data[:] = field.extended_data
        out_field.extended_data[:] /= self.norm_value

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
