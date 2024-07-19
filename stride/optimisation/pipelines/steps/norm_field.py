
import numpy as np

from .utils import name_from_op_name
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
        self.norm_guess_change = kwargs.pop('norm_guess_change', 0.5)
        self.norm_value = None

    def forward(self, field, **kwargs):
        variable = kwargs.pop('variable', None)
        global_norm = kwargs.pop('global_norm', self.global_norm)
        norm_guess_change = kwargs.pop('norm_guess_change', self.norm_guess_change)

        if self.norm_value is None or not global_norm:
            self.norm_value = np.max(np.abs(field.extended_data)) + 1e-31

        # work out guess change based on field value
        if variable is not None:
            min_val = np.min(variable.extended_data)
            max_val = np.max(variable.extended_data)

            mid_val = (max_val + min_val) / 2.0
            if variable.transform is not None:
                mid_val = variable.transform(mid_val)
            var_corr = mid_val * norm_guess_change / 100
        else:
            var_corr = 1.

        out_field = field.alike(name=name_from_op_name(self, field))
        out_field.extended_data[:] = field.extended_data
        out_field.extended_data[:] *= var_corr / self.norm_value

        return out_field

    def adjoint(self, d_field, field, **kwargs):
        raise NotImplementedError('No adjoint implemented for step %s' % self.__class__.__name__)
