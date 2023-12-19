
from ...problem import Scalar


__all__ = ['FunctionalValue']


class FunctionalValue(Scalar):
    """
    Container class for the calculated functional value and the residuals.

    Parameters
    ----------
    fun_value : float
        Scalar value of the functional.
    shot_id : int
        ID of the shot for which the value has been calculated.
    residuals : Data
        Calculated residuals.

    """

    def __init__(self, fun_value, shot_id, residuals=None, **kwargs):
        super().__init__(**kwargs)
        self.data[:] = fun_value

        self.shot_id = shot_id
        self.fun_value = fun_value
        if kwargs.pop('keep_residual', False):
            self.residuals = residuals

    def __repr__(self):
        return 'loss %e for shot %d' % (self.fun_value, self.shot_id)
