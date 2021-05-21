
from ...core import Variable


__all__ = ['FunctionalValue']


class FunctionalValue(Variable):
    """
    Container class for the calculated functional value and the residuals.

    Parameters
    ----------
    shot_id : int
        ID of the shot for which the value has been calculated.
    fun_value : float
        Scalar value of the functional.
    residuals : Data
        Calculated residuals.

    """

    def __init__(self, shot_id, fun_value, residuals, **kwargs):
        super().__init__(**kwargs)

        self.shot_id = shot_id
        self.fun_value = fun_value
        self.residuals = residuals

    def __repr__(self):
        return 'loss %e for shot %d' % (self.fun_value, self.shot_id)
