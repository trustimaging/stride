
from ...problem import Scalar


__all__ = ['FunctionalValue']


class FunctionalValue(Scalar):
    """
    Container class for the calculated functional value and the residuals.

    Parameters
    ----------
    value : float
        Scalar value of the functional.
    shot_id : int
        ID of the shot for which the value has been calculated.
    residuals : Data
        Calculated residuals.

    """

    def __init__(self, value, shot_id, residuals=None, **kwargs):
        super().__init__(**kwargs)
        self.data[:] = value

        self.shot_id = shot_id
        self.value = value
        if kwargs.pop('keep_residual', False):
            self.residuals = residuals

    def __repr__(self):
        return 'loss %e for shot %d' % (self.value, self.shot_id)
