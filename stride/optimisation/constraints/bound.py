
import numpy as np

from .constraint import Constraint


__all__ = ['Bound']


class Bound(Constraint):
    """
    Apply a bound constraint between some min and some max.

    Parameters
    ----------
    min : float, optional
        Min boundary.
    max : float, optional
        Max boundary.

    """

    def __init__(self, **kwargs):
        super().__init__()

        self.min = kwargs.pop('min', None)
        self.max = kwargs.pop('max', None)

    def project(self, variable, **kwargs):
        """
        Apply the projection.

        Parameters
        ----------
        variable : Variable
            Variable to project.
        min : float, optional
            Min boundary.
        max : float, optional
            Max boundary.

        Returns
        -------
        Variable
            Updated variable.

        """

        output = variable.copy()
        variable_data = output.extended_data

        min = kwargs.pop('min', self.min)
        max = kwargs.pop('max', self.max)

        if self.min is not None or self.max is not None:
            variable_data = np.clip(variable_data, min, max)

        output.extended_data[:] = variable_data

        return output
