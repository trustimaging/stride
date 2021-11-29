
import numpy as np

from .constraint import Constraint


__all__ = ['Thresholding']


class Thresholding(Constraint):
    """
    Apply a sparsity constraint through thresholding.

    Parameters
    ----------
    value : float
        Thresholding value.
    mode : str, optional
        Type of thresholding, from ``soft``, ``hard``, ``garrote``.
        Default is ``soft``.
    substitute : float, optional
        Substitute value, defaults to 0.

    """

    def __init__(self, value, **kwargs):
        super().__init__()

        self.value = value
        self.mode = kwargs.pop('mode', 'soft')
        self.substitute = kwargs.pop('substitute', 0.)

    def project(self, variable, **kwargs):
        """
        Apply the projection.

        Parameters
        ----------
        variable : Variable
            Variable to project.
        value : float
            Thresholding value.
        mode : str, optional
            Type of thresholding, from ``soft``, ``hard``, ``garrote``.
            Default is ``soft``.
        substitute : float, optional
            Substitute value, defaults to 0.

        Returns
        -------
        Variable
            Updated variable.

        """

        value = kwargs.pop('value', self.value)
        mode = kwargs.pop('mode', self.mode)
        substitute = kwargs.pop('substitute', self.substitute)

        output = variable.copy()
        variable_data = output.extended_data

        if mode == 'soft':
            variable_data = self.soft(variable_data, value, substitute)
        elif mode == 'garrote':
            variable_data = self.nn_garrote(variable_data, value, substitute)
        elif mode == 'hard':
            variable_data = self.hard(variable_data, value, substitute)
        else:
            raise ValueError('Thresholding mode should be soft, hard or garrote')

        output.extended_data[:] = variable_data

        return output

    @staticmethod
    def soft(data, value, substitute=0):
        data = np.asarray(data)
        magnitude = np.absolute(data)

        with np.errstate(divide='ignore'):
            # divide by zero okay as np.inf values get clipped, so ignore warning.
            thresholded = (1 - value / (magnitude + 1e-31))
            thresholded.clip(min=0, max=None, out=thresholded)
            thresholded = data * thresholded

        if substitute == 0:
            return thresholded
        else:
            cond = np.less(magnitude, value)
            return np.where(cond, substitute, thresholded)

    @staticmethod
    def nn_garrote(data, value, substitute=0):
        data = np.asarray(data)
        magnitude = np.absolute(data)

        with np.errstate(divide='ignore'):
            # divide by zero okay as np.inf values get clipped, so ignore warning.
            thresholded = (1 - value ** 2 / (magnitude ** 2 + 1e-31))
            thresholded.clip(min=0, max=None, out=thresholded)
            thresholded = data * thresholded

        if substitute == 0:
            return thresholded
        else:
            cond = np.less(magnitude, value)
            return np.where(cond, substitute, thresholded)

    @staticmethod
    def hard(data, value, substitute=0):
        data = np.asarray(data)
        cond = np.less(np.absolute(data), value)

        return np.where(cond, substitute, data)
