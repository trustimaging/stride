
from abc import ABC, abstractmethod


__all__ = ['FunctionalBase', 'FunctionalValue']


class FunctionalBase(ABC):
    """
    Base class for the implementation of functionals or loss functions. A functional calculates
    a scalar value given some modelled and some observed data, as well as the residual and the adjoint source.

    """

    @abstractmethod
    def apply(self, shot, modelled, observed, **kwargs):
        """
        Calculate the functional.

        Parameters
        ----------
        shot : Shot
            Shot for which the functional is calculated.
        modelled : Data
            Data of the modelled.
        observed : Data
            Data of the observed.

        Returns
        -------
        FunctionalValue
            Value of the functional and the residual.
        Data
            Adjoint source.

        """
        pass

    def get_grad(self, variables, **kwargs):
        """
        The functional might contain components of the gradient that need to be calculated.

        Parameters
        ----------
        variables : VariableList
            Updated list of variables.

        Returns
        -------

        """
        return variables


class FunctionalValue:
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

    def __init__(self, shot_id, fun_value, residuals):
        self.shot_id = shot_id
        self.fun_value = fun_value
        self.residuals = residuals

    def __repr__(self):
        return 'loss %e for shot %d' % (self.fun_value, self.shot_id)
