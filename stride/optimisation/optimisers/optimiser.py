
from abc import ABC, abstractmethod


__all__ = ['LocalOptimiser']


class LocalOptimiser(ABC):
    """
    Base class for a local optimiser. It takes the value of the gradient and applies
    it to the variable.

    Parameters
    ----------
    variable : Variable
        Variable to which the optimiser refers.
    kwargs
        Extra parameters to be used by the class.

    """

    def __init__(self, variable, **kwargs):
        self.variable = variable

    @abstractmethod
    def apply(self, grad, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        grad : Data
            Gradient to apply.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        pass
