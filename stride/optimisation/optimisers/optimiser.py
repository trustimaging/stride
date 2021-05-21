
from abc import ABC, abstractmethod


__all__ = ['LocalOptimiser']


async def noop(*args, **kwargs):
    return args


class LocalOptimiser(ABC):
    """
    Base class for a local optimiser. It takes the value of the gradient and applies
    it to the variable.

    Parameters
    ----------
    variable : Variable
        Variable to which the optimiser refers.
    process_grad : callable, optional
        Optional processing function to apply on the gradient prior to applying it.
    process_model : callable, optional
        Optional processing function to apply on the model after updating it.
    kwargs
        Extra parameters to be used by the class.

    """

    def __init__(self, variable, **kwargs):
        self.variable = variable
        self._process_grad = kwargs.pop('process_grad', noop)
        self._process_model = kwargs.pop('process_model', noop)

    def clear_grad(self):
        """
        Clear the internal gradient buffers of the varible.

        Returns
        -------

        """
        self.variable.clear_grad()

    @abstractmethod
    def step(self, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        pass
