

from .optimiser import LocalOptimiser


__all__ = ['GradientDescent']


class GradientDescent(LocalOptimiser):
    """
    Implementation of a gradient descent update.

    Parameters
    ----------
    variable : Variable
        Variable to which the optimiser refers.
    step_size : float, optional
        Step size for the update, defaults to 1.
    kwargs
        Extra parameters to be used by the class.

    """

    def __init__(self, variable, **kwargs):
        super().__init__(variable, **kwargs)

    async def pre_process(self, grad=None, processed_grad=None, **kwargs):
        processed_grad = await super().pre_process(grad=grad,
                                                   processed_grad=processed_grad,
                                                   **kwargs)
        return processed_grad

    def update_variable(self, step_size, variable, direction):
        variable.data[:] -= step_size * direction.data
        return variable
