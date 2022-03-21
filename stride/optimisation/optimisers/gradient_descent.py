
import numpy as np

import mosaic

from .optimiser import LocalOptimiser


__all__ = ['GradientDescent']


class GradientDescent(LocalOptimiser):
    """
    Implementation of a gradient descent update.

    Parameters
    ----------
    variable : Variable
        Variable to which the optimiser refers.
    step : float, optional
        Step size for the update, defaults to 1.
    kwargs
        Extra parameters to be used by the class.

    """

    def __init__(self, variable, **kwargs):
        super().__init__(variable, **kwargs)

        self.step_size = kwargs.pop('step_size', 1.)

    async def step(self, step_size=None, grad=None, processed_grad=None, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        step_size : float, optional
            Step size to use for this application, defaults to instance step.
        grad : Data, optional
            Gradient to use for the step, defaults to variable gradient.
        processed_grad : Data, optional
            Processed gradient to use for the step, defaults to processed variable gradient.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        step_size = self.step_size if step_size is None else step_size

        processed_grad = await self.pre_process(grad=grad,
                                                processed_grad=processed_grad,
                                                **kwargs)
        direction = processed_grad

        self.variable -= step_size*direction

        await self.post_process(**kwargs)

        return self.variable
