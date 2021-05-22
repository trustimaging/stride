
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

    async def step(self, step_size=None, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        step_size : float, optional
            Step size to use for this application, defaults to instance step.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        step_size = step_size or self.step_size
        grad = self.variable.process_grad(**kwargs)
        grad = await self._process_grad(grad)

        min_grad = np.min(grad.extended_data)
        max_grad = np.max(grad.extended_data)

        min_var = np.min(self.variable.extended_data)
        max_var = np.max(self.variable.extended_data)

        runtime = mosaic.runtime()
        runtime.logger.info('Updating variable %s, gradient in range [%e, %e]' %
                            (self.variable.name, min_grad, max_grad))
        runtime.logger.info('\t variable range before update [%e, %e]' %
                            (min_var, max_var))

        self.variable -= step_size*grad
        self.variable = await self._process_model(self.variable)

        min_var = np.min(self.variable.extended_data)
        max_var = np.max(self.variable.extended_data)

        runtime.logger.info('\t variable range after update [%e, %e]' %
                            (min_var, max_var))

        return self.variable
