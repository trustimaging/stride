
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

    async def step(self, step_size=None, direction=None, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        step_size : float, optional
            Step size to use for this application, defaults to instance step.
        direction : Data, optional
            Direction to use for the step, defaults to variable gradient.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        step_size = step_size or self.step_size

        if direction is None:
            grad = self.variable.process_grad(**kwargs)
            direction = await self._process_grad(grad)

        min_dir = np.min(direction.extended_data)
        max_dir = np.max(direction.extended_data)

        min_var = np.min(self.variable.extended_data)
        max_var = np.max(self.variable.extended_data)

        runtime = mosaic.runtime()
        runtime.logger.info('Updating variable %s, direction in range [%e, %e]' %
                            (self.variable.name, min_dir, max_dir))
        runtime.logger.info('\t variable range before update [%e, %e]' %
                            (min_var, max_var))

        self.variable -= step_size*direction
        self.variable = await self._process_model(self.variable)

        min_var = np.min(self.variable.extended_data)
        max_var = np.max(self.variable.extended_data)

        runtime.logger.info('\t variable range after update [%e, %e]' %
                            (min_var, max_var))

        return self.variable
