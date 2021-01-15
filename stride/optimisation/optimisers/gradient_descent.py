

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
    kwargs : dict
        Extra parameters to be used by the class.

    """

    def __init__(self, variable, **kwargs):
        super().__init__(variable, **kwargs)

        self.step = kwargs.pop('step', 1.)

    def apply(self, grad, **kwargs):
        """
        Apply the optimiser.

        Parameters
        ----------
        grad : Data
            Gradient to apply.
        step : float, optional
            Step size to use for this application, defaults to instance step.
        kwargs : dict
            Extra parameters to be used by the method.

        Returns
        -------
        Variable
            Updated variable.

        """
        step = kwargs.get('step', self.step)

        self.variable -= step*grad

        return self.variable
