

from .optimiser import LocalOptimiser


__all__ = ['GradientDescent']


class GradientDescent(LocalOptimiser):

    def __init__(self, variable, **kwargs):
        super().__init__(variable, **kwargs)

        self.step = kwargs.pop('step', 1.)

    def apply(self, grad, **kwargs):
        step = kwargs.get('step', self.step)

        self.variable -= step*grad

        return self.variable
