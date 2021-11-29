
import numpy as np

import mosaic

from ...core import Operator
from ...problem import Scalar


__all__ = ['SmoothTime']


@mosaic.tessera
class SmoothTime(Operator):
    """
    Time smoothing constraint:

    f = 1/2 ||d variable / dt||^2

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.residual = None
        self.spacing = kwargs.pop('spacing', None)

    async def forward(self, variable, **kwargs):
        grad = self._grad(variable, **kwargs)
        self.residual = grad

        fun_data = 0.5 * np.sum(grad.extended_data ** 2)
        fun = Scalar()
        fun.data[:] = fun_data

        return fun

    async def adjoint(self, d_fun, variable, **kwargs):
        grad = self._grad(self.residual, **kwargs)
        grad_variable = -np.asarray(d_fun) * grad

        self.residual = None

        return grad_variable

    def _grad(self, variable, **kwargs):
        axis = kwargs.pop('axis', 0)
        dt = kwargs.pop('spacing', self.spacing)

        if dt is None:
            if variable.time_dependent:
                dt = variable.time.step
            elif variable.slow_time_dependent:
                dt = variable.slow_time.step
            else:
                raise ValueError('spacing needs to be provided')

        variable_data = variable.extended_data

        grad_data = np.gradient(variable_data, dt,
                                axis=axis, edge_order=1)*dt
        grad = variable.alike(name='derivative', data=grad_data)

        return grad
