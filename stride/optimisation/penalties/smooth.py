
import numpy as np

import mosaic

from ...core import Operator
from ...problem import Scalar


__all__ = ['Smooth']


@mosaic.tessera
class Smooth(Operator):
    """
    Smoothing constraint:

    f = 1/2 ||variable||^2

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, variable, **kwargs):
        fun_data = 0.5 * np.sum(variable.extended_data ** 2)
        fun = Scalar()
        fun.data[:] = fun_data

        return fun

    def adjoint(self, d_fun, variable, **kwargs):
        grad_variable = np.asarray(d_fun) * variable

        return grad_variable
