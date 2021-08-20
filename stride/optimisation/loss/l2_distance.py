
import numpy as np

import mosaic

from .functional import FunctionalValue
from ...core import Operator


__all__ = ['L2DistanceLoss']


@mosaic.tessera
class L2DistanceLoss(Operator):
    """
    L2-Norm of the difference between observed and modelled data:

    f = 1/2 ||modelled - observed||^2

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.residual = None

    async def forward(self, modelled, observed, **kwargs):
        problem = kwargs.pop('problem', None)

        residual_data = modelled.data-observed.data
        residual = modelled.alike(name='residual', data=residual_data)
        self.residual = residual

        fun_data = 0.5 * np.sum(residual.data ** 2)
        fun = FunctionalValue(fun_data, problem.shot_id, residual)

        return fun

    async def adjoint(self, d_fun, modelled, observed, **kwargs):
        grad_modelled = None
        if modelled.needs_grad:
            grad_modelled = +np.asarray(d_fun) * self.residual

        grad_observed = None
        if observed.needs_grad:
            grad_observed = -np.asarray(d_fun) * self.residual

        self.residual = None

        return grad_modelled, grad_observed
