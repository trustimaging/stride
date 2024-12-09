
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

    Parameters
    ----------
    d_sample : int, optional
        Factor by which the time dimension of the traces are downsampled prior to calculating
        optimal assignments. Defaults to 4.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.residual = None

        # undersampling
        self.d_sample = kwargs.get('d_sample', 4)

    def forward(self, modelled, observed, **kwargs):
        problem = kwargs.pop('problem', None)
        shot_id = problem.shot_id if problem is not None \
            else kwargs.pop('shot_id', 0)

        residual_data = modelled.data-observed.data
        residual_data_sampled = residual_data[:, ::self.d_sample]

        residual = observed.alike(name='residual', data=residual_data)
        residual_sampled = observed.alike(name='residual', data=residual_data_sampled,
                                          shape=residual_data_sampled.shape,
                                          extended_shape=residual_data_sampled.shape)
        self.residual = residual

        fun_data = 0.5 * np.sum(residual.data ** 2)
        fun = FunctionalValue(fun_data, shot_id, residual_sampled, **kwargs)

        return fun

    def adjoint(self, d_fun, modelled, observed, **kwargs):
        grad_modelled = None
        if modelled.needs_grad:
            grad_modelled = +np.asarray(d_fun) * self.residual.copy(name='modelledresidual')

        grad_observed = None
        if observed.needs_grad:
            grad_observed = -np.asarray(d_fun) * self.residual.copy(name='observedresidual')

        self.residual = None

        return grad_modelled, grad_observed
