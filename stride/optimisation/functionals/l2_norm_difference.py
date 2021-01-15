
import numpy as np

from .functional import FunctionalBase, FunctionalValue


__all__ = ['Functional']


class Functional(FunctionalBase):

    def apply(self, shot, modelled, observed):
        residual_data = adjoint_source_data = modelled.data-observed.data

        residual = modelled.alike('residual')
        adjoint_source = modelled.alike('adjoint_source')

        residual.data[:] = residual_data
        adjoint_source.data[:] = adjoint_source_data

        fun = np.sum(residual.data**2)
        fun = FunctionalValue(shot.id, fun, residual)

        return fun, adjoint_source
