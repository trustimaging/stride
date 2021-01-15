
import numpy as np

from .functional import FunctionalBase, FunctionalValue


__all__ = ['Functional']


class Functional(FunctionalBase):
    """
    L2-Norm of the difference between observed and modelled data:

    f = ||modelled - observed||^2

    """

    def apply(self, shot, modelled, observed):
        """
        Calculate the functional.

        Parameters
        ----------
        shot : Shot
            Shot for which the functional is calculated.
        modelled : Data
            Data of the modelled.
        observed : Data
            Data of the observed.

        Returns
        -------
        FunctionalValue
            Value of the functional and the residual.
        Data
            Adjoint source.

        """

        residual_data = adjoint_source_data = modelled.data-observed.data

        residual = modelled.alike('residual')
        adjoint_source = modelled.alike('adjoint_source')

        residual.data[:] = residual_data
        adjoint_source.data[:] = adjoint_source_data

        fun = np.sum(residual.data**2)
        fun = FunctionalValue(shot.id, fun, residual)

        return fun, adjoint_source
