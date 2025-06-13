
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from .constraint import Constraint


__all__ = ['ChambolleTV1']


class ChambolleTV1(Constraint):
    """
    Apply a total-variation constraint using Chambolle's projection
    algorithm.

    Parameters
    ----------
    weight : float, optional
        Denoising weight.
    max_iter : int, optional
        Maximum number of denoising iterations.
    tol : float, optional
        Relative tolerance in loss function change.

    """

    def __init__(self, **kwargs):
        super().__init__()

        self.weight = kwargs.pop('weight', 0.1)
        self.max_iter = kwargs.pop('max_iter', 200)
        self.tol = kwargs.pop('tol', 2e-4)
        self.axis = kwargs.pop('axis', None)

    def project(self, variable, axis=None, **kwargs):
        """
        Apply the projection.

        Parameters
        ----------
        variable : Variable
            Variable to project.
        weight : float, optional
            Denoising weight.
        max_iter : int, optional
            Maximum number of denoising iterations.
        tol : float, optional
            Relative tolerance in loss function change.
        axis : int, optional
            Axis across which to make the projection, defaults to none.

        Returns
        -------
        Variable
            Updated variable.

        """

        output = variable.copy()
        variable_data = output.extended_data

        axis = axis or self.axis

        if axis is not None:
            for i in range(variable_data.shape[axis]):
                slice_i = [slice(0, None) for _ in variable_data.shape]
                slice_i[axis] = i
                slice_i = tuple(slice_i)

                variable_data[slice_i] = self._project(variable_data[slice_i], **kwargs)

        else:
            variable_data = self._project(variable_data, **kwargs)

        output.extended_data[:] = variable_data

        return output

    def _project(self, variable_data, **kwargs):
        weight = kwargs.pop('weight', self.weight)
        max_iter = kwargs.pop('max_iter', self.max_iter)
        tol = kwargs.pop('tol', self.tol)

        norm_before = np.sqrt(np.sum(variable_data**2)) + 1e-31

        variable_data = denoise_tv_chambolle(variable_data, weight,
                                             eps=tol, max_num_iter=max_iter)

        norm_after = np.sqrt(np.sum(variable_data ** 2)) + 1e-31

        variable_data *= norm_before/norm_after

        return variable_data
