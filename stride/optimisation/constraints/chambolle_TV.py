
from skimage.restoration import denoise_tv_chambolle

from .constraint import Constraint


__all__ = ['ChambolleTV']


class ChambolleTV(Constraint):
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

        variable_data = denoise_tv_chambolle(variable_data, weight,
                                             eps=tol, n_iter_max=max_iter)

        return variable_data
