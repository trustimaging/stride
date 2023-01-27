
import numpy as np

from .constraint import Constraint


__all__ = ['ProximalTVp']


class ProximalTVp(Constraint):
    """
    Apply a total-variation constraint of order p using proximal operators
    provided by the package prox_tv.

    Parameters
    ----------
    weight : float or tuple, optional
        Weight given to TV constraint in each direction.
    axes : tuple, optional
        List of axes across which to apply the projection.
    order : int or tuple, optional
        Order of the TV norm to use.
    n_threads : int, optional
        Number of C threads to use.
    max_iter : int, optional
        Maximum number of denoising iterations.

    """

    def __init__(self, **kwargs):
        super().__init__()

        self.weight = kwargs.pop('weight', 0.1)
        self.axes = kwargs.pop('axes', None)
        self.order = kwargs.pop('order', 1)
        self.n_threads = kwargs.pop('n_threads', 1)
        self.max_iter = kwargs.pop('max_iter', 0)

    def project(self, variable, **kwargs):
        """
        Apply the projection.

        Parameters
        ----------
        variable : Variable
            Variable to project.
        weight : float or tuple, optional
            Weight given to TV constraint in each direction.
        axes : tuple, optional
            List of axes across which to apply the projection.
        order : int or tuple, optional
            Order of the TV norm to use.
        n_threads : int, optional
            Number of C threads to use.
        max_iter : int, optional
            Maximum number of denoising iterations.

        Returns
        -------
        Variable
            Updated variable.

        """
        try:
            import prox_tv
        except ImportError:
            raise RuntimeError('proximal_TVp requires prox_tv to execute')

        output = variable.copy()
        variable_data = output.extended_data

        weight = kwargs.pop('weight', self.weight)
        axes = kwargs.pop('axes', self.axes)
        order = kwargs.pop('order', self.order)
        n_threads = kwargs.pop('n_threads', self.n_threads)
        max_iter = kwargs.pop('max_iter', self.max_iter)

        if isinstance(axes, (tuple, list)):
            axes = [ax+1 for ax in axes]
        elif axes is None:
            axes = list(range(1, variable_data.ndim+1))

        ndim = len(axes)

        if isinstance(weight, tuple):
            weight = list(weight)
        elif not isinstance(weight, list):
            weight = [weight]*ndim
        weight = [w*2. for w in weight]

        if isinstance(order, tuple):
            order = list(order)
        elif not isinstance(order, list):
            order = [order]*ndim

        data_norm = np.linalg.norm(variable_data)

        # first pass
        variable_data = prox_tv.tvgen(variable_data, weight, axes, order,
                                      n_threads=n_threads, max_iters=max_iter)
        variable_data *= data_norm / np.linalg.norm(variable_data)

        for i in axes:
            variable_data = np.flip(variable_data, axis=i-1)

        # second pass
        variable_data = prox_tv.tvgen(variable_data, weight, axes, order,
                                      n_threads=n_threads, max_iters=max_iter)
        variable_data *= data_norm / np.linalg.norm(variable_data)

        for i in reversed(axes):
            variable_data = np.flip(variable_data, axis=i-1)

        output.extended_data[:] = variable_data

        return output
