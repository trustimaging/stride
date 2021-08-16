
import numpy as np


class Dykstra:
    """
    Applies Dykstra's projection algorithm, finding the closest point
    (in the Euclidean sense) that falls within the intersection of
    a series of (theoretically) convex constraints.

    The loss function used is 1/2 ||variable - variable_projection||^2.

    Parameters
    ----------
    constraints : array of Constraint
        Array of constraints for which to find the intersection
    min_iter : int, optional
        Minimum number of iterations in the loop, defaults to ``1``.
    max_iter : int, optional
        Maximum number of iterations in the loop, defaults to ``1000``.
    tol : float, optional
        Tolerance to determine convergence. The algorithm will stop
        if the the relative change in the loss function is below this
        value. Defaults to ``1e-6``.

    """

    def __init__(self, *constraints, **kwargs):
        self.constraints = constraints

        self.min_iter = max(1, kwargs.pop('min_iter', 1))
        self.max_iter = kwargs.pop('max_iter', 1000)
        self.tol = kwargs.pop('tol', 1e-6)

    def step(self, variable, **kwargs):
        """
        Apply the algorithm on a given variable.

        Parameters
        ----------
        variable : Variable
            Variable on which the algorithm will be applied.
        min_iter : int, optional
            Minimum number of iterations in the loop, defaults to ``1``.
        max_iter : int, optional
            Maximum number of iterations in the loop, defaults to ``1000``.
        tol : float, optional
            Tolerance to determine convergence. The algorithm will stop
            if the the relative change in the loss function is below this
            value. Defaults to ``1e-6``.

        Returns
        -------

        """

        min_iter = max(1, kwargs.pop('min_iter', self.min_iter))
        max_iter = kwargs.pop('max_iter', self.max_iter)
        tol = kwargs.pop('tol', self.tol)

        num_constraints = len(self.constraints)

        x = variable.copy()
        x_0 = x.copy()
        y = [np.zeros(x.extended_shape) for _ in range(num_constraints)]

        loss = []

        for it in range(max_iter):

            for p in range(num_constraints):
                # Update iterate
                prev_x = x.copy()
                x = self.constraints[p].project(prev_x - y[p])

                # Update increment
                prev_y = y[p].copy()
                y[p][:] = (x - (prev_x - prev_y)).extended_data

            # Calculate loss
            loss.append(0.5 * np.sum((x_0.extended_data - x.extended_data)**2))

            # Continue if we have not reached min iterations
            if it < min_iter:
                continue

            # Check stopping criteria
            loss_delta = abs(loss[it] - loss[it-1]) / loss[it]
            if loss_delta < tol:
                break

        return x
