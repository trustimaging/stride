
import numpy as np

from stride.problem_definition import ScalarField


__all__ = ['Vp']


class Vp(ScalarField):
    """
    Class representing longitudinal speed of sound. The scalar field is enriched with a gradient and
    a preconditioner.

    For reference on the arguments see `~stride.problem_definition.data.ScalarField`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grad = ScalarField(self.name + '_grad', grid=self.grid)
        self.prec = ScalarField(self.name + '_prec', grid=self.grid)

        self.grad.fill(0.)
        self.prec.fill(1.)

    def update_problem(self, problem):
        """
        Use the current value of the variable to update the problem.

        Parameters
        ----------
        problem : Problem or SubProblem
            Problem to update

        Returns
        -------
        Problem or SubProblem
            Updated problem.

        """
        problem.medium[self.name].extended_data[:] = self.extended_data.copy()

        return problem

    def get_grad(self):
        """
        Process the accumulated gradient and preconditioner to produce a final gradient.

        Returns
        -------
        Data
            Final gradient.

        """
        grad = self.grad
        prec = self.prec

        prec += 1e-6 * np.max(prec.data) + 1e-31
        grad /= prec

        return grad
