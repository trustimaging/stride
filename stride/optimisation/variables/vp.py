
import numpy as np

from stride.problem_definition import ScalarField


__all__ = ['Vp']


class Vp(ScalarField):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grad = ScalarField(self.name + '_grad', grid=self.grid)
        self.prec = ScalarField(self.name + '_prec', grid=self.grid)

        self.grad.fill(0.)
        self.prec.fill(1.)

    def update_problem(self, problem):
        problem.medium[self.name].extended_data[:] = self.extended_data.copy()

        return problem

    def get_grad(self):
        grad = self.grad
        prec = self.prec

        # TODO is this the best value?
        prec += 1e-6 * np.max(prec.data)
        grad /= prec

        return grad
