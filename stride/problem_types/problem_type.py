
from abc import ABC, abstractmethod


__all__ = ['ProblemTypeBase']


class ProblemTypeBase(ABC):

    space_order = -1
    time_order = -1
    undersampling_factor = 1

    def __init__(self):
        self._problem = None

        self._state_operator = None
        self._adjoint_operator = None

    def set_problem(self, problem):
        self._problem = problem

    def set_grad(self, wrt):
        gradient_update = []

        for variable in wrt:
            method = getattr(self, 'set_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            update = method(variable)
            gradient_update += update

        return gradient_update

    def get_grad(self, wrt):
        for variable in wrt:
            method = getattr(self, 'get_grad_' + variable.name, None)

            if method is None:
                raise ValueError('Variable %s not implemented' % variable.name)

            method(variable)

        return wrt

    @abstractmethod
    def before_state(self, save_wavefield=False):
        pass

    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def after_state(self, save_wavefield=False):
        pass

    @abstractmethod
    def before_adjoint(self, wrt, adjoint_source, wavefield):
        pass

    @abstractmethod
    def adjoint(self):
        pass

    @abstractmethod
    def after_adjoint(self, wrt):
        pass
