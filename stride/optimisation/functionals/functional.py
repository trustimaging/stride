

__all__ = ['FunctionalBase', 'FunctionalValue']


class FunctionalBase:

    # return fun, residual, adjoint_source
    def apply(self, shot, modelled, observed):
        pass

    def gradient(self, variables):
        return variables


class FunctionalValue:

    def __init__(self, shot_id, fun_value, residuals):
        self.shot_id = shot_id
        self.fun_value = fun_value
        self.residuals = residuals

    def __repr__(self):
        return 'loss %e for shot %d' % (self.fun_value, self.shot_id)
