
# Some sort of iteration-wide object contains statistics about that iteration
# It contains an iteration-wide functional value, it also contains the functional
# values of all shots inside. Per-shot information also includes the residuals
# for that iteration (probably only temporarily).
# We might also want to store some other stuff, such as gradients, maybe only the
# last one as well
# We might also want to contain these iteration objects somewhere else?
# For example, the Optimisation. That object tracks and saves to disk the progress of
# the inversion. It will be the place where statistics about each iteration, each block,
# configuration, and completed blocks and iterations are preserved.

# Should we create a functional instance at the functional level, and therefore
# use it at the runner level, or should we do something else at the iteration level
# Or actually maybe both?
# At the runner level we construct the output of a functional for a certain shot
# At the iteration level, we group all of these into one


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
