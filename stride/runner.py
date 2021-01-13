

from mosaic import tessera


__all__ = ['Runner']


@tessera
class Runner:

    def __init__(self):
        self.problem = None
        self.block = None

        self.problem_type = None
        self.functional = None

    def set_problem(self, problem):
        self.logger.info('(ShotID %d) Preparing to run shot' % problem.shot_id)

        self.problem = problem

        if self.problem_type is None or self.problem_type.__class__ != problem.problem_type:
            self.problem_type = problem.problem_type()

        if self.block is not None:
            wavelets = self.problem.shot.wavelets
            wavelets = self.block.pipelines.wavelets.apply(wavelets)
            self.problem.shot.wavelets = wavelets

            observed = self.problem.shot.observed
            observed = self.block.pipelines.wavelets.apply(observed)
            self.problem.shot.observed = observed

        self.problem_type.set_problem(problem)

    def set_block(self, block):
        self.logger.info('Preparing to run block %d' % block.id)

        self.block = block

        if self.functional is None or self.functional.__class__ != block.functional:
            self.functional = block.functional()

    def run_state(self, save_wavefield=False):
        self.problem_type.before_state(save_wavefield=save_wavefield)

        self.logger.info('(ShotID %d) Running state equation for shot' % self.problem.shot_id)
        self.problem_type.state()

        self.logger.info('(ShotID %d) Completed state equation run for shot' % self.problem.shot_id)
        traces, wavefield = self.problem_type.after_state(save_wavefield=save_wavefield)

        if save_wavefield is True and self.block is not None:
            wavefield = self.block.pipelines.wavefield.apply(wavefield)

        return traces, wavefield

    def run_functional(self, modelled):
        if self.functional is None:
            raise ValueError('No functional was given to the runner instance')

        observed = self.problem.shot.observed
        if self.block is not None:
            modelled, observed = self.block.pipelines.traces.apply(modelled, observed)

        fun, adjoint_source = self.functional.apply(self.problem.shot, modelled, observed)

        if self.block is not None:
            adjoint_source = self.block.pipelines.adjoint_source.apply(adjoint_source)

        self.logger.info('(ShotID %d) Functional value: %s' % (self.problem.shot_id, fun))

        return fun, adjoint_source

    def run_adjoint(self, wrt, adjoint_source, wavefield):
        wrt.grad.fill(0.)
        wrt.prec.fill(0.)

        self.problem_type.before_adjoint(wrt, adjoint_source, wavefield)

        self.logger.info('(ShotID %d) Running adjoint equation for shot' % self.problem.shot_id)
        self.problem_type.adjoint()

        self.logger.info('(ShotID %d) Completed adjoint equation run shot' % self.problem.shot_id)

        wrt = self.problem_type.after_adjoint(wrt)
        wrt = self.functional.gradient(wrt)

        for variable in wrt:
            variable.grad, variable.prec = self.block.pipelines.\
                local_gradient.apply(variable.grad, variable.prec)

        return wrt

    def run_gradient(self, wrt):
        traces, wavefield = self.run_state(save_wavefield=True)

        fun, adjoint_source = self.run_functional(traces)

        wrt = self.run_adjoint(wrt, adjoint_source, wavefield)

        return fun, wrt

    def _sum_grad_prec(self, grad_problem, grad_fun):
        grad = []

        for each_problem, each_fun in zip(grad_problem, grad_fun):
            grad.append([each_problem[0]+each_fun[0],
                         each_problem[1]+each_fun[1]])

        return grad
