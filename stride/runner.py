

from mosaic import tessera


__all__ = ['Runner']


@tessera
class Runner:
    """
    The Runner acts as a manager of the forward and inverse runs in Stride. The Runner takes care
    of instantiating a problem type and, when needed, a functional; it takes care of setting
    up the sub-problem to run and the optimisation block; and it acts as an interface to
    execute forward, adjoint and gradient runs on these.

    The Runner is also responsible for all necessary processing of wavelets, observed and modelled
    data, as well as local-level actions on the gradients.

    """

    def __init__(self):
        self.problem = None
        self.block = None

        self.problem_type = None
        self.functional = None

    def set_problem(self, problem, **kwargs):
        """
        Set up the problem or sub-problem that needs to be run.

        Parameters
        ----------
        problem : SubProblem or Problem
            Problem on which the physics will be executed
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        self.logger.info('(ShotID %d) Preparing to run shot' % problem.shot_id)

        self.problem = problem

        if self.problem_type is None or self.problem_type.__class__ != problem.problem_type.__class__:
            self.problem_type = problem.problem_type

        if self.block is not None:
            wavelets = self.problem.shot.wavelets
            wavelets = self.block.pipelines.wavelets.apply(wavelets)
            self.problem.shot.wavelets = wavelets

            observed = self.problem.shot.observed
            observed = self.block.pipelines.wavelets.apply(observed)
            self.problem.shot.observed = observed

        self.problem_type.set_problem(problem, **kwargs)

    def set_block(self, block, **kwargs):
        """
        Set up the optimisation block for the inversion.

        Parameters
        ----------
        block : Block
            Relevant optimisation block.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------

        """
        self.logger.info('Preparing to run block %d' % block.id)

        self.block = block

        if self.functional is None or self.functional.__class__ != block.functional.__class__:
            self.functional = block.functional

    def run_state(self, save_wavefield=False, **kwargs):
        """
        Run all the necessary hooks on the problem type to execute the state or forward.

        Parameters
        ----------
        save_wavefield : bool, optional
            Whether or not the wavefield needs to be stored, defaults to False.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        Traces
            Time traces produced by the state run.
        Data or None
            Wavefield produced by the state run, if any.

        """
        self.problem_type.before_state(save_wavefield=save_wavefield, **kwargs)

        self.logger.info('(ShotID %d) Running state equation for shot' % self.problem.shot_id)
        self.problem_type.state(**kwargs)

        self.logger.info('(ShotID %d) Completed state equation run for shot' % self.problem.shot_id)
        traces, wavefield = self.problem_type.after_state(save_wavefield=save_wavefield, **kwargs)

        if save_wavefield is True and self.block is not None:
            wavefield = self.block.pipelines.wavefield.apply(wavefield)

        return traces, wavefield

    def run_functional(self, modelled, **kwargs):
        """
        Use some ``modelled`` data to calculate a functional value for the
        present SubProblem.

        Parameters
        ----------
        modelled : Traces
            Time traces to compare with the observed data in the shot.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        FunctionalValue
            Object containing information about the shot, the value of the functional
            and the residuals.
        Traces
            Generated adjoint source.

        """
        if self.functional is None:
            raise ValueError('No functional was given to the runner instance')

        observed = self.problem.shot.observed
        if self.block is not None:
            modelled, observed = self.block.pipelines.traces.apply(modelled, observed)

        fun, adjoint_source = self.functional.apply(self.problem.shot, modelled, observed, **kwargs)

        if self.block is not None:
            adjoint_source = self.block.pipelines.adjoint_source.apply(adjoint_source)

        self.logger.info('(ShotID %d) Functional value: %s' % (self.problem.shot_id, fun))

        return fun, adjoint_source

    def run_adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Run all the necessary hooks on the problem type to execute the adjoint problem.

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.
        adjoint_source : Traces
            Adjoint source to use in the adjoint propagation.
        wavefield : Data
            Stored wavefield from the forward run, to use as needed.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        VariableList
            Updated variable list with gradients added to them, if any.

        """
        wrt.grad.fill(0.)
        wrt.prec.fill(0.)

        self.problem_type.before_adjoint(wrt, adjoint_source, wavefield, **kwargs)

        self.logger.info('(ShotID %d) Running adjoint equation for shot' % self.problem.shot_id)
        self.problem_type.adjoint(wrt, adjoint_source, wavefield, **kwargs)

        self.logger.info('(ShotID %d) Completed adjoint equation run shot' % self.problem.shot_id)

        wrt = self.problem_type.after_adjoint(wrt, adjoint_source, wavefield, **kwargs)
        wrt = self.functional.get_grad(wrt, **kwargs)

        for variable in wrt:
            variable.grad, variable.prec = self.block.pipelines.\
                local_gradient.apply(variable.grad, variable.prec)

        return wrt

    def run_gradient(self, wrt, **kwargs):
        """
        Execute the state, functional and adjoint in order to calculate
        the gradients for a series of variables.

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.
        kwargs
            Extra parameters to be used by the method.

        Returns
        -------
        FunctionalValue
            Object containing information about the shot, the value of the functional
            and the residuals.
        VariableList
            Updated variable list with gradients added to them, if any.

        """
        traces, wavefield = self.run_state(save_wavefield=True, **kwargs)

        fun, adjoint_source = self.run_functional(traces, **kwargs)

        wrt = self.run_adjoint(wrt, adjoint_source, wavefield, **kwargs)

        return fun, wrt

    @staticmethod
    def _sum_grad_prec(grad_problem, grad_fun):
        grad = []

        for each_problem, each_fun in zip(grad_problem, grad_fun):
            grad.append([each_problem[0]+each_fun[0],
                         each_problem[1]+each_fun[1]])

        return grad
