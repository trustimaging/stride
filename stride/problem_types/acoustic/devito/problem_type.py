
import devito
import numpy as np

from stride.problem_definition import ScalarField
from ...operators.devito import GridDevito, OperatorDevito
from ...problem_type import ProblemTypeBase


__all__ = ['ProblemType']


# TODO We need to be able to check stability and dispersion conditions


class ProblemType(ProblemTypeBase):
    """
    This class represents the second-order isotropic acoustic wave equation,
    implemented using Devito.

    """

    space_order = 10
    time_order = 2
    undersampling_factor = 4
    kernel = 'OT4'

    def __init__(self):
        super().__init__()

        self._grad = None

        self._max_wavelet = 0.
        self._src_scale = 0.

        self._grid = GridDevito(self.space_order, self.time_order)
        self._state_operator = OperatorDevito(self.space_order, self.time_order, grid=self._grid)
        self._adjoint_operator = OperatorDevito(self.space_order, self.time_order, grid=self._grid)

    def set_problem(self, problem, **kwargs):
        """
        Set up the problem or sub-problem that needs to be run.

        Parameters
        ----------
        problem : SubProblem or Problem
            Problem on which the physics will be executed

        Returns
        -------

        """
        super().set_problem(problem)

        self._grid.set_problem(problem)
        self._state_operator.set_problem(problem)
        self._adjoint_operator.set_problem(problem)

    def before_state(self, save_wavefield=False, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------
        save_wavefield : bool, optional
            Whether or not the wavefield needs to be stored, defaults to False.

        Returns
        -------

        """
        time = self._problem.time
        space = self._problem.space
        shot = self._problem.shot

        num_sources = shot.num_sources
        num_receivers = shot.num_receivers

        # If there's no previous operator, generate one
        if self._state_operator.operator is None:
            # Define variables
            src = self._grid.sparse_time_function('src', num=num_sources)
            rec = self._grid.sparse_time_function('rec', num=num_receivers)

            p = self._grid.time_function('p')
            m = self._grid.function('m')

            # Create damping layer
            if np.max(space.extra) > 0:
                damp = self._grid.function('damp')
                damp.data[:] = self._problem.medium.damping(0.003 * np.log(1.0 / 0.001) / np.max(space.extra))

            else:
                damp = devito.Constant('damp')
                damp.data = 0.

            # Create stencil
            stencil = self._iso_stencil(self._grid.grid, p, m, damp,
                                        direction='forward')

            # Define the source injection function to generate the corresponding code
            src_term = src.inject(field=p.forward, expr=src * time.step**2 / m)
            rec_term = rec.interpolate(expr=p)

            kwargs = {
                'dt': time.step,
                'p': p,
                'm': m,
                'damp': damp,
                'src': src,
                'rec': rec,
            }

            # Define the saving of the wavefield
            if save_wavefield:
                p_saved = self._grid.undersampled_time_function('p_saved',
                                                                factor=self.undersampling_factor)

                update_saved = [devito.Eq(p_saved, self._saved(p, m))]
                kwargs['p_saved'] = p_saved

            else:
                update_saved = []

            # Compile the operator
            self._state_operator.set_operator(stencil + src_term + rec_term + update_saved,
                                              name='acoustic_iso_state')
            self._state_operator.compile()

            # Prepare arguments
            self._state_operator.arguments(**kwargs)

        else:
            # If the source/receiver size has changed, then create new functions for them
            # and generate the arguments again
            changed_args = False

            if num_sources != self._grid.vars.src.npoint:
                changed_args = True
                self._grid.sparse_time_function('src', num=num_sources, cached=False)

            if num_receivers != self._grid.vars.rec.npoint:
                changed_args = True
                self._grid.sparse_time_function('rec', num=num_receivers, cached=False)

            if changed_args:
                self._state_operator.arguments(src=self._grid.vars.src, rec=self._grid.vars.rec)

        # Clear all buffers
        self._grid.vars.src.data_with_halo.fill(0.)
        self._grid.vars.rec.data_with_halo.fill(0.)
        self._grid.vars.p.data_with_halo.fill(0.)

        if save_wavefield:
            self._grid.vars.p_saved.data_with_halo.fill(0.)

        # Set medium parameters
        medium = self._problem.medium

        self._grid.vars.m.data_with_halo[:] = 1 / self._grid.with_halo(medium.vp.extended_data)**2

        # Set geometry
        self._src_scale = 1000. / (np.max(medium.vp.extended_data)**2 * time.step**2)
        self._max_wavelet = np.max(np.abs(shot.wavelets.data))
        self._grid.vars.src.data[:] = shot.wavelets.data.T * self._src_scale / self._max_wavelet

        self._grid.vars.src.coordinates.data[:] = shot.source_coordinates
        self._grid.vars.rec.coordinates.data[:] = shot.receiver_coordinates

    def state(self, **kwargs):
        """
        Run the state or forward problem.

        Returns
        -------

        """
        self._state_operator.run()

    def after_state(self, save_wavefield=False, **kwargs):
        """
        Clean up after the state run and retrieve the time traces.

        If requested, also provide a saved wavefield.

        Parameters
        ----------
        save_wavefield : bool, optional
            Whether or not the wavefield needs to be stored, defaults to False.

        Returns
        -------
        Traces
            Time traces produced by the state run.
        Data or None
            Wavefield produced by the state run, if any.

        """
        if save_wavefield:
            wavefield_data = np.asarray(self._grid.vars.p_saved.data, dtype=np.float32)
            wavefield_data *= self._max_wavelet / self._src_scale

            wavefield = ScalarField('p_dt2', shape=wavefield_data.shape)
            wavefield.data[:] = wavefield_data

        else:
            wavefield = None

        traces = self._problem.shot.observed.alike('modelled')
        traces_data = np.array(self._grid.vars.rec.data).T * self._max_wavelet / self._src_scale
        traces.data[:] = traces_data

        return traces, wavefield

    def before_adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Prepare the problem type to run the adjoint problem.

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.
        adjoint_source : Traces
            Adjoint source to use in the adjoint propagation.
        wavefield : Data
            Stored wavefield from the forward run, to use as needed.

        Returns
        -------

        """
        time = self._problem.time
        space = self._problem.space
        shot = self._problem.shot

        num_sources = shot.num_sources
        num_receivers = shot.num_receivers

        # If there's no previous operator, generate one
        if self._adjoint_operator.operator is None:
            # Define variables
            src = self._grid.sparse_time_function('src', num=num_sources)
            rec = self._grid.sparse_time_function('rec', num=num_receivers)

            p_a = self._grid.time_function('p_a')
            p_saved = self._grid.undersampled_time_function('p_saved',
                                                            factor=self.undersampling_factor)
            m = self._grid.function('m')

            # Properly create damping layer
            if np.max(space.extra) > 0:
                damp = self._grid.function('damp')
                damp.data[:] = self._problem.medium.damping(0.005 * np.log(1.0 / 0.001) / np.max(space.extra))

            else:
                damp = devito.Constant('damp')
                damp.data = 0.

            # Create stencil
            stencil = self._iso_stencil(self._grid.grid, p_a, m, damp,
                                        direction='backward')

            # Define the source injection function to generate the corresponding code
            rec_term = rec.inject(field=p_a.backward, expr=-rec * time.step ** 2 / m)

            kwargs = {
                'dt': time.step,
                'p_a': p_a,
                'p_saved': p_saved,
                'm': m,
                'damp': damp,
                'rec': rec,
            }

            # Define gradient
            gradient_update = self.set_grad(wrt)

            # Compile the operator
            self._adjoint_operator.set_operator(stencil + rec_term + gradient_update,
                                                name='acoustic_iso_adjoint')
            self._adjoint_operator.compile()

            # Prepare arguments
            self._adjoint_operator.arguments(**kwargs)

        else:
            # If the source/receiver size has changed, then create new functions for them
            # and generate the arguments again
            changed_args = False

            if num_sources != self._grid.vars.src.npoint:
                changed_args = True
                self._grid.sparse_time_function('src', num=num_sources, cached=False)

            if num_receivers != self._grid.vars.rec.npoint:
                changed_args = True
                self._grid.sparse_time_function('rec', num=num_receivers, cached=False)

            if changed_args:
                self._adjoint_operator.arguments(src=self._grid.vars.src, rec=self._grid.vars.rec)

        # Clear all buffers
        self._grid.vars.src.data_with_halo.fill(0.)
        self._grid.vars.rec.data_with_halo.fill(0.)
        self._grid.vars.p_a.data_with_halo.fill(0.)

        for variable in wrt:
            self._grid.vars['grad_'+variable.name].data_with_halo.fill(0.)

        # Set prior wavefield
        self._grid.vars.p_saved.data[:] = wavefield.data

        # Set medium parameters
        medium = self._problem.medium

        self._grid.vars.m.data_with_halo[:] = 1 / self._grid.with_halo(medium.vp.extended_data)**2

        # Set geometry
        self._grid.vars.rec.data[:] = adjoint_source.data.T

        self._grid.vars.src.coordinates.data[:] = shot.source_coordinates
        self._grid.vars.rec.coordinates.data[:] = shot.receiver_coordinates

    def adjoint(self, **kwargs):
        """
        Run the adjoint problem.

        Returns
        -------

        """
        self._adjoint_operator.run()

    def after_adjoint(self, wrt, **kwargs):
        """
        Clean up after the adjoint run and retrieve the time gradients (if needed).

        Parameters
        ----------
        wrt : VariableList
            List of variables for which the inverse problem is being solved.

        Returns
        -------
        VariableList
            Updated variable list with gradients added to them.

        """
        return self.get_grad(wrt, **kwargs)

    def set_grad_vp(self, vp, **kwargs):
        """
        Prepare the problem type to calculate the gradients wrt Vp.

        Parameters
        ----------
        vp : Vp
            Vp variable to calculate the gradient.

        Returns
        -------
        tuple
            Tuple of gradient and preconditioner updates.

        """
        p = self._grid.vars.p_saved
        p_a = self._grid.vars.p_a

        grad = self._grid.function('grad_vp')
        grad_update = devito.Inc(grad, -p * p_a)

        prec = self._grid.function('prec_vp')
        prec_update = devito.Inc(prec, +p * p)

        return grad_update, prec_update

    def get_grad_vp(self, vp, **kwargs):
        """
        Retrieve the gradients calculated wrt to the input.

        The variable is updated inplace.

        Parameters
        ----------
        vp : Vp
            Vp variable to calculate the gradient.

        Returns
        -------

        """
        variable_grad = self._grid.vars.grad_vp
        variable_grad = np.asarray(variable_grad.data, dtype=np.float32)

        variable_prec = self._grid.vars.prec_vp
        variable_prec = np.asarray(variable_prec.data, dtype=np.float32)

        variable_grad *= 2 / vp.extended_data**3
        variable_prec *= 4 / vp.extended_data**6

        vp.grad += variable_grad
        vp.prec += variable_prec

    def _biharmonic(self, field, weight):
        space_dims = [d for d in field.dimensions if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        laplace = field.laplace

        biharmonic = []
        for d in derivs:
            for expr in laplace.args:
                expr = getattr(expr.evaluate * weight, d)
                biharmonic.append(expr)

        return sum(biharmonic)._eval_deriv

    def _laplacian(self, field, m):
        if self.kernel not in ['OT2', 'OT4']:
            raise ValueError("Unrecognized kernel")

        time = self._problem.time

        bi_harmonic = self._biharmonic(field, 1 / m) if self.kernel == 'OT4' else 0
        laplacian = field.laplace + time.step**2/12 * bi_harmonic

        return laplacian

    def _saved(self, field, m):
        if self.kernel not in ['OT2', 'OT4']:
            raise ValueError("Unrecognized kernel")

        time = self._problem.time

        bi_harmonic = field.biharmonic(m**(-2)) if self.kernel == 'OT4' else 0
        saved = field.dt2 + time.step**2/12 * bi_harmonic

        return saved

    def _iso_stencil(self, grid, field, m, damp, direction='forward'):
        # Forward or backward
        forward = direction == 'forward'

        # Define time step to be updated
        u_next = field.forward if forward else field.backward
        u_dt = field.dt if forward else field.dt.T

        # Get the spacial FD
        laplacian = self._laplacian(field, m)

        # Define PDE and update rule
        eq_time = devito.solve(m * field.dt2 - laplacian + damp*u_dt, u_next)

        # Define coefficients
        # dims = grid.dimensions
        #
        # weights = np.array([0., 0., 0.0274017,
        #                     -0.223818, 1.64875, -2.90467,
        #                     1.64875, -0.223818, 0.0274017,
        #                     0., 0.])
        #
        # coeffs = [devito.Coefficient(2, field, d, weights/d.spacing**2)
        #           for d in dims]
        # subs = devito.Substitutions(*coeffs)

        # Time-stepping stencil.
        stencil = [devito.Eq(u_next, eq_time,
                             subdomain=grid.subdomains['physical_domain'],
                             coefficients=None)]

        return stencil
