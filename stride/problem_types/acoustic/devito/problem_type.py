
import devito
import numpy as np

import mosaic

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

    def __init__(self):
        super().__init__()

        self.kernel = 'OT4'
        self.drp = True

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

        self.drp = kwargs.get('drp', True)
        self.check_conditions()

        self._grid.set_problem(problem)
        self._state_operator.set_problem(problem)
        self._adjoint_operator.set_problem(problem)

    def check_conditions(self):
        """
        Check CFL and dispersion conditions, and select appropriate OT method.

        Returns
        -------

        """
        time = self._problem.time
        space = self._problem.space
        shot = self._problem.shot

        runtime = mosaic.runtime()

        # Get speed of sound bounds
        medium = self._problem.medium

        vp_min = np.min(medium.vp.extended_data)
        vp_max = np.max(medium.vp.extended_data)

        # Figure out propagated bandwidth
        wavelets = shot.wavelets.data

        if not time.num % 2:
            num_freqs = (time.num + 1) // 2
        else:
            num_freqs = time.num // 2 + 1

        wavelets_fft = np.fft.fft(wavelets, axis=-1)[:, :num_freqs]
        freqs = np.fft.fftfreq(time.num, time.step)[:num_freqs]

        wavelets_fft = np.mean(np.abs(wavelets_fft), axis=0)
        wavelets_fft = 20 * np.log10(wavelets_fft / np.max(wavelets_fft))

        f_min = 0
        for f in range(num_freqs):
            if wavelets_fft[f] > -10:
                f_min = freqs[f]
                break

        f_max = num_freqs
        for f in reversed(range(num_freqs)):
            if wavelets_fft[f] > -10:
                f_max = freqs[f]
                break

        runtime.logger.info('Estimated bandwidth for the propagated '
                            'wavelet %.3f-%.3f MHz' % (f_min / 1e6, f_max / 1e6))

        # Check for dispersion
        h = max(*space.spacing)

        if self.drp:
            runtime.logger.info('Using DRP scheme')

            h_max = vp_min / (3 * f_max)

        else:
            runtime.logger.info('Deactivating DRP scheme')

            h_max = vp_min / (5 * f_max)

        if h > h_max:
            runtime.logger.warn('Spatial grid spacing (%.3f mm) is '
                                'higher than dispersion limit (%.3f mm)' % (h / 1e-3, h_max / 1e-3))

        # Check for instability
        dt = time.step

        dt_max_OT2 = self._dt_max(2.0 / np.pi, h, vp_max)
        dt_max_OT4 = self._dt_max(3.6 * np.pi, h, vp_max)
        dt_max_OT4_OP = self._dt_max(4.0 / np.pi, h, vp_max)

        recompile = False
        if dt <= dt_max_OT2:
            runtime.logger.info('Time grid spacing (%.3f \u03BCs) is '
                                'below OT2 limit (%.3f \u03BCs)' % (dt / 1e-6, dt_max_OT2 / 1e-6))

            if self.kernel != 'OT2':
                recompile = True

            self.kernel = 'OT2'

        elif dt <= dt_max_OT4:
            runtime.logger.info('Time grid spacing (%.3f \u03BCs) is '
                                'above OT2 limit (%.3f \u03BCs), '
                                'switching to OT4' % (dt / 1e-6, dt_max_OT2 / 1e-6))

            if self.kernel != 'OT4':
                recompile = True

            self.kernel = 'OT4'

        elif dt <= dt_max_OT4_OP:
            runtime.logger.info('Time grid spacing (%.3f \u03BCs) is'
                                'above OT4 limit (%.3f \u03BCs), '
                                'switching to OT4_OP' % (dt / 1e-6, dt_max_OT4 / 1e-6))

            if self.kernel != 'OT4_OP':
                recompile = True

            self.kernel = 'OT4_OP'

        else:
            runtime.logger.warn('Time grid spacing (%.3f \u03BCs) is '
                                'above OT4_OP limit (%.3f \u03BCs)' % (dt / 1e-6, dt_max_OT4_OP / 1e-6))

            if self.kernel != 'OT4_OP':
                recompile = True

            self.kernel = 'OT4_OP'

        if recompile:
            self._state_operator.operator = None

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

            p = self._grid.time_function('p', coefficients='symbolic' if self.drp else 'standard')
            m = self._grid.function('m', coefficients='symbolic' if self.drp else 'standard')

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

            op_kwargs = {
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
                                              name='acoustic_iso_state',
                                              **kwargs.get('devito_config', {}))
            self._state_operator.compile()

            # Prepare arguments
            self._state_operator.arguments(**{**op_kwargs, **kwargs.get('devito_args', {})})

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

            p_a = self._grid.time_function('p_a', coefficients='symbolic' if self.drp else 'standard')
            p_saved = self._grid.undersampled_time_function('p_saved',
                                                            factor=self.undersampling_factor)
            m = self._grid.function('m', coefficients='symbolic' if self.drp else 'standard')

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

            op_kwargs = {
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
                                                name='acoustic_iso_adjoint',
                                                **kwargs.get('devito_config', {}))
            self._adjoint_operator.compile()

            # Prepare arguments
            self._adjoint_operator.arguments(**{**op_kwargs, **kwargs.get('devito_args', {})})

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

    def _laplacian(self, field, laplacian, m):
        if self.kernel not in ['OT2', 'OT4', 'OT4_OP']:
            raise ValueError("Unrecognized kernel")

        time = self._problem.time

        if self.drp:
            bi_harmonic = laplacian.laplace.evaluate

        else:
            bi_harmonic = field.biharmonic(1/m)

        if self.kernel == 'OT2':
            bi_harmonic = 0

        elif self.kernel == 'OT4':
            bi_harmonic = time.step**2/12 * bi_harmonic

        else:
            bi_harmonic = time.step**2/16 * bi_harmonic

        laplacian_subs = field.laplace + time.step**2/12 * bi_harmonic

        return laplacian_subs

    def _saved(self, field, m):
        if self.kernel not in ['OT2', 'OT4', 'OT4_OP']:
            raise ValueError("Unrecognized kernel")

        time = self._problem.time

        # bi_harmonic = field.biharmonic(m**(-2)) if self.kernel == 'OT4' else 0
        bi_harmonic = 0
        saved = field.dt2 + time.step**2/12 * bi_harmonic

        return saved

    def _iso_stencil(self, grid, field, m, damp, direction='forward'):
        # Forward or backward
        forward = direction == 'forward'

        # Define time step to be updated
        u_next = field.forward if forward else field.backward
        u_dt = field.dt if forward else field.dt.T

        # Get the spacial FD
        laplacian = self._grid.function('laplacian', coefficients='symbolic')
        laplacian_subs = self._laplacian(field, laplacian, m)

        # Define PDE and update rule
        eq_time = devito.solve(m * field.dt2 - laplacian_subs + damp*u_dt, u_next)

        # Define coefficients
        if self.drp:
            dims = grid.dimensions

            weights_1 = np.array([
                -7.936507937e-4,
                +9.920634921e-3,
                -5.952380952e-2,
                +0.238095238100,
                -0.833333333333,
                +0.0,
                +0.833333333333,
                -0.238095238100,
                +5.952380952e-2,
                -9.920634921e-3,
                +7.936507937e-4,
            ])

            weights_2 = np.array([
                +0.0043726804,
                -0.0281145606,
                +0.1068406382,
                -0.3705141600,
                +1.8617697535,
                -3.1487087031,
                +1.8617697535,
                -0.3705141600,
                +0.1068406382,
                -0.0281145606,
                +0.0043726804,
            ])

            coeffs_field_1 = [devito.Coefficient(1, field, d, weights_1/d.spacing)
                              for d in dims]
            coeffs_field_2 = [devito.Coefficient(2, field, d, weights_2/d.spacing**2)
                              for d in dims]

            coeffs_lap_1 = [devito.Coefficient(1, laplacian, d, weights_1/d.spacing)
                            for d in dims]
            coeffs_lap_2 = [devito.Coefficient(2, laplacian, d, weights_2/d.spacing**2)
                            for d in dims]

            coeffs_m_1 = [devito.Coefficient(1, m, d, weights_1/d.spacing)
                          for d in dims]
            coeffs_m_2 = [devito.Coefficient(2, m, d, weights_2/d.spacing**2)
                          for d in dims]

            coeffs = coeffs_field_1 + coeffs_field_2 + \
                     coeffs_lap_1 + coeffs_lap_2 + \
                     coeffs_m_1 + coeffs_m_2
            subs = devito.Substitutions(*coeffs)

            # Time-stepping stencil
            laplacian_term = devito.Eq(laplacian, 1/m * field.laplace.evaluate,
                                       subdomain=grid.subdomains['physical_domain'],
                                       coefficients=subs)

            stencil = devito.Eq(u_next, eq_time,
                                subdomain=grid.subdomains['physical_domain'],
                                coefficients=subs)

            return [laplacian_term, stencil]

        else:
            # Time-stepping stencil
            stencil = devito.Eq(u_next, eq_time,
                                subdomain=grid.subdomains['physical_domain'])

            return [stencil]

    def _dt_max(self, k, h, vp_max):
        space = self._problem.space

        return k * h / vp_max * 1 / np.sqrt(space.dim)
