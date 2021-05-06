
import os
import devito
from devito.types import Buffer
import numpy as np
import scipy.signal

import mosaic
from mosaic.utils import camel_case

from stride.utils import fft
from stride.problem_definition import ScalarField
from ..operators.devito import GridDevito, OperatorDevito
from ..problem_type import ProblemTypeBase
from .. import boundaries


__all__ = ['AcousticDevito']


class AcousticDevito(ProblemTypeBase):
    """
    This class represents the second-order isotropic acoustic wave equation,
    implemented using Devito.

    """

    space_order = 10
    time_order = 2

    def __init__(self):
        super().__init__()

        self.kernel = 'OT4'
        self.drp = False
        self.undersampling_factor = 4
        self.boundary_type = 'sponge_boundary_2'
        self.interpolation_type = 'linear'
        self.attenuation_power = 0

        self._grad = None

        self._max_wavelet = 0.
        self._src_scale = 0.
        self._bandwidth = 0.

        self._grid = None
        self._state_operator = None
        self._adjoint_operator = None
        self._boundary = None

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

        if self._grid is None:
            self._grid = GridDevito(self.space_order, self.time_order)

        if self._state_operator is None:
            self._state_operator = OperatorDevito(self.space_order, self.time_order, grid=self._grid)

        if self._adjoint_operator is None:
            self._adjoint_operator = OperatorDevito(self.space_order, self.time_order, grid=self._grid)

        if self._boundary is None:
            self.boundary_type = kwargs.get('boundary_type', self.boundary_type)

            if isinstance(self.boundary_type, str):
                boundaries_module = boundaries.devito
                self._boundary = getattr(boundaries_module, camel_case(self.boundary_type))(self._grid)

            else:
                self._boundary = self.boundary_type

        self._grid.set_problem(problem)
        self._state_operator.set_problem(problem)
        self._adjoint_operator.set_problem(problem)
        self._boundary.set_problem(problem)

        self.interpolation_type = kwargs.get('interpolation_type', 'linear')
        self.attenuation_power = kwargs.get('attenuation_power', 0)
        self.drp = kwargs.get('drp', False)
        preferred_kernel = kwargs.get('kernel', None)
        preferred_undersampling = kwargs.get('undersampling', None)

        self.check_conditions(preferred_kernel, preferred_undersampling)

        # Recompile every time if using hicks
        if self.interpolation_type == 'hicks':
            self._state_operator.operator = None
            self._adjoint_operator.operator = None

        runtime = mosaic.runtime()
        runtime.logger.info('(ShotID %d) Selected time stepping scheme %s' %
                            (self._problem.shot_id, self.kernel,))

    def check_conditions(self, preferred_kernel=None, preferred_undersampling=None):
        """
        Check CFL and dispersion conditions, and select appropriate OT method.

        Parameters
        ----------
        preferred_kernel : str, optional
            Preferred kernel to run, defaults to internally calculated.
        preferred_undersampling : int, optional
            Preferred amount of undersampling to use, defaults to internally calculated.

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
        f_min, f_centre, f_max = fft.bandwidth(wavelets, time.step, cutoff=-10)

        self._bandwidth = (f_min, f_centre, f_max)

        runtime.logger.info('(ShotID %d) Estimated bandwidth for the propagated '
                            'wavelet %.3f-%.3f MHz' % (self._problem.shot_id, f_min / 1e6, f_max / 1e6))

        # Check for dispersion
        if self.drp is True:
            self.drp = False

            runtime.logger.warn('(ShotID %d) DRP weights are not implemented in this version of stride' %
                                self._problem.shot_id)

        h = max(*space.spacing)

        wavelength = vp_min / f_max
        ppw = wavelength / h
        ppw_max = 5

        h_max = wavelength / ppw_max

        if h > h_max:
            runtime.logger.warn('(ShotID %d) Spatial grid spacing (%.3f mm | %d PPW) is '
                                'higher than dispersion limit (%.3f mm | %d PPW)' %
                                (self._problem.shot_id, h / 1e-3, ppw, h_max / 1e-3, ppw_max))
        else:
            runtime.logger.info('(ShotID %d) Spatial grid spacing (%.3f mm | %d PPW) is '
                                'below dispersion limit (%.3f mm | %d PPW)' %
                                (self._problem.shot_id, h / 1e-3, ppw, h_max / 1e-3, ppw_max))

        # Check for instability
        dt = time.step

        dt_max_OT2 = self._dt_max(2.0 / np.pi, h, vp_max)
        dt_max_OT4 = self._dt_max(3.6 / np.pi, h, vp_max)

        crossing_factor = dt*vp_max / h * 100

        recompile = False
        if dt <= dt_max_OT2:
            runtime.logger.info('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                                'below OT2 limit (%.3f \u03BCs)' %
                                (self._problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT2 / 1e-6))

            selected_kernel = 'OT2'

        elif dt <= dt_max_OT4:
            runtime.logger.info('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                                'above OT2 limit (%.3f \u03BCs)'
                                % (self._problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT2 / 1e-6))

            selected_kernel = 'OT4'

        else:
            runtime.logger.warn('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                                'above OT4 limit (%.3f \u03BCs)'
                                % (self._problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT4 / 1e-6))

            selected_kernel = 'OT4'

        selected_kernel = selected_kernel if preferred_kernel is None else preferred_kernel

        if self.kernel != selected_kernel:
            recompile = True

        self.kernel = selected_kernel

        # Select undersampling level
        f_max *= 4
        dt_max = 1 / f_max

        undersampling = min(max(2, int(dt_max / dt)), 10) if preferred_undersampling is None else preferred_undersampling

        if self.undersampling_factor != undersampling:
            recompile = True

        self.undersampling_factor = undersampling

        runtime.logger.info('(ShotID %d) Selected undersampling level %d' %
                            (self._problem.shot_id, undersampling,))

        # Maybe recompile
        if recompile:
            self._state_operator.operator = None
            self._adjoint_operator.operator = None

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
        medium = self._problem.medium

        num_sources = shot.num_sources
        num_receivers = shot.num_receivers

        # If there's no previous operator, generate one
        if self._state_operator.operator is None:
            save = Buffer(save_wavefield) if type(save_wavefield) is int else None

            # Define variables
            src = self._grid.sparse_time_function('src', num=num_sources,
                                                  coordinates=shot.source_coordinates,
                                                  interpolation_type=self.interpolation_type)
            rec = self._grid.sparse_time_function('rec', num=num_receivers,
                                                  coordinates=shot.receiver_coordinates,
                                                  interpolation_type=self.interpolation_type)

            p = self._grid.time_function('p', coefficients='symbolic' if self.drp else 'standard', save=save)

            vp = self._grid.function('vp', coefficients='symbolic' if self.drp else 'standard')
            vp2 = self._grid.function('vp2', coefficients='symbolic' if self.drp else 'standard')
            inv_vp2 = self._grid.function('inv_vp2', coefficients='symbolic' if self.drp else 'standard')

            if 'rho' in medium.fields:
                rho = self._grid.function('rho', coefficients='symbolic' if self.drp else 'standard')
                buoy = self._grid.function('buoy', coefficients='symbolic' if self.drp else 'standard')

            else:
                rho = buoy = None

            if 'alpha' in medium.fields:
                alpha = self._grid.function('alpha', coefficients='symbolic' if self.drp else 'standard')

            else:
                alpha = None

            # Create stencil
            stencil = self._iso_stencil(p, vp, vp2, inv_vp2,
                                        rho=rho, buoy=buoy, alpha=alpha,
                                        direction='forward')

            # Define the source injection function to generate the corresponding code
            # pressure_to_mass_acc = time.step / (vp * space.spacing**3)
            # solve_p_dt = time.step**2 * vp**2
            src_term = src.inject(field=p.forward, expr=src * time.step**2 * vp2)
            rec_term = rec.interpolate(expr=p)

            op_kwargs = {
                'dt': time.step,
            }

            # Define the saving of the wavefield
            if save_wavefield is True:
                p_saved = self._grid.undersampled_time_function('p_saved',
                                                                factor=self.undersampling_factor)

                update_saved = [devito.Eq(p_saved, self._saved(p, vp, vp2, inv_vp2))]

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

        if save_wavefield is True:
            self._grid.vars.p_saved.data_with_halo.fill(0.)

        # Set medium parameters
        vp_with_halo = self._grid.with_halo(medium.vp.extended_data)
        self._grid.vars.vp.data_with_halo[:] = vp_with_halo
        self._grid.vars.vp2.data_with_halo[:] = vp_with_halo**2
        self._grid.vars.inv_vp2.data_with_halo[:] = 1 / vp_with_halo**2

        if 'rho' in medium.fields:
            rho_with_halo = self._grid.with_halo(medium.rho.extended_data)
            self._grid.vars.rho.data_with_halo[:] = rho_with_halo
            self._grid.vars.buoy.data_with_halo[:] = 1 / rho_with_halo

        if 'alpha' in medium.fields:
            db_cm_to_np_m = np.log(10)/20 * 1e2

            alpha_with_halo = self._grid.with_halo(medium.alpha.extended_data)*db_cm_to_np_m
            self._grid.vars.alpha.data_with_halo[:] = alpha_with_halo

        # Set geometry and wavelet
        wavelets = shot.wavelets.data

        self._src_scale = 1000.
        self._max_wavelet = np.max(np.abs(wavelets)) + 1e-31

        window = scipy.signal.get_window(('tukey', 0.01), time.num, False)  # TODO This should happen in a pipeline
        window = window.reshape((time.num, 1))

        self._grid.vars.src.data[:] = wavelets.T * self._src_scale / self._max_wavelet * window

        if self.interpolation_type == 'linear':
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
            if save_wavefield is True:
                wavefield_data = np.asarray(self._grid.vars.p_saved.data_with_halo, dtype=np.float32)

            else:
                wavefield_data = np.asarray(self._grid.vars.p.data, dtype=np.float32)

            wavefield_slice = kwargs.pop('wavefield_slice', None)
            if wavefield_slice is not None:
                wavefield_data = wavefield_data[wavefield_slice]

            wavefield_data *= self._max_wavelet / self._src_scale

            wavefield = ScalarField('p_dt2',
                                    data=wavefield_data,
                                    shape=wavefield_data.shape)

            self._grid.deallocate('p_saved')

            if os.environ.get('STRIDE_DUMP_WAVEFIELD', None) == 'yes':
                wavefield.dump(path=self._problem.output_folder,
                               project_name=self._problem.name)

        else:
            wavefield = None

        traces_data = np.asarray(self._grid.vars.rec.data, dtype=np.float32).T
        traces_data *= self._max_wavelet / self._src_scale
        traces = self._problem.shot.observed.alike('modelled', data=traces_data)

        self._grid.deallocate('p')
        self._grid.deallocate('src')
        self._grid.deallocate('rec')
        self._grid.deallocate('vp')
        self._grid.deallocate('vp2')
        self._grid.deallocate('inv_vp2')
        self._grid.deallocate('rho')
        self._grid.deallocate('buoy')
        self._grid.deallocate('alpha')
        self._boundary.deallocate()

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
        medium = self._problem.medium

        num_sources = shot.num_sources
        num_receivers = shot.num_receivers

        # If there's no previous operator, generate one
        if self._adjoint_operator.operator is None:
            # Define variables
            rec = self._grid.sparse_time_function('rec', num=num_receivers,
                                                  coordinates=shot.receiver_coordinates,
                                                  interpolation_type=self.interpolation_type)

            p_a = self._grid.time_function('p_a', coefficients='symbolic' if self.drp else 'standard')
            p_saved = self._grid.undersampled_time_function('p_saved',
                                                            factor=self.undersampling_factor)

            vp = self._grid.function('vp', coefficients='symbolic' if self.drp else 'standard')
            vp2 = self._grid.function('vp2', coefficients='symbolic' if self.drp else 'standard')
            inv_vp2 = self._grid.function('inv_vp2', coefficients='symbolic' if self.drp else 'standard')

            if 'rho' in medium.fields:
                rho = self._grid.function('rho', coefficients='symbolic' if self.drp else 'standard')
                buoy = self._grid.function('buoy', coefficients='symbolic' if self.drp else 'standard')

            else:
                rho = buoy = None

            if 'alpha' in medium.fields:
                alpha = self._grid.function('alpha', coefficients='symbolic' if self.drp else 'standard')

            else:
                alpha = None

            # Create stencil
            stencil = self._iso_stencil(p_a, vp, vp2, inv_vp2,
                                        rho=rho, buoy=buoy, alpha=alpha,
                                        direction='backward')

            # Define the source injection function to generate the corresponding code
            rec_term = rec.inject(field=p_a.backward, expr=-rec * time.step**2 * vp2)

            op_kwargs = {
                'dt': time.step,
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
        self._boundary.clear()

        for variable in wrt:
            self._grid.vars['grad_'+variable.name].data_with_halo.fill(0.)

        # Set prior wavefield
        # Currently, we need to use this trick to ensure no copy is made
        # of the wavefield
        class Allocator:
            @staticmethod
            def alloc(*args, **kwargs):
                return wavefield.data, None

        wavefield_data = devito.data.Data(self._grid.vars.p_saved.shape_allocated,
                                          self._grid.vars.p_saved.dtype,
                                          modulo=self._grid.vars.p_saved._mask_modulo,
                                          allocator=Allocator)

        self._grid.vars.p_saved._data = wavefield_data

        # Set medium parameters
        vp_with_halo = self._grid.with_halo(medium.vp.extended_data)
        self._grid.vars.vp.data_with_halo[:] = vp_with_halo
        self._grid.vars.vp2.data_with_halo[:] = vp_with_halo**2
        self._grid.vars.inv_vp2.data_with_halo[:] = 1 / vp_with_halo**2

        if 'rho' in medium.fields:
            rho_with_halo = self._grid.with_halo(medium.rho.extended_data)
            self._grid.vars.rho.data_with_halo[:] = rho_with_halo
            self._grid.vars.buoy.data_with_halo[:] = 1 / rho_with_halo

        if 'alpha' in medium.fields:
            db_cm_to_np_m = np.log(10)/20 * 1e2

            alpha_with_halo = self._grid.with_halo(medium.alpha.extended_data)*db_cm_to_np_m
            self._grid.vars.alpha.data_with_halo[:] = alpha_with_halo

        # Set geometry and adjoint source
        window = scipy.signal.get_window(('tukey', 0.01), time.num, False)  # TODO This should happen in a pipeline
        window = window.reshape((time.num, 1))

        self._grid.vars.rec.data[:] = adjoint_source.data.T * window

        if self.interpolation_type == 'linear':
            self._grid.vars.rec.coordinates.data[:] = shot.receiver_coordinates

    def adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Run the adjoint problem.

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
        self._adjoint_operator.run()

    def after_adjoint(self, wrt, adjoint_source, wavefield, **kwargs):
        """
        Clean up after the adjoint run and retrieve the time gradients (if needed).

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
        VariableList
            Updated variable list with gradients added to them.

        """
        wavefield.deallocate()
        self._grid.deallocate('p_saved')
        self._grid.deallocate('p_a')
        self._grid.deallocate('vp')
        self._grid.deallocate('vp2')
        self._grid.deallocate('inv_vp2')
        self._grid.deallocate('rho')
        self._grid.deallocate('buoy')
        self._grid.deallocate('alpha')
        self._boundary.deallocate()

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

        self._grid.deallocate('grad_vp')
        self._grid.deallocate('prec_vp')

    def _symbolic_coefficients(self, *functions):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _weights(self):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _diff_op(self, field, **kwargs):
        rho = kwargs.pop('rho', None)
        buoy = kwargs.pop('buoy', None)

        if buoy is None:
            return field.laplace

        else:
            return field.laplace + rho * devito.grad(buoy).dot(devito.grad(field))

    def _laplacian(self, field, laplacian, vp, vp2, inv_vp2, **kwargs):
        if self.kernel not in ['OT2', 'OT4']:
            raise ValueError("Unrecognized kernel")

        time = self._problem.time

        if self.kernel == 'OT2':
            bi_harmonic = 0

        else:
            bi_harmonic = time.step**2/12 * vp2*self._diff_op(field, **kwargs)

        laplacian_update = field + bi_harmonic

        return laplacian_update

    def _saved(self, field, vp, vp2, inv_vp2, *kwargs):
        return field.dt2

    def _iso_stencil(self, field, vp, vp2, inv_vp2, direction='forward', **kwargs):
        # Forward or backward
        forward = direction == 'forward'

        # Define time step to be updated
        u_next = field.forward if forward else field.backward

        # Get the spatial FD
        laplacian = self._grid.function('laplacian', coefficients='symbolic' if self.drp else 'standard')
        laplacian_update = self._laplacian(field, laplacian, vp, vp2, inv_vp2, **kwargs)

        if self.kernel == 'OT2':
            laplacian_term = self._diff_op(laplacian_update, **kwargs)
        else:
            laplacian_term = self._diff_op(laplacian, **kwargs)

        # Get the subs
        if self.drp:
            extra_functions = ()
            if kwargs.get('rho', None) is not None:
                extra_functions = (kwargs.get('rho'), kwargs.get('buoy'))

            subs = self._symbolic_coefficients(field, laplacian, vp, vp2, inv_vp2,
                                               *extra_functions)
        else:
            subs = None

        # Get the attenuation term
        alpha = kwargs.get('alpha', None)
        if alpha is not None:
            if self.attenuation_power == 0:
                u = field
                u_dt = u.dt if direction == 'forward' else u.dt.T

                attenuation_term = 2*alpha/vp * u_dt
            elif self.attenuation_power == 2:
                u = -field.laplace
                u_dt = u.dt if direction == 'forward' else u.dt.T

                attenuation_term = 2*alpha*vp * u_dt
            else:
                raise ValueError('The "attenuation_exponent" can only take values (0, 2).')
        else:
            attenuation_term = 0

        # Set up the boundary
        medium = self._problem.medium
        boundary_term, eq_before, eq_after = self._boundary.apply(field, medium.vp.extended_data,
                                                                  direction=direction, subs=subs,
                                                                  f_centre=self._bandwidth[1])

        # Define PDE and update rule
        # TODO The only way to make the PML work is to use OT2 in the boundary,
        #      a PML formulation including the extra term is needed for this.
        eq_interior = devito.solve(inv_vp2*field.dt2 - laplacian_term + attenuation_term, u_next)
        eq_boundary = devito.solve(inv_vp2*field.dt2 - laplacian_term + attenuation_term + boundary_term, u_next)

        # Time-stepping stencil
        stencils = []

        if self.kernel != 'OT2':
            stencil_laplacian = devito.Eq(laplacian, laplacian_update,
                                          subdomain=self._grid.full,
                                          coefficients=subs)
            stencils.append(stencil_laplacian)

        stencil_interior = devito.Eq(u_next, eq_interior,
                                     subdomain=self._grid.interior,
                                     coefficients=subs)
        stencils.append(stencil_interior)

        stencil_boundary = [devito.Eq(u_next, eq_boundary,
                                      subdomain=dom,
                                      coefficients=subs) for dom in self._grid.pml]
        stencils += stencil_boundary

        return eq_before + stencils + eq_after

    def _dt_max(self, k, h, vp_max):
        space = self._problem.space

        return k * h / vp_max * 1 / np.sqrt(space.dim)
