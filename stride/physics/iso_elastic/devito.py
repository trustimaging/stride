
import os
import devito
from devito.types import Buffer
import numpy as np
import scipy.signal

import mosaic
from mosaic.utils import camel_case

from stride.utils import fft
from stride.problem import ScalarField
from ..common.devito import GridDevito, OperatorDevito
from ..problem_type import ProblemTypeBase
from .. import boundaries


__all__ = ['IsoElasticDevito']


@mosaic.tessera
class IsoElasticDevito(ProblemTypeBase):
    """

    Parameters
    ----------
    name : str, optional
        Name of the PDE, defaults to an automatic name.
    grid : Grid, optional
        Existing grid, if not provided one will be created. Either a grid or
        space, time and slow_time need to be provided.
    space : Space, optional
    time : Time, optional
    slow_time : SlowTime, optional

    Notes
    -----

    For forward execution of the PDE, the following parameters can be used:

        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.
        save_wavefield : bool or int, optional
            Whether or not to solve the forward wavefield, defaults to True when
            a gradient is expected, and to False otherwise. An integer number N can
            also be provided, in which case the last N timesteps of the wavefield
            will be saved.
        boundary_type : str, optional
            Type of boundary for the wave equation (``sponge_boundary_2`` or
            ``complex_frequency_shift_PML_2``), defaults to ``sponge_boundary_2``.
        interpolation_type : str, optional
            Type of source/receiver interpolation (``linear`` or ``hicks``), defaults
            to ``linear``.
        attenuation_power : int, optional
            Power of the attenuation law if attenuation is given (``0`` or ``2``),
            defaults to ``0``.
        drp : bool, optional
            Whether or not to use dispersion-relation preserving coefficients (only
            available in some versions of Stride). Defaults to False.
        kernel : str, optional
            Type of time kernel to use (``OT2`` for 2nd order in time or ``OT4`` for 4th
            order in time). If not given, it is automatically decided given the time spacing.
        undersampling : int, optional
            Amount of undersampling in time when saving the forward wavefield. If not given,
            it is calculated given the bandwidth.


    """

    space_order = 10
    time_order = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kernel = 'OT4'
        self.drp = False
        self.undersampling_factor = 4
        self.boundary_type = 'sponge_boundary_2'
        self.interpolation_type = 'linear'
        self.attenuation_power = 0

        self.wavefield = None

        self._max_wavelet = 0.
        self._src_scale = 0.
        self._bandwidth = 0.

        self.dev_grid = GridDevito(self.space_order, self.time_order, **kwargs)

        kwargs.pop('grid', None)
        self.state_operator = OperatorDevito(self.space_order, self.time_order,
                                             name='acoustic_iso_state',
                                             grid=self.dev_grid,
                                             **kwargs)
        self.adjoint_operator = OperatorDevito(self.space_order, self.time_order,
                                               name='acoustic_iso_adjoint',
                                               grid=self.dev_grid,
                                               **kwargs)
        self.boundary = None

    def clear_operators(self):
        self.state_operator.devito_operator = None
        self.adjoint_operator.devito_operator = None

    # forward

    async def before_forward(self, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.
        save_wavefield : bool or int, optional
            Whether or not to solve the forward wavefield, defaults to True when
            a gradient is expected, and to False otherwise. An integer number N can
            also be provided, in which case the last N timesteps of the wavefield
            will be saved.
        boundary_type : str, optional
            Type of boundary for the wave equation (``sponge_boundary_2`` or
            ``complex_frequency_shift_PML_2``), defaults to ``sponge_boundary_2``.
        interpolation_type : str, optional
            Type of source/receiver interpolation (``linear`` or ``hicks``), defaults
            to ``linear``.
        attenuation_power : int, optional
            Power of the attenuation law if attenuation is given (``0`` or ``2``),
            defaults to ``0``.
        drp : bool, optional
            Whether or not to use dispersion-relation preserving coefficients (only
            available in some versions of Stride). Defaults to False.
        kernel : str, optional
            Type of time kernel to use (``OT2`` for 2nd order in time or ``OT4`` for 4th
            order in time). If not given, it is automatically decided given the time spacing.
        undersampling : int, optional
            Amount of undersampling in time when saving the forward wavefield. If not given,
            it is calculated given the bandwidth.


        Returns
        -------

        """
        problem = kwargs.get('problem')
        shot = problem.shot

        self._check_problem(wavelets, vp, rho=rho, alpha=alpha, **kwargs)

        num_sources = shot.num_sources
        num_receivers = shot.num_receivers

        save_wavefield = kwargs.get('save_wavefield', False) # In acoustic case we have three parameters which need gradients, not needed yet
        if save_wavefield is False:
            save_wavefield = vp.needs_grad
            if rho is not None:
                save_wavefield |= rho.needs_grad
            if alpha is not None:
                save_wavefield |= alpha.needs_grad

        # If there's no previous operator, generate one
        if self.state_operator.devito_operator is None:
            save = Buffer(save_wavefield) if type(save_wavefield) is int else None

            # Define variables, each devito container has a method in the dev_grid object
            src = self.dev_grid.sparse_time_function('src', num=num_sources, # Set the sources
                                                     coordinates=shot.source_coordinates,
                                                     interpolation_type=self.interpolation_type)
            rec = self.dev_grid.sparse_time_function('rec', num=num_receivers, # Set the receivers
                                                     coordinates=shot.receiver_coordinates,
                                                     interpolation_type=self.interpolation_type)

            p = self.dev_grid.time_function('p', coefficients='symbolic' if self.drp else 'standard', save=save) # what is this?

            # Create stencil
            stencil = self._stencil(p, wavelets, vp, rho=rho, alpha=alpha, direction='forward') # the stencil contains the sympy expression that devito will solve

            # Define the source injection function to generate the corresponding code
            # pressure_to_mass_acc = time.step / (vp * space.spacing**3)
            # solve_p_dt = time.step**2 * vp**2
            vp2 = self.dev_grid.vars.vp2
            src_term = src.inject(field=p.forward, expr=src * self.time.step**2 * vp2) # this is designed to match the source injection from k-wave
            rec_term = rec.interpolate(expr=p)

            # Define the saving of the wavefield
            if save_wavefield is True:
                p_saved = self.dev_grid.undersampled_time_function('p_saved',
                                                                   factor=self.undersampling_factor)
                update_saved = [devito.Eq(p_saved, self._saved(p))]

            else:
                update_saved = []

            # Compile the operator
            self.state_operator.set_operator(stencil + src_term + rec_term + update_saved,
                                             **kwargs)
            self.state_operator.compile()

        else:
            # If the source/receiver size has changed, then create new functions for them
            if num_sources != self.dev_grid.vars.src.npoint:
                self.dev_grid.sparse_time_function('src', num=num_sources, cached=False)

            if num_receivers != self.dev_grid.vars.rec.npoint:
                self.dev_grid.sparse_time_function('rec', num=num_receivers, cached=False)

        # Clear all buffers
        self.dev_grid.vars.src.data_with_halo.fill(0.)
        self.dev_grid.vars.rec.data_with_halo.fill(0.)
        self.dev_grid.vars.p.data_with_halo.fill(0.)

        if save_wavefield is True:
            self.dev_grid.vars.p_saved.data_with_halo.fill(0.)

        # Set medium parameters
        vp_with_halo = self.dev_grid.with_halo(vp.extended_data)
        self.dev_grid.vars.vp.data_with_halo[:] = vp_with_halo
        self.dev_grid.vars.vp2.data_with_halo[:] = vp_with_halo**2
        self.dev_grid.vars.inv_vp2.data_with_halo[:] = 1 / vp_with_halo**2

        if rho is not None:
            rho_with_halo = self.dev_grid.with_halo(rho.extended_data)
            self.dev_grid.vars.rho.data_with_halo[:] = rho_with_halo
            self.dev_grid.vars.buoy.data_with_halo[:] = 1 / rho_with_halo

        if alpha is not None:
            db_to_neper = 100 * (1e-6 / (2*np.pi))**self.attenuation_power / (20 * np.log10(np.exp(1)))

            alpha_with_halo = self.dev_grid.with_halo(alpha.extended_data)*db_to_neper
            self.dev_grid.vars.alpha.data_with_halo[:] = alpha_with_halo

        # Set geometry and wavelet
        wavelets = wavelets.data

        self._src_scale = 1000
        self._max_wavelet = np.max(np.abs(wavelets)) + 1e-31

        window = scipy.signal.get_window(('tukey', 0.01), self.time.num, False)
        window = window.reshape((self.time.num, 1))

        self.dev_grid.vars.src.data[:] = wavelets.T * self._src_scale / self._max_wavelet * window

        if self.interpolation_type == 'linear':
            self.dev_grid.vars.src.coordinates.data[:] = shot.source_coordinates
            self.dev_grid.vars.rec.coordinates.data[:] = shot.receiver_coordinates

    async def run_forward(self, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Run the state or forward problem.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        functions = dict(
            src=self.dev_grid.vars.src,
            rec=self.dev_grid.vars.rec,
        )

        self.state_operator.run(dt=self.time.step,
                                **functions,
                                **kwargs.get('devito_args', {}))

    async def after_forward(self, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Clean up after the state run and retrieve the time traces.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------
        Traces
            Time traces produced by the state run.

        """
        problem = kwargs.pop('problem')
        shot = problem.shot

        save_wavefield = kwargs.get('save_wavefield', False)
        if save_wavefield is False:
            save_wavefield = vp.needs_grad
            if rho is not None:
                save_wavefield |= rho.needs_grad
            if alpha is not None:
                save_wavefield |= alpha.needs_grad

        if save_wavefield:
            if save_wavefield is True:
                wavefield_data = np.asarray(self.dev_grid.vars.p_saved.data_with_halo, dtype=np.float32)

            else:
                wavefield_data = np.asarray(self.dev_grid.vars.p.data, dtype=np.float32)

            wavefield_slice = kwargs.pop('wavefield_slice', None)
            if wavefield_slice is not None:
                wavefield_data = wavefield_data[wavefield_slice]

            wavefield_data *= self._max_wavelet / self._src_scale

            self.wavefield = ScalarField(name='p_dt2',
                                         data=wavefield_data,
                                         shape=wavefield_data.shape)

            if os.environ.get('STRIDE_DUMP_WAVEFIELD', None) == 'yes':
                self.wavefield.dump(path=problem.output_folder,
                                    project_name=problem.name)

        else:
            self.wavefield = None

        traces_data = np.asarray(self.dev_grid.vars.rec.data, dtype=np.float32).T
        traces_data *= self._max_wavelet / self._src_scale
        traces = shot.observed.alike(name='modelled', data=traces_data)

        self.dev_grid.deallocate('p')
        self.dev_grid.deallocate('src')
        self.dev_grid.deallocate('rec')
        self.dev_grid.deallocate('vp')
        self.dev_grid.deallocate('vp2')
        self.dev_grid.deallocate('inv_vp2')
        self.dev_grid.deallocate('rho')
        self.dev_grid.deallocate('buoy')
        self.dev_grid.deallocate('alpha')
        self.boundary.deallocate()

        return traces

    # adjoint

    async def before_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """

        Parameters
        ----------
        adjoint_source : Traces
            Adjoint source.
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        pass

    async def run_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Run the adjoint problem.

        Parameters
        ----------
        adjoint_source : Traces
            Adjoint source.
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        pass

    async def after_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Clean up after the adjoint run and retrieve the time gradients (if needed).

        Parameters
        ----------
        adjoint_source : Traces
            Adjoint source.
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional speed of sound fo the medium, in [m/s].
        rho : ScalarField, optional
            Density of the medium, defaults to homogeneous, in [kg/m^3].
        alpha : ScalarField, optional
            Attenuation coefficient of the medium, defaults to 0, in [Np/m].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------
        tuple of gradients
            Tuple with the gradients of the variables that need them

        """
        pass
    # gradients

    async def prepare_grad_vp(self, vp, **kwargs):
        """
        Prepare the problem type to calculate the gradients wrt Vp.

        Parameters
        ----------
        vp : ScalarField
            Vp variable to calculate the gradient.

        Returns
        -------
        tuple
            Tuple of gradient and preconditioner updates.

        """
        pass

    async def init_grad_vp(self, vp, **kwargs):
        """
        Initialise buffers in the problem type to calculate the gradients wrt Vp.

        Parameters
        ----------
        vp : ScalarField
            Vp variable to calculate the gradient.

        Returns
        -------

        """
        pass

    async def get_grad_vp(self, vp, **kwargs):
        """
        Retrieve the gradients calculated wrt to the input.

        The variable is updated inplace.

        Parameters
        ----------
        vp : ScalarField
            Vp variable to calculate the gradient.

        Returns
        -------
        ScalarField
            Gradient wrt Vp.

        """
        pass

    # utils

    def _check_problem(self, wavelets, vp, rho=None, alpha=None, **kwargs):
        problem = kwargs.get('problem')

        recompile = False

        boundary_type = kwargs.get('boundary_type', 'sponge_boundary_2')
        if boundary_type != self.boundary_type or self.boundary is None:
            recompile = True
            self.boundary_type = boundary_type

            if isinstance(self.boundary_type, str):
                boundaries_module = boundaries.devito
                self.boundary = getattr(boundaries_module, camel_case(self.boundary_type))(self.dev_grid)

            else:
                self.boundary = self.boundary_type

        interpolation_type = kwargs.get('interpolation_type', 'linear')
        if interpolation_type != self.interpolation_type:
            recompile = True
            self.interpolation_type = interpolation_type

        attenuation_power = kwargs.get('attenuation_power', 0)
        if attenuation_power != self.attenuation_power:
            recompile = True
            self.attenuation_power = attenuation_power

        drp = kwargs.get('drp', False)
        if drp != self.drp:
            recompile = True
            self.drp = drp

        preferred_kernel = kwargs.get('kernel', None)
        preferred_undersampling = kwargs.get('undersampling', None)

        self._check_conditions(wavelets, vp, rho, alpha,
                               preferred_kernel, preferred_undersampling,
                               **kwargs)

        # Recompile every time if using hicks
        if self.interpolation_type == 'hicks' or recompile:
            self.state_operator.devito_operator = None
            self.adjoint_operator.devito_operator = None

        runtime = mosaic.runtime()
        runtime.logger.info('(ShotID %d) Selected time stepping scheme %s' %
                            (problem.shot_id, self.kernel,))

    def _check_conditions(self, wavelets, vp, rho=None, alpha=None,
                          preferred_kernel=None, preferred_undersampling=None, **kwargs):
        runtime = mosaic.runtime()

        problem = kwargs.get('problem')

        # Get speed of sound bounds
        vp_min = np.min(vp.extended_data)
        vp_max = np.max(vp.extended_data)

        # Figure out propagated bandwidth
        wavelets = wavelets.data
        f_min, f_centre, f_max = fft.bandwidth(wavelets, self.time.step, cutoff=-10)

        self._bandwidth = (f_min, f_centre, f_max)

        runtime.logger.info('(ShotID %d) Estimated bandwidth for the propagated '
                            'wavelet %.3f-%.3f MHz' % (problem.shot_id, f_min / 1e6, f_max / 1e6))

        # Check for dispersion
        if self.drp is True:
            self.drp = False

            runtime.logger.warn('(ShotID %d) DRP weights are not implemented in this version of stride' %
                                problem.shot_id)

        h = max(*self.space.spacing)

        wavelength = vp_min / f_max
        ppw = wavelength / h
        ppw_max = 5

        h_max = wavelength / ppw_max

        if h > h_max:
            runtime.logger.warn('(ShotID %d) Spatial grid spacing (%.3f mm | %d PPW) is '
                                'higher than dispersion limit (%.3f mm | %d PPW)' %
                                (problem.shot_id, h / 1e-3, ppw, h_max / 1e-3, ppw_max))
        else:
            runtime.logger.info('(ShotID %d) Spatial grid spacing (%.3f mm | %d PPW) is '
                                'below dispersion limit (%.3f mm | %d PPW)' %
                                (problem.shot_id, h / 1e-3, ppw, h_max / 1e-3, ppw_max))

        # Check for instability
        dt = self.time.step

        dt_max_OT2 = self._dt_max(2.0 / np.pi, h, vp_max)
        dt_max_OT4 = self._dt_max(3.6 / np.pi, h, vp_max)

        crossing_factor = dt*vp_max / h * 100

        recompile = False
        if dt <= dt_max_OT2:
            runtime.logger.info('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                                'below OT2 limit (%.3f \u03BCs)' %
                                (problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT2 / 1e-6))

            selected_kernel = 'OT2'

        elif dt <= dt_max_OT4:
            runtime.logger.info('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                                'above OT2 limit (%.3f \u03BCs)'
                                % (problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT2 / 1e-6))

            selected_kernel = 'OT4'

        else:
            runtime.logger.warn('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                                'above OT4 limit (%.3f \u03BCs)'
                                % (problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT4 / 1e-6))

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
                            (problem.shot_id, undersampling,))

        # Maybe recompile
        if recompile:
            self.state_operator.operator = None
            self.adjoint_operator.operator = None

    def _stencil(self, field, wavelets, vp, rho=None, alpha=None, direction='forward', **kwargs):
        # Prepare medium functions
        vp_fun, vp2_fun, inv_vp2_fun, rho_fun, buoy_fun, alpha_fun = self._medium_functions(vp, rho, alpha, **kwargs)

        # Forward or backward
        forward = direction == 'forward'

        # Define time step to be updated
        u_next = field.forward if forward else field.backward

        # Get the spatial FD
        laplacian = self.dev_grid.function('laplacian',
                                           coefficients='symbolic' if self.drp else 'standard')
        laplacian_update = self._laplacian(field, laplacian, vp_fun, vp2_fun, inv_vp2_fun,
                                           rho=rho_fun, buoy=buoy_fun, alpha=alpha_fun,
                                           **kwargs)

        if self.kernel == 'OT2':
            laplacian_term = self._diff_op(laplacian_update,
                                           rho=rho_fun, buoy=buoy_fun, alpha=alpha_fun,
                                           **kwargs)
        else:
            laplacian_term = self._diff_op(laplacian,
                                           rho=rho_fun, buoy=buoy_fun, alpha=alpha_fun,
                                           **kwargs)

        # Get the subs
        if self.drp:
            extra_functions = ()
            if rho_fun is not None:
                extra_functions = (rho_fun, buoy_fun)

            subs = self._symbolic_coefficients(field, laplacian, vp_fun, vp2_fun, inv_vp2_fun,
                                               *extra_functions)
        else:
            subs = None

        # Get the attenuation term
        if alpha_fun is not None:
            if self.attenuation_power == 0:
                u = field
                u_dt = u.dt if direction == 'forward' else u.dt.T

                attenuation_term = 2*alpha_fun/vp_fun * u_dt
            elif self.attenuation_power == 2:
                u = -field.laplace
                u_dt = u.dt if direction == 'forward' else u.dt.T

                attenuation_term = 2*alpha_fun*vp_fun * u_dt
            else:
                raise ValueError('The "attenuation_exponent" can only take values (0, 2).')
        else:
            attenuation_term = 0

        # Set up the boundary
        boundary_term, eq_before, eq_after = self.boundary.apply(field, vp.extended_data,
                                                                 direction=direction, subs=subs,
                                                                 f_centre=self._bandwidth[1])

        # Define PDE and update rule
        # TODO The only way to make the PML work is to use OT2 in the boundary,
        #      a PML formulation including the extra term is needed for this.
        eq_interior = devito.solve(inv_vp2_fun*field.dt2 - laplacian_term + attenuation_term, u_next)
        eq_boundary = devito.solve(inv_vp2_fun*field.dt2 - laplacian_term + attenuation_term + boundary_term, u_next)

        # Time-stepping stencil
        stencils = []

        if self.kernel != 'OT2':
            stencil_laplacian = devito.Eq(laplacian, laplacian_update,
                                          subdomain=self.dev_grid.full,
                                          coefficients=subs)
            stencils.append(stencil_laplacian)

        stencil_interior = devito.Eq(u_next, eq_interior,
                                     subdomain=self.dev_grid.interior,
                                     coefficients=subs)
        stencils.append(stencil_interior)

        stencil_boundary = [devito.Eq(u_next, eq_boundary,
                                      subdomain=dom,
                                      coefficients=subs) for dom in self.dev_grid.pml]
        stencils += stencil_boundary

        return eq_before + stencils + eq_after

    def _medium_functions(self, vp, rho=None, alpha=None, **kwargs):
        _kwargs = dict(coefficients='symbolic' if self.drp else 'standard')

        vp_fun = self.dev_grid.function('vp', **_kwargs)
        vp2_fun = self.dev_grid.function('vp2', **_kwargs)
        inv_vp2_fun = self.dev_grid.function('inv_vp2', **_kwargs)

        if rho is not None:
            rho_fun = self.dev_grid.function('rho', **_kwargs)
            buoy_fun = self.dev_grid.function('buoy', **_kwargs)
        else:
            rho_fun = buoy_fun = None

        if alpha is not None:
            alpha_fun = self.dev_grid.function('alpha', **_kwargs)
        else:
            alpha_fun = None

        return vp_fun, vp2_fun, inv_vp2_fun, rho_fun, buoy_fun, alpha_fun

    def _laplacian(self, field, laplacian, vp, vp2, inv_vp2, **kwargs):
        if self.kernel not in ['OT2', 'OT4']:
            raise ValueError("Unrecognized kernel")

        if self.kernel == 'OT2':
            bi_harmonic = 0

        else:
            bi_harmonic = self.time.step**2/12 * vp2*self._diff_op(field, **kwargs)

        laplacian_update = field + bi_harmonic

        return laplacian_update

    def _diff_op(self, field, **kwargs):
        rho = kwargs.pop('rho', None)
        buoy = kwargs.pop('buoy', None)

        if buoy is None:
            return field.laplace

        else:
            if self.drp:
                return field.laplace + rho * devito.grad(buoy, shift=-0.5).dot(devito.grad(field, shift=-0.5))
            else:
                return rho * devito.div(buoy * devito.grad(field, shift=+0.5), shift=-0.5)

    def _saved(self, field, *kwargs):
        return field.dt2

    def _symbolic_coefficients(self, *functions):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _weights(self):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _dt_max(self, k, h, vp_max):
        return k * h / vp_max * 1 / np.sqrt(self.space.dim)