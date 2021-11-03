
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
            !!!!!!!!!!!!!!!!!!!!!!!.
        rho : ScalarField, optional
            !!!!!!!!!!!!!!!!!!!!!!!.
        alpha : ScalarField, optional
            !!!!!!!!!!!!!!!!!!!!!!!.
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

        self.kernel = 'OT2'
        self.drp = False
        self.undersampling_factor = 4
        self.boundary_type = 'sponge_boundary_2'
        self.interpolation_type = 'linear'
        self.attenuation_power = 0

        self.wavefield = None

        self._max_wavelet = 0.
        self._src_scale = 0.
        self._bandwidth = 0.

        # config_devito(**kwargs)

        self.dev_grid = GridDevito(self.space_order, self.time_order, **kwargs)

        kwargs.pop('grid', None)
        self.state_operator = OperatorDevito(self.space_order, self.time_order,
                                             name='elastic_iso_state',
                                             grid=self.dev_grid,
                                             **kwargs)
        self.adjoint_operator = OperatorDevito(self.space_order, self.time_order,
                                               name='elastic_iso_adjoint',
                                               grid=self.dev_grid,
                                               **kwargs)
        self.boundary = None


    def clear_operators(self):
        self.state_operator.devito_operator = None
        self.adjoint_operator.devito_operator = None

    # forward

    async def before_forward(self, wavelets, vp, vs, rho, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        rho : ScalarField, optional
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        alpha : ScalarField, optional
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

        num_sources = shot.num_sources
        num_receivers = shot.num_receivers

        # If there's no previous operator, generate one
        if self.state_operator.devito_operator is None:

            # Define variables
            src = self.dev_grid.sparse_time_function('src', num=num_sources,
                                                     coordinates=shot.source_coordinates,
                                                     interpolation_type=self.interpolation_type)
            rec_tau = self.dev_grid.sparse_time_function('rec_tau', num=num_receivers,
                                                     coordinates=shot.receiver_coordinates,
                                                     interpolation_type=self.interpolation_type)

            s = self.time.step
            # Create stencil
            # save_wavefield = kwargs.get('save_wavefield', False)
            # save = Buffer(save_wavefield) if type(save_wavefield) is int else None
            vel = self.dev_grid.vector_time_function('vel')
            tau = self.dev_grid.tensor_time_function('tau')

            # Absorbing boundaries
            damp = devito.Function(name="damp", grid=self.dev_grid.devito_grid)
            eqs = [devito.Eq(damp, 1.0)]
            padsizes = [self.space.absorbing for _ in range(self.dev_grid.devito_grid.dim)]
            for (nbl, nbr), d in zip(padsizes, damp.dimensions):
                dampcoeff = 0.008635*self.space.spacing[0] # 0.2763102111592855 (water_example)
                # left
                dim_l = devito.SubDimension.left(name='abc_%s_l' % d.name, parent=d, thickness=nbl)
                pos = devito.Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
                val = - dampcoeff * (pos - devito.sin(2 * np.pi * pos) / (2 * np.pi))
                eqs += [devito.Inc(damp.subs({d: dim_l}), val / d.spacing)]
                # right
                dampcoeff = 0.008635*self.space.spacing[0] # 0.2763102111592855 (water_example)
                dim_r = devito.SubDimension.right(name='abc_%s_r' % d.name, parent=d, thickness=nbr)
                pos = devito.Abs((nbr - (d.symbolic_max - dim_r) + 1) / float(nbr))
                val = - dampcoeff * (pos - devito.sin(2 * np.pi * pos) / (2 * np.pi))
                eqs += [devito.Inc(damp.subs({d: dim_r}), val / d.spacing)]
            devito.Operator(eqs, name='initdamp')()

            # Define the source injection function using a pressure disturbance
            src_xx = src.inject(field=tau.forward[0, 0], expr=s * src)
            src_zz = src.inject(field=tau.forward[1, 1], expr=s * src)

            rec_term = rec_tau.interpolate(expr=tau[0, 0] + tau[1, 1])
            # rec_term += rec_v1.interpolate(expr=v[1])  # Placeholder for vel_x receiver
            # rec_term += rec_v2.interpolate(expr=v[0])  # Placeholder for vel_y receiver

            # Set up parameters as functions
            lam_fun = self.dev_grid.function('lam_fun')
            mu_fun = self.dev_grid.function('mu_fun')
            byn_fun = self.dev_grid.function('byn_fun')
            # devito.Function(name='lam_fun', grid=self.dev_grid.devito_grid, space_order=self.dev_grid.space_order, parameter=True)
            # devito.Function(name='mu_fun', grid=self.dev_grid.devito_grid, space_order=self.dev_grid.space_order, parameter=True)
            # devito.Function(name='byn_fun', grid=self.dev_grid.devito_grid, space_order=self.dev_grid.space_order, parameter=True)

            # Compile the operator
            # velocity (first derivative vel w.r.t. time, first order euler method), s: time_spacing
            u_v = devito.Eq(vel.forward,
                            damp*(vel + s * byn_fun * devito.div(tau)),
                            grid=self.dev_grid,
                            coefficients=None)

            # stress (first derivative tau w.r.t. time, first order euler method)
            u_tau = devito.Eq(tau.forward,
                              damp*(tau + s * (lam_fun * devito.diag(devito.div(vel.forward)) + mu_fun * (devito.grad(vel.forward) + devito.grad(vel.forward).T))))

            self.state_operator.set_operator([u_v] + [u_tau] + src_xx + src_zz + rec_term,
                                             **kwargs)
            self.state_operator.compile()

        else:
            # If the source/receiver size has changed, then create new functions for them
            if num_sources != self.dev_grid.vars.src.npoint:
                self.dev_grid.sparse_time_function('src', num=num_sources, cached=False)

            if num_receivers != self.dev_grid.vars.rec_tau.npoint:
                self.dev_grid.sparse_time_function('rec', num=num_receivers, cached=False)

        # Clear all buffers
        self.dev_grid.vars.src.data_with_halo.fill(0.)
        self.dev_grid.vars.rec_tau.data_with_halo.fill(0.)
        self.dev_grid.vars.vel[0].data_with_halo.fill(0.)
        self.dev_grid.vars.vel[1].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[0,0].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[0,1].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[1,0].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[1,1].data_with_halo.fill(0.)
        # self.boundary.clear()

        # Set medium parameters
        vp_with_halo = self.dev_grid.with_halo(vp.extended_data)
        vs_with_halo = self.dev_grid.with_halo(vs.extended_data)
        rho_with_halo = self.dev_grid.with_halo(rho.extended_data)

        lam_with_halo = rho_with_halo * (vp_with_halo ** 2 - 2. * vs_with_halo ** 2)
        mu_with_halo = self.dev_grid.with_halo(rho_with_halo * vs_with_halo ** 2)
        byn_with_halo = self.dev_grid.with_halo(1 / rho_with_halo)

        self.dev_grid.vars.lam_fun.data_with_halo[:] = lam_with_halo
        self.dev_grid.vars.mu_fun.data_with_halo[:] = mu_with_halo
        self.dev_grid.vars.byn_fun.data_with_halo[:] = byn_with_halo

        # Set geometry and wavelet
        wavelets = wavelets.data

        self._src_scale = 1000
        self._max_wavelet = np.max(np.abs(wavelets)) + 1e-31

        window = scipy.signal.get_window(('tukey', 0.01), self.time.num, False)
        window = window.reshape((self.time.num, 1))

        self.dev_grid.vars.src.data[:] = wavelets.T * self._src_scale / self._max_wavelet * window

        if self.interpolation_type == 'linear':
            self.dev_grid.vars.src.coordinates.data[:] = shot.source_coordinates
            self.dev_grid.vars.rec_tau.coordinates.data[:] = shot.receiver_coordinates

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
            )

        self.state_operator.run(dt=self.time.step,
                                **functions,
                                **kwargs.pop('devito_args', {}))

    async def after_forward(self, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Clean up after the state run and retrieve the time traces.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        rho : ScalarField, optional
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        alpha : ScalarField, optional
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------
        Traces
            Time traces produced by the state run.

        """
        problem = kwargs.pop('problem')
        shot = problem.shot

        self.wavefield = None

        traces_data = np.asarray(self.dev_grid.vars.rec_tau.data, dtype=np.float32).T
        traces_data *= self._max_wavelet / self._src_scale
        traces = shot.observed.alike(name='modelled', data=traces_data)

        self.dev_grid.deallocate('p')
        self.dev_grid.deallocate('laplacian')
        self.dev_grid.deallocate('src')
        self.dev_grid.deallocate('rec')
        self.dev_grid.deallocate('vp')
        self.dev_grid.deallocate('vp2')
        self.dev_grid.deallocate('inv_vp2')
        self.dev_grid.deallocate('rho')
        self.dev_grid.deallocate('buoy')
        self.dev_grid.deallocate('alpha')
        # self.boundary.deallocate()

        return traces

    # adjoint

    async def before_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Not implemented
        """
        pass

    async def run_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Not implemented
        """
        pass

    async def after_adjoint(self, **kwargs):
        """
        Not implemented
        """
        pass
    # gradients

    async def prepare_grad_vp(self, **kwargs):
        """
        Not implemented
        """
        pass

    async def init_grad_vp(self, **kwargs):
        """
        Not implemented
        """
        pass

    async def get_grad_vp(self, **kwargs):
        """
        Not implemented
        """
        pass

    # utils

    def _check_problem(self, wavelets, vp, vs, density, **kwargs):
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

        self._check_conditions(wavelets, vp, vs, density,
                               preferred_kernel, preferred_undersampling,
                               **kwargs)

        # Recompile every time if using hicks
        if self.interpolation_type == 'hicks' or recompile:
            self.state_operator.devito_operator = None
            self.adjoint_operator.devito_operator = None

        runtime = mosaic.runtime()
        runtime.logger.info('(ShotID %d) Selected time stepping scheme %s' %
                            (problem.shot_id, self.kernel,))

    def _check_conditions(self, wavelets, vp, vs, density,
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

    def _medium_functions(self, vp, rho=None, alpha=None, **kwargs):
        _kwargs = dict(coefficients='symbolic' if self.drp else 'standard')

        v_fun = devito.VectorTimeFunction(name='v', grid=self.grid, space_order=self.space_order)
        tau_fun = devito.TensorTimeFunction(name='t', grid=self.grid, space_order=self.space_order)

        # vp_fun = self.dev_grid.function('vp', **_kwargs)
        # vp2_fun = self.dev_grid.function('vp2', **_kwargs)
        # inv_vp2_fun = self.dev_grid.function('inv_vp2', **_kwargs)

        return v_fun, tau_fun

    def _saved(self, field, *kwargs):
        return field.dt2

    def _symbolic_coefficients(self, *functions):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _weights(self):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _dt_max(self, k, h, vp_max):
        return k * h / vp_max * 1 / np.sqrt(self.space.dim)
