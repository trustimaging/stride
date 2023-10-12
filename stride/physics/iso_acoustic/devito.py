
import os
import glob
import shutil
import tempfile
import warnings
import numpy as np
import scipy.signal

import mosaic
from mosaic.utils import camel_case, at_exit
from mosaic.comms.compression import maybe_compress, decompress

from stride.utils import fft
from stride.problem import StructuredData
from ..common.devito import GridDevito, OperatorDevito, config_devito, devito
from ..problem_type import ProblemTypeBase
from .. import boundaries


__all__ = ['IsoAcousticDevito']


warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


@mosaic.tessera
class IsoAcousticDevito(ProblemTypeBase):
    """
    This class represents the second-order isotropic acoustic wave equation,
    implemented using Devito.

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
        problem : Problem
            Sub-problem being solved by the PDE.
        save_wavefield : bool, optional
            Whether or not to solve the forward wavefield, defaults to True when
            a gradient is expected, and to False otherwise.
        save_bounds : tuple of int, optional
            If saving the wavefield, specify the ``(min timestep, max timestep)``
            where the wavefield should be saved
        save_undersampling : int, optional
            Amount of undersampling in time when saving the forward wavefield. If not given,
            it is calculated given the bandwidth.
        save_compression : str, optional
            Compression applied to saved wavefield, only available with DevitoPRO. Defaults to no
            compression in 2D and `bitcomp` in 3D.
        boundary_type : str, optional
            Type of boundary for the wave equation (``sponge_boundary_2`` or
            ``complex_frequency_shift_PML_2``), defaults to ``sponge_boundary_2``.
            Note that ``complex_frequency_shift_PML_2`` boundaries have lower OT4 stability
            limit than other boundaries.
        interpolation_type : str, optional
            Type of source/receiver interpolation (``linear`` for bi-/tri-linear or ``hicks`` for sinc
            interpolation), defaults to ``linear``.
        attenuation_power : int, optional
            Power of the attenuation law if attenuation is given (``0`` or ``2``),
            defaults to ``0``.
        drp : bool, optional
            Whether or not to use dispersion-relation preserving coefficients (only
            available in some versions of Stride). Defaults to False.
        kernel : str, optional
            Type of time kernel to use (``OT2`` for 2nd order in time or ``OT4`` for 4th
            order in time). If not given, it is automatically decided given the time spacing.
        diff_source : bool, optional
            Whether the source should be injected as is, or as its 1st time derivative. Defaults to
            False, leaving it unchanged.
        adaptive_boxes : bool, optional
            Whether to activate adaptive boxes (requires DevitoPRO and only
            available in some versions of Stride). Defaults to False.
        platform : str, optional
            Platform on which to run the operator, ``None`` to run on the CPU or ``nvidia-acc`` to run on
            the GPU with OpenACC. Defaults to ``None``.
        devito_config : dict, optional
            Additional keyword arguments to configure Devito before operator generation.
        devito_args : dict, optional
            Additional keyword arguments used when calling the generated operator.


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
        self.adaptive_boxes = False

        self._wavefield = None

        self._bandwidth = 0.

        self._cached_operator = kwargs.pop('cached_operator', False)
        cached_name = self.__class__.__name__.lower()
        try:
            warehouse = mosaic.get_local_warehouse()
        except AttributeError:
            self._cached_operator = False

        if not self._cached_operator or ('%s_dev_grid' % cached_name) not in warehouse:
            config_devito(**kwargs)

            dev_grid = kwargs.pop('dev_grid', None)
            self.dev_grid = dev_grid or GridDevito(self.space_order, self.time_order, **kwargs)

            kwargs.pop('grid', None)
            self.state_operator = OperatorDevito(self.space_order, self.time_order,
                                                 name='acoustic_iso_state',
                                                 grid=self.dev_grid,
                                                 **kwargs)
            self.adjoint_operator = OperatorDevito(self.space_order, self.time_order,
                                                   name='acoustic_iso_adjoint',
                                                   grid=self.dev_grid,
                                                   **kwargs)

            if self._cached_operator:
                warehouse['%s_dev_grid' % cached_name] = self.dev_grid
                warehouse['%s_state_operator' % cached_name] = self.state_operator
                warehouse['%s_adjoint_operator' % cached_name] = self.adjoint_operator

        else:
            self.dev_grid = warehouse['%s_dev_grid' % cached_name]
            self.state_operator = warehouse['%s_state_operator' % cached_name]
            self.adjoint_operator = warehouse['%s_adjoint_operator' % cached_name]

        self.boundary = None

        self._cache_folder = None

        self._sub_ops = []

        self._cached_subdomains = None

    def clear_operators(self):
        self.state_operator.devito_operator = None
        self.adjoint_operator.devito_operator = None

    def add_sub_op(self, sub_op):
        sub_op = sub_op(grid=self.grid, parent_grid=self.dev_grid.devito_grid, dtype=self.dev_grid.dtype)
        self._sub_ops.append(sub_op)

    @property
    def wavefield(self):
        if self._wavefield is None:
            return None

        wavefield_data = np.asarray(self._wavefield.data, dtype=np.float32)
        wavefield = StructuredData(name='p',
                                   data=wavefield_data,
                                   shape=wavefield_data.shape)

        return wavefield

    @property
    def subdomains(self):
        return self._cached_subdomains

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
        problem : Problem
            Sub-problem being solved by the PDE.
        save_wavefield : bool, optional
            Whether or not to solve the forward wavefield, defaults to True when
            a gradient is expected, and to False otherwise.
        save_bounds : tuple of int, optional
            If saving the wavefield, specify the ``(min timestep, max timestep)``
            where the wavefield should be saved
        save_undersampling : int, optional
            Amount of undersampling in time when saving the forward wavefield. If not given,
            it is calculated given the bandwidth.
        save_compression : str, optional
            Compression applied to saved wavefield, only available with DevitoPRO. Defaults to no
            compression in 2D and `bitcomp` in 3D.
        boundary_type : str, optional
            Type of boundary for the wave equation (``sponge_boundary_2`` or
            ``complex_frequency_shift_PML_2``), defaults to ``sponge_boundary_2``.
            Note that ``complex_frequency_shift_PML_2`` boundaries have lower OT4 stability
            limit than other boundaries.
        interpolation_type : str, optional
            Type of source/receiver interpolation (``linear`` for bi-/tri-linear or ``hicks`` for sinc
            interpolation), defaults to ``linear``.
        attenuation_power : int, optional
            Power of the attenuation law if attenuation is given (``0`` or ``2``),
            defaults to ``0``.
        drp : bool, optional
            Whether or not to use dispersion-relation preserving coefficients (only
            available in some versions of Stride). Defaults to False.
        kernel : str, optional
            Type of time kernel to use (``OT2`` for 2nd order in time or ``OT4`` for 4th
            order in time). If not given, it is automatically decided given the time spacing.
        diff_source : bool, optional
            Whether the source should be injected as is, or as its 1st time derivative. Defaults to
            False, leaving it unchanged.
        adaptive_boxes : bool, optional
            Whether to activate adaptive boxes (requires DevitoPRO and only
            available in some versions of Stride). Defaults to False.
        platform : str, optional
            Platform on which to run the operator, ``None`` to run on the CPU or ``nvidia-acc`` to run on
            the GPU with OpenACC. Defaults to ``None``.
        devito_config : dict, optional
            Additional keyword arguments to configure Devito before operator generation.
        devito_args : dict, optional
            Additional keyword arguments used when calling the generated operator.


        Returns
        -------

        """
        problem = kwargs.get('problem')
        shot = problem.shot

        self._check_problem(wavelets, vp, rho=rho, alpha=alpha, **kwargs)

        num_sources = shot.num_points_sources
        num_receivers = shot.num_points_receivers

        save_wavefield = kwargs.get('save_wavefield', False)
        if save_wavefield is False:
            save_wavefield = vp.needs_grad
            if rho is not None:
                save_wavefield |= rho.needs_grad
            if alpha is not None:
                save_wavefield |= alpha.needs_grad

        platform = kwargs.get('platform', 'cpu')
        is_nvidia = platform is not None and 'nvidia' in platform

        diff_source = kwargs.pop('diff_source', False)
        save_compression = kwargs.get('save_compression',
                                      'bitcomp' if self.space.dim > 2 else None)
        save_compression = save_compression if is_nvidia and devito.pro_available else None

        # If there's no previous operator, generate one
        if self.state_operator.devito_operator is None:
            # Define variables
            src = self.dev_grid.sparse_time_function('src', num=num_sources,
                                                     coordinates=shot.source_coordinates,
                                                     interpolation_type=self.interpolation_type)
            rec = self.dev_grid.sparse_time_function('rec', num=num_receivers,
                                                     coordinates=shot.receiver_coordinates,
                                                     interpolation_type=self.interpolation_type)

            p = self.dev_grid.time_function('p', coefficients='symbolic' if self.drp else 'standard')

            # Create stencil
            stencil = self._stencil(p, wavelets, vp, rho=rho, alpha=alpha, direction='forward',
                                    save_wavefield=save_wavefield, **kwargs)

            # Define the source injection function to generate the corresponding code
            # pressure_to_density = 1 / vp**2
            # density_to_density_rate = 2 * vp / spacing
            # FDTD_scale = step**2 * vp**2
            vp_fun = self.dev_grid.vars.vp
            src_scale = 2 * self.time.step**2 * vp_fun / max(*self.space.spacing)

            if not diff_source:
                src_scale /= self.time.step

            src_term = src.inject(field=p.forward, expr=src * src_scale)
            rec_term = rec.interpolate(expr=p)

            # Define the saving of the wavefield
            if save_wavefield is True:
                space_order = None if self._needs_grad(rho, alpha) else 0
                layers = devito.HostDevice if is_nvidia else devito.NoLayers
                p_saved = self.dev_grid.undersampled_time_function('p_saved',
                                                                   bounds=kwargs.pop('save_bounds', None),
                                                                   factor=self.undersampling_factor,
                                                                   space_order=space_order,
                                                                   layers=layers,
                                                                   compression=save_compression)

                if not is_nvidia:
                    self.logger.perf('(ShotID %d) Expected wavefield size %.4f GB' %
                                     (problem.shot_id,
                                      np.prod(p_saved.shape_allocated)*p_saved.dtype().itemsize/1024**3))

                if self._needs_grad(wavelets, rho, alpha):
                    p_saved_expr = p
                else:
                    p_saved_expr = p.dt2
                abox, full, interior, boundary = self.subdomains
                update_saved = [devito.Eq(p_saved, p_saved_expr, subdmomain=interior)]
                devicecreate = (self.dev_grid.vars.p, self.dev_grid.vars.p_saved,)

            else:
                update_saved = []
                devicecreate = (self.dev_grid.vars.p,)

            # Compile the operator
            kwargs['devito_config'] = kwargs.get('devito_config', {})
            kwargs['devito_config']['devicecreate'] = devicecreate

            if self.attenuation_power == 2:
                kwargs['devito_config']['opt'] = 'noop'

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
        self.dev_grid.vars.rec.data_with_halo.fill(0.)
        self.dev_grid.vars.p.data_with_halo.fill(0.)
        self.boundary.clear()

        # Set medium parameters
        vp_with_halo = self.dev_grid.with_halo(vp.extended_data)
        self.dev_grid.vars.vp.data_with_halo[:] = vp_with_halo

        if rho is not None:
            self.logger.perf('(ShotID %d) Using inhomogeneous density' % problem.shot_id)

            rho_with_halo = self.dev_grid.with_halo(rho.extended_data)
            self.dev_grid.vars.rho.data_with_halo[:] = rho_with_halo
            self.dev_grid.vars.buoy.data_with_halo[:] = 1/rho_with_halo

        if alpha is not None:
            self.logger.perf('(ShotID %d) Using attenuation with power %d' % (problem.shot_id, self.attenuation_power))

            db_to_neper = 100 * (1e-6 / (2*np.pi))**self.attenuation_power / (20 * np.log10(np.exp(1)))

            alpha_with_halo = self.dev_grid.with_halo(alpha.extended_data)*db_to_neper
            self.dev_grid.vars.alpha.data_with_halo[:] = alpha_with_halo

        # Set geometry and wavelet
        wavelets = wavelets.data

        if diff_source:
            wavelets = np.gradient(wavelets, self.time.step, axis=-1)

        window = scipy.signal.get_window(('tukey', 0.001), self.time.num, False)
        window = window.reshape((self.time.num, 1))

        self.dev_grid.vars.src.data[:] = wavelets.T * window

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """

        functions = dict(
            vp=self.dev_grid.vars.vp,
            src=self.dev_grid.vars.src,
            rec=self.dev_grid.vars.rec,
        )

        devito_args = kwargs.get('devito_args', {})

        if 'p_saved' in self.dev_grid.vars:
            if self._wavefield is None:
                self._wavefield = self.dev_grid.func('p_saved')

            functions['p_saved'] = self._wavefield

            if 'nbits_compression' in kwargs or 'nbits' in devito_args:
                devito_args['nbits'] = kwargs.get('nbits_compression',
                                                  devito_args.get('nbits', 9))

        self.state_operator.run(dt=self.time.step,
                                **functions,
                                **devito_args)

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
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
            if os.environ.get('STRIDE_DUMP_WAVEFIELD', None) == 'yes':
                self.wavefield.dump(path=problem.output_folder,
                                    project_name=problem.name)

            cache_forward = kwargs.pop('cache_forward', False)
            cache_location = kwargs.pop('cache_location', None)
            if cache_forward:
                slices = [slice(extra, -extra) for extra in self.space.extra]
                slices = (slice(0, None),) + tuple(slices)
                inner_wavefield = self._wavefield.data[slices]

                if cache_location is None:
                    inner_wavefield = [maybe_compress(inner_wavefield[t].copy())
                                       for t in range(inner_wavefield.shape[0])]

                    self._wavefield = inner_wavefield

                else:
                    prev_cache = glob.glob(os.path.join(cache_location, 'stride-*'))
                    if len(prev_cache):
                        self._cache_folder = prev_cache[0]

                    if self._cache_folder is None:
                        self._cache_folder = tempfile.mkdtemp(prefix='stride-', dir=cache_location)

                        def _rm_tmpdir():
                            shutil.rmtree(self._cache_folder, ignore_errors=True)

                        at_exit.add(_rm_tmpdir)

                    try:
                        filename = os.path.join(self._cache_folder,
                                                '%s-%s-%05d.npy' % (problem.name, 'P', shot.id))
                        np.save(filename, inner_wavefield)
                    except:
                        shutil.rmtree(self._cache_folder, ignore_errors=True)
                        raise

                    self._wavefield = None

                self.dev_grid.deallocate('p_saved')

        else:
            self._wavefield = None

        traces_data = np.asarray(self.dev_grid.vars.rec.data, dtype=np.float32).T
        traces = shot.observed.alike(name='modelled', data=traces_data, shape=None, extended_shape=None, inner=None)

        deallocate = kwargs.get('deallocate', False)
        if deallocate:
            self.boundary.deallocate()
            self.dev_grid.deallocate('p')
            self.dev_grid.deallocate('laplacian')
            self.dev_grid.deallocate('src')
            self.dev_grid.deallocate('rec')
            self.dev_grid.deallocate('vp')
            self.dev_grid.deallocate('rho')
            self.dev_grid.deallocate('buoy')
            self.dev_grid.deallocate('alpha', collect=True)

        return traces

    # adjoint

    async def before_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Prepare the problem type to run the adjoint problem.

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        problem = kwargs.get('problem')
        shot = problem.shot

        num_sources = shot.num_points_sources
        num_receivers = shot.num_points_receivers

        # If there's no previous operator, generate one
        if self.adjoint_operator.devito_operator is None:
            # Define variables
            src = self.dev_grid.sparse_time_function('src', num=num_sources,
                                                     coordinates=shot.source_coordinates,
                                                     interpolation_type=self.interpolation_type)
            rec = self.dev_grid.sparse_time_function('rec', num=num_receivers,
                                                     coordinates=shot.receiver_coordinates,
                                                     interpolation_type=self.interpolation_type)

            p_a = self.dev_grid.time_function('p_a', coefficients='symbolic' if self.drp else 'standard')

            # Create stencil
            stencil = self._stencil(p_a, wavelets, vp, rho=rho, alpha=alpha, direction='backward', **kwargs)

            # Define the source injection function to generate the corresponding code
            vp2 = self.dev_grid.vars.vp**2
            rec_term = rec.inject(field=p_a.backward, expr=-rec * self.time.step**2 * vp2)

            if wavelets.needs_grad:
                src_term = src.interpolate(expr=p_a)
            else:
                src_term = []

            # Define gradient
            gradient_update = await self.prepare_grad(wavelets, vp, rho, alpha)

            # Compile the operator
            kwargs['devito_config'] = kwargs.get('devito_config', {})
            kwargs['devito_config']['devicecreate'] = (self.dev_grid.vars.p_a,)

            if self.attenuation_power == 2:
                kwargs['devito_config']['opt'] = 'noop'

            self.adjoint_operator.set_operator(stencil + rec_term + src_term + gradient_update,
                                               **kwargs)
            self.adjoint_operator.compile()

        else:
            # If the receiver size has changed, then create new functions for it
            if num_receivers != self.dev_grid.vars.rec.npoint:
                self.dev_grid.sparse_time_function('rec', num=num_receivers, cached=False)

        # Clear all buffers
        self.dev_grid.vars.src.data_with_halo.fill(0.)
        self.dev_grid.vars.p_a.data_with_halo.fill(0.)
        self.boundary.clear()
        await self.init_grad(wavelets, vp, rho, alpha)

        # Set wavefield if necessary
        cache_forward = kwargs.pop('cache_forward', False)
        cache_location = kwargs.pop('cache_location', None)
        if cache_forward:
            slices = [slice(extra, -extra) for extra in self.space.extra]
            slices = (slice(0, None),) + tuple(slices)

            wavefield = self.dev_grid.func('p_saved')

            if cache_location is None:
                inner_wavefield = np.asarray([np.frombuffer(decompress(*each), dtype=np.float32)
                                              for each in self._wavefield])
                inner_wavefield = inner_wavefield.reshape((inner_wavefield.shape[0],) + self.space.shape)
                wavefield.data[slices] = inner_wavefield

            else:
                filename = os.path.join(self._cache_folder,
                                        '%s-%s-%05d.npy' % (problem.name, 'P', shot.id))
                wavefield.data[slices] = np.load(filename)

                os.remove(filename)

            self._wavefield = wavefield

        # Set medium parameters
        vp_with_halo = self.dev_grid.with_halo(vp.extended_data)
        self.dev_grid.vars.vp.data_with_halo[:] = vp_with_halo

        if rho is not None:
            rho_with_halo = self.dev_grid.with_halo(rho.extended_data)
            self.dev_grid.vars.rho.data_with_halo[:] = rho_with_halo
            self.dev_grid.vars.buoy.data_with_halo[:] = 1/rho_with_halo

        if alpha is not None:
            db_to_neper = 100 * (1e-6 / (2*np.pi))**self.attenuation_power / (20 * np.log10(np.exp(1)))

            alpha_with_halo = self.dev_grid.with_halo(alpha.extended_data)*db_to_neper
            self.dev_grid.vars.alpha.data_with_halo[:] = alpha_with_halo

        # Set geometry and adjoint source
        window = scipy.signal.get_window(('tukey', 0.001), self.time.num, False)
        window = window.reshape((self.time.num, 1))

        self.dev_grid.vars.rec.data[:] = adjoint_source.data.T * window

        if self.interpolation_type == 'linear':
            self.dev_grid.vars.src.coordinates.data[:] = shot.source_coordinates
            self.dev_grid.vars.rec.coordinates.data[:] = shot.receiver_coordinates

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        functions = dict(
            vp=self.dev_grid.vars.vp,
            rec=self.dev_grid.vars.rec,
            p_saved=self._wavefield,
        )

        devito_args = kwargs.get('devito_args', {})

        if wavelets.needs_grad:
            functions['src'] = self.dev_grid.vars.src

        self.adjoint_operator.run(dt=self.time.step,
                                  **functions,
                                  **devito_args)

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
            Attenuation coefficient of the medium, defaults to 0, in [dB/cm].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------
        tuple of gradients
            Tuple with the gradients of the variables that need them

        """
        platform = kwargs.get('platform', 'cpu')
        deallocate = kwargs.get('deallocate', False)

        if platform and 'nvidia' in platform or deallocate:
            self._wavefield = None
            devito.clear_cache(force=True)

        if deallocate:
            self.boundary.deallocate()
            self.dev_grid.deallocate('p_a')
            self.dev_grid.deallocate('p_saved')
            self.dev_grid.deallocate('laplacian')
            self.dev_grid.deallocate('src')
            self.dev_grid.deallocate('rec')
            self.dev_grid.deallocate('vp')
            self.dev_grid.deallocate('rho')
            self.dev_grid.deallocate('buoy')
            self.dev_grid.deallocate('alpha', collect=True)

        return await self.get_grad(wavelets, vp, rho, alpha, **kwargs)

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
        p = self.dev_grid.vars.p_saved
        p_a = self.dev_grid.vars.p_a

        abox, full, interior, boundary = self.subdomains

        wavelets, _, rho, alpha = kwargs.get('wrt')
        if self._needs_grad(wavelets, rho, alpha):
            p_dt2 = self.dev_grid.undersampled_time_derivative(p, self.undersampling_factor,
                                                               bounds=kwargs.pop('save_bounds', None),
                                                               deriv_order=2, fd_order=2)

            p_dt2_fun = self.dev_grid.function('p_dt2', space_order=0)
            p_dt2_update = (devito.Eq(p_dt2_fun, p_dt2, subdomain=interior),)
        else:
            p_dt2 = p
            p_dt2_fun = p_dt2
            p_dt2_update = ()

        grad = self.dev_grid.function('grad_vp', space_order=0)
        grad_update = devito.Inc(grad, p_dt2_fun * p_a, subdomain=interior)

        prec = self.dev_grid.function('prec_vp', space_order=0)
        prec_update = devito.Inc(prec, p_dt2_fun * p_dt2_fun, subdomain=interior)

        return p_dt2_update + (grad_update, prec_update)

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
        grad = self.dev_grid.function('grad_vp')
        grad.data_with_halo.fill(0.)

        prec = self.dev_grid.function('prec_vp')
        prec.data_with_halo.fill(0.)

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
        variable_grad = self.dev_grid.vars.grad_vp
        variable_grad = np.asarray(variable_grad.data, dtype=np.float32)

        variable_prec = self.dev_grid.vars.prec_vp
        variable_prec = np.asarray(variable_prec.data, dtype=np.float32)

        variable_grad *= -2 / vp.extended_data**3
        variable_prec *= +4 / vp.extended_data**6 * self.time.step**2

        deallocate = kwargs.pop('deallocate', False)
        if deallocate:
            self.dev_grid.deallocate('grad_vp')
            self.dev_grid.deallocate('prec_vp')

        grad = vp.alike(name='vp_grad', data=variable_grad)
        grad.prec = vp.alike(name='vp_prec', data=variable_prec)

        return grad

    async def prepare_grad_rho(self, rho, **kwargs):
        """
        Prepare the problem type to calculate the gradients wrt rho.

        Parameters
        ----------
        rho : ScalarField
            Density variable to calculate the gradient.

        Returns
        -------
        tuple
            Tuple of gradient and preconditioner updates.

        """
        p = self.dev_grid.vars.p_saved
        p_a = self.dev_grid.vars.p_a
        buoy = self.dev_grid.vars.buoy

        abox, full, interior, boundary = self.subdomains

        grad_term = - devito.grad(buoy, shift=-0.5).dot(devito.grad(p, shift=-0.5)) \
                    - buoy * p.laplace
        grad_rho_fun = self.dev_grid.function('grad_rho_fun', space_order=0)
        grad_term_update = (devito.Eq(grad_rho_fun, grad_term, subdomain=interior),)

        grad = self.dev_grid.function('grad_rho', space_order=0)
        grad_update = devito.Inc(grad, grad_rho_fun * p_a, subdomain=interior)

        prec = self.dev_grid.function('prec_rho', space_order=0)
        prec_update = devito.Inc(prec, grad_rho_fun * grad_rho_fun, subdomain=interior)

        return grad_term_update + (grad_update, prec_update)

    async def init_grad_rho(self, rho, **kwargs):
        """
        Initialise buffers in the problem type to calculate the gradients wrt rho.

        Parameters
        ----------
        rho : ScalarField
            Density variable to calculate the gradient.

        Returns
        -------

        """
        grad = self.dev_grid.function('grad_rho')
        grad.data_with_halo.fill(0.)

        prec = self.dev_grid.function('prec_rho')
        prec.data_with_halo.fill(0.)

    async def get_grad_rho(self, rho, **kwargs):
        """
        Retrieve the gradients calculated wrt to rho.

        The variable is updated inplace.

        Parameters
        ----------
        rho : ScalarField
            Density variable to calculate the gradient.

        Returns
        -------
        ScalarField
            Gradient wrt Density.

        """
        variable_grad = self.dev_grid.vars.grad_rho
        variable_grad = np.asarray(variable_grad.data, dtype=np.float32)

        variable_prec = self.dev_grid.vars.prec_rho
        variable_prec = np.asarray(variable_prec.data, dtype=np.float32)

        deallocate = kwargs.pop('deallocate', False)
        if deallocate:
            self.dev_grid.deallocate('grad_rho')
            self.dev_grid.deallocate('prec_rho')

        grad = rho.alike(name='rho_grad', data=variable_grad)
        grad.prec = rho.alike(name='rho_prec', data=variable_prec)

        return grad

    # utils

    def _check_problem(self, wavelets, vp, rho=None, alpha=None, **kwargs):
        problem = kwargs.get('problem')

        recompile = False

        cached_name = self.__class__.__name__.lower()
        try:
            warehouse = mosaic.get_local_warehouse()
        except AttributeError:
            warehouse = {}

        if not self._cached_operator or ('%s_boundary' % cached_name) not in warehouse:
            boundary_type = kwargs.get('boundary_type', 'sponge_boundary_2')
            if boundary_type != self.boundary_type or self.boundary is None:
                recompile = True
                self.boundary_type = boundary_type

                if isinstance(self.boundary_type, str):
                    boundaries_module = boundaries.devito
                    self.boundary = getattr(boundaries_module, camel_case(self.boundary_type))(self.dev_grid)

                else:
                    self.boundary = self.boundary_type

            if self._cached_operator:
                warehouse['%s_boundary' % cached_name] = self.boundary

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

            adaptive_boxes = kwargs.pop('adaptive_boxes', self.adaptive_boxes)
            if adaptive_boxes != self.adaptive_boxes:
                recompile = True
                self.adaptive_boxes = adaptive_boxes

        else:
            self.boundary = warehouse['%s_boundary' % cached_name]

        preferred_kernel = kwargs.get('kernel', None)
        preferred_undersampling = kwargs.get('save_undersampling', None)

        self._check_conditions(wavelets, vp, rho, alpha,
                               preferred_kernel, preferred_undersampling,
                               **kwargs)

        # Recompile every time if using hicks or if there are sub ops
        if self.interpolation_type == 'hicks' or len(self._sub_ops) or recompile:
            self.state_operator.devito_operator = None
            self.adjoint_operator.devito_operator = None

        self.logger.perf('(ShotID %d) Selected time stepping scheme %s' %
                         (problem.shot_id, self.kernel,))

    def _check_conditions(self, wavelets, vp, rho=None, alpha=None,
                          preferred_kernel=None, preferred_undersampling=None, **kwargs):
        problem = kwargs.get('problem')

        # Get speed of sound bounds
        vp_min = np.min(vp.extended_data)
        vp_max = np.max(vp.extended_data)

        # Figure out propagated bandwidth
        wavelets = wavelets.data
        if wavelets.ndim > 1:
            f_mins = []
            f_centres = []
            f_maxs = []
            for i in range(wavelets.shape[0]):
                if np.any(wavelets[i]):
                    # only run calculations on non-zero wavelets
                    f_min, f_centre, f_max = fft.bandwidth(wavelets[i], self.time.step, cutoff=-10)
                    f_mins.append(f_min)
                    f_centres.append(f_centre)
                    f_maxs.append(f_max)

            f_min = np.min(f_mins)
            f_max = np.max(f_maxs)
            f_centre = np.median(f_centres)

        self._bandwidth = (f_min, f_centre, f_max)

        self.logger.perf('(ShotID %d) Estimated bandwidth for the propagated '
                         'wavelet %.3f-%.3f MHz' % (problem.shot_id, f_min / 1e6, f_max / 1e6))

        # Check for dispersion
        if self.drp is True:
            self.drp = False

            self.logger.warn('(ShotID %d) DRP weights are not implemented in this version of stride' %
                             problem.shot_id)

        h = max(*self.space.spacing)

        wavelength = vp_min / f_max
        ppw = wavelength / h
        ppw_max = 5

        h_max = wavelength / ppw_max

        if h > h_max:
            self.logger.warn('(ShotID %d) Spatial grid spacing (%.3f mm | %.3f PPW) is '
                             'higher than dispersion limit (%.3f mm | %.3f PPW)' %
                             (problem.shot_id, h / 1e-3, ppw, h_max / 1e-3, ppw_max))
        else:
            self.logger.perf('(ShotID %d) Spatial grid spacing (%.3f mm | %.3f PPW) is '
                             'below dispersion limit (%.3f mm | %.3f PPW)' %
                             (problem.shot_id, h / 1e-3, ppw, h_max / 1e-3, ppw_max))

        # Check for instability
        dt = self.time.step

        dt_max_OT2 = self._dt_max(2.0 / np.pi, h, vp_max)
        dt_max_OT4 = self._dt_max(3.6 / np.pi, h, vp_max)

        crossing_factor = dt*vp_max / h * 100

        recompile = False
        if dt <= dt_max_OT2:
            self.logger.perf('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                             'below OT2 limit (%.3f \u03BCs)' %
                             (problem.shot_id, dt / 1e-6, crossing_factor, dt_max_OT2 / 1e-6))

            selected_kernel = 'OT2'

        elif dt <= dt_max_OT4:
            self.logger.perf('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
                             'above OT2 limit (%.3f \u03BCs) and below OT4 limit (%.3f \u03BCs)'
                             % (problem.shot_id, dt / 1e-6, crossing_factor,
                                dt_max_OT2 / 1e-6, dt_max_OT4 / 1e-6))

            selected_kernel = 'OT4'

        else:
            self.logger.warn('(ShotID %d) Time grid spacing (%.3f \u03BCs | %d%%) is '
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

        undersampling = min(max(4, int(dt_max / dt)), 10) if preferred_undersampling is None else preferred_undersampling

        if self.undersampling_factor != undersampling:
            recompile = True

        self.undersampling_factor = undersampling

        self.logger.perf('(ShotID %d) Selected undersampling level %d' %
                         (problem.shot_id, undersampling,))

        # Maybe recompile
        if recompile:
            self.state_operator.operator = None
            self.adjoint_operator.operator = None

    def _stencil(self, field, wavelets, vp, rho=None, alpha=None, direction='forward',
                 save_wavefield=False, **kwargs):
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
                                           vp_fun, vp2_fun, inv_vp2_fun,
                                           rho=rho_fun, buoy=buoy_fun, alpha=alpha_fun,
                                           **kwargs)
        else:
            laplacian_term = self._diff_op(laplacian,
                                           vp_fun, vp2_fun, inv_vp2_fun,
                                           rho=rho_fun, buoy=buoy_fun, alpha=alpha_fun,
                                           **kwargs)

        # Get the subs
        if self.drp:
            extra_functions = ()
            if rho_fun is not None:
                extra_functions = (rho_fun, buoy_fun,)

            subs = self._symbolic_coefficients(field, laplacian, vp_fun,
                                               *extra_functions)
        else:
            subs = None

        # Get the attenuation term
        if alpha_fun is not None:
            if self.attenuation_power == 0:
                u = field
            elif self.attenuation_power == 2:
                u = -field.laplace
            else:
                raise ValueError('The "attenuation_exponent" can only take values (0, 2).')

            u_dt = u.dt if direction == 'forward' else u.dt.T
            attenuation_term = -2 * alpha_fun * vp_fun**(self.attenuation_power - 1) * u_dt
        else:
            attenuation_term = 0

        # Set up the boundary
        boundary_field = laplacian if self.kernel != 'OT2' and 'PML' in self.boundary_type else field
        boundary_term, eq_before, eq_after = self.boundary.apply(boundary_field, vp.extended_data,
                                                                 direction=direction, subs=subs,
                                                                 f_centre=self._bandwidth[1])

        sub_befores = []
        sub_afters = []
        sub_exprs = []

        for sub_op in self._sub_ops:
            sub_term, sub_before, sub_after = sub_op.sub_stencil(p=field,
                                                                 wavelets=wavelets, vp=vp, rho=rho,
                                                                 dev_grid=self.dev_grid,
                                                                 **kwargs)

            sub_befores += sub_before
            sub_afters += sub_after
            if sub_term is not None:
                sub_exprs.append(sub_term)

        sub_exprs = sum(sub_exprs)

        # Define PDE and update rule
        eq_interior = devito.solve(field.dt2 - laplacian_term
                                   - vp2_fun*attenuation_term
                                   - vp2_fun*sub_exprs, u_next)
        eq_boundary = devito.solve(field.dt2 - laplacian_term
                                   - vp2_fun*attenuation_term
                                   + vp2_fun*boundary_term
                                   - vp2_fun*sub_exprs, u_next)

        # Prepare the subdomains
        abox, full, interior, boundary = self._subdomains(field, wavelets, vp_fun,
                                                          direction=direction,
                                                          save_wavefield=save_wavefield,
                                                          **kwargs)

        # Time-stepping stencil
        stencils = []

        if self.kernel != 'OT2':
            stencil_laplacian = devito.Eq(laplacian, laplacian_update,
                                          subdomain=abox,
                                          coefficients=subs)
            stencils.append(stencil_laplacian)

        stencil_interior = devito.Eq(u_next, eq_boundary,
                                     subdomain=abox,
                                     coefficients=subs)
        stencils.append(stencil_interior)

        # stencil_boundary = [devito.Eq(u_next, eq_boundary,
        #                               subdomain=dom,
        #                               coefficients=subs) for dom in boundary]
        # stencils += stencil_boundary

        return sub_befores + eq_before + stencils + eq_after + sub_afters

    def _medium_functions(self, vp, rho=None, alpha=None, **kwargs):
        _kwargs = {
            'coefficients': 'symbolic' if self.drp else 'standard',
        }

        vp_fun = self.dev_grid.function('vp', **_kwargs)
        vp2_fun = vp_fun**2
        inv_vp2_fun = 1/vp_fun**2

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
            bi_harmonic = self.time.step**2/12 * self._diff_op(field,
                                                               vp, vp2, inv_vp2,
                                                               **kwargs)

        laplacian_update = field + bi_harmonic

        return laplacian_update

    def _diff_op(self, field, vp, vp2, inv_vp2, **kwargs):
        rho = kwargs.pop('rho', None)
        buoy = kwargs.pop('buoy', None)

        if buoy is None:
            return vp2 * field.laplace

        else:
            if self.drp:
                return vp2 * (field.laplace + rho * devito.grad(buoy, shift=-0.5).dot(devito.grad(field, shift=-0.5)))
            else:
                return vp2 * rho * devito.div(buoy * devito.grad(field, shift=+0.5), shift=-0.5)

    def _subdomains(self, *args, **kwargs):
        problem = kwargs.get('problem')

        if self.adaptive_boxes:
            self.logger.warn('(ShotID %d) Adaptive boxes are not implemented in this version of stride' %
                             problem.shot_id)

        full = self.dev_grid.full
        interior = self.dev_grid.interior
        boundary = self.dev_grid.pml
        self._cached_subdomains = (full, full, interior, boundary)

        return full, full, interior, boundary

    def _symbolic_coefficients(self, *functions):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _weights(self):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _dt_max(self, k, h, vp_max):
        return k * h / vp_max * 1 / np.sqrt(self.space.dim)

    def _needs_grad(self, *wrt):
        return any(v is not None and v.needs_grad for v in wrt)
