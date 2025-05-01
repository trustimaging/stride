
import numpy as np
import scipy.signal

import mosaic

from ..common.devito import GridDevito, OperatorDevito, devito
from ..problem_type import ProblemTypeBase
from ..boundaries import boundaries_registry


__all__ = ['IsoElasticDevito']


@mosaic.tessera
class IsoElasticDevito(ProblemTypeBase):
    """
    This class represents the stress-strain formulation of the elastic wave equation, implemented using Devito from
    the tutorial https://slimgroup.github.io/Devito-Examples/tutorials/07_elastic_varying_parameters/.

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
            Compressional (acoustic) speed of sound of the medium, in [m/s].
        vs : ScalarField
            Transverse (shear) speed of sound of the medium, in [m/s].
        rho : ScalarField
            Density of the medium in [kg/m^3].
        problem : Problem
            Sub-problem being solved by the PDE.

    """

    space_order = 10
    time_order = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.boundary_type = 'sponge_boundary_1'
        self.interpolation_type = 'linear'

        self.wavefield = None

        self._max_wavelet = 0.
        self._src_scale = 0.

        dev_grid = kwargs.pop('dev_grid', None)
        self.dev_grid = dev_grid or GridDevito(self.space_order, self.time_order, **kwargs)

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

    def before_forward(self, wavelets, vp, vs, rho, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional (acoustic) speed of sound of the medium, in [m/s].
        vs : ScalarField
            Transverse (shear) speed of sound of the medium, in [m/s].
        rho : ScalarField
            Density of the medium in [kg/m^3].
        problem : Problem
            Sub-problem being solved by the PDE.


        Returns
        -------

        """
        problem = kwargs.get('problem')
        shot = problem.shot

        num_sources = shot.num_points_sources
        num_receivers = shot.num_points_receivers

        eps_coords = 1e-3 * np.array(self.space.spacing).reshape((1, -1))
        source_coordinates = shot.source_coordinates + eps_coords
        receiver_coordinates = shot.receiver_coordinates + eps_coords

        # If there's no previous operator, generate one
        if self.state_operator.devito_operator is None:

            # Define variables
            src = self.dev_grid.sparse_time_function('src', num=num_sources,
                                                     coordinates=source_coordinates,
                                                     interpolation_type=self.interpolation_type)
            rec_tau = self.dev_grid.sparse_time_function('rec_tau', num=num_receivers,
                                                         coordinates=receiver_coordinates,
                                                         interpolation_type=self.interpolation_type)

            step = self.time.step
            # Create stencil
            # TODO: save wavefield during simulation
            # save_wavefield = kwargs.get('save_wavefield', False)
            # save = Buffer(save_wavefield) if type(save_wavefield) is int else None
            vel = self.dev_grid.vector_time_function('vel')
            tau = self.dev_grid.tensor_time_function('tau')

            # Absorbing boundaries
            self.boundary = boundaries_registry[self.boundary_type](self.dev_grid)
            _, _, _ = self.boundary.apply(vel, vp.extended_data, damping_coefficient=1/step)

            # Define the source injection function using a pressure disturbance
            src_term = src.inject(field=tau.forward.diagonal(), expr=step * src)

            rec_term = rec_tau.interpolate(expr=tau[0, 0] + tau[1, 1])
            # rec_term += rec_vel.interpolate(expr=devito.div(vel))  # Placeholder for vel receiver

            # Set up parameters as functions
            lam_fun = self.dev_grid.function('lam_fun')
            mu_fun = self.dev_grid.function('mu_fun')
            byn_fun = self.dev_grid.function('byn_fun')

            # Compile the operator
            # velocity (first derivative vel w.r.t. time, first order euler method), step: time_spacing
            u_v = devito.Eq(vel.forward, self.boundary.damp * (vel + step * byn_fun * devito.div(tau)),
                            grid=self.dev_grid, coefficients=None)

            # stress (first derivative tau w.r.t. time, first order euler method)
            u_tau = devito.Eq(tau.forward,
                              self.boundary.damp * (
                                      tau + step * (lam_fun * devito.diag(devito.div(vel.forward))
                                                    + mu_fun * (devito.grad(vel.forward) +
                                                                devito.grad(vel.forward).transpose(inner=False))))
                              )

            self.state_operator.set_operator([u_v] + [u_tau] + src_term + rec_term,
                                             **kwargs)
            self.state_operator.compile()

        else:
            # If the source/receiver size has changed, then create new functions for them
            if num_sources != self.dev_grid.vars.src.npoint:
                self.dev_grid.sparse_time_function('src', num=num_sources, cached=False)

            if num_receivers != self.dev_grid.vars.rec_tau.npoint:
                self.dev_grid.sparse_time_function('rec_tau', num=num_receivers, cached=False)

        # Clear all buffers
        self.dev_grid.vars.src.data_with_halo.fill(0.)
        self.dev_grid.vars.rec_tau.data_with_halo.fill(0.)
        self.dev_grid.vars.vel[0].data_with_halo.fill(0.)
        self.dev_grid.vars.vel[1].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[0, 0].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[0, 1].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[1, 0].data_with_halo.fill(0.)
        self.dev_grid.vars.tau[1, 1].data_with_halo.fill(0.)
        self.boundary.clear()

        # Set medium parameters
        vp_with_halo = self.dev_grid.with_halo(vp.extended_data)
        vs_with_halo = self.dev_grid.with_halo(vs.extended_data)
        rho_with_halo = self.dev_grid.with_halo(rho.extended_data)

        lam_with_halo = rho_with_halo * (vp_with_halo ** 2 - 2. * vs_with_halo ** 2)
        mu_with_halo = rho_with_halo * vs_with_halo ** 2
        byn_with_halo = 1 / rho_with_halo

        self.dev_grid.vars.lam_fun.data_with_halo[:] = lam_with_halo
        self.dev_grid.vars.mu_fun.data_with_halo[:] = mu_with_halo
        self.dev_grid.vars.byn_fun.data_with_halo[:] = byn_with_halo

        # Set geometry and wavelet
        wavelets = wavelets.data

        window = scipy.signal.get_window(('tukey', 0.01), self.time.num, False)
        window = window.reshape((self.time.num, 1))

        self.dev_grid.vars.src.data[:] = wavelets.T * window

        if self.interpolation_type == 'linear':
            self.dev_grid.vars.src.coordinates.data[:] = source_coordinates
            self.dev_grid.vars.rec_tau.coordinates.data[:] = receiver_coordinates

    def run_forward(self, wavelets, vp, vs, rho, **kwargs):
        """
        Run the state or forward problem.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional (acoustic) speed of sound of the medium, in [m/s].
        vs : ScalarField
            Transverse (shear) speed of sound of the medium, in [m/s].
        rho : ScalarField
            Density of the medium in [kg/m^3].
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        functions = dict(
            src=self.dev_grid.vars.src,
            rec_tau=self.dev_grid.vars.rec_tau,
        )

        self.state_operator.run(dt=self.time.step,
                                **functions,
                                **kwargs.pop('devito_args', {}))

    def after_forward(self, wavelets, vp, vs, rho, **kwargs):
        """
        Clean up after the state run and retrieve the time traces.

        Parameters
        ----------
        wavelets : Traces
            Source wavelets.
        vp : ScalarField
            Compressional (acoustic) speed of sound of the medium, in [m/s].
        vs : ScalarField
            Transverse (shear) speed of sound of the medium, in [m/s].
        rho : ScalarField
            Density of the medium in [kg/m^3].
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
        traces = shot.observed.alike(name='modelled', data=traces_data)

        self.dev_grid.deallocate('p')
        self.dev_grid.deallocate('laplacian')
        self.dev_grid.deallocate('src')
        self.dev_grid.deallocate('rec')
        self.dev_grid.deallocate('vp')
        self.dev_grid.deallocate('vp2')
        self.dev_grid.deallocate('inv_vp2')

        return traces

    # adjoint

    def before_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Not implemented
        """
        pass

    def run_adjoint(self, adjoint_source, wavelets, vp, rho=None, alpha=None, **kwargs):
        """
        Not implemented
        """
        pass

    def after_adjoint(self, **kwargs):
        """
        Not implemented
        """
        pass

    # gradients

    def prepare_grad_vp(self, **kwargs):
        """
        Not implemented
        """
        pass

    def init_grad_vp(self, **kwargs):
        """
        Not implemented
        """
        pass

    def get_grad_vp(self, **kwargs):
        """
        Not implemented
        """
        pass

    # utils

    def _check_problem(self):
        raise NotImplementedError('Check problem not implemented for elastic propagator')

    def _check_conditions(self):
        raise NotImplementedError('Check conditions not implemented for elastic propagator')

    def _saved(self):
        raise NotImplementedError('Saved not implemented for elastic propagator')

    def _symbolic_coefficients(self):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _weights(self):
        raise NotImplementedError('DRP weights are not implemented in this version of stride')

    def _dt_max(self):
        raise NotImplementedError('dt_max not implemented for elastic propagator')
