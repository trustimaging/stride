
import functools
import numpy as np

import mosaic

from stride.problem import Traces
from ..common.devito import GridDevito, OperatorDevito, config_devito, devito
from ..problem_type import ProblemTypeBase


__all__ = ['MarmottantDevito']


@mosaic.tessera
class MarmottantDevito(ProblemTypeBase):
    """
    This class represents the bubble oscillatory behaviour through the Marmottant model,
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

        x_0 : ParticleField
            Spatial location of the bubble population, in [m].
        r_0 : SparseField
            Initial radius of the bubble population, in [m].
        vp : float or ScalarField
            The speed of sound in the surrounding liquid, in [m/s].
        rho : float or ScalarField
            The density of the surrounding liquid, in [kg/m^3].
        sigma : float or ScalarField
            The surface tension of the surrounding liquid, in [N/m].
        mu : float or ScalarField
            The viscosity of the surrounding liquid, in [Pa*s].
        p_0 : float
            The ambient pressure, in [Pa].
        p : Traces or devito.TimeFunction
            The excitation pressure, in [Pa].
        kappa : float
            The polytropic gas exponent, in [-].
        kappa_s : float
            The surface dilatational viscosity from the monolayer, in [N].
        chi : float
            The elastic compression modulus of the monolayer, in [N/m].
        r_buckle : float
            The buckling radius of the bubble, in [m].
        r_break : float
            The break radius of the bubble, in [m].
        interpolation_type : str, optional
            Type of bubble interpolation (``linear`` for bi-/tri-linear or ``hicks`` for sinc
            interpolation), defaults to ``linear``.
        problem : Problem
            Sub-problem being solved by the PDE.

    """

    space_order = 0
    time_order = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kernel = None
        self.sigma = None
        self.interpolation_type = 'hicks'

        config_devito(**kwargs)

        dev_grid = kwargs.pop('dev_grid', None)
        self.dev_grid = dev_grid or GridDevito(self.space_order, self.time_order,
                                               dtype=kwargs.pop('dtype', np.float32),
                                               **kwargs)

        kwargs.pop('grid', None)
        self.state_operator = OperatorDevito(self.space_order, self.time_order,
                                             name='marmottant_state',
                                             grid=self.dev_grid,
                                             **kwargs)
        self.adjoint_operator = OperatorDevito(self.space_order, self.time_order,
                                               name='marmottant_adjoint',
                                               grid=self.dev_grid,
                                               **kwargs)

    def clear_operators(self):
        self.state_operator.devito_operator = None
        self.adjoint_operator.devito_operator = None

    # forward

    async def before_forward(self, r_0, x_0=None,
                             vp=1540., rho=997, sigma=0.073, mu=0.002,
                             p_0=101325, p=0., kappa=1.07, kappa_s=5E-9, chi=0.4,
                             r_buckle=None, r_break=None, **kwargs):
        """
        Prepare the problem type to run the state or forward problem.

        Parameters
        ----------
        x_0 : ParticleField
            Spatial location of the bubble population, in [m].
        r_0 : SparseField
            Initial radius of the bubble population, in [m].
        vp : float or ScalarField
            The speed of sound in the surrounding liquid, in [m/s].
        rho : float or ScalarField
            The density of the surrounding liquid, in [kg/m^3].
        sigma : float or ScalarField
            The surface tension of the surrounding liquid, in [N/m].
        mu : float or ScalarField
            The viscosity of the surrounding liquid, in [Pa*s].
        p_0 : float
            The ambient pressure, in [Pa].
        p : Traces
            The excitation pressure, in [Pa].
        kappa : float
            The polytropic gas exponent, in [-].
        kappa_s : float
            The surface dilatational viscosity from the monolayer, in [N].
        chi : float
            The elastic compression modulus of the monolayer, in [N/m].
        r_buckle : float
            The buckling radius of the bubble, in [m].
        r_break : float
            The break radius of the bubble, in [m].
        interpolation_type : str, optional
            Type of bubble interpolation (``linear`` for bi-/tri-linear or ``hicks`` for sinc
            interpolation), defaults to ``linear``.
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        # self._check_problem(*args, **kwargs)
        self.interpolation_type = kwargs.pop('interpolation_type', self.interpolation_type)

        # Stencil
        init_terms, stencil = self._stencil(r_0, x_0=x_0,
                                            vp=vp, rho=rho, sigma=sigma, mu=mu,
                                            p_0=p_0, p=p, kappa=kappa, kappa_s=kappa_s, chi=chi,
                                            r_buckle=r_buckle, r_break=r_break, direction='forward', **kwargs)

        # Operator
        self.state_operator.set_operator(init_terms + stencil, **kwargs)
        self.state_operator.compile()

    async def run_forward(self, *args, **kwargs):
        """
        Run the state or forward problem.

        Parameters
        ----------
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------

        """
        functions = dict()

        self.state_operator.run(dt=self.time.step,
                                **functions,
                                **kwargs.pop('devito_args', {}))

    async def after_forward(self, *args, **kwargs):
        """
        Clean up after the state run and retrieve the time traces.

        Parameters
        ----------
        problem : Problem
            Sub-problem being solved by the PDE.

        Returns
        -------
        ScalarField
            Final distribution of mass.

        """

        r_data = np.array(self.dev_grid.vars.r_out_sparse.data, dtype=np.float32)
        r = Traces(name='r', transducer_ids=list(range(r_data.shape[1])), data=r_data.T, grid=self.grid)

        return r

    # utils

    def _check_problem(self, *args, **kwargs):
        pass

    def sub_stencil(self, **kwargs):
        parent_grid = kwargs.get('dev_grid')
        num_inner = kwargs.pop('num_inner', 2)
        parent_grid.num_inner = num_inner
        self.dev_grid.num_inner = num_inner

        if (num_inner % 2) != 0:
            self.logger.warn('Number of inner-loop iterations (num_inner=%d) '
                             'should be an even number' % num_inner)

        # Stencil
        init_terms, stencil = self._stencil(**kwargs)

        # Transfer r and r_dt to injections
        r_0 = kwargs.get('r_0')
        x_0 = kwargs.get('x_0')
        p_out = kwargs.get('p')
        num_bubbles = r_0.num

        r = self.dev_grid.vars.r_sparse
        r_dt = self.dev_grid.vars.r_dt_sparse
        r_0_fun = self.dev_grid.vars.r_0_sparse

        p_dim = devito.Dimension(name='p_bubble')
        r_out = parent_grid.time_function('r_out_sparse',
                                          dimensions=(parent_grid.devito_grid.stepping_dim, p_dim),
                                          time_dimension=parent_grid.devito_grid.time_dim,
                                          shape=(p_out.shape[0], num_bubbles))

        r_saved = parent_grid.time_function('r_saved',
                                            dimensions=(parent_grid.devito_grid.time_dim, p_dim),
                                            time_dimension=parent_grid.devito_grid.time_dim,
                                            shape=(self.time.num, num_bubbles))

        v_saved = parent_grid.time_function('v_saved',
                                            dimensions=(parent_grid.devito_grid.time_dim, p_dim),
                                            time_dimension=parent_grid.devito_grid.time_dim,
                                            shape=(self.time.num, num_bubbles))

        v_inject = parent_grid.sparse_time_function('v_inject', num=num_bubbles, p_dim=p_dim,
                                                    dimensions=(parent_grid.devito_grid.stepping_dim, p_dim),
                                                    time_dimension=parent_grid.devito_grid.time_dim,
                                                    nt=p_out.shape[0],
                                                    interpolation_type=self.interpolation_type,
                                                    coordinates=x_0.data,
                                                    cached=self.interpolation_type == 'linear')

        r_out.data[0, :] = r_0.data.T
        r_out.data[1, :] = r_0.data.T
        r_out.data[2, :] = r_0.data.T

        r_saved.data[0, :] = r_0.data.T

        v_inject.data.fill(0.)
        if self.interpolation_type == 'linear':
            v_inject.coordinates.data[:] = x_0.data

        implicit_dims = (parent_grid.devito_grid.time_dim, self.dev_grid.devito_grid.time_dim)

        d_V = max(*self.space.spacing)**self.space.dim
        if self.space.dim == 3:
            inject_scale = 4 * np.pi / d_V
            v_dt2 = 2 * r_out * r_out.dt**2 + r_out**2 * r_out.dt2
        elif self.space.dim == 2:
            # extra dx: area to volume
            inject_scale = 2 * np.pi / d_V
            v_dt2 = (r_out.dt**2 + r_out * r_out.dt2) * max(*self.space.spacing)
        else:
            raise RuntimeError('Only 2 and 3 dimensions can be used')

        transfer_terms = [
            devito.Eq(r_out.forward, (r.forward+1)*r_0_fun, implicit_dims=implicit_dims),
            devito.Eq(r_saved, r_out.forward),
            devito.Eq(v_inject, v_dt2),
            devito.Eq(v_saved, v_inject),
        ]

        # Inject source
        try:
            rho = parent_grid.vars.rho
        except AttributeError:
            rho = self.dev_grid.vars.rho_sparse
        vp2 = parent_grid.vars.vp2

        inject_term = v_inject.inject(field=p_out.forward, expr=vp2 * self.time.step**2 * rho * inject_scale * v_inject)

        return 0., init_terms, stencil + transfer_terms + inject_term

    def _stencil(self, r_0=None, r=None, x_0=None,
                 vp=1540., rho=997, sigma=0.073, mu=0.002,
                 p_0=101325, p=0., kappa=1.07, kappa_s=5E-9, chi=0.4,
                 r_buckle=None, r_break=None, direction='forward', **kwargs):
        num_bubbles = r_0.num

        if direction == 'forward':
            var_name = 'r'
        else:
            var_name = 's'

        # State functions
        var = self._make_time_function(name=var_name, num=num_bubbles)
        var_dt = self._make_time_function(name='%s_dt' % var_name, num=num_bubbles)

        if direction == 'backward':
            r_fun = self._make_saved_time_function(name='r', num=num_bubbles)
            r_dt_fun = self._make_saved_time_function(name='r_dt', num=num_bubbles)
            r_dt2_fun = self._make_saved_time_function(name='r_dt2', num=num_bubbles)

        k_0_1 = self._make_time_function(name='k_0_1', num=num_bubbles)
        k_0_2 = self._make_time_function(name='k_0_2', num=num_bubbles)
        k_0_3 = self._make_time_function(name='k_0_3', num=num_bubbles)
        k_0_4 = self._make_time_function(name='k_0_4', num=num_bubbles)

        k_1_1 = self._make_time_function(name='k_1_1', num=num_bubbles)
        k_1_2 = self._make_time_function(name='k_1_2', num=num_bubbles)
        k_1_3 = self._make_time_function(name='k_1_3', num=num_bubbles)
        k_1_4 = self._make_time_function(name='k_1_4', num=num_bubbles)

        # Property functions
        r_0_fun = self._make_function(name='r_0', num=num_bubbles)
        r_0_fun.data[:] = r_0.data.T

        interp_terms = []

        p_fun, p_dense_fun, interp_term = self._make_interp_time_function('p', p, x_0, num_bubbles, **kwargs)
        if interp_term is not None:
            interp_terms.append(interp_term)

        init_terms = []

        vp_fun, vp_dense_fun, interp_term = self._make_interp_function('vp', vp, x_0, num_bubbles)
        if interp_term is not None:
            init_terms.append(interp_term)

        rho_fun, rho_dense_fun, interp_term = self._make_interp_function('rho', rho, x_0, num_bubbles)
        if interp_term is not None:
            init_terms.append(interp_term)

        sigma_fun, sigma_dense_fun, interp_term = self._make_interp_function('sigma', sigma, x_0, num_bubbles)
        if interp_term is not None:
            init_terms.append(interp_term)

        mu_fun, mu_dense_fun, interp_term = self._make_interp_function('mu', mu, x_0, num_bubbles)
        if interp_term is not None:
            init_terms.append(interp_term)

        t_0 = self.time.step
        if self.dev_grid.num_inner is not None:
            t_0 = t_0/self.dev_grid.num_inner

        functions = dict(
            r_0=r_0_fun / r_0_fun,
            vp=vp_fun / (r_0_fun / t_0),
            rho=rho_fun / (p_0 * t_0**2 / r_0_fun**2),
            sigma=sigma_fun / (p_0 * r_0_fun),
            mu=mu_fun / (p_0 * t_0),
            p_0=p_0 / p_0,
            p=p_fun / p_0,
            kappa=kappa,
            kappa_s=kappa_s / (p_0 * r_0_fun**2),
            chi=chi / (p_0 * r_0_fun),
            r_buckle=r_buckle / r_0_fun,
            r_break=r_break / r_0_fun,
        )

        if direction == 'backward':
            functions['r'] = r_fun
            functions['r_dt'] = r_dt_fun
            functions['r_dt2'] = r_dt2_fun

        # Time stepping
        parent_grid = kwargs.get('dev_grid', None)
        implicit_dims = (self.dev_grid.devito_grid.time_dim,)
        if parent_grid is not None:
            implicit_dims = (parent_grid.devito_grid.time_dim,) + implicit_dims

        if direction == 'forward':
            step = +1
            var_next = var.forward
            var_dt_next = var_dt.forward

            op_0 = functools.partial(self._op_0, **functions)
            op_1 = functools.partial(self._op_1, **functions)
        else:
            step = -1
            var_next = var.backward
            var_dt_next = var_dt.backward

            op_0 = functools.partial(self._adj_op_0, **functions)
            op_1 = functools.partial(self._adj_op_1, **functions)

        eq_0_1 = devito.Eq(k_0_1, step * op_0(var, var_dt), implicit_dims=implicit_dims)
        eq_0_2 = devito.Eq(k_0_2, step * op_0(var + k_0_1 / 2, var_dt + k_1_1 / 2), implicit_dims=implicit_dims)
        eq_0_3 = devito.Eq(k_0_3, step * op_0(var + k_0_2 / 2, var_dt + k_1_2 / 2), implicit_dims=implicit_dims)
        eq_0_4 = devito.Eq(k_0_4, step * op_0(var + k_0_3, var_dt + k_1_3), implicit_dims=implicit_dims)
        eq_0 = devito.Eq(var_next, var + 1 / 6 * (k_0_1 + 2 * k_0_2 + 2 * k_0_3 + k_0_4), implicit_dims=implicit_dims)

        eq_1_1 = devito.Eq(k_1_1, step * op_1(var, var_dt), implicit_dims=implicit_dims)
        eq_1_2 = devito.Eq(k_1_2, step * op_1(var + k_0_1 / 2, var_dt + k_1_1 / 2), implicit_dims=implicit_dims)
        eq_1_3 = devito.Eq(k_1_3, step * op_1(var + k_0_2 / 2, var_dt + k_1_2 / 2), implicit_dims=implicit_dims)
        eq_1_4 = devito.Eq(k_1_4, step * op_1(var + k_0_3, var_dt + k_1_3), implicit_dims=implicit_dims)
        eq_1 = devito.Eq(var_dt_next, var_dt + 1 / 6 * (k_1_1 + 2 * k_1_2 + 2 * k_1_3 + k_1_4), implicit_dims=implicit_dims)

        if parent_grid is None:
            var_out = self._make_saved_time_function(name='%s_out' % var_name, num=num_bubbles)
            output_eq = [devito.Eq(var_out, (var + 1)*r_0_fun, implicit_dims=implicit_dims)]
        else:
            output_eq = []

        # Stencil
        stencil = [
            eq_0_1, eq_1_1,
            eq_0_2, eq_1_2,
            eq_0_3, eq_1_3,
            eq_0_4, eq_1_4,
            eq_0, eq_1,
        ] + output_eq

        # Initialise state functions
        var_dt.data.fill(0.)
        var.data.fill(0.)

        if parent_grid is None:
            var_out.data.fill(0.)
            var_out.data[0, :] = r_0.data.T
            var_out.data[1, :] = r_0.data.T

        if direction == 'backward':
            r_fun.data[:] = r.data.T/r_0.data.T - 1
            r_dt_fun.data[:] = np.gradient(r_fun.data, self.time.step, axis=0)
            r_dt2_fun.data[:] = np.gradient(r_dt_fun.data, self.time.step, axis=0)

        k_0_1.data.fill(0.)
        k_0_2.data.fill(0.)
        k_0_3.data.fill(0.)
        k_0_4.data.fill(0.)

        k_1_1.data.fill(0.)
        k_1_2.data.fill(0.)
        k_1_3.data.fill(0.)
        k_1_4.data.fill(0.)

        return init_terms, interp_terms + stencil

    def _make_function(self, name, num, **kwargs):
        p_dim = devito.Dimension(name='p_bubble')

        return self.dev_grid.function('%s_sparse' % name,
                                      dimensions=(p_dim,),
                                      shape=(num,),
                                      space_order=self.space_order,
                                      time_order=self.time_order,
                                      **kwargs)

    def _make_time_function(self, name, num, **kwargs):
        p_dim = devito.Dimension(name='p_bubble')
        t_dim = self.dev_grid.devito_grid.time_dim
        step_dim = self.dev_grid.devito_grid.stepping_dim

        return self.dev_grid.time_function('%s_sparse' % name,
                                           dimensions=(step_dim, p_dim),
                                           time_dimension=t_dim,
                                           shape=(self.time_order+1, num),
                                           space_order=self.space_order,
                                           time_order=self.time_order,
                                           **kwargs)

    def _make_saved_time_function(self, name, num, **kwargs):
        p_dim = devito.Dimension(name='p_bubble')
        t_dim = self.dev_grid.devito_grid.time_dim

        return self.dev_grid.time_function('%s_sparse' % name,
                                           dimensions=(t_dim, p_dim),
                                           time_dimension=t_dim,
                                           shape=(self.time.num, num),
                                           space_order=self.space_order,
                                           time_order=self.time_order,
                                           **kwargs)

    def _make_interp_function(self, name, value, x_0, num, **kwargs):
        if not hasattr(value, 'data'):
            fun = self._make_function(name, num)
            fun.data.fill(value)

            dense_fun = None
            interp_term = None
        else:
            p_dim = devito.Dimension(name='p_bubble')
            fun = self.dev_grid.sparse_function('%s_sparse' % name,
                                                num=num, p_dim=p_dim,
                                                space_order=self.space_order,
                                                time_order=self.time_order,
                                                interpolation_type=self.interpolation_type,
                                                coordinates=x_0.data,
                                                cached=self.interpolation_type == 'linear')

            dense_fun = self.dev_grid.function('%s_dense' % name,
                                               space_order=self.space_order,
                                               time_order=self.time_order)
            with_halo = self.dev_grid.with_halo(value.extended_data)
            dense_fun.data[:] = with_halo

            interp_term = fun.interpolate(expr=dense_fun)

            if x_0 is None:
                raise ValueError('Bubble location x_0 needs to be provided when'
                                 'property %s is not a scalar' % name)

            fun.data.fill(0.)
            if self.interpolation_type == 'linear':
                fun.coordinates.data[:] = x_0.data

        return fun, dense_fun, interp_term

    def _make_interp_time_function(self, name, value, x_0, num, **kwargs):
        if not isinstance(value, devito.TimeFunction):
            fun = self._make_saved_time_function(name, num=num, save=self.time.num)
            fun.data[:] = value.data.T

            dense_fun = None
            interp_term = None
        else:
            parent_grid = kwargs.get('dev_grid', None)
            p_dim = devito.Dimension(name='p_bubble')

            fun = parent_grid.sparse_time_function('%s_sparse' % name,
                                                   p_dim=p_dim,
                                                   dimensions=(parent_grid.devito_grid.stepping_dim, p_dim),
                                                   time_dimension=parent_grid.devito_grid.time_dim,
                                                   num=num,
                                                   nt=value.shape[0],
                                                   interpolation_type=self.interpolation_type,
                                                   coordinates=x_0.data,
                                                   cached=self.interpolation_type == 'linear')

            dense_fun = value

            interp_term = fun.interpolate(expr=dense_fun.forward)

            if x_0 is None:
                raise ValueError('Bubble location x_0 needs to be provided when'
                                 'property %s is not a scalar' % name)

            fun.data.fill(0.)
            if self.interpolation_type == 'linear':
                fun.coordinates.data[:] = x_0.data

        return fun, dense_fun, interp_term

    @staticmethod
    def _step(t, a, x):
        return (devito.sign(a * (x - t)) + 1) / 2

    def _surface_tension_0(self, x, **kwargs):
        sigma = kwargs.get('sigma')
        chi = kwargs.get('chi')
        r_buckle = kwargs.get('r_buckle')
        r_break = kwargs.get('r_break')

        return self._step(r_buckle, 1, x) * self._step(r_break, -1, x) * chi * (x**2 / r_buckle**2 - 1) + \
               self._step(r_break, 1, x) * sigma

    def _surface_tension(self, x, **kwargs):
        sigma = kwargs.get('sigma')
        chi = kwargs.get('chi')
        r_buckle = kwargs.get('r_buckle')

        sigma_0 = self._surface_tension_0(1, **kwargs)
        c = 2 * chi * np.e / sigma * devito.sqrt(1 + sigma / (2 * chi))
        b = - devito.log(sigma_0 / sigma) / devito.exp(c * (1 - 1 / r_buckle))

        return sigma * devito.exp(-b * devito.exp(c * (1 - x / r_buckle)))

    def _op_0(self, r, r_dt, **kwargs):
        return r_dt

    def _op_1(self, r, r_dt, **kwargs):
        r_0 = kwargs.get('r_0')
        vp = kwargs.get('vp')
        rho = kwargs.get('rho')
        mu = kwargs.get('mu')
        p_0 = kwargs.get('p_0')
        kappa = kwargs.get('kappa')
        kappa_s = kwargs.get('kappa_s')
        p = kwargs.get('p')

        return 1 / (r+1) * 1 / rho * (
            + (p_0 + 2 * self._surface_tension_0(r_0, **kwargs) / r_0) * ((r+1) / r_0)**(-3 * kappa) * (1 - 3 * kappa / vp * r_dt)
            - p_0
            - 2 * self._surface_tension(r+1, **kwargs) / (r+1)
            - 4 * mu * r_dt / (r+1)
            - 4 * kappa_s * r_dt / (r+1)**2
            - p
        ) - 3/2 * r_dt**2 / (r+1)
