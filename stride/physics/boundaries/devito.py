import scipy.special
import sympy
from ..common.devito import devito
import numpy as np

from .boundary import Boundary


class SpongeBoundary1(Boundary):

    def __init__(self, grid):
        """
        Sponge boundary for elastic codes
        """
        super().__init__(grid)

        self.damp = None

    def apply(self, field, velocity, direction='forward', **kwargs):
        space = self._grid.space
        time = self._grid.time

        reflection_coefficient = 10**(-(np.log10(max(*space.absorbing)) - 1)/np.log10(2) - 3)
        reflection_coefficient = kwargs.pop('reflection_coefficient', reflection_coefficient)

        if np.max(space.extra) > 0:
            damp = self._grid.function('damp')
            damp.data[:] = self.damping(velocity=velocity, reflection_coefficient=reflection_coefficient, mask=True) * time.step
        else:
            damp = 0

        self.damp = damp
        return None, [], []


class SpongeBoundary2(Boundary):
    """
    Sponge boundary layer for a second-order equation as proposed in
    https://doi.org/10.1088/1742-2140/aaa4da.

    """

    def __init__(self, grid):
        super().__init__(grid)

        self.damp = None

    def apply(self, field, velocity, direction='forward', **kwargs):
        space = self._grid.space
        time = self._grid.time

        reflection_coefficient = 10 ** (-(np.log10(max(*space.absorbing)) - 1) / np.log10(2) - 3)
        reflection_coefficient = kwargs.pop('reflection_coefficient', reflection_coefficient)

        if np.max(space.extra) > 0:
            damp = self._grid.function('damp')
            damp.data[:] = 7 * self.damping(velocity=velocity, reflection_coefficient=reflection_coefficient) * time.step
        else:
            damp = 0

        self.damp = damp

        u_dt = field.dtc if direction == 'forward' else field.dtc.T

        damping_term = 2*damp*u_dt + damp**2*field

        return damping_term, [], []


class ComplexFrequencyShiftPML2(Boundary):
    """
    Complex frequency shift PML for a second-order equation as presented
    in https://doi.org/10.1121/1.4938270.

    """

    def __init__(self, grid):
        super().__init__(grid)

    def apply(self, field, velocity, direction='forward', **kwargs):
        space = self._grid.space

        reflection_coefficient = 10**(-(np.log10(max(*space.absorbing)) - 1)/np.log10(2) - 3)
        reflection_coefficient = kwargs.pop('reflection_coefficient', reflection_coefficient)
        f_centre = kwargs.pop('f_centre')
        subs = kwargs.pop('subs', None)

        dimensions = self._grid.devito_grid.dimensions
        shape = space.extended_shape

        eq_preparation = []
        eq_before = []
        eq_after = []
        boundary_term = []

        for dim_i in range(space.dim):
            dim = dimensions[dim_i]

            # Create damping functions
            sigma_i = self._grid.function('sigma_%d' % dim_i, space_order=2,
                                          dimensions=(dim,), shape=(shape[dim_i],))
            alpha_i = self._grid.function('alpha_%d' % dim_i, space_order=2,
                                          dimensions=(dim,), shape=(shape[dim_i],))
            sigma_di = self._grid.function('sigma_d%d' % dim_i, space_order=2,
                                           dimensions=(dim,), shape=(shape[dim_i],))
            alpha_di = self._grid.function('alpha_d%d' % dim_i, space_order=2,
                                           dimensions=(dim,), shape=(shape[dim_i],))

            # Fill functions
            sigma_i_data = self.damping(dimensions=(dim_i,), velocity=velocity,
                                        damping_type='power', power_degree=3,
                                        reflection_coefficient=reflection_coefficient,
                                        assign=True)
            alpha_i_data = self.damping(dimensions=(dim_i,), velocity=velocity,
                                        damping_type='power', power_degree=2,
                                        damping_coefficient=0.01*np.pi*f_centre,
                                        mask=True, assign=True)

            sigma_i.data_with_halo[:] = np.pad(sigma_i_data, ((2, 2),), mode='edge')
            alpha_i.data_with_halo[:] = np.pad(alpha_i_data, ((2, 2),), mode='edge')

            # Calculate their derivative
            eq_sigma_di = devito.Eq(sigma_di, devito.Derivative(sigma_i, (dim, 1)))
            eq_alpha_di = devito.Eq(alpha_di, devito.Derivative(alpha_i, (dim, 1)))

            # Create the auxiliary fields
            u_3 = self._grid.time_function('u_3_%d' % dim_i, time_order=1, space_order=2)
            u_2 = self._grid.time_function('u_2_%d' % dim_i, time_order=1, space_order=2)
            u_1 = self._grid.time_function('u_1_%d' % dim_i, time_order=1, space_order=2)

            # Prepare the various derivatives depending on whether we are going
            # forward or backward
            u_di = devito.Derivative(field, (dim, 1))
            u_di2 = devito.Derivative(field, (dim, 2))

            u_3_dt = u_3.dt if direction == 'forward' else u_3.dt.T
            u_2_dt = u_2.dt if direction == 'forward' else u_2.dt.T
            u_1_dt = u_1.dt if direction == 'forward' else u_1.dt.T

            u_3_next = u_3.forward if direction == 'forward' else u_3.backward
            u_2_next = u_2.forward if direction == 'forward' else u_2.backward
            u_1_next = u_1.forward if direction == 'forward' else u_1.backward

            u_di = u_di if direction == 'forward' else u_di.T

            # Calculate the auxiliary fields
            pde_3 = u_3_dt + (alpha_i + sigma_i) * u_3 - sigma_i ** 2 * (sigma_di + alpha_di) * u_di
            pde_2 = u_2_dt + (alpha_i + sigma_i) * u_2 - sigma_i * (2 * sigma_di * u_di + alpha_di * u_di + sigma_i * u_di2) + u_3_next  # noqa: E501
            pde_1 = u_1_dt + (alpha_i + sigma_i) * u_1 - (sigma_di * u_di + 2 * sigma_i * u_di2) + u_2_next

            # with the corresponding stencils
            pml_domain = (self._grid.pml_left[dim_i], self._grid.pml_right[dim_i])

            stencil_3 = [devito.Eq(u_3_next, devito.solve(pde_3, u_3_next, ),
                                   subdomain=dom, coefficients=subs) for dom in pml_domain]
            stencil_2 = [devito.Eq(u_2_next, devito.solve(pde_2, u_2_next, ),
                                   subdomain=dom, coefficients=subs) for dom in pml_domain]
            stencil_1 = [devito.Eq(u_1_next, devito.solve(pde_1, u_1_next, ),
                                   subdomain=dom, coefficients=subs) for dom in pml_domain]

            eq_preparation += [eq_sigma_di, eq_alpha_di]
            eq_before += stencil_3 + stencil_2 + stencil_1
            boundary_term.append(u_1_next)

        return sum(boundary_term), eq_preparation + eq_before, eq_after

    def clear(self):
        space = self._grid.space

        for dim_i in range(space.dim):
            self._grid.vars['u_3_%d' % dim_i].data_with_halo.fill(0.)
            self._grid.vars['u_2_%d' % dim_i].data_with_halo.fill(0.)
            self._grid.vars['u_1_%d' % dim_i].data_with_halo.fill(0.)

    def deallocate(self):
        space = self._grid.space

        for dim_i in range(space.dim):
            self._grid.deallocate('u_3_%d' % dim_i)
            self._grid.deallocate('u_2_%d' % dim_i)
            self._grid.deallocate('u_1_%d' % dim_i)


class _HybridBoundary2(Boundary):
    """
    Hybrid Higdon boundary condition as presented
    in https://10.0.4.69/jge/gxz102.

    """

    def __init__(self, grid):
        super().__init__(grid)

    def apply(self, field, velocity, direction='forward', **kwargs):
        subs = kwargs.pop('subs', None)
        velocity_fun = kwargs.pop('velocity_fun', None)

        space = self._grid.space
        time = self._grid.time
        t = self._grid.devito_grid.stepping_dim
        ht = t.spacing
        dimensions = self._grid.devito_grid.dimensions
        spacings = self._grid.devito_grid.spacing_symbols

        alpha_1 = 0.
        alpha_2 = np.pi / 4
        cos_1 = np.cos(alpha_1)
        cos_2 = np.cos(alpha_2)

        # Create necessary functions
        u0 = field
        u0_next = u0.forward if direction == 'forward' else u0.backward
        u0_prev = u0.backward if direction == 'forward' else u0.forward

        u1 = self._grid.time_function('u1', time_order=2, space_order=2)
        u1.data_with_halo.fill(0.)
        u1_next = u1.forward if direction == 'forward' else u1.backward

        c = self._grid.function('c', space_order=2)

        sigma = self._grid.function('sigma', space_order=0)
        sigma_data = self.damping(velocity=velocity,
                                  damping_type='cosine',
                                  damping_coefficient=1,
                                  assign=True)
        sigma.data_with_halo[:] = sigma_data

        reflection_coefficient = 10 ** (-(np.log10(max(*space.absorbing)) - 1) / np.log10(2) - 3)
        reflection_coefficient = kwargs.pop('reflection_coefficient', reflection_coefficient)
        if np.max(space.extra) > 0:
            damp = self._grid.function('damp')
            damp.data[:] = self.damping(velocity=velocity,
                                        reflection_coefficient=reflection_coefficient) * 1e-1 * time.step
        else:
            damp = 0
        u_dt = u0.backward.dt if direction == 'forward' else u0.dt.T
        damping_term = 2 * damp * u_dt

        # Prepare sub-domains
        domains = {}
        # PMLCentre
        for domain in self._grid.pml_centres:
            dims = [domain.side if dim_i == domain.dim_i else 0
                    for dim_i in range(space.dim)]
            domains[tuple(dims)] = domain

        # PMLCentreCorner
        for domain in self._grid.pml_centre_corners:
            dims = [side if side != 'middle' else 0
                    for side in domain.sides]
            domains[tuple(dims)] = domain

        # PMLCorner
        for domain in self._grid.pml_corners:
            dims = domain.sides
            domains[tuple(dims)] = domain

        # Prepare stencils
        def expand(expr_):
            return sympy.expand(expr_._evaluate())

        def replace(expr_, f, g):
            return expr_.replace(
                lambda expr: hasattr(expr, 'name') and expr.name == f.name,
                lambda expr: g.func(*expr.args)
            )

        stencils = {}
        for dim_i in range(space.dim):
            dim = dimensions[dim_i]
            spacing = spacings[dim_i]

            # Create OWWE stencils
            x0_left = {d: d + spacing if d == dim else d for d in dimensions}
            x0_right = {d: d - spacing if d == dim else d for d in dimensions}

            # u_dt = expand(u1.dt)
            #
            # def diff_op_left():
            #     u_dx2 = replace(expand(devito.Derivative(u1, (dim, 2), x0=x0_left)), u1, u0)
            #     u_dtdx = expand(devito.Derivative(u1, (dim, 1)))
            #
            #     return c ** 2 / (cos_1 * cos_2) * u_dx2 \
            #          - c * (cos_1 + cos_2) / (cos_1 * cos_2) * u_dtdx
            #
            # def diff_op_right():
            #     u_dx2 = replace(expand(devito.Derivative(u1, (dim, 2), x0=x0_right)), u1, u0)
            #     u_dtdx = expand(-devito.Derivative(u1, (dim, 1), x0=x0_right))
            #
            #     return c ** 2 / (cos_1 * cos_2) * u_dx2 \
            #          - c * (cos_1 + cos_2) / (cos_1 * cos_2) * u_dtdx
            #
            # d_left = diff_op_left()
            # d_right = diff_op_right()
            #
            # stencil_left = u_dt + replace(d_left, c, velocity_fun)
            # stencil_right = u_dt + replace(d_right, c, velocity_fun)
            #
            # stencils[dim_i] = (stencil_left, stencil_right)

            u_dt2 = replace(expand(u1.dt2), u1, u0)
            k = 2 * time.step / space.spacing[0]**2 * 0.5
            u0_ = k/2 * u0.subs(x0_left) + (1 - k) * u0 + k/2 * u0.subs(x0_right)
            u_dt2 = (u0.forward - 2*u0_ + u0.backward)/time.step**2

            def diff_op_left(u, o=1):
                u = expand(u)
                u_dx2 = expand(devito.Derivative(u, (dim, 2), x0=x0_left))
                u_dtdx = expand(devito.Derivative(u.dt(x0=t - o*ht), (dim, 1)))

                return - c ** 2 / (cos_1 * cos_2) * u_dx2 \
                       + c * (cos_1 + cos_2) / (cos_1 * cos_2) * u_dtdx

            def diff_op_right(u, o=1):
                u = expand(u)
                u_dx2 = expand(devito.Derivative(u, (dim, 2), x0=x0_right))
                u_dtdx = expand(-devito.Derivative(u.dt(x0=t - o*ht), (dim, 1), x0=x0_right))

                return - c ** 2 / (cos_1 * cos_2) * u_dx2 \
                       + c * (cos_1 + cos_2) / (cos_1 * cos_2) * u_dtdx

            d1_left = 0*time.step**2/12 * replace(replace(diff_op_left(u1), u1, u0), c, velocity_fun)
            d1_right = 0*time.step**2/12 * replace(replace(diff_op_right(u1), u1, u0), c, velocity_fun)

            d0_left = replace(replace(diff_op_left(u1), u1, u0), c, velocity_fun) + replace(diff_op_left(u1_next), c, velocity_fun)
            d0_right = replace(replace(diff_op_right(u1), u1, u0), c, velocity_fun) + replace(diff_op_right(u1_next), c, velocity_fun)

            d0_left = u_dt2 - d0_left
            d0_right = u_dt2 - d0_right

            stencils[dim_i] = ((d0_left, d0_right), (d1_left, d1_right))

        # Prepare update expressions
        eq_after = []
        for dims, domain in domains.items():
            update_u1 = 0
            update_u0 = 0
            for dim_i, dim in enumerate(dims):
                if dim == 0:
                    continue

                d0, d1 = stencils[dim_i]

                if dim == 'left':
                    update_u1 += d1[0]
                    update_u0 += d0[0]
                else:
                    update_u1 += d1[1]
                    update_u0 += d0[1]

            #  + velocity_fun ** 2 / (cos_1 * cos_2) * damping_term
            # u_owwe1 = devito.solve(u_owwe, u1_next)
            # u_owwe0 = devito.solve(u0.dt(fd_order=1) - u1_next, u0_next)
            # update = [
            #     devito.Eq(u1_next, u_owwe1,
            #               subdomain=domain, coefficients=subs),
            #     devito.Eq(u0_next, (1 - sigma)*u0_next + sigma*u_owwe0,
            #               subdomain=domain, coefficients=subs)
            # ]

            update_u0 = devito.solve(update_u0, u0_next)
            update = [
                devito.Eq(u1_next, update_u1,
                          subdomain=domain, coefficients=subs),
                devito.Eq(u0_next, (1 - sigma)*u0_next + sigma*update_u0,
                          subdomain=domain, coefficients=subs)
            ]

            eq_after.append(update)

        return 0, [], eq_after


class _HybridBoundary2(Boundary):
    """
    Hybrid Higdon boundary condition as presented
    in https://10.0.4.69/jge/gxz102.

    """

    def __init__(self, grid):
        super().__init__(grid)

    def apply(self, field, velocity, direction='forward', **kwargs):
        subs = kwargs.pop('subs', None)
        velocity_fun = kwargs.pop('velocity_fun', None)

        space = self._grid.space
        time = self._grid.time
        t_dim = self._grid.devito_grid.time_dim
        t = self._grid.devito_grid.stepping_dim
        ht = t.spacing
        dimensions = self._grid.devito_grid.dimensions
        spacings = self._grid.devito_grid.spacing_symbols

        c_integer = np.array(space.spacing) / time.step
        vp = velocity[0, 0, 0]

        # Create necessary functions
        u_tmp = self._grid.time_function('u_tmp', time_order=2, space_order=2)
        u_next = field.forward if direction == 'forward' else field.backward
        u_prev = field.backward if direction == 'forward' else field.forward
        u_tmp_next = u_tmp.forward if direction == 'forward' else u_tmp.backward

        sigma = self._grid.function('sigma', space_order=0)
        sigma_data = self.damping(velocity=velocity,
                                  damping_type='power',
                                  power_degree=1,
                                  damping_coefficient=1,
                                  assign=True)
        # sigma_data.fill(1)
        sigma.data_with_halo[:] = sigma_data

        reflection_coefficient = 10 ** (-(np.log10(max(*space.absorbing)) - 1) / np.log10(2) - 3)
        reflection_coefficient = kwargs.pop('reflection_coefficient', reflection_coefficient)
        if np.max(space.extra) > 0:
            damp = self._grid.function('damp', space_order=0)
            damp.data[:] = self.damping(velocity=velocity,
                                        damping_coefficient=1,
                                        damping_type='cosine',
                                        assign=True,
                                        mask=True)
        else:
            damp = 0

        pt = devito.Dimension(name='pt')
        plane_0 = self._grid.function('plane_0', space_order=0,
                                      dimensions=(pt, dimensions[1], dimensions[2]),
                                      shape=(100, space.extended_shape[1], space.extended_shape[2]))
        plane_1 = self._grid.time_function('plane_1', space_order=0, time_order=10,
                                           dimensions=(t, dimensions[0], dimensions[2]),
                                           shape=(10, space.extended_shape[0], space.extended_shape[2]))

        # Prepare sub-domains
        domains = {}
        # PMLCentre
        for domain in self._grid.pml_centres:
            dims = [domain.side if dim_i == domain.dim_i else 0
                    for dim_i in range(space.dim)]
            domains[tuple(dims)] = domain

        # # PMLCentreCorner
        # for domain in self._grid.pml_centre_corners:
        #     dims = [side if side != 'middle' else 0
        #             for side in domain.sides]
        #     domains[tuple(dims)] = domain

        # # PMLCorner
        # for domain in self._grid.pml_corners:
        #     dims = domain.sides
        #     domains[tuple(dims)] = domain

        # Prepare update expressions
        eq_after = []
        for dims, domain in domains.items():
            p = 0

            for dim_i, dim in enumerate(dims):
                if dim_i != 0:
                    continue

                if dim == 0:
                    continue

                if dim == 'left':
                    s = +1
                    continue
                else:
                    s = -1

                arg_0 = {d: space.absorbing[0]+0 if d == dimensions[dim_i] else d for d in dimensions}
                arg_1 = {d: space.absorbing[0]+1 if d == dimensions[dim_i] else d for d in dimensions}
                arg_2 = {d: space.absorbing[0]+2 if d == dimensions[dim_i] else d for d in dimensions}
                arg_3 = {d: space.absorbing[0]+3 if d == dimensions[dim_i] else d for d in dimensions}
                arg_4 = {d: space.absorbing[0]+4 if d == dimensions[dim_i] else d for d in dimensions}

                p += 1/3*(+ u_next.subs(arg_0) * (0.15 - 1)*(0.15 - 2)/((0 - 1)*(0 - 2))
                          + u_next.subs(arg_1) * (0.15 - 0)*(0.15 - 2)/((1 - 0)*(1 - 2))
                          + u_next.subs(arg_2) * (0.15-0)*(0.15-1)/((2-0)*(2-1))) \
                   + 1/3*(+ field.subs(arg_0) * (0.30 - 1)*(0.30 - 2)/((0 - 1)*(0 - 2))
                          + field.subs(arg_1) * (0.30 - 0)*(0.30 - 2)/((1 - 0)*(1 - 2))
                          + field.subs(arg_2) * (0.30-0)*(0.30-1)/((2-0)*(2-1))) \
                   + 1/3*(+ u_prev.subs(arg_0) * (0.45 - 1)*(0.45 - 2)/((0 - 1)*(0 - 2))
                          + u_prev.subs(arg_1) * (0.45 - 0)*(0.45 - 2)/((1 - 0)*(1 - 2))
                          + u_prev.subs(arg_2) * (0.45-0)*(0.45-1)/((2-0)*(2-1)))

            if p == 0:
                continue

            derived_dim = [
                devito.CustomDimension('dtmp%d' % i,
                                       0 if i == 0 else d.symbolic_min,
                                       int(space.absorbing[i]) if i == 0 else d.symbolic_max,
                                       int(space.absorbing[i]) if i == 0 else d.symbolic_max - d.symbolic_min,
                                       d)
                for i, d in enumerate(dimensions)
            ]
            derived_dim = tuple(derived_dim)

            dist = (space.absorbing[0] - derived_dim[0])*space.spacing[0]

            o = devito.types.Symbol(name='o', dtype=np.int32)
            d = devito.types.Symbol(name='d', dtype=np.float32)

            from devito.symbolics import INT
            from devito.finite_differences.elementary import floor

            u_sub = u_next.subs({d: v for d, v in zip(dimensions, derived_dim)})
            sigma_sub = sigma.subs({d: v for d, v in zip(dimensions, derived_dim)})
            damp_sub = damp.subs({d: v for d, v in zip(dimensions, derived_dim)})

            update = [
                devito.Eq(plane_0.subs({pt: t_dim % 100}), p),
                devito.Eq(o, (t_dim - INT(floor(dist/(1500*time.step)))) % 100,
                          implicit_dims=(t, t_dim) + derived_dim),
                devito.Eq(d, sympy.exp(-50*1/0.15*dist),
                          implicit_dims=(t, t_dim) + derived_dim),
                devito.Eq(u_sub,
                          (1-sigma_sub)*u_sub + sigma_sub*plane_0.subs({pt: o}).subs({d: v for d, v in zip(dimensions, derived_dim)}),
                          implicit_dims=(t, t_dim) + derived_dim),
            ]
            eq_after.append(update)

        return 0, [], eq_after

    def clear(self):
        self._grid.vars.u_tmp.data_with_halo.fill(0.)
