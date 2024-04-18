
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
            damp.data[:] = self.damping(velocity=velocity,
                                        reflection_coefficient=reflection_coefficient,
                                        mask=True, **kwargs) * time.step
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
            damp.data[:] = 7 * self.damping(velocity=velocity,
                                            reflection_coefficient=reflection_coefficient, **kwargs) * time.step
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
        abox = kwargs.pop('abox', None)

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
            pml_domain = [abox.intersection(dom) for dom in pml_domain] \
                if devito.pro_available and isinstance(abox, devito.ABox) else pml_domain

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
