import numpy as np
import sympy as sp
from devito import TimeFunction, Coefficient, Eq, Substitutions, solve, Operator
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model

# Define symbol for laplacian replacement
H = sp.symbols('H')

# Define a physical size
Lx = 2000
Lz = Lx
h = 10
Nx = int(Lx / h) + 1
Nz = Nx

shape = (Nx, Nz)  # Number of grid point
spacing = (h, h)  # Grid spacing in m. The domain size is now 2km by 2km
origin = (0., 0.)

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :121] = 1.5
v[:, 121:] = 4.0

# With the velocity and model size defined, we can create the seismic model that
# encapsulates these properties. We also define the size of the absorbing layer as 10 grid points
nbl = 10
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=20, nbl=nbl, bcs="damp")

t0 = 0.  # Simulation starts a t=0
tn = 500.  # Simulation lasts 0.5 seconds (500 ms)
dt = 1.0  # Time step of 0.2ms

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.015  # Source peak frequency is 25Hz (0.025 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 800.  # Depth is 800m

order = 10
u_DRP = TimeFunction(name="u_DRP", grid=model.grid, time_order=2, space_order=order, coefficients='symbolic')
u_tmp = TimeFunction(name="u_tmp", grid=model.grid, time_order=2, space_order=order)

pde_DRP = model.m * u_DRP.dt2 - H + model.damp * u_DRP.dt

# Define our custom FD coefficients:
x, z = model.grid.dimensions

# Lower layer
weights_l = np.array([0., 0., 0.0274017,
                      -0.223818, 1.64875, -2.90467,
                      1.64875, -0.223818, 0.0274017,
                      0., 0.])

ux_l_coeffs = Coefficient(2, u_DRP, x, weights_l / x.spacing ** 2)
uz_l_coeffs = Coefficient(2, u_DRP, z, weights_l / z.spacing ** 2)

# Define our custom FD coefficients:
x, z = model.grid.dimensions
# Upper layer
weights_u = np.array([2.00462e-03, -1.63274e-02, 7.72781e-02,
                      -3.15476e-01, 1.77768e+00, -3.05033e+00,
                      1.77768e+00, -3.15476e-01, 7.72781e-02,
                      -1.63274e-02, 2.00462e-03])
# Lower layer
weights_l = np.array([0., 0., 0.0274017,
                      -0.223818, 1.64875, -2.90467,
                      1.64875, -0.223818, 0.0274017,
                      0., 0.])
# Create the Devito Coefficient objects:
ux_l_coeffs = Coefficient(2, u_DRP, x, weights_l / x.spacing ** 2)
uz_l_coeffs = Coefficient(2, u_DRP, z, weights_l / z.spacing ** 2)

# And the replacement rules:
coeffs_l = Substitutions(ux_l_coeffs, uz_l_coeffs)

# line below would work
laplace = u_DRP.laplace

# laplace = laplace.evaluate.xreplace(coeffs_l.rules).dx2 + laplace.evaluate.xreplace(coeffs_l.rules).dy2
# laplace = u_DRP.laplace + dt**2/12 * laplace

print(laplace.evaluate.xreplace(coeffs_l.rules).xreplace({u_DRP: u_tmp}))
input()

# # but this one wouldn't
# laplace = u_DRP.laplace + dt**2/12 * u_DRP.biharmonic(1/model.m)

stencil_l = Eq(u_DRP.forward, solve(pde_DRP, u_DRP.forward).subs({H: laplace}), coefficients=coeffs_l)

# Source term:
src_term = src.inject(field=u_DRP.forward, expr=src * dt ** 2 / model.m)

# Create the operator
op = Operator([stencil_l] + src_term, subs=model.spacing_map)

op(time=time_range.num - 1, dt=dt)
