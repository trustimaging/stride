
import os
import gc
import sympy
import logging
import functools
import itertools
import numpy as np
import scipy.special

import mosaic
from mosaic.types import Struct

from . import import_devito as devito
from ...problem.base import Gridded


__all__ = ['OperatorDevito', 'GridDevito', 'config_devito']


class FullDomain(devito.SubDomain):

    name = 'full_domain'

    def __init__(self, space_order, extra):
        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        return {dimension: dimension for dimension in dimensions}


class InteriorDomain(devito.SubDomain):

    name = 'interior_domain'

    def __init__(self, space_order, extra):
        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        return {dimension: ('middle', extra, extra)
                for dimension, extra in zip(dimensions, self.extra)}


class PMLSide(devito.SubDomain):

    def __init__(self, space_order, extra, dim, side):
        self.dim_i = dim
        self.side = side
        self.name = 'pml_side_' + side + str(dim)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: dimension for dimension in dimensions}
        domain[dimensions[self.dim_i]] = (self.side, self.extra[self.dim_i])

        return domain


class PMLCentre(devito.SubDomain):

    def __init__(self, space_order, extra, dim, side):
        self.dim_i = dim
        self.side = side
        self.name = 'pml_centre_' + side + str(dim)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: ('middle', extra, extra)
                  for dimension, extra in zip(dimensions, self.extra)}
        domain[dimensions[self.dim_i]] = (self.side, self.extra[self.dim_i])

        return domain


class PMLCorner(devito.SubDomain):

    def __init__(self, space_order, extra, *sides):
        self.sides = sides
        self.name = 'pml_corner_' + '_'.join(sides)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: (side, extra)
                  for dimension, side, extra in zip(dimensions, self.sides, self.extra)}

        return domain


class PMLCentreCorner(devito.SubDomain):

    def __init__(self, space_order, extra, *sides):
        self.sides = sides
        self.name = 'pml_centre_corner_' + '_'.join(sides)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: (side, extra) if side != 'middle' else ('middle', extra, extra)
                  for dimension, side, extra in zip(dimensions, self.sides, self.extra)}

        return domain


class PMLPartial(devito.SubDomain):

    def __init__(self, space_order, extra, dim, side):
        self.dim_i = dim
        self.side = side
        self.name = 'pml_partial_' + side + str(dim)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: ('middle', extra, extra)
                  for dimension, extra in zip(dimensions, self.extra)}

        domain[dimensions[0]] = dimensions[0]
        if len(dimensions) > 2:
            domain[dimensions[1]] = dimensions[1]

        domain[dimensions[self.dim_i]] = (self.side, self.extra[self.dim_i])

        return domain


def _cached(func):

    @functools.wraps(func)
    def cached_wrapper(self, *args, **kwargs):
        name = args[0]
        cached = kwargs.get('cached', True)
        replace_cached = kwargs.get('replace_cached', True)

        if cached is True:
            fun = self.vars.get(name, None)
            if fun is not None:
                _args, _kwargs = self.cached_args[name]

                same_args = True
                for arg, _arg in zip(args, _args):
                    try:
                        same_args = same_args and bool(arg == _arg)
                    except ValueError:
                        pass

                for arg, _arg in zip(kwargs.values(), _kwargs.values()):
                    try:
                        same_args = same_args and bool(arg == _arg)
                    except ValueError:
                        pass

                if same_args:
                    return fun

        elif replace_cached:
            self.vars.pop(name, None)
            self.cached_args.pop(name, None)

        if name not in self.cached_funcs:
            self.cached_funcs[name] = cached_wrapper

        fun = func(self, *args, **kwargs)

        if replace_cached:
            self.vars[name] = fun
            self.cached_args[name] = (args, kwargs)

        return fun

    return cached_wrapper


class GridDevito(Gridded):
    """
    Instances of this class encapsulate the Devito grid, and interact with it by
    generating appropriate functions on demand.

    Instances will also keep a cache of created Devito functions under the ``vars``
    attribute, which can be accessed by name using dot notation.

    Parameters
    ----------
    space_order : int
        Default space order of the discretisation for functions of the grid.
    time_order : int
        Default time order of the discretisation for functions of the grid.
    time_dim : Time, optional
        Time dimension on which to step. or ``False`` if no time dependency.
    grid : Grid, optional
        Existing grid, if not provided one will be created. Either a grid or
        space, time and slow_time need to be provided.
    space : Space, optional
    time : Time, optional
    slow_time : SlowTime, optional

    """

    def __init__(self, space_order, time_order, time_dim=None, **kwargs):
        super().__init__(**kwargs)

        self.vars = Struct()
        self.cached_args = Struct()
        self.cached_funcs = Struct()

        self.space_order = space_order
        self.time_order = time_order

        self.time_dim = time_dim if time_dim is not None else self.time
        self._time_under_count = 0

        grid_kwargs = dict()

        space = self.space

        if space is None:
            origin = (0,)
            extended_shape = (1,)
            extended_extent = (1,)
        else:
            extra = space.absorbing

            origin = space.pml_origin
            extended_shape = space.extended_shape
            extended_extent = tuple(np.array(space.spacing) * (np.array(space.extended_shape) - 1))

            self.full = FullDomain(space_order, extra)
            self.interior = InteriorDomain(space_order, extra)
            self.pml_left = tuple()
            self.pml_right = tuple()
            self.pml_centres = tuple()
            self.pml_partials = tuple()
            self.pml_centre_corners = tuple()

            for dim in range(space.dim):
                self.pml_left += (PMLSide(space_order, extra, dim, 'left'),)
                self.pml_right += (PMLSide(space_order, extra, dim, 'right'),)
                self.pml_centres += (PMLCentre(space_order, extra, dim, 'left'),
                                     PMLCentre(space_order, extra, dim, 'right'))
                self.pml_partials += (PMLPartial(space_order, extra, dim, 'left'),
                                      PMLPartial(space_order, extra, dim, 'right'))

                for sides in itertools.product(['left', 'right'], repeat=space.dim-1):
                    sides = list(sides)
                    sides.insert(dim, 'middle')
                    self.pml_centre_corners += (PMLCentreCorner(space_order, extra, *sides),)

            self.pml_corners = [PMLCorner(space_order, extra, *sides)
                                for sides in itertools.product(['left', 'right'],
                                                               repeat=space.dim)]
            self.pml_corners = tuple(self.pml_corners)

            self.pml = self.pml_partials

            grid_kwargs['subdomains'] = (self.full, self.interior,) + \
                                         self.pml + self.pml_left + self.pml_right + \
                                         self.pml_centres + self.pml_corners + self.pml_centre_corners

        dimensions = None
        time_dimension = None

        parent_grid = kwargs.pop('parent_grid', None)
        if parent_grid is not None:
            dimensions = parent_grid.dimensions
            time_dimension = devito.TimeDimension(name='time_inner',
                                                  spacing=devito.types.Scalar(name='dt_inner', is_const=True))
            self.num_inner = kwargs.pop('num_inner', 1)
        else:
            self.num_inner = None

        self.dtype = kwargs.pop('dtype', np.float32)

        self.devito_grid = devito.Grid(extent=extended_extent,
                                       shape=extended_shape,
                                       origin=origin,
                                       dimensions=dimensions,
                                       time_dimension=time_dimension,
                                       dtype=self.dtype,
                                       **grid_kwargs)

    @_cached
    def symbol(self, name, dtype=np.float32, **kwargs):
        """
        Create a Devito Function with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        dtype : data-type, optional
            Data type for the symbol, defaults to float32.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.Symbol
            Generated symobl.

        """
        fun = devito.Symbol(name=name,
                            dtype=dtype,
                            **kwargs)

        return fun

    @_cached
    def function(self, name, space_order=None, **kwargs):
        """
        Create a Devito Function with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.Function
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order

        fun = devito.Function(name=name,
                              grid=kwargs.pop('grid', self.devito_grid),
                              space_order=space_order,
                              dtype=kwargs.pop('dtype', self.dtype),
                              **kwargs)

        return fun

    @_cached
    def time_function(self, name, space_order=None, time_order=None, **kwargs):
        """
        Create a Devito TimeFunction with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        time_order : int, optional
            Time order of the discretisation, defaults to the grid time order.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.TimeFunction
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order
        time_order = self.time_order if time_order is None else time_order

        fun = devito.TimeFunction(name=name,
                                  grid=kwargs.pop('grid', self.devito_grid),
                                  time_order=time_order,
                                  space_order=space_order,
                                  dtype=kwargs.pop('dtype', self.dtype),
                                  **kwargs)

        return fun

    @_cached
    def vector_function(self, name, space_order=None, **kwargs):
        """
        Create a Devito VectorFunction with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.VectorFunction
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order

        fun = devito.VectorFunction(name=name,
                                    grid=kwargs.pop('grid', self.devito_grid),
                                    space_order=space_order,
                                    dtype=kwargs.pop('dtype', self.dtype),
                                    **kwargs)

        return fun

    @_cached
    def vector_time_function(self, name, space_order=None, time_order=None, **kwargs):
        """
        Create a Devito VectorTimeFunction with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        time_order : int, optional
            Time order of the discretisation, defaults to the grid time order.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.VectorTimeFunction
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order
        time_order = self.time_order if time_order is None else time_order

        fun = devito.VectorTimeFunction(name=name,
                                        grid=kwargs.pop('grid', self.devito_grid),
                                        time_order=time_order,
                                        space_order=space_order,
                                        dtype=kwargs.pop('dtype', self.dtype),
                                        **kwargs)

        return fun

    @_cached
    def tensor_time_function(self, name, space_order=None, time_order=None, **kwargs):
        """
        Create a Devito TensorTimeFunction with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        time_order : int, optional
            Time order of the discretisation, defaults to the grid time order.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.TimeFunction
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order
        time_order = self.time_order if time_order is None else time_order

        fun = devito.TensorTimeFunction(name=name,
                                        grid=kwargs.pop('grid', self.devito_grid),
                                        time_order=time_order,
                                        space_order=space_order,
                                        dtype=kwargs.pop('dtype', self.dtype),
                                        **kwargs)

        return fun

    @_cached
    def undersampled_time_function(self, name, factor, time_bounds=None,
                                   space_order=None, time_order=None, **kwargs):
        """
        Create an undersampled version of a Devito function with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        factor : int
            Undersampling factor.
        time_bounds : tuple, optional
            Timestep bounds in which the function is sampled, defaults to all timesteps.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        time_order : int, optional
            Time order of the discretisation, defaults to the grid time order.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.Function
            Generated function.

        """
        time_bounds = time_bounds or (0, self.time.extended_num-1)

        time_under, buffer_size = self._time_undersampled('time_under', factor, time_bounds)

        compression = kwargs.pop('compression', None)

        fun = self.time_function(name,
                                 space_order=space_order,
                                 time_order=time_order,
                                 time_dim=time_under,
                                 save=buffer_size,
                                 dtype=kwargs.pop('dtype', self.dtype),
                                 compression=compression,
                                 **kwargs)

        space_dims = fun.dimensions[1:]

        return fun.func(time_under - int(time_bounds[0] // factor), *space_dims)

    def undersampled_time_derivative(self, fun, factor, time_bounds=None, offset=None,
                                     deriv_order=1, fd_order=1):
        offset = offset or (0, 0)

        time_under, buffer_size = self._time_undersampled('time_under_d', factor, time_bounds, offset)

        deriv = devito.Derivative(fun, (fun.dimensions[0], deriv_order), fd_order=fd_order)
        deriv = deriv.xreplace({fun.dimensions[0]: time_under})

        return deriv

    def _time_undersampled(self, name, factor, time_bounds=None, offset=None):
        time_bounds = time_bounds or (0, self.time.extended_num - 1)
        offset = offset or (0, 0)

        time_dim = self.devito_grid.time_dim

        condition = sympy.And(devito.symbolics.CondEq(time_dim % factor, 0),
                              devito.Ge(time_dim, time_bounds[0] + offset[0]),
                              devito.Le(time_dim, time_bounds[1] - offset[1]), )

        time_under = devito.ConditionalDimension('%s%d' % (name, self._time_under_count),
                                                 parent=time_dim,
                                                 factor=factor,
                                                 condition=condition)
        self._time_under_count += 1

        # buffer_size = (time_bounds[1] - time_bounds[0] + factor) // factor + 1
        # TODO Force larger buffer size to prevent devito issue
        buffer_size = (self.time.extended_num - 1 - 0 + factor) // factor + 1

        return time_under, buffer_size

    @_cached
    def sparse_time_function(self, name, num=1, space_order=None, time_order=None,
                             coordinates=None, interpolation_type='linear', **kwargs):
        """
        Create a Devito SparseTimeFunction with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        num : int, optional
            Number of points in the function, defaults to 1.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        time_order : int, optional
            Time order of the discretisation, defaults to the grid time order.
        coordinates : ndarray, optional
            Spatial coordinates of the sparse points (num points, dimensions), only
            needed when interpolation is not linear.
        interpolation_type : str, optional
            Type of interpolation to perform (``linear``, ``sinc``, or ``hicks``), defaults
            to ``linear``, computationally more efficient but less accurate.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.SparseTimeFunction
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order
        time_order = self.time_order if time_order is None else time_order
        time_bounds = kwargs.pop('time_bounds', (0, self.time.extended_num))
        smooth = kwargs.pop('smooth', False)

        # Define variables
        p_dim = kwargs.pop('p_dim', devito.Dimension(name='p_%s' % name))

        sparse_kwargs = dict(name=name,
                             grid=kwargs.pop('grid', self.devito_grid),
                             dimensions=kwargs.get('dimensions', (self.devito_grid.time_dim, p_dim)),
                             npoint=num,
                             nt=time_bounds[1]-0,
                             space_order=space_order,
                             time_order=time_order,
                             dtype=kwargs.pop('dtype', self.dtype))
        sparse_kwargs.update(kwargs)

        if interpolation_type == 'linear':
            fun = devito.SparseTimeFunction(**sparse_kwargs)

        elif interpolation_type == 'sinc':
            r = sparse_kwargs.pop('r', 7)
            fun = devito.SparseTimeFunction(interpolation='sinc', r=r,
                                            coordinates=coordinates,
                                            **sparse_kwargs)

        elif interpolation_type == 'hicks':
            r = sparse_kwargs.pop('r', 7)

            reference_gridpoints, coefficients = self._calculate_hicks(coordinates, smooth=smooth)

            fun = devito.PrecomputedSparseTimeFunction(r=r+1,
                                                       gridpoints=reference_gridpoints,
                                                       interpolation_coeffs=coefficients,
                                                       **sparse_kwargs)

        else:
            raise ValueError('Only "linear" and "hicks" interpolations are allowed.')

        return fun

    @_cached
    def sparse_function(self, name, num=1, space_order=None,
                        coordinates=None, interpolation_type='linear', **kwargs):
        """
        Create a Devito SparseFunction with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        num : int, optional
            Number of points in the function, defaults to 1.
        space_order : int, optional
            Space order of the discretisation, defaults to the grid space order.
        coordinates : ndarray, optional
            Spatial coordinates of the sparse points (num points, dimensions), only
            needed when interpolation is not linear.
        interpolation_type : str, optional
            Type of interpolation to perform (``linear`` or ``hicks``), defaults
            to ``linear``, computationally more efficient but less accurate.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.SparseTimeFunction
            Generated function.

        """
        space_order = self.space_order if space_order is None else space_order

        # Define variables
        p_dim = kwargs.pop('p_dim', devito.Dimension(name='p_%s' % name))

        sparse_kwargs = dict(name=name,
                             grid=kwargs.pop('grid', self.devito_grid),
                             dimensions=kwargs.get('dimensions', (p_dim,)),
                             npoint=num,
                             space_order=space_order,
                             dtype=kwargs.pop('dtype', self.dtype))
        sparse_kwargs.update(kwargs)

        if interpolation_type == 'linear':
            fun = devito.SparseFunction(**sparse_kwargs)

        elif interpolation_type == 'hicks':
            r = sparse_kwargs.pop('r', 7)

            reference_gridpoints, coefficients = self._calculate_hicks(coordinates)

            fun = devito.PrecomputedSparseFunction(r=r,
                                                   gridpoints=reference_gridpoints,
                                                   interpolation_coeffs=coefficients,
                                                   **sparse_kwargs)

        else:
            raise ValueError('Only "linear" and "hicks" interpolations are allowed.')

        return fun

    def func(self, name, cached=False):
        """
        Re-instantiate devito function, if ``name`` is cached.

        Parameters
        ----------
        name : str
            Name of the function.
        cached : bool, optional
            Whether to cache the result of the func call, defaults to ``False``.

        Returns
        -------

        """
        func = self.cached_funcs[name]
        args, kwargs = self.cached_args[name]
        return func(self, *args, **kwargs,
                    cached=cached, replace_cached=cached)

    def deallocate(self, name, collect=False):
        """
        Remove internal references to data buffers, if ``name`` is cached.

        Parameters
        ----------
        name : str
            Name of the function.
        collect : bool, optional
            Whether to garbage collect after deallocate, defaults to ``False``.

        Returns
        -------

        """
        if name in self.vars:
            if hasattr(self.vars[name], 'is_VectorValued') and self.vars[name].is_VectorValued:
                for dim in range(self.space.dim):
                    del self.vars[name][dim]._data
                    self.vars[name][dim]._data = None
            else:
                del self.vars[name]._data
                self.vars[name]._data = None

            if collect:
                gc.collect()

    def delete(self, name, collect=False):
        """
        Remove internal references to devito function, if ``name`` is cached.

        Parameters
        ----------
        name : str
            Name of the function.
        collect : bool, optional
            Whether to garbage collect after deallocate, defaults to ``False``.

        Returns
        -------

        """
        if name in self.vars:
            del self.vars[name]
            del self.cached_funcs[name]
            del self.cached_args[name]

            if collect:
                devito.clear_cache(force=True)

    def with_halo(self, data, value=None, time_dependent=False, is_vector=False, **kwargs):
        """
        Pad ndarray with appropriate halo given the grid space order.

        Parameters
        ----------
        data : ndarray
            Array to pad
        value : float, optional
            Value used for the filling, defaults to edge value.
        time_dependent : bool, optional
            Whether the array should be considered time dependent,
            defaults to ``False``.
        is_vector : bool, optional
            Whether the array should be considered a vector field,
            defaults to ``False``.

        Returns
        -------
        ndarray
            Padded array.

        """
        space_order = kwargs.pop('space_order', self.space_order)
        pad_widths = [[space_order, space_order]
                      for _ in self.space.shape]

        if time_dependent:
            pad_widths.insert(0, [0, 0])

        if is_vector:
            pad_widths.insert(0, [0, 0])

        if value is None:
            return np.pad(data, pad_widths, mode='edge')
        else:
            return np.pad(data, pad_widths, mode='constant', constant_values=value)

    def _calculate_hicks(self, coordinates, smooth=False):
        space = self.space

        # Calculate the reference gridpoints and offsets
        grid_coordinates = (coordinates - np.array(space.pml_origin)) / np.array(space.spacing)
        reference_gridpoints = np.floor(grid_coordinates).astype(np.int32)
        offsets = grid_coordinates - reference_gridpoints

        # Pre-calculate stuff
        kaiser_b = 4.14
        kaiser_half_width = 3
        kaiser_den = scipy.special.iv(0, kaiser_b)
        kaiser_extended_width = kaiser_half_width/0.99

        # Calculate coefficients
        r = 2*kaiser_half_width+1
        num = coordinates.shape[0]
        coefficients = np.zeros((num, space.dim, r+1))

        for grid_point in range(-kaiser_half_width, kaiser_half_width+1):
            index = kaiser_half_width + grid_point

            x = offsets - grid_point

            weights = (x / kaiser_extended_width)**2
            weights[weights > 1] = 1
            b_weights = scipy.special.iv(0, kaiser_b * np.sqrt(1 - weights)) / kaiser_den

            coefficients[:, :, index] = np.sinc(x) * b_weights

        # Smooth if needed
        if smooth:
            n = kaiser_half_width-1
            a = coefficients[:, :, n]
            b = coefficients[:, :, n+1]
            c = coefficients[:, :, n+2]
            coefficients[:, :, n-1] = coefficients[:, :, n-1] + a*0.01
            coefficients[:, :, n] = a*0.98 + b*0.03
            coefficients[:, :, n+1] = b*0.94 + (a+c)*0.01
            coefficients[:, :, n+2] = c*0.98 + b*0.03
            coefficients[:, :, n+3] = coefficients[:, :, n+3] + c*0.01

        return reference_gridpoints, coefficients


class OperatorDevito:
    """
    Instances of this class encapsulate Devito operators, how to configure them and how to run them.


    Parameters
    ----------
    grid : GridDevito, optional
        Predefined GridDevito. A new one will be created unless specified.
    name : str
            Name to give to the operator, defaults to ``kernel``.
    """

    def __init__(self, *args, grid=None, name='kernel', **kwargs):
        self.name = name

        self.devito_operator = None
        self.devito_context = {}

        self.grid = GridDevito(*args, **kwargs) if grid is None else grid

    def set_operator(self, op, **kwargs):
        """
        Set up a Devito operator from a list of operations.

        Parameters
        ----------
        op : list
            List of operations to be given to the devito.Operator instance.
        kwargs : optional
            Configuration parameters to set for Devito overriding defaults.

        Returns
        -------

        """
        from mosaic.utils.logger import log_level

        platform = kwargs.pop('platform', None)
        devito_config = kwargs.pop('devito_config', {})

        subs = self.grid.devito_grid.spacing_map

        if self.grid.time_dim:
            time = self.grid.time_dim
            time_spacing = self.grid.devito_grid.time_dim.spacing
            subs = {**subs, **{time_spacing: devito_config.get('dt', time.step)}}

        if platform is None or platform == 'cpu':
            default_config = {
                'name': self.name,
                'subs': subs,
                'opt': 'advanced',
            }

        elif platform == 'cpu-icc':
            default_config = {
                'name': self.name,
                'subs': subs,
                'opt': 'advanced',
                'compiler': 'icc',
            }

        elif platform == 'cpu-nvc':
            default_config = {
                'name': self.name,
                'subs': subs,
                'opt': 'advanced',
                'compiler': 'nvc',
            }

        elif platform == 'nvidia-acc' or platform == 'nvidia-nvc':
            default_config = {
                'name': self.name,
                'subs': subs,
                'opt': 'advanced',
                'autotuning': 'off',
                'compiler': 'nvc',
                'language': 'openacc',
                'platform': 'nvidiaX',
            }

        elif platform == 'nvidia-cuda' and devito.pro_available:
            default_config = {
                'name': self.name,
                'subs': subs,
                'opt': 'advanced',
                'compiler': 'cuda',
                'language': 'cuda',
                'platform': 'nvidiaX',
            }

        elif platform == 'amd-hip' and devito.pro_available:
            default_config = {
                'name': self.name,
                'subs': subs,
                'opt': 'advanced',
                'compiler': 'hip',
                'language': 'hip',
                'platform': 'amdgpuX',
            }

        else:
            raise ValueError('Unrecognised platform %s' % platform)

        default_config.update(devito_config)

        context = {'log-level': 'DEBUG' if log_level in ['perf', 'debug'] else 'INFO'}
        compiler_config = {}
        for key, value in default_config.items():
            if key in devito.configuration and key != 'opt':
                context[key] = value
            else:
                compiler_config[key] = value

        self.devito_context = context

        logger = mosaic.logger()
        logger.perf('Operator `%s` instance configuration:' % self.name)

        for key, value in default_config.items():
            logger.perf('\t * %s=%s' % (key, value))

        with devito.switchconfig(**self.devito_context):
            self.devito_operator = devito.Operator(op, **compiler_config)

    def compile(self):
        """
        Compile the operator.

        Returns
        -------

        """
        self.devito_operator.cfunction

    def run(self, **kwargs):
        """
        Run the operator.

        Returns
        -------

        """

        default_kwargs = {}

        for arg in self.devito_operator.parameters:
            if arg.name in self.grid.vars:
                default_kwargs[arg.name] = self.grid.vars[arg.name]

        autotune = kwargs.pop('autotune', None)
        default_kwargs.update(kwargs)

        if self.grid.time_dim:
            time = self.grid.time_dim

            default_kwargs['dt'] = default_kwargs.get('dt', time.step)

            if self.grid.num_inner is not None:
                default_kwargs['dt_inner'] = default_kwargs.get('dt_inner', time.step/self.grid.num_inner)
                default_kwargs['time_inner_m'] = default_kwargs.get('time_inner_m', 0)
                default_kwargs['time_inner_M'] = default_kwargs.get('time_inner_M', self.grid.num_inner-1)

        runtime_context = {}
        runtime_kwargs = {}
        for key, value in default_kwargs.items():
            if key in devito.configuration:
                runtime_context[key] = value
            else:
                runtime_kwargs[key] = value

        with devito.switchconfig(**self.devito_context, **runtime_context):
            if autotune is None:
                try:
                    tuned = self.devito_operator._state['autotuning'][-1]['tuned']
                    runtime_kwargs.update(tuned)
                    runtime_kwargs['autotune'] = 'off'
                except KeyError:
                    pass

            self.devito_operator.apply(**runtime_kwargs)


def config_devito(**kwargs):
    from mosaic.utils.logger import log_level

    # global devito config
    default_config = {
        'autotuning': ['aggressive', 'runtime'],
        'develop-mode': False,
        'mpi': False,
        'log-level': 'DEBUG' if log_level in ['perf', 'debug'] else 'INFO',
    }

    compiler = os.getenv('DEVITO_COMPILER', None)
    if compiler is not None:
        default_config['compiler'] = compiler

    language = os.getenv('DEVITO_LANGUAGE', 'openmp')
    if language is not None:
        default_config['language'] = language

    devito_config = kwargs.pop('devito_config', {})
    default_config.update(devito_config)

    logger = mosaic.logger()
    logger.perf('Default Devito configuration:')

    for key, value in default_config.items():
        logger.perf('\t * %s=%s' % (key, value))

        devito.parameters.configuration[key] = value

    # fix devito logging
    devito_logger = logging.getLogger('Devito')
    devito_logger.setLevel(logging.DEBUG if log_level in ['perf', 'debug'] else logging.INFO)
    devito.logger.logger = devito_logger

    class RerouteFilter(logging.Filter):

        def __init__(self):
            super().__init__()

        def filter(self, record):
            _logger = mosaic.logger()

            if record.levelno == devito.logger.PERF:
                _logger.perf(record.msg)

            elif record.levelno == logging.ERROR:
                _logger.error(record.msg)

            elif record.levelno == logging.WARNING:
                _logger.warning(record.msg)

            elif record.levelno == logging.DEBUG:
                _logger.debug(record.msg)

            else:
                _logger.perf(record.msg)

            return False

    devito_logger.addFilter(RerouteFilter())

    runtime = mosaic.runtime()
    if runtime is not None and runtime.mode == 'local':
        devito_logger.propagate = False
