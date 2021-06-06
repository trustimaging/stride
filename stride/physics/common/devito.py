
import os
import gc
import devito
import logging
import functools
import itertools
import numpy as np
import scipy.special

import mosaic
from mosaic.types import Struct

from ...problem.base import Gridded


__all__ = ['OperatorDevito', 'GridDevito']


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
        self.dim = dim
        self.side = side
        self.name = 'pml_side_' + side + str(dim)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: dimension for dimension in dimensions}
        domain[dimensions[self.dim]] = (self.side, self.extra[self.dim])

        return domain


class PMLCentre(devito.SubDomain):

    def __init__(self, space_order, extra, dim, side):
        self.dim = dim
        self.side = side
        self.name = 'pml_centre_' + side + str(dim)

        super().__init__()

        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        domain = {dimension: ('middle', extra, extra)
                  for dimension, extra in zip(dimensions, self.extra)}
        domain[dimensions[self.dim]] = (self.side, self.extra[self.dim])

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


class PMLPartial(devito.SubDomain):

    def __init__(self, space_order, extra, dim, side):
        self.dim = dim
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

        domain[dimensions[self.dim]] = (self.side, self.extra[self.dim])

        return domain


def _cached(func):

    @functools.wraps(func)
    def cached_wrapper(self, *args, **kwargs):
        name = args[0]
        cached = kwargs.pop('cached', True)

        if cached is True:
            fun = self.vars.get(name, None)
            if fun is not None:
                _args, _kwargs = self._args[name]

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

        fun = func(self, *args, **kwargs)

        self.vars[name] = fun
        self._args[name] = (args, kwargs)

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
    grid : Grid, optional
        Existing grid, if not provided one will be created. Either a grid or
        space, time and slow_time need to be provided.
    space : Space, optional
    time : Time, optional
    slow_time : SlowTime, optional

    """

    def __init__(self, space_order, time_order, **kwargs):
        super().__init__(**kwargs)

        self.vars = Struct()
        self._args = Struct()

        self.space_order = space_order
        self.time_order = time_order

        space = self.space
        extra = space.absorbing

        extended_extent = tuple(np.array(space.spacing) * (np.array(space.extended_shape) - 1))

        self.full = FullDomain(space_order, extra)
        self.interior = InteriorDomain(space_order, extra)
        self.pml_left = tuple()
        self.pml_right = tuple()
        self.pml_centres = tuple()
        self.pml_partials = tuple()

        for dim in range(space.dim):
            self.pml_left += (PMLSide(space_order, extra, dim, 'left'),)
            self.pml_right += (PMLSide(space_order, extra, dim, 'right'),)
            self.pml_centres += (PMLCentre(space_order, extra, dim, 'left'),
                                 PMLCentre(space_order, extra, dim, 'right'))
            self.pml_partials += (PMLPartial(space_order, extra, dim, 'left'),
                                  PMLPartial(space_order, extra, dim, 'right'))

        self.pml_corners = [PMLCorner(space_order, extra, *sides)
                            for sides in itertools.product(['left', 'right'],
                                                           repeat=space.dim)]
        self.pml_corners = tuple(self.pml_corners)

        self.pml = self.pml_partials

        self.devito_grid = devito.Grid(extent=extended_extent,
                                       shape=space.extended_shape,
                                       origin=space.pml_origin,
                                       subdomains=(self.full, self.interior,) +
                                                   self.pml + self.pml_left + self.pml_right +
                                                   self.pml_centres + self.pml_corners,
                                       dtype=np.float32)

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
            Type of interpolation to perform (``linear`` or ``hicks``), defaults
            to ``linear``, computationally more efficient but less accurate.
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.SparseTimeFunction
            Generated function.

        """
        space_order = space_order or self.space_order
        time_order = time_order or self.time_order

        # Define variables
        p_dim = devito.Dimension(name='p_%s' % name)

        sparse_kwargs = dict(name=name,
                             grid=self.devito_grid,
                             dimensions=(self.devito_grid.time_dim, p_dim),
                             npoint=num,
                             nt=self.time.extended_num,
                             space_order=space_order,
                             time_order=time_order,
                             dtype=np.float32)
        sparse_kwargs.update(kwargs)

        if interpolation_type == 'linear':
            fun = devito.SparseTimeFunction(**sparse_kwargs)

        elif interpolation_type == 'hicks':
            r = sparse_kwargs.pop('r', 7)

            reference_gridpoints, coefficients = self._calculate_hicks(coordinates)

            fun = devito.PrecomputedSparseTimeFunction(r=r,
                                                       gridpoints=reference_gridpoints,
                                                       interpolation_coeffs=coefficients,
                                                       **sparse_kwargs)

        else:
            raise ValueError('Only "linear" and "hicks" interpolations are allowed.')

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
        space_order = space_order or self.space_order

        fun = devito.Function(name=name,
                              grid=self.devito_grid,
                              space_order=space_order,
                              dtype=np.float32,
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
        space_order = space_order or self.space_order
        time_order = time_order or self.time_order

        fun = devito.TimeFunction(name=name,
                                  grid=self.devito_grid,
                                  time_order=time_order,
                                  space_order=space_order,
                                  dtype=np.float32,
                                  **kwargs)

        return fun

    @_cached
    def undersampled_time_function(self, name, factor, space_order=None, time_order=None, **kwargs):
        """
        Create an undersampled version of a Devito function with parameters provided.

        Parameters
        ----------
        name : str
            Name of the function.
        factor : int,=
            Undersampling factor.
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
        time_under = devito.ConditionalDimension('time_under',
                                                 parent=self.devito_grid.time_dim,
                                                 factor=factor)

        buffer_size = (self.time.extended_num + factor - 1) // factor

        return self.time_function(name,
                                  space_order=space_order,
                                  time_order=time_order,
                                  time_dim=time_under,
                                  save=buffer_size,
                                  **kwargs)

    def deallocate(self, name):
        """
        Remove internal references to data buffers, if ``name`` is cached.

        Parameters
        ----------
        name : str
            Name of the function.

        Returns
        -------

        """
        if name in self.vars:
            del self.vars[name]._data
            self.vars[name]._data = None
            gc.collect()

    def with_halo(self, data):
        """
        Pad ndarray with appropriate halo given the grid space order.

        Parameters
        ----------
        data : ndarray
            Array to pad

        Returns
        -------
        ndarray
            Padded array.

        """
        pad_widths = [[self.space_order, self.space_order]
                      for _ in self.space.shape]

        return np.pad(data, pad_widths, mode='edge')

    def _calculate_hicks(self, coordinates):
        space = self.space

        # Calculate the reference gridpoints and offsets
        grid_coordinates = (coordinates - np.array(space.pml_origin)) / np.array(space.spacing)
        reference_gridpoints = np.round(grid_coordinates).astype(np.int32)
        offsets = grid_coordinates - reference_gridpoints

        # Pre-calculate stuff
        kaiser_b = 4.14
        kaiser_half_width = 3
        kaiser_den = scipy.special.iv(0, kaiser_b)
        kaiser_extended_width = kaiser_half_width/0.99

        # Calculate coefficients
        r = 2*kaiser_half_width+1
        num = coordinates.shape[0]
        coefficients = np.zeros((num, space.dim, r))

        for grid_point in range(-kaiser_half_width, kaiser_half_width+1):
            index = kaiser_half_width + grid_point

            x = grid_point + offsets

            weights = (x / kaiser_extended_width)**2
            weights[weights > 1] = 1
            weights = scipy.special.iv(0, kaiser_b * np.sqrt(1 - weights)) / kaiser_den

            coefficients[:, :, index] = np.sinc(x) * weights

        return reference_gridpoints - kaiser_half_width, coefficients


class OperatorDevito:
    """
    Instances of this class encapsulate Devito operators, how to configure them and how to run them.


    Parameters
    ----------
    grid : GridDevito, optional
        Predefined GridDevito. A new one will be created unless specified.
    """

    def __init__(self, *args, grid=None, name='kernel', **kwargs):
        self.name = name

        self.devito_operator = None
        self.kwargs = {}

        if grid is None:
            self.grid = GridDevito(*args, **kwargs)
        else:
            self.grid = grid

        # fix devito logging
        devito_logger = logging.getLogger('devito')
        devito.logger.logger = devito_logger

        class RerouteFilter(logging.Filter):

            def __init__(self):
                super().__init__()

            def filter(self, record):
                _runtime = mosaic.runtime()

                if record.levelno == devito.logger.PERF:
                    _runtime.logger.info(record.msg)

                elif record.levelno == logging.ERROR:
                    _runtime.logger.error(record.msg)

                elif record.levelno == logging.WARNING:
                    _runtime.logger.warning(record.msg)

                elif record.levelno == logging.DEBUG:
                    _runtime.logger.debug(record.msg)

                else:
                    _runtime.logger.info(record.msg)

                return False

        devito_logger.addFilter(RerouteFilter())

        runtime = mosaic.runtime()
        if runtime.mode == 'local':
            devito_logger.propagate = False

        # global devito config
        default_config = {
            'autotuning': ['aggressive', 'runtime'],
            'develop-mode': False,
            'mpi': False,
            'log-level': 'DEBUG',
        }

        compiler = os.getenv('DEVITO_COMPILER', None)
        if compiler is not None:
            default_config['compiler'] = compiler

        devito_config = kwargs.pop('devito_config', {})
        default_config.update(devito_config)

        runtime = mosaic.runtime()
        runtime.logger.info('Operator `%s` default configuration:' % self.name)

        for key, value in default_config.items():
            runtime.logger.info('\t * %s=%s' % (key, value))

            devito.parameters.configuration[key] = value

    def set_operator(self, op, **kwargs):
        """
        Set up a Devito operator from a list of operations.

        Parameters
        ----------
        op : list
            List of operations to be given to the devito.Operator instance.
        name : str
            Name to give to the operator, defaults to ``kernel``.
        kwargs : optional
            Configuration parameters to set for Devito overriding defaults.

        Returns
        -------

        """

        default_config = {
            'name': self.name,
            'subs': self.grid.devito_grid.spacing_map,
            'opt': 'advanced',
            'platform': os.getenv('DEVITO_PLATFORM', None),
            'language': os.getenv('DEVITO_LANGUAGE', 'openmp'),
            'compiler': os.getenv('DEVITO_COMPILER', None),
        }

        devito_config = kwargs.pop('op_config', {})
        default_config.update(devito_config)

        runtime = mosaic.runtime()
        runtime.logger.info('Operator `%s` instance configuration:' % self.name)

        for key, value in default_config.items():
            if key == 'name':
                continue

            runtime.logger.info('\t * %s=%s' % (key, value))

        self.devito_operator = devito.Operator(op, **default_config)

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
        time = self.grid.time

        kwargs['time_m'] = kwargs.get('time_m', 0)
        kwargs['time_M'] = kwargs.get('time_M', time.extended_num - 1)

        self.devito_operator.apply(**kwargs)
