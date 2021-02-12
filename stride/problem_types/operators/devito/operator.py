
import os
import devito
import functools
import numpy as np

import mosaic
from mosaic.types import Struct


__all__ = ['OperatorDevito', 'GridDevito']


class PhysicalDomain(devito.SubDomain):

    name = 'physical_domain'

    def __init__(self, space_order, extra):
        super(PhysicalDomain, self).__init__()
        self.space_order = space_order
        self.extra = extra

    def define(self, dimensions):
        return {dimension: dimension for dimension in dimensions}


def _cached(func):

    @functools.wraps(func)
    def cached_wrapper(self, *args, **kwargs):
        name = args[0]
        cached = kwargs.pop('cached', True)

        if cached is True:
            fun = self.vars.get(name, None)
            if fun is not None:
                return fun

        fun = func(self, *args, **kwargs)

        self.vars[name] = fun

        return fun

    return cached_wrapper


class GridDevito:
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
    grid : devito.Grid, optional
        Predefined Devito grid. A new one will be created unless specified.

    """

    def __init__(self, space_order, time_order, grid=None):
        self._problem = None

        self.vars = Struct()

        self.space_order = space_order
        self.time_order = time_order

        self.grid = grid

    # TODO The grid needs to be re-created if the space or time extent has changed
    def set_problem(self, problem):
        """
        Set up the problem or sub-problem that will be run on this grid.

        Parameters
        ----------
        problem : SubProblem or Problem
            Problem on which the physics will be executed

        Returns
        -------

        """
        self._problem = problem

        if self.grid is None:
            space = problem.space

            extended_extent = tuple(np.array(space.spacing) * (np.array(space.extended_shape) - 1))
            physical_domain = PhysicalDomain(self.space_order, space.extra)
            self.grid = devito.Grid(extent=extended_extent,
                                    shape=space.extended_shape,
                                    origin=space.pml_origin,
                                    subdomains=physical_domain,
                                    dtype=np.float32)

    @_cached
    def sparse_time_function(self, name, num=1, space_order=None, time_order=None, **kwargs):
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
        kwargs
            Additional arguments for the Devito constructor.

        Returns
        -------
        devito.SparseTimeFunction
            Generated function.

        """
        time = self._problem.time

        space_order = space_order or self.space_order
        time_order = time_order or self.time_order

        # Define variables
        p_dim = devito.Dimension(name='p_%s' % name)
        fun = devito.SparseTimeFunction(name=name,
                                        grid=self.grid,
                                        dimensions=(self.grid.time_dim, p_dim),
                                        npoint=num,
                                        nt=time.extended_num,
                                        space_order=space_order,
                                        time_order=time_order,
                                        dtype=np.float32,
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
        space_order = space_order or self.space_order

        fun = devito.Function(name=name,
                              grid=self.grid,
                              space_order=space_order,
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
                                  grid=self.grid,
                                  time_order=time_order,
                                  space_order=space_order,
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
        time = self._problem.time

        time_under = devito.ConditionalDimension('time_under',
                                                 parent=self.grid.time_dim,
                                                 factor=factor)

        buffer_size = (time.extended_num + factor - 1) // factor

        return self.time_function(name,
                                  space_order=space_order,
                                  time_order=time_order,
                                  time_dim=time_under,
                                  save=buffer_size,
                                  **kwargs)

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
                      for _ in self._problem.space.shape]

        return np.pad(data, pad_widths, mode='edge')


class OperatorDevito:
    """
    Instances of this class encapsulate Devito operators, how to configure them and how to run them.


    Parameters
    ----------
    space_order : int
        Default space order of the discretisation for functions of the grid.
    time_order : int
        Default time order of the discretisation for functions of the grid.
    grid : GridDevito, optional
        Predefined GridDevito. A new one will be created unless specified.
    """

    def __init__(self, space_order, time_order, grid=None):
        self._problem = None

        self.operator = None
        self.kwargs = {}

        self.space_order = space_order
        self.time_order = time_order

        if grid is None:
            self.grid = GridDevito(space_order, time_order)
        else:
            self.grid = grid

        # TODO In local mode, devito PERF is logged as ERROR
        # devito.logger.PERF = 21
        # devito.logger.logger_registry['PERF'] = 21
        #
        # import logging
        # logging.addLevelName(21, "PERF")

        runtime = mosaic.runtime()
        if runtime.mode == 'local':
            devito.logger.logger.propagate = False

    def set_problem(self, problem):
        """
        Set up the problem or sub-problem that will be run with this operator.

        Parameters
        ----------
        problem : SubProblem or Problem
            Problem on which the physics will be executed

        Returns
        -------

        """
        self._problem = problem

    def set_operator(self, op, name='kernel', **kwargs):
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
            'autotuning': ['aggressive', 'runtime'],
            'develop-mode': False,
            'mpi': False,
            'log-level': 'DEBUG',
        }

        for key, value in default_config.items():
            if key in kwargs:
                value = kwargs[key]
                default_config[key] = value
                del kwargs[key]

            devito.parameters.configuration[key] = value

        default_kwargs = {
            'name': name,
            'subs': self.grid.grid.spacing_map,
            'opt': 'advanced',
            'platform': os.getenv('DEVITO_PLATFORM', None),
            'language': os.getenv('DEVITO_LANGUAGE', 'openmp'),
            'compiler': os.getenv('DEVITO_COMPILER', None),
        }

        default_kwargs.update(kwargs)

        runtime = mosaic.runtime()
        runtime.logger.info('Operator `%s` configuration:' % name)

        for key, value in default_config.items():
            runtime.logger.info('\t * %s=%s' % (key, value))

        for key, value in default_kwargs.items():
            if key == 'name':
                continue

            runtime.logger.info('\t * %s=%s' % (key, value))

        self.operator = devito.Operator(op, **default_kwargs)

    def compile(self):
        """
        Compile the operator.

        Returns
        -------

        """
        # compiler_flags = os.getenv('DEVITO_COMP_FLAGS', '').split(',')
        # compiler_flags = [each.strip() for each in compiler_flags]
        # self.operator._compiler.cflags += compiler_flags
        self.operator.cfunction

    def arguments(self, **kwargs):
        """
        Prepare Devito arguments.

        Parameters
        ----------
        kwargs : optional
            Arguments to pass to Devito.

        Returns
        -------

        """
        time = self._problem.time

        kwargs['time_m'] = kwargs.get('time_m', 0)
        kwargs['time_M'] = kwargs.get('time_M', time.extended_num - 1)

        self.kwargs.update(kwargs)

    def run(self):
        """
        Run the operator.

        Returns
        -------

        """
        self.operator.apply(**self.kwargs)
