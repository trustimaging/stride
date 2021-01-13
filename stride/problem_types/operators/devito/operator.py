
import os
import devito
import numpy as np

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

    def __init__(self, space_order, time_order, grid=None):
        self._problem = None

        self.vars = Struct()

        self.space_order = space_order
        self.time_order = time_order

        self.grid = grid

    # TODO The grid needs to be re-created if the space or time extent has changed
    def set_problem(self, problem):
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
        space_order = space_order or self.space_order

        fun = devito.Function(name=name,
                              grid=self.grid,
                              space_order=space_order,
                              **kwargs)

        return fun

    @_cached
    def time_function(self, name, space_order=None, time_order=None, **kwargs):
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
        time = self._problem.time

        time_under = devito.ConditionalDimension('time_under',
                                                 parent=self.grid.time_dim,
                                                 factor=factor)

        buffer_size = (time.extended_num + factor - 1) // factor

        return self.time_function(name,
                                  space_order=space_order,
                                  time_order=time_order,
                                  time_dim=time_under,
                                  save=buffer_size)

    def with_halo(self, data):
        pad_widths = [[self.space_order, self.space_order]
                      for _ in self._problem.space.shape]

        return np.pad(data, pad_widths, mode='edge')


class OperatorDevito:

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

    def set_problem(self, problem):
        self._problem = problem

    def set_operator(self, op, config=None):
        default_config = {
            'autotuning': ['aggressive', 'runtime'],
            'opt': 'advanced',
            'compiler': os.getenv('DEVITO_COMPILER', 'gcc'),
            'develop-mode': False,
            'language': os.getenv('DEVITO_LANGUAGE', 'openmp'),
            'mpi': False,
            'log-level': 'ERROR',
        }

        config = config or {}
        default_config.update(config)

        for key, value in default_config.items():
            devito.parameters.configuration[key] = value

        devito.parameters.configuration['opt-options']['blockinner'] = True

        self.operator = devito.Operator(op,
                                        subs=self.grid.grid.spacing_map,
                                        platform=os.getenv('DEVITO_PLATFORM', None))

    def compile(self):
        compiler_flags = os.getenv('DEVITO_COMP_FLAGS', '').split(',')
        compiler_flags = [each.strip() for each in compiler_flags]
        self.operator._compiler.cflags += compiler_flags
        self.operator.cfunction

    def arguments(self, **kwargs):
        time = self._problem.time

        kwargs['time_m'] = kwargs.get('time_m', 0)
        kwargs['time_M'] = kwargs.get('time_M', time.extended_num - 1)

        self.kwargs.update(kwargs)

    def run(self):
        self.operator.apply(**self.kwargs)
