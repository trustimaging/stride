
import numpy as np
from collections import OrderedDict

import mosaic
from mosaic.types import Struct

from .. import optimisation
from .. import Runner
from .pipelines import default_pipelines
from stride.problem_definition.base import Saved


__all__ = ['Iteration', 'Block', 'Optimisation']


_magic_ops = [
    '__add__',
    '__sub__',
    '__mul__',
    '__pow__',
    '__truediv__',
    '__floordiv__',
    '__iadd__',
    '__isub__',
    '__imul__',
    '__ipow__',
    '__itruediv__',
    '__ifloordiv__',
    '__radd__',
    '__rsub__',
    '__rmul__',
    '__rtruediv__',
    '__rfloordiv__',
]


class CallableList:

    def __init__(self, items=None):
        self.items = items

    def __contains__(self, item):
        return item in self.items

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def __getattribute__(self, item):
        try:
            if item in _magic_ops:
                raise AttributeError('Magic method')

            return super().__getattribute__(item)

        except AttributeError:

            first_variable = next(self.__iter__())

            if not hasattr(first_variable, item):
                raise AttributeError('Class %s does not have method %s' %
                                     (first_variable.__class__.__name__, item))

            if not callable(getattr(first_variable, item)):
                result_list = []
                for variable in self:
                    result_list.append(getattr(variable, item))

                return CallableList(result_list)

            else:
                def list_method(*args, **kwargs):
                    arg_list = [[]] * len(self)
                    kwarg_list = [{}] * len(self)

                    for arg in args:
                        if isinstance(arg, (list, CallableList)) and len(arg) == len(self):
                            for index in range(len(self)):
                                arg_list[index].append(arg[index])
                        else:
                            for index in range(len(self)):
                                arg_list[index].append(arg)

                    for key, arg in kwargs.items():
                        if isinstance(arg, (list, CallableList)) and len(arg) == len(self):
                            for index in range(len(self)):
                                kwarg_list[index][key] = arg[index]
                        else:
                            for index in range(len(self)):
                                kwarg_list[index][key] = arg

                    result_list = []
                    for index, elem in zip(range(len(self)), self):
                        method = getattr(elem, item)
                        result_list.append(method(*arg_list[index], **kwarg_list[index]))

                    return result_list

                return list_method

    def __setattr__(self, key, value):
        if key == 'items':
            super().__setattr__(key, value)

        else:
            value_list = []

            if isinstance(value, (list, CallableList)) and len(value) == len(self):
                for index in range(len(self)):
                    value_list.append(value[index])

            else:
                for index in range(len(self)):
                    value_list.append(value)

            for index, elem in zip(range(len(self)), self):
                setattr(elem, key, value_list[index])

    def __setstate__(self, state):
        self.items = state['items']

    @staticmethod
    def magic_op(item):
        def magic_wrap(self, *args, **kwargs):
            return self.__getattribute__(item)(*args, **kwargs)

        return magic_wrap


for op in _magic_ops:
    setattr(CallableList, op, CallableList.magic_op(op))


class VariableList(CallableList):

    def __init__(self):
        super().__init__()

        self.items = Struct()

    def __len__(self):
        return len(self.items.keys())

    def __iter__(self):
        return iter(list(self.items.values()))

    def __getitem__(self, item):
        try:
            return self.items[item]

        except AttributeError:
            return list(self.items.values())[item]


class Iteration:

    def __init__(self, id, **kwargs):
        self.id = id

        self._fun = OrderedDict()

    @property
    def fun_value(self):
        return sum([each.fun_value for each in self._fun.values()])

    def add_fun(self, fun):
        self._fun[fun.shot_id] = fun


class Block:

    def __init__(self, id, **kwargs):
        self.id = id

        self.functional = kwargs.pop('functional', None)
        self.pipelines = Struct()
        self.select_shots = dict()

        self._num_iterations = None
        self._iterations = OrderedDict()

    @property
    def num_iterations(self):
        return self._num_iterations

    def iterations(self):
        for index in range(self._num_iterations):
            iteration = Iteration(index)
            self._iterations[index] = iteration

            yield iteration

    def config(self, **kwargs):
        self._num_iterations = kwargs.pop('num_iterations', 1)
        self.select_shots = kwargs.pop('select_shots', {})

        # Process wavelets
        wavelets = kwargs.pop('wavelets', {})
        if callable(wavelets):
            self.pipelines.wavelets = wavelets(**kwargs)
        else:
            self.pipelines.wavelets = default_pipelines.ProcessWavelets(**kwargs, **wavelets)

        # Process wavefield
        wavefield = kwargs.pop('wavefield', {})
        if callable(wavefield):
            self.pipelines.wavefield = wavefield(**kwargs)
        else:
            self.pipelines.wavefield = default_pipelines.ProcessWavefield(**kwargs, **wavefield)

        # Process traces
        traces = kwargs.pop('traces', {})
        if callable(traces):
            self.pipelines.traces = traces(**kwargs)
        else:
            self.pipelines.traces = default_pipelines.ProcessTraces(**kwargs, **traces)

        # Process adjoint source
        adjoint_source = kwargs.pop('adjoint_source', {})
        if callable(adjoint_source):
            self.pipelines.adjoint_source = adjoint_source(**kwargs)
        else:
            self.pipelines.adjoint_source = default_pipelines.ProcessAdjointSource(**kwargs, **adjoint_source)

        # Process local gradient
        local_gradient = kwargs.pop('local_gradient', {})
        if callable(local_gradient):
            self.pipelines.local_gradient = local_gradient(**kwargs)
        else:
            self.pipelines.local_gradient = default_pipelines.ProcessLocalGradient(**kwargs, **local_gradient)

        # Process global gradient
        global_gradient = kwargs.pop('global_gradient', {})
        if callable(global_gradient):
            self.pipelines.global_gradient = global_gradient(**kwargs)
        else:
            self.pipelines.global_gradient = default_pipelines.ProcessGlobalGradient(**kwargs, **global_gradient)

        # Process model iteration
        model_iteration = kwargs.pop('model_iteration', {})
        if callable(model_iteration):
            self.pipelines.model_iteration = model_iteration(**kwargs)
        else:
            self.pipelines.model_iteration = default_pipelines.ProcessModelIteration(**kwargs, **model_iteration)

        # Process model block
        model_block = kwargs.pop('model_block', {})
        if callable(model_block):
            self.pipelines.model_block = model_block(**kwargs)
        else:
            self.pipelines.model_block = default_pipelines.ProcessModelBlock(**kwargs, **model_block)


class Optimisation(Saved):

    def __init__(self, name='optimisation', **kwargs):
        super().__init__(name, **kwargs)

        functional = kwargs.pop('functional', 'l2_norm_difference')

        if callable(functional):
            self.functional = functional

        else:
            functional_module = getattr(optimisation.functionals, functional)
            self.functional = functional_module.Functional

        self._num_blocks = None
        self._variables = VariableList()
        self._optimisers = OrderedDict()
        self._blocks = OrderedDict()

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        if isinstance(value, list):
            for variable_value in value:
                self._variables.items[variable_value.name] = variable_value

        else:
            self._variables = value

    def add(self, variable, optimiser):
        if variable.name not in self._variables:
            self._variables.items[variable.name] = variable
            self._optimisers[variable.name] = optimiser

    def blocks(self, num):
        self._num_blocks = num

        for index in range(num):
            block = Block(index, functional=self.functional)
            self._blocks[index] = block

            yield block

    def apply_optimiser(self, updated_variable, block=None, iteration=None, **kwargs):
        variable = self._variables[updated_variable.name]
        optimiser = self._optimisers[updated_variable.name]

        grad = updated_variable.get_grad()
        grad = block.pipelines.global_gradient.apply(grad)

        min_grad = np.min(grad.extended_data)
        max_grad = np.max(grad.extended_data)

        min_var = np.min(variable.extended_data)
        max_var = np.max(variable.extended_data)

        print('Updating variable %s, gradient in range [%e, %e]' %
              (variable.name, min_grad, max_grad))
        print('\t variable range before update [%e, %e]' %
              (min_var, max_var))

        variable = optimiser.apply(grad, iteration=iteration, block=block, **kwargs)
        self._variables.items[updated_variable.name] = variable

        block.pipelines.model_iteration.apply(variable)

        min_var = np.min(variable.extended_data)
        max_var = np.max(variable.extended_data)

        print('\t variable range after update [%e, %e]' %
              (min_var, max_var))

        return variable

    def dump_variable(self, updated_variable, problem):
        variable = self._variables[updated_variable.name]
        variable.dump(path=problem.output_folder,
                      project_name=problem.name)

    async def run(self, block, problem, dump=True):
        runtime = mosaic.runtime()

        # Create an array of runners
        runners = await Runner.remote(len=runtime.num_workers)

        tasks = await runners.set_block(block)
        await mosaic.gather(tasks)

        for iteration in block.iterations():
            runtime.logger.info('Starting iteration %d (out of %d), '
                                'block %d (out of %d)' %
                                (iteration.id, block.num_iterations,
                                 block.id, self.num_blocks))

            fun, updated_variables = await problem.inverse(runners, self.variables,
                                                           needs_grad=True,
                                                           block=block, iteration=iteration)

            for updated_variable in updated_variables:
                self.apply_optimiser(updated_variable,
                                     block=block, iteration=iteration)

                if dump is True:
                    self.dump_variable(updated_variable,
                                       problem=problem)

            runtime.logger.info('Done iteration %d (out of %d), '
                                'block %d (out of %d) - Total loss %e' %
                                (iteration.id, block.num_iterations, block.id,
                                 self.num_blocks, iteration.fun_value))
            runtime.logger.info('====================================================================')

        for variable in self.variables:
            block.pipelines.model_block.apply(variable)
