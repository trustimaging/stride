
import numpy as np
from collections import OrderedDict

import mosaic
from mosaic.types import Struct
from mosaic.utils import camel_case

from .. import optimisation
from .. import Runner
from .pipelines import default_pipelines
from stride.problem_definition.base import Saved


__all__ = ['Iteration', 'Block', 'Optimisation', 'VariableList', 'CallableList']


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
    """
    Class representing a series of objects that contain common interfaces.

    For example, let's say that we create a certain class a a callable list using it:

    >>> class Klass:
    >>>     def __init__(self, value):
    >>>         self.value = value
    >>>
    >>>     def print(self):
    >>>         print('Value: ', self.value)
    >>>
    >>> callable_list = CallableList([Klass(1), Klass(2)])

    then we can call the common method in all elements of the list by doing:

    >>> callable_list.print()
    Value: 1
    Value: 2

    but also that we can set attributes accordingly:

    >>> callable_list.value = [3, 4]
    >>> callable_list.print()
    Value: 3
    Value: 4

    A callable list can be iterated and ``len(callable_list)`` is also valid.

    The result of an attribute access on a callable list is another callable list,
    ensuring composability of operations.


    Parameters
    ----------
    items : iterable
        Items in the callable list.

    """

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
    """
    A variable list is a specific type of callable list in which the items are a Struct instead of
    a Python list.

    """

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
    """
    Objects of this class contain information about the iteration, such as the value of the functional.

    Parameters
    ----------
    id : int
        Numerical ID of the iteration.

    """

    def __init__(self, id, **kwargs):
        self.id = id

        self._fun = OrderedDict()

    @property
    def fun_value(self):
        """
        Functional value for this iteration across all shots.

        """
        return sum([each.fun_value for each in self._fun.values()])

    def add_fun(self, fun):
        """
        Add a functional value for a particular shot to the iteration.

        Parameters
        ----------
        fun : FunctionalValue

        Returns
        -------

        """
        self._fun[fun.shot_id] = fun


class Block:
    """
    A block determines a set of conditions that is maintained over a number of iterations,
    such as the frequency band used or the step size applied.

    These can be given to the block through the default ``Block.config``, which will take care
    of creating and configuring the ``pipeline`` that implement these conditions.

    Pipelines can be accessed through ``Block.pipelines`` using dot notation.

    The iteration loop of the block can be started using the generator ``Block.iterations`` as:

    >>> for iteration in block.iterations():
    >>>     pass

    Parameters
    ----------
    id : int
        Numerical ID of the block.
    functional : Functional
        Functional class to be used in the inversion.

    """

    def __init__(self, id, **kwargs):
        self.id = id

        self.functional = kwargs.pop('functional', None)
        self.pipelines = Struct()
        self.select_shots = dict()

        self._num_iterations = None
        self._iterations = OrderedDict()

    @property
    def num_iterations(self):
        """
        Number of iterations in the block.

        """
        return self._num_iterations

    def iterations(self):
        """
        Generator of iterations.

        Returns
        -------
        iterable
            Iteration iterables.

        """
        for index in range(self._num_iterations):
            iteration = Iteration(index)
            self._iterations[index] = iteration

            yield iteration

    def config(self, **kwargs):
        """
        Configure the block appropriately.

        Parameters
        ----------
        num_iterations : int, optional
            Number of iterations in the block, defaults to 1.
        select_shots : dict, optional
            Rules to select shots in each iteration, defaults to selecting all.
        wavelets : callable or dict
            Pipeline class to process the wavelets or dictionary with configuration for the default pipeline.
        wavefield : callable or dict
            Pipeline class to process the wavefield or dictionary with configuration for the default pipeline.
        traces : callable or dict
            Pipeline class to process the traces or dictionary with configuration for the default pipeline.
        adjoint_source : callable or dict
            Pipeline class to process the adjoint source or dictionary with configuration for the default pipeline.
        local_gradient : callable or dict
            Pipeline class to process the local gradient or dictionary with configuration for the default pipeline.
        global_gradient : callable or dict
            Pipeline class to process the global gradient or dictionary with configuration for the default pipeline.
        model_iteration : callable or dict
            Pipeline class to process the model after each iteration or dictionary with configuration for the default pipeline.
        model_block : callable or dict
            Pipeline class to process the model after each block or dictionary with configuration for the default pipeline.

        Returns
        -------

        """
        self._num_iterations = kwargs.pop('num_iterations', 1)
        self.select_shots = kwargs.pop('select_shots', {})

        # Process wavelets
        wavelets = kwargs.pop('wavelets', {})
        if isinstance(wavelets, dict):
            self.pipelines.wavelets = default_pipelines.ProcessWavelets(**kwargs, **wavelets)
        else:
            self.pipelines.wavelets = wavelets

        # Process wavefield
        wavefield = kwargs.pop('wavefield', {})
        if isinstance(wavefield, dict):
            self.pipelines.wavefield = default_pipelines.ProcessWavefield(**kwargs, **wavefield)
        else:
            self.pipelines.wavefield = wavefield

        # Process traces
        traces = kwargs.pop('traces', {})
        if isinstance(traces, dict):
            self.pipelines.traces = default_pipelines.ProcessTraces(**kwargs, **traces)
        else:
            self.pipelines.traces = traces

        # Process adjoint source
        adjoint_source = kwargs.pop('adjoint_source', {})
        if isinstance(adjoint_source, dict):
            self.pipelines.adjoint_source = default_pipelines.ProcessAdjointSource(**kwargs, **adjoint_source)
        else:
            self.pipelines.adjoint_source = adjoint_source

        # Process local gradient
        local_gradient = kwargs.pop('local_gradient', {})
        if isinstance(local_gradient, dict):
            self.pipelines.local_gradient = default_pipelines.ProcessLocalGradient(**kwargs, **local_gradient)
        else:
            self.pipelines.local_gradient = local_gradient

        # Process global gradient
        global_gradient = kwargs.pop('global_gradient', {})
        if isinstance(global_gradient, dict):
            self.pipelines.global_gradient = default_pipelines.ProcessGlobalGradient(**kwargs, **global_gradient)
        else:
            self.pipelines.global_gradient = global_gradient

        # Process model iteration
        model_iteration = kwargs.pop('model_iteration', {})
        if isinstance(model_iteration, dict):
            self.pipelines.model_iteration = default_pipelines.ProcessModelIteration(**kwargs, **model_iteration)
        else:
            self.pipelines.model_iteration = model_iteration

        # Process model block
        model_block = kwargs.pop('model_block', {})
        if isinstance(model_block, dict):
            self.pipelines.model_block = default_pipelines.ProcessModelBlock(**kwargs, **model_block)
        else:
            self.pipelines.model_block = model_block


class Optimisation(Saved):
    """
    Objects of this class act as managers of a local optimisation process.

    Optimisations are performed with respect of a given ``functional``
    (e.g. L2-norm of the difference between observed and modelled data), over
    one or more optimisation variables (such as longitudinal speed of sound or attenuation).
    Variables are updated according to an associated local optimiser (such as gradient descent or momentum).

    Variables can be added to the optimisation through ``Optimisation.add(variable, optimiser)``.

    The general convention is to divide the optimisation process in blocks and iterations,
    although that doesn't have to be the case. A block determines a set of conditions
    that is maintained over a number of iterations, such as the frequency band used or the
    step size applied.

    Blocks are generated through ``Optimisation.blocks``:

    >>> for block in optimisation.blocks(num_blocks):
    >>>     block.config(...)
    >>>

    The default running behaviour of the optimisation is obtained when calling ``Optimisation.run(block, problem)``:

    >>> for block in optimisation.blocks(num_blocks):
    >>>     block.config(...)
    >>>     await optimisation.run(block, problem)

    but iterations can also be run manually:

    >>> for block in optimisation.blocks(num_blocks):
    >>>     block.config(...)
    >>>
    >>>     for iteration in block.iterations():
    >>>         pass

    Parameters
    ----------
    name : str, optional
        Optional name for the optimisation object.
    functional : str or object, optional
        Name of the functional to be used, or object defining that functional, defaults to ``l2_norm_difference``.

    """

    def __init__(self, name='optimisation', **kwargs):
        super().__init__(name, **kwargs)

        functional = kwargs.pop('functional', 'l2_norm_difference')

        if isinstance(functional, str):
            functional_module = getattr(optimisation.functionals, functional)
            self.functional = getattr(functional_module, camel_case(functional))()

        else:
            self.functional = functional

        self._num_blocks = None
        self._variables = VariableList()
        self._optimisers = OrderedDict()
        self._blocks = OrderedDict()

    @property
    def num_blocks(self):
        """
        Get number of blocks.

        """
        return self._num_blocks

    @property
    def variables(self):
        """
        Access the variables.

        """
        return self._variables

    @variables.setter
    def variables(self, value):
        """
        Set the variables

        """
        if isinstance(value, list):
            for variable_value in value:
                self._variables.items[variable_value.name] = variable_value

        else:
            self._variables = value

    def add(self, variable, optimiser):
        """
        Add a variable to the optimisation.

        Parameters
        ----------
        variable : Variable
            Variable to add to the optimisation.
        optimiser : LocalOptimiser
            Optimiser associated with the given variable.

        Returns
        -------

        """
        if variable.name not in self._variables:
            self._variables.items[variable.name] = variable
            self._optimisers[variable.name] = optimiser

    def blocks(self, num):
        """
        Generator for the blocks of the optimisation.

        Parameters
        ----------
        num : int
            Number of blocks to generate.

        Returns
        -------
        iterable
            Blocks iterable.

        """
        self._num_blocks = num

        for index in range(num):
            block = Block(index, functional=self.functional)
            self._blocks[index] = block

            yield block

    def apply_optimiser(self, updated_variable, block=None, iteration=None, **kwargs):
        """
        Apply an optimiser to its associated variable.

        Parameters
        ----------
        updated_variable : Variable
            Container for the gradient of the variable.
        block : Block
            Block in which the optimisation is at the moment.
        iteration : Iteration
            Iteration in which the optimisation is at the moment.
        kwargs
            Additional arguments for the optimiser.

        Returns
        -------
        Variable
            Updated variable.

        """
        runtime = mosaic.runtime()

        variable = self._variables[updated_variable.name]
        optimiser = self._optimisers[updated_variable.name]

        grad = updated_variable.get_grad()
        grad = block.pipelines.global_gradient.apply(grad)

        min_grad = np.min(grad.extended_data)
        max_grad = np.max(grad.extended_data)

        min_var = np.min(variable.extended_data)
        max_var = np.max(variable.extended_data)

        runtime.logger.info('Updating variable %s, gradient in range [%e, %e]' %
                            (variable.name, min_grad, max_grad))
        runtime.logger.info('\t variable range before update [%e, %e]' %
                            (min_var, max_var))

        variable = optimiser.apply(grad, iteration=iteration, block=block, **kwargs)
        self._variables.items[updated_variable.name] = variable

        block.pipelines.model_iteration.apply(variable)

        min_var = np.min(variable.extended_data)
        max_var = np.max(variable.extended_data)

        runtime.logger.info('\t variable range after update [%e, %e]' %
                            (min_var, max_var))

        return variable

    def dump_variable(self, updated_variable, problem):
        """
        Dump the updated value of a variable to disk.

        Parameters
        ----------
        updated_variable : Variable
            Container for the gradient of the variable.
        problem : Problem
            Problem being executed.

        Returns
        -------

        """
        variable = self._variables[updated_variable.name]
        variable.dump(path=problem.output_folder,
                      project_name=problem.name)

    async def run(self, block, problem, dump=True):
        """
        Run the default a block with default settings.

        Parameters
        ----------
        block : Block
            Block to run.
        problem : Problem
            Problem being run.
        dump : bool, optional
            Whether or not to dump the updated variable at each iteration, defaults to True.

        Returns
        -------

        """
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
