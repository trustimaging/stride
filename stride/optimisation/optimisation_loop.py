
from collections import OrderedDict

from ..problem.base import Saved
from .loss.functional import FunctionalValue


__all__ = ['Iteration', 'Block', 'OptimisationLoop']


class Iteration:
    """
    Objects of this class contain information about the iteration, such as the value of the functional.

    Parameters
    ----------
    id : int
        Numerical ID of the iteration.
    abs_id : int
        Numerical ID of the iteration in absolute, global terms.
    block : Block
        Block to which the iteration belongs.
    opt_loop : OptimisationLoop
        Loop to which the iteration belongs.

    """

    def __init__(self, id, abs_id, block, opt_loop):
        self.id = id
        self.abs_id = abs_id

        self._block = block
        self._optimisation_loop = opt_loop
        self._fun = OrderedDict()
        self._step_fun = OrderedDict()

    @property
    def fun_value(self):
        """
        Functional value for this iteration across all shots.

        """
        return sum([each.fun_value for each in self._fun.values()])

    @property
    def step_fun_value(self):
        """
        Functional value for this iteration across all shots
        after step.

        """
        return sum([each.fun_value for each in self._step_fun.values()])

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

    def add_step_fun(self, fun):
        """
        Add a functional value, after step, for a particular shot to the iteration.

        Parameters
        ----------
        fun : FunctionalValue

        Returns
        -------

        """
        self._step_fun[fun.shot_id] = fun

    def __get_desc__(self, **kwargs):
        description = {
            'id': self.id,
            'abs_id': self.abs_id,
            'functional_values': [],
        }

        for fun in self._fun.values():
            description['functional_values'].append({
                'shot_id': fun.shot_id,
                'fun_value': fun.fun_value,
            })

        return description

    def __set_desc__(self, description):
        self.id = description.id
        self.abs_id = description.abs_id

        for fun_desc in description.functional_values:
            fun = FunctionalValue(fun_desc.fun_value, fun_desc.shot_id)
            self._fun[fun.shot_id] = fun


class Block:
    """
    A block determines a set of conditions that is maintained over a number of iterations,
    such as the frequency band used or the step size applied.

    The iteration loop of the block can be started using the generator ``Block.iterations`` as:

    >>> for iteration in block.iterations(num_iterations, *iterators):
    >>>     pass

    Parameters
    ----------
    id : int
        Numerical ID of the block.
    opt_loop : OptimisationLoop
        Loop to which the block belongs.

    """

    def __init__(self, id, opt_loop, **kwargs):
        self.id = id
        self._optimisation_loop = opt_loop

        self._num_iterations = None
        self._iterations = OrderedDict()
        self._current_iteration = None
        self.restart = False

    @property
    def num_iterations(self):
        """
        Number of iterations in the block.

        """
        return self._num_iterations

    @property
    def current_iteration(self):
        """
        Get current active iteration.

        """
        return self._current_iteration

    @property
    def fun_value(self):
        """
        Functional value for this block across all iterations.

        """
        return sum([each.fun_value for each in self._iterations.values()])

    def clear(self):
        """
        Clear the block.

        Returns
        -------

        """
        self._num_iterations = None
        self._iterations = OrderedDict()
        self._current_iteration = None

    def iterations(self, num, *iters, restart=None, restart_id=-1):
        """
        Generator of iterations.

        Parameters
        ----------
        num : int
            Number of iterations to generate.
        iters : tuple, optional
            Any other iterables to zip with the iterations.
        restart : int or bool, optional
            Whether or not attempt to restart the loop from a previous
            iteration. Defaults to the value given to the loop.
        restart_id : int, optional
            If an integer greater than zero, it will restart from
            a specific iteration. Otherwise, it will restart from the latest
            available iteration.

        Returns
        -------
        iterable
            Iteration iterables.

        """
        loop_restart = self._optimisation_loop.restart
        restart = loop_restart if restart is None else restart
        self.restart = restart

        if restart is False:
            self.clear()
        else:
            if type(restart_id) is int and restart_id < 0:
                iteration = Iteration(self._current_iteration.id+1, self._optimisation_loop.running_id,
                                      self, self._optimisation_loop)

                self._iterations[self._current_iteration.id+1] = iteration
                self._optimisation_loop.running_id += 1
                self._current_iteration = iteration

            elif type(restart_id) is int and restart_id >= 0:
                if restart_id not in self._iterations:
                    raise ValueError('Iteration %d does not exist, so loop cannot be '
                                     'restarted from that point' % restart_id)

                self._current_iteration = self._iterations[restart_id]

                for index in range(restart_id+1, self._num_iterations):
                    if index in self._iterations:
                        del self._iterations[index]

            if self._current_iteration is not None:
                self._optimisation_loop.running_id = self._current_iteration.abs_id+1

        if self._num_iterations is None:
            self._num_iterations = num

        for zipped in zip(range(self._num_iterations), *iters):
            index = zipped[0]

            if self._current_iteration is not None \
                    and index < self._current_iteration.id:
                continue

            if index not in self._iterations:
                iteration = Iteration(index, self._optimisation_loop.running_id,
                                      self, self._optimisation_loop)

                self._iterations[index] = iteration
                self._optimisation_loop.running_id += 1

            self._current_iteration = self._iterations[index]

            if len(zipped) > 1:
                yield (self._iterations[index],) + zipped[1:]
            else:
                yield self._iterations[index]

            self._optimisation_loop.started = True
            self._optimisation_loop.dump()

    def __get_desc__(self, **kwargs):
        description = {
            'id': self.id,
            'num_iterations': self._num_iterations,
            'current_iteration': self._current_iteration.__get_desc__(),
            'iterations': [],
        }

        for iteration in self._iterations.values():
            description['iterations'].append(iteration.__get_desc__())

        return description

    def __set_desc__(self, description):
        self.id = description.id
        self._num_iterations = description.num_iterations

        for iter_desc in description.iterations:
            iteration = Iteration(iter_desc.id, iter_desc.abs_id,
                                  self, self._optimisation_loop)
            iteration.__set_desc__(iter_desc)
            self._iterations[iteration.id] = iteration

        self._current_iteration = self._iterations[description.current_iteration.id]


class OptimisationLoop(Saved):
    """
    Objects of this class act as managers of a local optimisation process.

    The general convention is to divide the optimisation process in blocks and iterations,
    although that doesn't have to be the case. A block determines a set of conditions
    that is maintained over a number of iterations, such as the frequency band used or the
    step size applied.

    Blocks are generated through ``Optimisation.blocks``:

    >>> for block in optimisation.blocks(num_blocks, *iterators):
    >>>     block.config(...)
    >>>

    The default running behaviour of the optimisation is obtained when calling ``Optimisation.run(block, problem)``:

    >>> for block in optimisation.blocks(num_blocks, *iterators):
    >>>     block.config(...)
    >>>     await optimisation.run(block, problem)

    but iterations can also be run manually:

    >>> for block in optimisation.blocks(num_blocks, *iterators):
    >>>     block.config(...)
    >>>
    >>>     for iteration in block.iterations(num_iterations, *iterators):
    >>>         pass

    Parameters
    ----------
    name : str, optional
        Optional name for the optimisation object.

    """

    def __init__(self, name='optimisation_loop', **kwargs):
        super().__init__(name=name, **kwargs)

        self._num_blocks = None
        self._blocks = OrderedDict()
        self._current_block = None
        self.restart = False
        self.running_id = 0
        self.started = False

        self._problem = kwargs.pop('problem', None)
        self._file_kwargs = {}

    @property
    def num_blocks(self):
        """
        Get number of blocks.

        """
        return self._num_blocks

    @property
    def current_block(self):
        """
        Get current active block.

        """
        return self._current_block

    @property
    def problem(self):
        """
        Access problem object.

        """
        return self._problem

    def clear(self):
        """
        Clear the loop.

        Returns
        -------

        """
        self._num_blocks = None
        self._blocks = OrderedDict()
        self._current_block = None
        self.running_id = 0

    def blocks(self, num, *iters, restart=False, restart_id=-1, **kwargs):
        """
        Generator for the blocks of the optimisation.

        Parameters
        ----------
        num : int
            Number of blocks to generate.
        iters : tuple, optional
            Any other iterables to zip with the blocks.
        restart : int or bool, optional
            Whether or not attempt to restart the loop from a previous
            block. Defaults to ``False``.
        restart_id : int, optional
            If an integer greater than zero, it will restart from
            a specific block. Otherwise, it will restart from the latest
            available block.

        Returns
        -------
        iterable
            Blocks iterable.

        """
        self.restart = restart

        if self.restart is False:
            self.clear()
        else:
            self._file_kwargs = kwargs
            try:
                load_kwargs = dict(path=self.problem.output_folder,
                                   project_name=self.problem.name, version=0)
                load_kwargs.update(kwargs)

                self.load(**load_kwargs)

                if type(restart_id) is int and restart_id >= 0:
                    if restart_id not in self._blocks:
                        raise ValueError('Block %d does not exist, so loop cannot be '
                                         'restarted from that point' % restart_id)

                    self._current_block = self._blocks[restart_id]
                    last_iter = self._current_block._iterations[self._current_block.num_iterations - 1]
                    self.running_id = last_iter.abs_id

                    if restart_id-1 in self._blocks:
                        prev_block = self._blocks[restart_id-1]
                        last_iter = prev_block._iterations[prev_block.num_iterations-1]
                        self.running_id = last_iter.abs_id

                    for index in range(restart_id+1, self._num_blocks):
                        if index in self._blocks:
                            del self._blocks[index]

            except (OSError, AttributeError):
                pass

        if self._num_blocks is None:
            self._num_blocks = num

        for zipped in zip(range(num), *iters):
            index = zipped[0]

            if self._current_block is not None \
                    and index < self._current_block.id:
                continue

            if index not in self._blocks:
                block = Block(index, self)
                self._blocks[index] = block

            self._current_block = self._blocks[index]

            if len(zipped) > 1:
                yield (self._blocks[index],) + zipped[1:]
            else:
                yield self._blocks[index]

            self.restart = False

    def dump(self, *args, **kwargs):
        """
        Dump latest version of the loop to a file.

        See :class:`~stride.problem.base.Saved` for more information on the parameters of this method.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        try:
            dump_kwargs = dict(path=self.problem.output_folder,
                               project_name=self.problem.name, version=0)
            dump_kwargs.update(self._file_kwargs)

            super().dump(*args, **dump_kwargs)
        except AttributeError:
            pass

    def __get_desc__(self, **kwargs):
        description = {
            'running_id': self.running_id,
            'num_blocks': self._num_blocks,
            'current_block': self._current_block.__get_desc__(),
            'blocks': [],
        }

        for block in self._blocks.values():
            description['blocks'].append(block.__get_desc__())

        return description

    def __set_desc__(self, description):
        self.running_id = description.running_id
        self._num_blocks = description.num_blocks

        for block_desc in description.blocks:
            block = Block(block_desc.id, self)
            block.__set_desc__(block_desc)
            self._blocks[block.id] = block

        self._current_block = self._blocks[description.current_block.id]
