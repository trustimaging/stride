
from collections import OrderedDict

from stride.problem.base import Saved


__all__ = ['Iteration', 'Block', 'OptimisationLoop']


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

    >>> for iteration in block.iterations(num_iterations, *iterators):
    >>>     pass

    Parameters
    ----------
    id : int
        Numerical ID of the block.

    """

    def __init__(self, id, **kwargs):
        self.id = id

        self._num_iterations = None
        self._iterations = OrderedDict()

    @property
    def num_iterations(self):
        """
        Number of iterations in the block.

        """
        return self._num_iterations

    def iterations(self, num, *iters):
        """
        Generator of iterations.

        Parameters
        ----------
        num : int
            Number of iterations to generate.
        iters : tuple, optional
            Any other iterables to zip with the iterations.

        Returns
        -------
        iterable
            Iteration iterables.

        """
        self._num_iterations = num

        for zipped in zip(range(self._num_iterations), *iters):
            index = zipped[0]

            iteration = Iteration(index)
            self._iterations[index] = iteration

            if len(zipped) > 1:
                yield (iteration,) + zipped[1:]
            else:
                yield iteration


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

    def __init__(self, name='optimisation', **kwargs):
        super().__init__(name=name, **kwargs)

        self._num_blocks = None
        self._blocks = OrderedDict()
        self._current_block = None

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

    def blocks(self, num, *iters):
        """
        Generator for the blocks of the optimisation.

        Parameters
        ----------
        num : int
            Number of blocks to generate.
        iters : tuple, optional
            Any other iterables to zip with the blocks.

        Returns
        -------
        iterable
            Blocks iterable.

        """
        self._num_blocks = num

        for zipped in zip(range(num), *iters):
            index = zipped[0]

            block = Block(index)
            self._blocks[index] = block
            self._current_block = block

            if len(zipped) > 1:
                yield (block,) + zipped[1:]
            else:
                yield block
