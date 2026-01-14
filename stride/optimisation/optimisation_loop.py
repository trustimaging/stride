
import numpy as np

import mosaic.types
from mosaic.file_manipulation import h5

from ..problem.base import Saved
from .loss.functional import FunctionalValue


__all__ = ['Iteration', 'Block', 'OptimisationLoop']


class IterationRun:
    """
    Objects of this class contain information about a specific repetition of the iteration.

    """

    def __init__(self, id, iteration):
        self.id = id

        self._iteration = iteration
        self.submitted_shots = []
        self.completed_shots = []
        self.losses = dict()

    @property
    def iteration(self):
        """
        Corresponding iteration.

        """
        return self._iteration

    @property
    def total_loss(self):
        """
        Functional value for this iteration across all shots.

        """
        return sum([each.value for each in self.losses.values()])

    def clear(self):
        """
        Clear memory.

        Returns
        -------

        """
        self.submitted_shots = []
        self.completed_shots = []
        for shot_id, loss in self.losses.items():
            self.losses[shot_id] = FunctionalValue(loss.value, shot_id)

    def __get_desc__(self, **kwargs):
        legacy = kwargs.get('legacy', False)

        if legacy:
            description = {
                'id': self.id,
                'submitted_shots': self.submitted_shots,
                'completed_shots': self.completed_shots,
                'losses': [],
            }

            for loss in self.losses.values():
                description['losses'].append({
                    'shot_id': loss.shot_id,
                    'value': loss.value,
                })

        else:
            shot_ids = []
            values = []
            for loss in self.losses.values():
                shot_ids.append(loss.shot_id)
                values.append(loss.value)

            description = {
                'id': self.id,
                'submitted_shots': np.asarray(self.submitted_shots, dtype=np.int32),
                'completed_shots': np.asarray(self.completed_shots, dtype=np.int32),
                'losses': {
                    'shot_id': np.asarray(shot_ids, dtype=np.int32),
                    'value': np.asarray(values, dtype=np.float32),
                },
            }

        return description

    def __set_desc__(self, description, **kwargs):
        self.id = description.id
        self.submitted_shots = description.submitted_shots.load().tolist()
        self.completed_shots = description.completed_shots.load().tolist()

        if isinstance(description.losses, mosaic.types.Struct):
            for idx in range(len(description.losses.shot_id)):
                loss = FunctionalValue(description.losses.value[idx], description.losses.shot_id[idx])
                self.losses[loss.shot_id] = loss
        else:
            for loss_desc in description.losses:
                loss = FunctionalValue(loss_desc.value, loss_desc.shot_id)
                self.losses[loss.shot_id] = loss


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
        self._runs = {
            0: IterationRun(0, self),
        }
        self._curr_run_idx = 0

    @property
    def curr_run(self):
        return self._runs[self._curr_run_idx]

    @property
    def prev_run(self):
        if self._curr_run_idx <= 0:
            return None
        return self._runs[self._curr_run_idx-1]

    @property
    def block(self):
        """
        Block of the iteration.

        """
        return self._block

    @property
    def block_id(self):
        """
        ID of the iteration block.

        """
        return self._block.id

    @property
    def loop(self):
        """
        Optimisation loop.

        """
        return self._optimisation_loop

    @property
    def total_loss(self):
        """
        Functional value for this iteration across all shots.

        """
        return self.curr_run.total_loss

    @property
    def total_loss_change(self):
        """
        Functional value change between last two runs

        """
        if self._curr_run_idx <= 0:
            return 0.

        curr_loss = self.curr_run.total_loss
        prev_loss = self.prev_run.total_loss

        return (curr_loss - prev_loss) / (prev_loss + 1e-6)

    @property
    def num_submitted(self):
        """
        Number of shots submitted in this iteration.

        """
        return len(self.curr_run.submitted_shots)

    @property
    def num_completed(self):
        """
        Number of shots completed in this iteration

        """
        return len(self.curr_run.completed_shots)

    def next_run(self):
        """
        Set up next iteration run.

        Returns
        -------

        """
        self._curr_run_idx += 1
        self._runs[self._curr_run_idx] = IterationRun(self._curr_run_idx, self)
        return self.curr_run

    def clear(self):
        """
        Clear iteration.

        Returns
        -------

        """
        self._runs = {
            0: IterationRun(0, self),
        }
        self._curr_run_idx = 0

    def clear_run(self):
        """
        Clear run memory.

        Returns
        -------

        """
        for run in self._runs.values():
            run.clear()

    def add_loss(self, fun):
        """
        Add a functional value for a particular shot to the iteration.

        Parameters
        ----------
        fun : FunctionalValue

        Returns
        -------

        """
        self.curr_run.losses[fun.shot_id] = fun

    def add_submitted(self, shot):
        """
        Add a submitted shot.

        Parameters
        ----------
        shot : Shot

        Returns
        -------

        """
        self.curr_run.submitted_shots.append(shot.id)

    def add_completed(self, shot):
        """
        Add a completed shot.

        Parameters
        ----------
        shot : Shot

        Returns
        -------

        """
        self.curr_run.completed_shots.append(shot.id)

    def append_iteration(self, *args, **kwargs):
        """
        Append the iteration to the corresponding OptimisationLoop file.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        loop = self._optimisation_loop
        block = self._block

        try:
            dump_kwargs = dict(parameter='optimisation_loop',
                               path=loop.problem.output_folder,
                               project_name=loop.problem.name, version=0)
            dump_kwargs.update(loop._file_kwargs)
            dump_kwargs.update(kwargs)
        except AttributeError:
            return

        if h5.file_exists(*args, **dump_kwargs):
            self_desc = self.__get_desc__(**kwargs)

            description = {
                'running_id': loop.running_id,
                'num_blocks': loop.num_blocks,
                'current_block_id': block.id,
                'blocks': {
                    str(block.id): {
                        'id': block.id,
                        'num_iterations': block.num_iterations,
                        'current_iteration_id': self.id,
                        'iterations': {
                            str(self.id): self_desc,
                        },
                    }
                },
            }

            with h5.HDF5(*args, **dump_kwargs, mode='a') as file:
                file.append(description)

        else:
            loop.dump(*args, **dump_kwargs)

    def __get_desc__(self, **kwargs):
        legacy = kwargs.get('legacy', False)

        if legacy:
            description = {
                'id': self.id,
                'abs_id': self.abs_id,
                'runs': [],
            }

            for run in self._runs.values():
                description['runs'].append(run.__get_desc__(**kwargs))

        else:
            description = {
                'id': self.id,
                'abs_id': self.abs_id,
                'runs': {},
            }

            for run in self._runs.values():
                description['runs'][str(run.id)] = run.__get_desc__(**kwargs)

        if kwargs.pop('no_runs', False):
            del description['runs']

        return description

    def __set_desc__(self, description, **kwargs):
        self.id = description.id
        self.abs_id = description.abs_id

        try:
            runs = description.runs
        except AttributeError:
            self.clear()
            return
        if isinstance(runs, mosaic.types.Struct):
            runs = runs.values()

        self._curr_run_idx = -1
        for run_desc in runs:
            self._curr_run_idx += 1
            run = IterationRun(self._curr_run_idx, self)
            run.__set_desc__(run_desc, **kwargs)
            self._runs[self._curr_run_idx] = run

    _serialisation_attrs = ['id', 'abs_id']

    def _serialisation_helper(self):
        state = {}

        for attr in self._serialisation_attrs:
            state[attr] = getattr(self, attr)

        state['_runs'] = dict()

        return state

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = cls.__new__(cls)
        instance._curr_run_idx = 0

        for attr, value in state.items():
            if attr == '_runs':
                setattr(instance, '_runs', {
                    0: IterationRun(0, instance),
                })
            else:
                setattr(instance, attr, value)

        return instance

    def __reduce__(self):
        state = self._serialisation_helper()
        return self._deserialisation_helper, (state,)


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
        self._iterations = dict()
        self._current_iteration = None

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
    def total_loss(self):
        """
        Functional value for this block across all iterations.

        """
        return sum([each.value for each in self._iterations.values()])

    def clear(self):
        """
        Clear the block.

        Returns
        -------

        """
        self._num_iterations = None
        self._iterations = dict()
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
            # dump the empty iteration to keep track of loop
            self._current_iteration.append_iteration(no_runs=True)

            if len(zipped) > 1:
                yield (self._current_iteration,) + zipped[1:]
            else:
                yield self._current_iteration

            self._optimisation_loop.started = True
            # dump completed iteration to enable restart
            self._current_iteration.append_iteration()

    def __get_desc__(self, **kwargs):
        legacy = kwargs.get('legacy', False)

        if legacy:
            description = {
                'id': self.id,
                'num_iterations': self._num_iterations,
                'current_iteration_id': self._current_iteration.id,
                'iterations': [],
            }

            for iteration in self._iterations.values():
                description['iterations'].append(iteration.__get_desc__(**kwargs))

        else:
            description = {
                'id': self.id,
                'num_iterations': self._num_iterations,
                'current_iteration_id': self._current_iteration.id,
                'iterations': {},
            }

            for iteration in self._iterations.values():
                description['iterations'][str(iteration.id)] = iteration.__get_desc__(**kwargs)

        return description

    def __set_desc__(self, description, **kwargs):
        self.id = description.id
        self._num_iterations = description.num_iterations

        iterations = description.iterations
        if isinstance(iterations, mosaic.types.Struct):
            iterations = iterations.values()

        for iter_desc in iterations:
            iteration = Iteration(iter_desc.id, iter_desc.abs_id,
                                  self, self._optimisation_loop)
            iteration.__set_desc__(iter_desc, **kwargs)
            self._iterations[iteration.id] = iteration

        self._current_iteration = self._iterations[description.current_iteration_id]


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
        self._blocks = dict()
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

    def clear(self, **kwargs):
        """
        Clear the loop.

        Returns
        -------

        """
        self._num_blocks = None
        self._blocks = dict()
        self._current_block = None
        self.running_id = 0

        self.remove_file(**kwargs)

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
        restart_id : int or tuple, optional
            If an integer greater than zero, it will restart from
            a specific iteration. If -1, it will restart from the latest
            iteration. If a tuple, it will restart from ``(block_id, iteration_id)``.

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

            except (OSError, AttributeError):
                self.clear()
                self.restart = False

            else:
                # restart with an absolute iteration_id
                if type(restart_id) is int and restart_id >= 0:
                    if restart_id >= self.running_id:
                        raise ValueError('Iteration %d does not exist, so loop cannot be '
                                         'restarted from that point' % restart_id)

                    block = self._current_block
                    iter = list(self._current_block._iterations.values())[-1]

                    found_iter = False
                    for block in self._blocks.values():
                        for iter in block._iterations.values():
                            if iter.abs_id >= restart_id:
                                found_iter = True
                                break
                        if found_iter:
                            break

                    self._current_block = block
                    block._current_iteration = iter
                    self.running_id = iter.abs_id+1

                # restart with a tuple (block_id, iteration_id)
                elif type(restart_id) is tuple:
                    block_id, iteration_id = restart_id

                    # point to requested block
                    if block_id not in self._blocks:
                        raise ValueError('Block %d does not exist, so loop cannot be '
                                         'restarted from that point' % block_id)

                    self._current_block = self._blocks[block_id]
                    block = self._current_block

                    # point to requested iteration
                    if iteration_id < 0:
                        last_iter = list(block._iterations.values())[-1]
                        self.running_id = last_iter.abs_id+1

                    else:
                        if iteration_id not in block._iterations:
                            raise ValueError('Iteration %d does not exist, so loop cannot be '
                                             'restarted from that point' % iteration_id)

                        block._current_iteration = block._iterations[iteration_id]
                        self.running_id = block._current_iteration.abs_id+1

                # access restarted block an iteration
                block = self._current_block
                iter = self._current_block._current_iteration

                # clear iteration
                iter.clear()

                # delete any other blocks
                for index in range(block.id+1, self._num_blocks):
                    if index in self._blocks:
                        del self._blocks[index]

                # delete any other iterations
                for index in range(iter.id+1, block._num_iterations):
                    if index in block._iterations:
                        del block._iterations[index]

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
                yield (self._current_block,) + zipped[1:]
            else:
                yield self._current_block

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
            dump_kwargs.update(kwargs)

            super().dump(*args, **dump_kwargs)
        except AttributeError:
            pass

    def remove_file(self, *args, **kwargs):
        try:
            load_kwargs = dict(parameter='optimisation_loop',
                               path=self.problem.output_folder,
                               project_name=self.problem.name, version=0)
            load_kwargs.update(kwargs)
        except AttributeError:
            return

        try:
            h5.rm(*args, **load_kwargs)
        except FileNotFoundError:
            pass

    def __get_desc__(self, **kwargs):
        legacy = kwargs.get('legacy', False)

        if legacy:
            description = {
                'running_id': self.running_id,
                'num_blocks': self._num_blocks,
                'current_block_id': self._current_block.id,
                'blocks': [],
            }

            for block in self._blocks.values():
                description['blocks'].append(block.__get_desc__(**kwargs))

        else:
            description = {
                'running_id': self.running_id,
                'num_blocks': self._num_blocks,
                'current_block_id': self._current_block.id,
                'blocks': {},
            }

            for block in self._blocks.values():
                description['blocks'][str(block.id)] = block.__get_desc__(**kwargs)

        return description

    def __set_desc__(self, description, **kwargs):
        self.running_id = description.running_id
        self._num_blocks = description.num_blocks

        blocks = description.blocks
        if isinstance(blocks, mosaic.types.Struct):
            blocks = blocks.values()

        for block_desc in blocks:
            block = Block(block_desc.id, self)
            block.__set_desc__(block_desc, **kwargs)
            self._blocks[block.id] = block

        self._current_block = self._blocks[description.current_block_id]
