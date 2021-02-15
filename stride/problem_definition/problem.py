
import os
import numpy as np

import mosaic

from .base import Gridded
from . import Medium, Transducers, Geometry, Acquisitions
from .. import Runner
from .. import problem_types
from .. import plotting


__all__ = ['Problem', 'SubProblem']


class Problem(Gridded):
    """
    The Problem is the object that fully defines the setting in which Stride works.

    The problem defines a medium with a set of fields (such as Vp or density), some
    transducers (such as a series of scalar point transducers), a geometry where those
    transducers are located in space, and the acquisitions that happen given that geometry.

    The problem also defines a problem type, which determines the physics of interest, such
    as the second-order isotropic acoustic wave equation. And a numerical implementation
    of those physics, such as through the finite-difference library Devito.

    Parameters
    ----------
    name : str
        Name of the problem.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined
    input_folder : str, optional
        Default folder from which files should be read, defaults to current working directory.
    output_folder : str, optional
        Default folder to which files should be written, defaults to current working directory.
    medium : Medium, optional
        Predefined Medium of the problem.
    transducers : Transducers, optional
        Predefined Transducers of the problem.
    geometry : Geometry, optional
        Predefined Geometry of the problem.
    acquisitions : Acquisitions, optional
        Predefined Acquisitions of the problem.
    problem_type : str or callable, optional
        Problem type that will be executed on this Problem, defaults to ``acoustic``.
    problem_implementation : str or callable, optional
        Implementation of the problem type that will be executed on this Problem, defaults to ``devito``.

    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.input_folder = kwargs.pop('input_folder', os.getcwd())
        self.output_folder = kwargs.pop('output_folder', os.getcwd())

        self.problem_config = {}

        medium = kwargs.pop('medium', None)
        if medium is None:
            medium = Medium(problem=self)

        self.medium = medium

        transducers = kwargs.pop('transducers', None)
        if transducers is None:
            transducers = Transducers(problem=self)

        self.transducers = transducers

        geometry = kwargs.pop('geometry', None)
        if geometry is None:
            geometry = Geometry(transducers=transducers, problem=self)

        self.geometry = geometry

        acquisitions = kwargs.pop('acquisitions', None)
        if acquisitions is None:
            acquisitions = Acquisitions(geometry=geometry, problem=self)

        self.acquisitions = acquisitions

        problem_type = kwargs.pop('problem_type', 'acoustic')
        problem_implementation = kwargs.pop('problem_implementation', 'devito')

        if callable(problem_type):
            self.problem_type = problem_type

        else:
            problem_module = getattr(problem_types, problem_type)
            problem_module = getattr(problem_module, problem_implementation)
            self.problem_type = problem_module.problem_type.ProblemType

    def load(self, *args, **kwargs):
        """
        Load all elements in the Problem.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        self.medium.load(*args, **kwargs)
        self.transducers.load(*args, **kwargs)
        self.geometry.load(*args, **kwargs)
        self.acquisitions.load(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """
        Dump all elements in the Problem.

        See :class:`~mosaic.file_manipulation.h5.HDF5` for more information on the parameters of this method.

        Returns
        -------

        """
        self.medium.dump(*args, **kwargs)
        self.transducers.dump(*args, **kwargs)
        self.geometry.dump(*args, **kwargs)
        self.acquisitions.dump(*args, **kwargs)

    def plot(self, **kwargs):
        """
        Plot all elements in the Problem.

        Parameters
        ----------
        kwargs
            Arguments for plotting the fields.

        Returns
        -------

        """
        kwargs['plot'] = False

        # Medium
        medium_axes = self.medium.plot(**kwargs)

        # Geometry
        geometry_axes = []
        for axis in medium_axes:
            geometry_axes.append(self.geometry.plot(axis=axis, title=None, **kwargs))

        plotting.show(geometry_axes)

        # Acquisitions
        acquisitions_axes = self.acquisitions.plot()
        plotting.show(acquisitions_axes)

    def sub_problem(self, shot_id):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method takes care of selecting creating a SubProblem instance and populating it
        appropriately.

        Parameters
        ----------
        shot_id : int
            ID of the shot for which this sub-problem will be generated.

        Returns
        -------
        SubProblem
            Newly created SubProblem instance.

        """

        if isinstance(shot_id, (np.int32, np.int64)):
            shot_id = int(shot_id)

        sub_problem = SubProblem(self.name,
                                 input_folder=self.input_folder,
                                 output_folder=self.output_folder,
                                 problem_type=self.problem_type,
                                 grid=self.grid)

        shot = self.acquisitions.get(shot_id)

        # Set up medium
        sub_problem.medium = self.medium.sub_problem(shot, sub_problem)

        # Set up transducers
        sub_problem.transducers = self.transducers.sub_problem(shot, sub_problem)

        # Set up geometry
        sub_problem.geometry = self.geometry.sub_problem(shot, sub_problem)

        # Set up acquisitions
        shot = shot.sub_problem(shot, sub_problem)
        sub_problem.shot = shot
        sub_problem.shot_id = shot.id
        sub_problem.acquisitions = self.acquisitions.sub_problem(shot, sub_problem)

        return sub_problem

    async def forward(self, shot_ids=None, dump=True, deallocate=True, **kwargs):
        """
        Run the problem forward with default parameters.

        This will generate a series of Runners, one per available worker, and
        distribute all the available shots in the Acquisitions across those
        Runners.

        Parameters
        ----------
        shot_ids : list, optional
            List of shot IDs to run forward, defaults to all of them.
        dump : bool, optional
            Whether or not the generated data should be dumped to disk, defaults to True.
        deallocate : bool, optional
            Whether or not to deallocate the generated data after each Shot is completed,
            defaults to True.

        Returns
        -------

        """
        runtime = mosaic.runtime()

        # Create an array of runners
        runners = await Runner.remote(len=runtime.num_workers)

        # Prepare sub-problems
        if dump is True:
            try:
                self.acquisitions.load(path=self.output_folder,
                                       project_name=self.name, version=0)
            except OSError:
                pass

        if shot_ids is None:
            shot_ids = self.acquisitions.remaining_shot_ids
            if not len(shot_ids):
                runtime.logger.warning('No need to run forward, observed already exists')
                return

        if not isinstance(shot_ids, list):
            shot_ids = [shot_ids]

        # Run sub-problems
        await runners.map(self.run_forward, shot_ids, dump=dump, deallocate=deallocate, **kwargs)

    async def run_forward(self, shot_id, runner, dump=True, deallocate=True, **kwargs):
        """
        Run a single shot forward in a given Runner using default parameters.

        This means that no wavefield will be generated and the resulting traces will be stored in the Shot.

        Parameters
        ----------
        shot_id : int
            ID of the shot to be run.
        runner : Runner
            Runner on which the shot will be run.
        dump : bool, optional
            Whether or not the generated data should be dumped to disk, defaults to True.
        deallocate : bool, optional
            Whether or not to deallocate the generated data after each Shot is completed,
            defaults to True.

        Returns
        -------

        """
        runtime = mosaic.runtime()

        sub_problem = self.sub_problem(shot_id)
        runtime.logger.info('\nGiving shot %d to %s' % (shot_id, runner.runtime_id))

        await runner.set_problem(sub_problem, **kwargs)

        task = await runner.run_state(save_wavefield=False, **kwargs)
        traces, _ = await task.result()

        runtime.logger.info('Shot %d retrieved' % sub_problem.shot_id)

        shot = self.acquisitions.get(shot_id)
        shot.observed.data[:] = traces.data

        if dump is True:
            shot.append_observed(path=self.output_folder,
                                 project_name=self.name)

            runtime.logger.info('Appended traces for shot %d to observed file' % sub_problem.shot_id)

        if deallocate is True:
            shot.observed.deallocate()

    async def inverse(self, runners, variables,
                      block=None, iteration=None, **kwargs):
        """
        Run the inverse problem with default parameters.

        This will generate a series of given Runners, one per available worker,
        select some shots according to the Block configuration and then distribute
        those across the Runners.

        As the Runners return the functional value and gradient, those are
        accumulated in per-variable buffers before returning them.

        Parameters
        ----------
        runners : ArrayProxy
            Runners on which to distribute the shots.
        variables : VariableList
            Variables on which the inverse problem is running.
        block : Block
            Block instance that determines the configuration of the inverse problem.
        iteration : Iteration
            Iteration instance.
        kwargs
            Extra arguments for ``run_inverse``.

        Returns
        -------
        Iteration
            Iteration updated with the functional values returned for all shots.
        VariableList
            Variables of the inverse problem with the accumulated gradient of all shots.

        """
        variables.grad.fill(0.)
        variables.prec.fill(0.)

        shot_ids = self.acquisitions.select_shot_ids(**block.select_shots)

        # Run sub-problems
        async for fun, vars in runners.map_as_completed(self.run_inverse, shot_ids, variables,
                                                        iteration=iteration, **kwargs):
            iteration.add_fun(fun)
            variables.grad += vars.grad
            variables.prec += vars.prec

        return iteration, variables

    async def run_inverse(self, shot_id, runner, variables,
                          needs_grad=True, **kwargs):
        """
        Run the inverse problem for a single shot with default parameters.

        This will generate a series of given Runners, one per available worker,
        select some shots according to the Block configuration and then distribute
        those across the Runners.

        As the Runners return the functional value and gradient, those are
        accumulated in per-variable buffers before returning them.

        Parameters
        ----------
        shot_id : int
            ID of the shot to be run.
        runner : Runner
            Runner on which the shot will be run.
        variables : VariableList
            Variables on which the inverse problem is running.
        needs_grad : bool
            Whether or not the gradient is needed or only the functional value.
        kwargs
            Extra arguments for ``run_inverse``.

        Returns
        -------
        FunctionalValue
            Value of the functional and other information about the inverse execution.
        VariableList
            Variables of the inverse problem with the gradient of this shots.

        """
        runtime = mosaic.runtime()

        sub_problem = self.sub_problem(shot_id)
        runtime.logger.info('\nGiving shot %d to %s' % (shot_id, runner.runtime_id))

        variables.update_problem(sub_problem)

        await runner.set_problem(sub_problem, **kwargs)

        if needs_grad is True:
            task = await runner.run_gradient(variables, **kwargs)
            fun, vars = await task.result()

        else:
            task_fwd = await runner.run_state(save_wavefield=False, **kwargs)
            task_fun = await runner.run_functional(task_fwd.outputs[0], **kwargs)
            fun = await task_fun.outputs[0].result()
            vars = variables

        runtime.logger.info('Gradient and functional for shot %d retrieved' % sub_problem.shot_id)

        return fun, vars


class SubProblem(Gridded):
    """
    The SubProblem is the object that fully defines how a specific Shot is to be run. The SubProblem
    resembles the Problem from which ir originates, but takes from it only those parts that
    are relevant for this particular Shot.

    The SubProblem defines a medium with a set of fields (such as Vp or density), some
    transducers (such as a series of scalar point transducers), a geometry where those
    transducers are located in space, and the acquisitions that happen given that geometry.

    The SubProblem also defines a problem type, which determines the physics of interest, such
    as the second-order isotropic acoustic wave equation. And a numerical implementation
    of those physics, such as through the finite-difference library Devito.

    Parameters
    ----------
    name : str
        Name of the problem.
    grid : Grid or any of Space or Time
        Grid on which the Problem is defined
    input_folder : str, optional
        Default folder from which files should be read, defaults to current working directory.
    output_folder : str, optional
        Default folder to which files should be written, defaults to current working directory.
    medium : Medium, optional
        Predefined Medium of the problem.
    transducers : Transducers, optional
        Predefined Transducers of the problem.
    geometry : Geometry, optional
        Predefined Geometry of the problem.
    acquisitions : Acquisitions, optional
        Predefined Acquisitions of the problem.
    problem_type : callable, optional
        Problem type that will be executed on this SubProblem, defaults to ``acoustic``.

    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shot = None
        self.shot_id = -1

        self.name = name
        self.input_folder = kwargs.pop('input_folder', os.getcwd())
        self.output_folder = kwargs.pop('output_folder', os.getcwd())

        self.problem_config = {}
        self.medium = kwargs.pop('medium', None)
        self.transducers = kwargs.pop('transducers', None)
        self.geometry = kwargs.pop('geometry', None)
        self.acquisitions = kwargs.pop('acquisitions', None)

        self.problem_type = kwargs.pop('problem_type', None)
