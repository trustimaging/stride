
import os
import numpy as np

import mosaic

from .base import Gridded
from .. import Runner
from .. import problem_types
from .. import plotting


__all__ = ['Problem', 'SubProblem']


class Problem(Gridded):

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = name
        self.input_folder = kwargs.pop('input_folder', os.getcwd())
        self.output_folder = kwargs.pop('output_folder', os.getcwd())

        self.problem_config = {}
        self.medium = kwargs.pop('medium', None)
        self.transducers = kwargs.pop('transducers', None)
        self.geometry = kwargs.pop('geometry', None)
        self.acquisitions = kwargs.pop('acquisitions', None)

        problem_type = kwargs.pop('problem_type', 'acoustic')
        problem_implementation = kwargs.pop('problem_implementation', 'devito')

        if callable(problem_type):
            self.problem_type = problem_type

        else:
            problem_module = getattr(problem_types, problem_type)
            problem_module = getattr(problem_module, problem_implementation)
            self.problem_type = problem_module.problem_type.ProblemType

    def load(self, *args, **kwargs):
        self.medium.load(*args, **kwargs)
        self.transducers.load(*args, **kwargs)
        self.geometry.load(*args, **kwargs)
        self.acquisitions.load(*args, **kwargs)

    def dump(self, *args, **kwargs):
        self.medium.dump(*args, **kwargs)
        self.transducers.dump(*args, **kwargs)
        self.geometry.dump(*args, **kwargs)
        self.acquisitions.dump(*args, **kwargs)

    def plot(self, **kwargs):
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

    async def forward(self, dump=True, deallocate=True):
        runtime = mosaic.runtime()

        # Create an array of runners
        runners = await Runner.remote(len=runtime.num_workers)

        # Prepare sub-problems
        try:
            self.acquisitions.load(path=self.output_folder,
                                   project_name=self.name, version=0)
        except OSError:
            pass

        shot_ids = self.acquisitions.remaining_shot_ids
        if not len(shot_ids):
            runtime.logger.warning('No need to run forward, observed already exists')
            return

        # Run sub-problems
        await runners.map(self.run_forward, shot_ids, dump=dump, deallocate=deallocate)

    async def run_forward(self, shot_id, runner, dump=True, deallocate=True):
        runtime = mosaic.runtime()

        sub_problem = self.sub_problem(shot_id)
        runtime.logger.info('Giving shot %d to %s' % (shot_id, runner.runtime_id))

        await runner.set_problem(sub_problem)

        task = await runner.run_state(save_wavefield=False, max_retries=0)
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
        runtime = mosaic.runtime()

        sub_problem = self.sub_problem(shot_id)
        runtime.logger.info('Giving shot %d to %s' % (shot_id, runner.runtime_id))

        variables.update_problem(sub_problem)

        await runner.set_problem(sub_problem)

        if needs_grad is True:
            task = await runner.run_gradient(variables)
            fun, vars = await task.result()

        else:
            task_fwd = await runner.run_state(save_wavefield=False)
            task_fun = await runner.run_functional(task_fwd.outputs[0])
            fun = await task_fun.outputs[0].result()
            vars = None

        runtime.logger.info('Gradient and functional for shot %d retrieved' % sub_problem.shot_id)

        return fun, vars


class SubProblem(Gridded):

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
