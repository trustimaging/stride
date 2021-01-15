
from abc import ABC, abstractmethod

from . import steps as steps_module


__all__ = ['Pipeline', 'PipelineStep']


class PipelineStep(ABC):

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class Pipeline:

    def __init__(self, steps=None, **kwargs):
        steps = steps or []

        self._steps = []
        for step in steps:
            if callable(step):
                self._steps.append(step(**kwargs))

            else:
                step_module = getattr(steps_module, step)
                step = getattr(step_module, 'Step')

                self._steps.append(step(**kwargs))

    def apply(self, *args, **kwargs):
        next_args = args

        for step in self._steps:
            next_args = step.apply(*next_args, **kwargs)
            next_args = (next_args,) if len(args) == 1 else next_args

        if len(args) == 1:
            return next_args[0]

        else:
            return next_args
