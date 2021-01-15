
from abc import ABC, abstractmethod

from . import steps as steps_module


__all__ = ['Pipeline', 'PipelineStep']


class PipelineStep(ABC):
    """
    Base class for processing steps in pipelines.

    """

    @abstractmethod
    def apply(self, *args, **kwargs):
        """
        Apply the processing step to the arguments.

        """
        pass


class Pipeline:
    """
    A pipeline represents a series of processing steps that will be applied
    in order to a series of inputs. Pipelines encode pre-processing or
    post-processing steps such as filtering time traces or smoothing a gradient.

    Parameters
    ----------
    steps : list, optional
        List of steps that form the pipeline. Steps can be callable or strings pointing
        to a default, pre-defined step.

    """

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
        """
        Apply all steps in the pipeline in order.

        """
        next_args = args

        for step in self._steps:
            next_args = step.apply(*next_args, **kwargs)
            next_args = (next_args,) if len(args) == 1 else next_args

        if len(args) == 1:
            return next_args[0]

        else:
            return next_args
