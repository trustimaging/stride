
from mosaic.utils import camel_case

from . import steps as steps_module


__all__ = ['Pipeline']


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
        self._no_grad = kwargs.pop('no_grad', False)

        steps = steps or []
        self._steps = []
        for step in steps:
            if isinstance(step, str):
                step_module = getattr(steps_module, step)
                step = getattr(step_module, camel_case(step))

                self._steps.append(step(**kwargs))

            else:
                self._steps.append(step)

    async def __call__(self, *args, **kwargs):
        """
        Apply all steps in the pipeline in order.

        """
        next_args = args

        needs_grad = dict()
        if self._no_grad:
            for arg in args:
                if hasattr(arg, 'needs_grad'):
                    needs_grad[arg.name] = arg.needs_grad
                    arg.needs_grad = False

        for step in self._steps:
            next_args = await step(*next_args, **kwargs)
            next_args = (next_args,) if len(args) == 1 else next_args

        if self._no_grad:
            for arg in next_args:
                if hasattr(arg, 'needs_grad'):
                    arg.needs_grad = needs_grad[arg.name]

        if len(args) == 1:
            return next_args[0]

        else:
            return next_args
