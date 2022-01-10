
from mosaic.utils import camel_case

from mosaic import tessera

from . import steps as steps_module
from ...core import Operator, no_grad


__all__ = ['Pipeline']


@tessera
class Pipeline(Operator):
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
        super().__init__(**kwargs)

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

    async def forward(self, *args, **kwargs):
        """
        Apply all steps in the pipeline in order.

        """
        next_args = args

        for step in self._steps:
            if self._no_grad:
                with no_grad(*next_args, **kwargs):
                    next_args = await step(*next_args, **kwargs)
            else:
                next_args = await step(*next_args, **kwargs)
            next_args = (next_args,) if len(args) == 1 else next_args

        if len(args) == 1:
            return next_args[0]

        else:
            return next_args

    async def adjoint(self, *args, **kwargs):
        if len(self._steps) < 1:
            return

        last_step = self._steps[-1]
        outputs = last_step.adjoint(*args, **kwargs)

        if len(outputs) == 1:
            return outputs[0]

        else:
            return outputs
