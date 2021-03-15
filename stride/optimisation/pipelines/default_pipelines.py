
from .pipeline import Pipeline


__all__ = ['ProcessWavelets', 'ProcessWavefield', 'ProcessTraces',
           'ProcessAdjointSource', 'ProcessLocalGradient', 'ProcessGlobalGradient',
           'ProcessModelIteration', 'ProcessModelBlock']

# TODO Some of these pipelines should be different for variables
# TODO A more flexible and intuitive way of configuring pipelines is needed
# TODO Default configuration of pipelines should be better defined


class ProcessWavelets(Pipeline):
    """
    Default pipeline to process wavelets before running the forward problem.

    **Default steps:**

    - ``filter_wavelets``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('filter_wavelets')

        super().__init__(steps, **kwargs)


class ProcessWavefield(Pipeline):
    """
    Default pipeline to process the wavefield after running the forward problem.

    **Default steps:**

    """
    pass


class ProcessTraces(Pipeline):
    """
    Default pipeline to process modelled and observed before running the functional.

    **Default steps:**

    - ``filter_traces``
    - ``norm_per_shot``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('filter_traces')
        steps.append('norm_per_shot')

        super().__init__(steps, **kwargs)


class ProcessAdjointSource(Pipeline):
    """
    Default pipeline to process adjoint source before running the adjoint problem.

    **Default steps:**

    - ``filter_traces``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('filter_traces')

        super().__init__(steps, **kwargs)


class ProcessLocalGradient(Pipeline):
    """
    Default pipeline to process the gradient locally before returning it.

    **Default steps:**

    """
    pass


class ProcessGlobalGradient(Pipeline):
    """
    Default pipeline to process the global gradient before updating the variable.

    **Default steps:**

    - ``mask``
    - ``smooth_field``
    - ``norm_field``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('mask')
        steps.append('smooth_field')
        steps.append('norm_field')

        super().__init__(steps, **kwargs)


class ProcessModelIteration(Pipeline):
    """
    Default pipeline to process the model after each iteration.

    **Default steps:**

    - ``clip``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('clip')

        super().__init__(steps, **kwargs)


class ProcessModelBlock(Pipeline):
    """
    Default pipeline to process the model after each block.

    **Default steps:**

    """
    pass
