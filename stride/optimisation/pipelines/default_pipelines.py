
from .pipeline import Pipeline


__all__ = ['ProcessWavelets', 'ProcessWavefield', 'ProcessTraces',
           'ProcessAdjointSource', 'ProcessLocalGradient', 'ProcessGlobalGradient',
           'ProcessModelIteration', 'ProcessModelBlock']

# TODO Some of these pipelines should be different for variables
# TODO A more flexible and intuitive way of configuring pipelines is needed
# TODO Default configuration of pipelines should be better defined


class ProcessWavelets(Pipeline):

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('filter_wavelets')

        super().__init__(steps, **kwargs)


class ProcessWavefield(Pipeline):
    pass


class ProcessTraces(Pipeline):

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('filter_traces')
        steps.append('norm_per_shot')

        super().__init__(steps, **kwargs)


class ProcessAdjointSource(Pipeline):

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('filter_traces')

        super().__init__(steps, **kwargs)


class ProcessLocalGradient(Pipeline):
    pass


class ProcessGlobalGradient(Pipeline):

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('mask')
        steps.append('smooth_field')
        steps.append('norm_field')

        super().__init__(steps, **kwargs)


class ProcessModelIteration(Pipeline):

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        steps.append('clip')

        super().__init__(steps, **kwargs)


class ProcessModelBlock(Pipeline):
    pass
