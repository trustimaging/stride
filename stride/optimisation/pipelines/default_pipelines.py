
import mosaic

from .pipeline import Pipeline


__all__ = ['ProcessWavelets', 'ProcessObserved', 'ProcessTraces',
           'ProcessGlobalGradient', 'ProcessModelIteration']

# TODO Default configuration of pipelines should be better defined


@mosaic.tessera
class ProcessWavelets(Pipeline):
    """
    Default pipeline to process wavelets before running the forward problem.

    **Default steps:**

    - ``filter_traces``

    """

    def __init__(self, steps=None, no_grad=False, **kwargs):
        steps = steps or []

        if kwargs.pop('check_traces', True):
            steps.append('check_traces')

        if kwargs.pop('filter_traces', True):
            steps.append('filter_traces')

        if kwargs.pop('resonance_filter', False):
            steps.append(('resonance_filter', False))

        super().__init__(steps, no_grad=no_grad, **kwargs)


@mosaic.tessera
class ProcessObserved(ProcessWavelets):
    """
    Default pipeline to process observed data before running the forward problem.

    **Default steps:**

    - ``filter_traces``

    """

    def __init__(self, steps=None, no_grad=False, **kwargs):
        steps = steps or []
        kwargs['resonance_filter'] = False
        super().__init__(steps, no_grad=no_grad, **kwargs)


@mosaic.tessera
class ProcessTraces(Pipeline):
    """
    Default pipeline to process modelled and observed before running the functional.

    **Default steps:**

    - ``mute_traces``
    - ``filter_traces``
    - ``norm_per_shot``

    """

    def __init__(self, steps=None, no_grad=False, **kwargs):
        steps = steps or []

        if kwargs.pop('check_traces', True):
            steps.append('check_traces')

        if kwargs.pop('filter_offsets', False):
            steps.append(('filter_offsets', False))  # do not raise if not present

        if kwargs.pop('mute_traces', True):
            steps.append('mute_traces')

        if kwargs.pop('filter_traces', True):
            steps.append('filter_traces')

        if kwargs.pop('time_weighting', True):
            steps.append(('time_weighting', False))

        if kwargs.pop('agc', False):
            steps.append(('agc', False))

        norm_per_shot = kwargs.pop('norm_per_shot', True)
        norm_per_trace = kwargs.pop('norm_per_trace', False)
        if norm_per_trace:
            steps.append('norm_per_trace')
        elif norm_per_shot:
            steps.append('norm_per_shot')

        scale_per_shot = kwargs.pop('scale_per_shot', False)
        scale_per_trace = kwargs.pop('scale_per_trace', False)
        if scale_per_shot:
            steps.append('scale_per_shot')
        elif scale_per_trace:
            steps.append('scale_per_trace')

        super().__init__(steps, no_grad=no_grad, **kwargs)


@mosaic.tessera
class ProcessGlobalGradient(Pipeline):
    """
    Default pipeline to process the global gradient before updating the variable.

    **Default steps:**

    - ``mask``
    - ``smooth_field``
    - ``norm_field``

    """

    def __init__(self, steps=None, no_grad=True, **kwargs):
        steps = steps or []

        mask = kwargs.pop('mask', True)
        if mask:
            steps.append('mask')

        smooth = kwargs.pop('smooth', True)
        if smooth:
            steps.append('smooth_field')

        norm = kwargs.pop('norm', True)
        if norm:
            steps.append('norm_field')

        super().__init__(steps, no_grad=no_grad, **kwargs)


@mosaic.tessera
class ProcessModelIteration(Pipeline):
    """
    Default pipeline to process the model after each iteration.

    **Default steps:**

    - ``clip``

    """

    def __init__(self, steps=None, no_grad=True, **kwargs):
        steps = steps or []
        steps.append('clip')

        super().__init__(steps, no_grad=no_grad, **kwargs)
