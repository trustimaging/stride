
import mosaic

from .pipeline import Pipeline


__all__ = ['ProcessWavelets', 'ProcessObserved', 'ProcessWaveletsObserved', 'ProcessTraces',
           'ProcessGlobalGradient', 'ProcessModelIteration']

# TODO Default configuration of pipelines should be better defined


@mosaic.tessera
class ProcessWavelets(Pipeline):
    """
    Default pipeline to process wavelets before running the forward problem.

    **Default steps:**

    - ``check_traces``
    - ``filter_traces``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []

        if kwargs.pop('check_traces', True):
            steps.append('check_traces')

        if kwargs.pop('filter_traces', True):
            steps.append('filter_traces')

        if kwargs.pop('fw3d_mode', False):
            steps.append('shift_traces')

        if kwargs.pop('resonance_filter', False):
            steps.append(('resonance_filter', False))

        super().__init__(steps, **kwargs)


@mosaic.tessera
class ProcessObserved(ProcessWavelets):
    """
    Default pipeline to process observed data before running the forward problem.

    **Default steps:**

    - ``check_traces``
    - ``filter_traces``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []
        kwargs['resonance_filter'] = False
        super().__init__(steps, **kwargs)


@mosaic.tessera
class ProcessWaveletsObserved(Pipeline):
    """
    Default pipeline to process wavelets and observed before running the forward problem,
    in steps that require both to be present.

    **Default steps:**

    - ``differentiate_traces``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []

        if kwargs.pop('differentiate_traces', True):
            steps.append(('differentiate_traces', False))

        super().__init__(steps, **kwargs)


@mosaic.tessera
class ProcessTraces(Pipeline):
    """
    Default pipeline to process modelled and observed before running the functional.

    **Default steps:**

    - ``check_traces``
    - ``mute_first_arrival``
    - ``mute_traces``
    - ``filter_traces``
    - ``norm_per_shot``
    - ``time_tweaking``
    - ``time_weighting``

    """

    def __init__(self, steps=None, **kwargs):
        steps = steps or []

        if kwargs.pop('check_traces', True):
            steps.append('check_traces')

        if kwargs.pop('filter_offsets', False):
            steps.append(('filter_offsets', False))  # do not raise if not present

        if kwargs.pop('mute_first_arrival', True):
            steps.append(('mute_first_arrival', False))

        if kwargs.pop('mute_traces', True):
            steps.append('mute_traces')

        if kwargs.pop('filter_traces', True):
            steps.append('filter_traces')

        agc = kwargs.pop('agc', False)
        if agc:
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

        if not agc and kwargs.pop('time_tweaking', True):
            steps.append(('time_tweaking', False))

        if kwargs.pop('time_weighting', True):
            steps.append(('time_weighting', False))

        super().__init__(steps, **kwargs)


@mosaic.tessera
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

        mask = kwargs.pop('mask_grad', True)
        if mask:
            steps.append('mask_field')

        smooth = kwargs.pop('smooth_grad', True)
        if smooth:
            steps.append('smooth_field')

        norm = kwargs.pop('norm_grad', True)
        if norm:
            steps.append('norm_field')

        super().__init__(steps, **kwargs)


@mosaic.tessera
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
