
import numpy as np

from ....core import Operator


class Scale(Operator):
    """
    Scale a series of time traces.

    Parameters
    ----------
    scale_to : Traces
        Reference traces to normalise to.
    relative_scale : bool, optional
        Whether to make the scaling relative to the norm of scale_to.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scale_to = kwargs.pop('scale_to', None)
        self.relative_scale = kwargs.pop('relative_scale', True)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)
        scale_to = kwargs.pop('scale_to', self.scale_to)
        relative_scale = kwargs.pop('relative_scale', self.relative_scale)

        if relative_scale:
            relative_scale = self._norm(scale_to, **kwargs) + 1e-31
        else:
            relative_scale = 1.

        scaled = tuple([self._apply(each, scale_to, relative_scale, **kwargs)
                        for each in traces])

        if len(scaled) == 1:
            scaled = scaled[0]

        return scaled

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        if len(d_traces) == 1:
            d_traces = d_traces[0]

        return d_traces

    def _norm(self, traces, **kwargs):
        norm_value = 0.

        for index in range(traces.extended_shape[0]):
            norm_value += np.sum(traces.extended_data[index] ** 2)

        return np.sqrt(norm_value / traces.extended_shape[0])

    def _apply(self, traces, scale_to, relative_scale, **kwargs):
        pass


class ScalePerShot(Scale):
    """
    Scale a series of time traces to the norm of the reference set.

    Parameters
    ----------
    scale_to : Traces
        Reference traces to normalise to.
    relative_scale : bool, optional
        Whether to make the scaling relative to the norm of scale_to.

    """

    def _apply(self, traces, scale_to, relative_scale, **kwargs):
        norm_value = self._norm(scale_to, **kwargs)

        out_traces = traces.alike(name='scaled_%s' % traces.name)
        out_traces.extended_data[:] = traces.extended_data * norm_value / relative_scale

        return out_traces


class ScalePerTrace(Scale):
    """
    Scale a series of time traces individually.

    Parameters
    ----------
    scale_to : Traces
        Reference traces to normalise to.
    relative_scale : bool, optional
        Whether to make the scaling relative to the norm of scale_to.

    """

    def _apply(self, traces, scale_to, relative_scale, **kwargs):
        out_traces = traces.alike(name='scaled_%s' % traces.name)

        norms = []
        for index in range(traces.extended_shape[0]):
            norm_value = np.sqrt(np.sum(scale_to.extended_data[index]**2))
            norms.append(norm_value)

        avg_norm = np.mean(norms)
        std_norm = np.std(norms)
        for index in range(traces.extended_shape[0]):
            # prevent outlier traces with large or small norms
            norm_value = max(avg_norm-std_norm, min(norms[index], avg_norm+std_norm))
            out_traces.extended_data[index] = traces.extended_data[index] * norm_value / relative_scale

        return out_traces
