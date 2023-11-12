import numpy as np

import mosaic

from ....core import Operator


class CheckTraces(Operator):
    """
    Check a set of time traces for NaNs, Inf, etc.

    Parameters
    ----------
    raise_incorrect : bool, optional
        Whether to raise an exception if there are incorrect traces.
        Defaults to True.
    filter_incorrect : bool, optional
        Whether to filter out traces that are incorrect. Defaults to False.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.raise_incorrect = kwargs.pop('raise_incorrect', True)
        self.filter_incorrect = kwargs.pop('filter_incorrect', False)

        self._num_traces = None

    def forward(self, *traces, **kwargs):
        self._num_traces = len(traces)

        filtered = []
        for each in traces:
            filtered.append(self._apply(each, **kwargs))

        if len(traces) > 1:
            return tuple(filtered)

        else:
            return filtered[0]

    def adjoint(self, *d_traces, **kwargs):
        d_traces = d_traces[:self._num_traces]

        self._num_traces = None

        if len(d_traces) > 1:
            return d_traces

        else:
            return d_traces[0]

    def _apply(self, traces, **kwargs):
        raise_incorrect = kwargs.pop('raise_incorrect', self.raise_incorrect)
        filter_incorrect = kwargs.pop('filter_incorrect', self.filter_incorrect)

        out_traces = traces.alike(name='checked_%s' % traces.name)
        filtered = traces.extended_data.copy()

        is_nan = np.any(np.isnan(filtered), axis=-1)
        is_inf = np.any(np.isinf(filtered), axis=-1)

        if np.any(is_nan) or np.any(is_inf):
            msg = 'Nan or inf detected in %s' % traces.name

            problem = kwargs.pop('problem', None)
            shot_id = problem.shot.id if problem is not None else kwargs.pop('shot_id', None)
            if shot_id is not None:
                msg = '(ShotID %d) ' % shot_id + msg

            if raise_incorrect:
                raise RuntimeError(msg)
            else:
                mosaic.logger().warn(msg)

            if filter_incorrect:
                filtered[is_nan | is_inf, :] = 0

        out_traces.extended_data[:] = filtered

        return out_traces
