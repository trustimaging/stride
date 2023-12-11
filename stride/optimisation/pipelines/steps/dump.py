
import numpy as np

from ....core import Operator


class Dump(Operator):
    """
    Dump a series of data objects.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._num_data = None

    def forward(self, *data, **kwargs):
        self._num_data = len(data)

        for each in data:
            self._apply(each, **kwargs)

        if len(data) == 1:
            data = data[0]

        return data

    def adjoint(self, *d_traces, **kwargs):
        d_data = d_traces[:self._num_data]

        for each in d_data:
            self._apply(each, prefix='adjoint_', **kwargs)

        if len(d_data) == 1:
            d_data = d_data[0]

        return d_data

    def _apply(self, data, prefix=None, **kwargs):
        problem = kwargs.pop('problem', None)
        if problem is None:
            return

        parameter = data.name
        if prefix:
            parameter = prefix + parameter

        data.dump(path=problem.output_folder,
                  project_name=problem.name,
                  version=0, parameter=parameter)
