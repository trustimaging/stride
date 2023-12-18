
from mosaic.utils import snake_case

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

    def adjoint(self, *d_data, **kwargs):
        d_data = d_data[:self._num_data]

        for each in d_data:
            self._apply(each, prefix='adjoint', **kwargs)

        if len(d_data) == 1:
            d_data = d_data[0]

        return d_data

    def _apply(self, data, prefix=None, **kwargs):
        problem = kwargs.pop('problem', None)
        if problem is None:
            return

        prev_step = kwargs.pop('prev_step', None)
        parameter = data.name.split('_')[-1].strip('_')
        if prev_step:
            prev_step = snake_case(prev_step.__class__.__name__)
            parameter = '%s_%s' % (prev_step, parameter)
        if prefix:
            parameter = '%s_%s' % (prefix, parameter)

        data.dump(path=problem.output_folder,
                  project_name=problem.name,
                  version=0, parameter=parameter)
