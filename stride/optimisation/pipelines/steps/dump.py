
import mosaic
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
        iteration = kwargs.pop('iteration', None)
        if problem is None or iteration is None:
            return

        shot_id = problem.shot_id if hasattr(problem, 'shot_id') else None
        prev_step = kwargs.pop('prev_step', None)
        parameter = data.name.split('_')[-1].strip('_')
        info = 'parameter %s' % parameter
        if prev_step:
            prev_step = snake_case(prev_step.__class__.__name__)
            parameter = '%s_%s' % (prev_step, parameter)
            info = '%s after step %s' % (info, prev_step)
        if prefix:
            parameter = '%s_%s' % (prefix, parameter)
            info = '%s %s' % (prefix, info)
        info = 'Dumping %s' % info
        if shot_id is not None:
            parameter = '%s-Shot%05d' % (parameter, shot_id)
            info = '(ShotID %d) %s' % (shot_id, info)

        logger = mosaic.logger()
        logger.perf(info)

        data.dump(path=problem.output_folder,
                  project_name=problem.name,
                  version=iteration.abs_id+1, parameter=parameter)
