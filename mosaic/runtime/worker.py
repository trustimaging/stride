
import os
import psutil

from .runtime import Runtime
from ..utils import LoggerManager
from ..profile import global_profiler


__all__ = ['Worker']


class Worker(Runtime):
    """
    Workers are the runtimes where tesserae live, and where tasks are executed on them.

    Workers are initialised and managed by the node runtimes.

    """

    is_worker = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        num_threads = kwargs.pop('num_threads', None)
        num_threads = num_threads or psutil.cpu_count()

        if num_threads is None:
            num_threads = psutil.cpu_count()
        self._num_threads = num_threads

        os.environ['OMP_NUM_THREADS'] = str(self._num_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(self._num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self._num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self._num_threads)

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        self.logger = LoggerManager()

        if self.mode == 'local':
            self.logger.set_local(format=self.mode)
        else:
            runtime_id = 'head' if self.mode == 'interactive' else 'monitor'
            self.logger.set_remote(runtime_id=runtime_id, format=self.mode)

    def set_profiler(self):
        """
        Set up profiling.

        Returns
        -------

        """
        global_profiler.set_remote('monitor')
        super().set_profiler()
