
import os
import time
import asyncio

import mosaic
from .runtime import Runtime, RuntimeProxy
from ..utils import LoggerManager
from ..utils import subprocess
from ..profile import profiler, global_profiler


__all__ = ['Head']


class Head(Runtime):
    """
    The head is the main runtime, where the user entry point is executed.

    """

    is_head = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def init(self, **kwargs):
        """
        Asynchronous counterpart of ``__init__``.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        # Start monitor if necessary and handshake in reverse
        monitor_address = kwargs.get('monitor_address', None)
        if not self.is_monitor and monitor_address is None:
            await self.init_monitor(**kwargs)

            path = os.path.join(os.getcwd(), 'mosaic-workspace')
            if not os.path.exists(path):
                os.makedirs(path)

            filename = os.path.join(path, 'monitor.key')

            while not os.path.exists(filename):
                await asyncio.sleep(0.1)

            with open(filename, 'r') as file:
                file.readline()

                _ = file.readline().split('=')[1].strip()
                parent_address = file.readline().split('=')[1].strip()
                parent_port = file.readline().split('=')[1].strip()
                pubsub_port = file.readline().split('=')[1].strip()

                kwargs['monitor_address'] = parent_address
                kwargs['monitor_port'] = int(parent_port)
                kwargs['pubsub_port'] = int(pubsub_port)

        await super().init(**kwargs)

        # Wait for workers to be ready
        tic = time.time()
        num_workers = kwargs.pop('num_workers')
        timeout = 180
        while len(self.workers) < num_workers:
            if timeout is not None and (time.time() - tic) > timeout:
                raise RuntimeError('Timed out while waiting for %d workers to connect' % num_workers)
            await asyncio.sleep(0.1)

    async def init_monitor(self, **kwargs):
        """
        Init monitor runtime.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        def start_monitor(*args, **extra_kwargs):
            kwargs.update(extra_kwargs)
            kwargs['dump_init'] = True
            mosaic.init('monitor', *args, **kwargs, wait=True)

        monitor_proxy = RuntimeProxy(name='monitor')
        monitor_subprocess = subprocess(start_monitor)(name=monitor_proxy.uid,
                                                       daemon=False)
        monitor_subprocess.start_process()
        monitor_proxy.subprocess = monitor_subprocess

        self._monitor = monitor_proxy

    async def stop(self, sender_id=None):
        """
        Stop runtime.

        Parameters
        ----------
        sender_id : str

        Returns
        -------

        """
        # Send final profile updates before closing
        if profiler.tracing:
            await self.send_profile()

        if self._monitor.subprocess is not None:
            await self._monitor.stop()
            self._monitor.subprocess.join_process()

        await super().stop(sender_id)
        # os._exit(0)

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        self.logger = LoggerManager()
        self.logger.set_local(format=self.mode)

    def set_profiler(self):
        """
        Set up profiling.

        Returns
        -------

        """
        global_profiler.set_remote('monitor')
        super().set_profiler()
