
import os
import uuid
import asyncio
import inspect
import weakref
import functools
import concurrent.futures

import mosaic
from .utils import set_main_thread


__all__ = ['EventLoop', 'Future', 'gather']


class Future:
    """
    A local future associated with an EventLoop.

    Parameters
    ----------
    name : str, optional
        Name to give to the future, defaults to ``anon``.
    loop : EventLoop, optional
        Loop associated with the future, defaults to current loop.

    """

    def __init__(self, name='anon', loop=None):
        self._future = asyncio.Future()
        self._loop = loop or mosaic.get_event_loop()

        self._name = name
        self._uid = '%s-%s-%s' % ('fut',
                                  name,
                                  uuid.uuid4().hex)

    @property
    def uid(self):
        """
        Access the UID of the future.

        """
        return self._uid

    @property
    def state(self):
        """
        Check the state of the future (``pending``, ``done``, ``cancelled``).

        """
        if self._future.cancelled():
            return 'cancelled'

        elif self._future.done():
            return 'done'

        else:
            return 'pending'

    @property
    def future(self):
        """
        The wrapped future

        """
        return self._future

    def __repr__(self):
        return "<%s object at %s, uid=%s, state=%s>" % \
               (self.__class__.__name__, id(self), self.uid, self.state)

    def __await__(self):
        return (yield from self._future.__await__())

    def result(self):
        """
        Get the future result.

        Returns
        -------

        """
        return self._future.result()

    def exception(self):
        """
        Get the future exception.

        Returns
        -------

        """
        return self._future.exception()

    def set_result(self, result):
        """
        Set the future result.

        Parameters
        ----------
        result : object

        Returns
        -------

        """
        self._future.set_result(result)

    def set_exception(self, exc):
        """
        Set the future exception.

        Parameters
        ----------
        exc : Exception

        Returns
        -------

        """
        self._future.set_exception(exc)

    def done(self):
        """
        Check whether the future is done.

        Returns
        -------

        """
        return self._future.done()

    def cancelled(self):
        """
        Check whether the future is cancelled.

        Returns
        -------

        """
        return self._future.cancelled()

    def add_done_callback(self, fun):
        """
        Add done callback.

        Parameters
        ----------
        fun : callable

        Returns
        -------

        """
        self._future.add_done_callback(fun)


class EventLoop:
    """
    The event loop encapsulates the asyncio (or equivalent) event loop, which
    will run in a separate thread.

    It provides helper functions to run things within the loop, in an executor,
    and to call functions after a period of time or every fixed amount of time.

    Parameters
    ----------
    loop : asyncio loop, optional
        Asyncio event loop to use internally, defaults to new loop.

    """

    def __init__(self, loop=None):
        self._loop = loop or asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # TODO Figure out the best way to set this
        num_workers = int(os.environ.get('OMP_NUM_THREADS', 2))
        self._executor = concurrent.futures.ThreadPoolExecutor(1)

        self._stop = asyncio.Event()

        self._recurring_tasks = weakref.WeakSet()

    def get_event_loop(self):
        """
        Access the internal loop.

        Returns
        -------
        asyncio loop

        """
        return self._loop

    def run_forever(self):
        """
        Run event loop forever.

        Returns
        -------

        """
        async def main():
            await self._stop.wait()

        return self._loop.run_until_complete(main())

    def stop(self):
        """
        Stop the event loop.

        Returns
        -------

        """
        try:
            if self._stop.is_set():
                return

            self._stop.set()

            for task in list(self._recurring_tasks):
                if not task.done():
                    task.cancel()

            tasks = asyncio.Task.all_tasks()
            pending = [task for task in tasks if not task.done()]

            for task in pending:
                task.cancel()

            while len(pending):
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                pending = [task for task in tasks if not task.done()]

            self._loop.stop()
            self._loop.close()
            self._executor.shutdown()
            self._executor.shutdown(wait=True)

            asyncio.set_event_loop(None)

        except RuntimeError:
            pass

    def __del__(self):
        self.stop()

    def run(self, coro, args=(), kwargs=None, wait=False):
        """
        Schedule a function in the event loop from synchronous code.

        The call can be waited or returned immediately.

        Parameters
        ----------
        coro : callable
            Function to execute in the loop.
        args : tuple, optional
            Set of arguments for the function.
        kwargs : optional
            Set of keyword arguments for the function.
        wait : bool, optional
            Whether or not to wait for the call to end, defaults to False.

        Returns
        -------
        Return value from call or concurrent.futures.Future, depending on whether it is waited or not.

        """
        if self._stop.is_set():
            return

        kwargs = kwargs or {}

        if not inspect.iscoroutine(coro) and not inspect.iscoroutinefunction(coro):
            coro = asyncio.coroutine(coro)

        if not self._loop.is_running():
            return self._loop.run_until_complete(coro(*args, **kwargs))

        future = self._loop.create_task(coro(*args, **kwargs))
        return future

    def run_in_executor(self, callback, args=(), kwargs=None):
        """
        Run function in a thread executor.

        Parameters
        ----------
        callback : callable
            Function to execute.
        args : tuple, optional
            Set of arguments for the function.
        kwargs : optional
            Set of keyword arguments for the function.

        Returns
        -------
        asyncio.Future

        """
        if self._stop.is_set():
            return

        callback = functools.partial(callback, *args, **kwargs)
        future = self._loop.run_in_executor(self._executor, callback)

        return future

    def wrap_future(self, future):
        """
        Wrap a concurrent.futures.Future to be compatible
        with asyncio.

        Parameters
        ----------
        future : concurrent.futures.Future

        Returns
        -------
        asyncio.Future

        """
        return asyncio.wrap_future(future, loop=self._loop)

    def timeout(self, coro, timeout, args=(), kwargs=None):
        """
        Run function after a certain ``timeout`` in seconds.

        Parameters
        ----------
        coro : callable
            Function to execute in the loop.
        timeout : float
            Time to wait before execution in seconds.
        args : tuple, optional
            Set of arguments for the function.
        kwargs : optional
            Set of keyword arguments for the function.

        Returns
        -------
        concurrent.futures.Future

        """
        kwargs = kwargs or {}

        async def _timeout():
            await asyncio.sleep(timeout)
            await self.run(coro, args=args, kwargs=kwargs)

        future = asyncio.run_coroutine_threadsafe(_timeout(), self._loop)
        self._recurring_tasks.add(future)

        return future

    def interval(self, coro, interval, args=(), kwargs=None):
        """
        Run function every ``interval`` in seconds, starting after ``interval`` seconds.

        Parameters
        ----------
        coro : callable
            Function to execute in the loop.
        interval : float
            Time to wait between executions in seconds.
        args : tuple, optional
            Set of arguments for the function.
        kwargs : optional
            Set of keyword arguments for the function.

        Returns
        -------
        concurrent.futures.Future

        """
        kwargs = kwargs or {}

        async def _interval():
            while not self._stop.is_set():
                await asyncio.sleep(interval)
                await self.run(coro, args=args, kwargs=kwargs)

        future = asyncio.run_coroutine_threadsafe(_interval(), self._loop)
        self._recurring_tasks.add(future)

        return future

    def set_main_thread(self):
        """
        Set loop thread as main thread.

        Returns
        -------

        """
        self._loop.call_soon_threadsafe(set_main_thread)


def gather(tasks):
    """
    Wait for the termination of a group of tasks concurrently.

    Parameters
    ----------
    tasks : list
        Set of tasks to wait.

    Returns
    -------
    list
        Set of results from the task list.

    """
    if not isinstance(tasks, list):
        return tasks

    else:
        return asyncio.gather(*tasks)
