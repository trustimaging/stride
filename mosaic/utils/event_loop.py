
import os
import uuid
import asyncio
import inspect
import functools
import threading
import concurrent.futures

import mosaic
from .utils import set_main_thread


__all__ = ['EventLoop', 'Future', 'gather']


# Most remote objects behave similarly in that they have an UID
# they register themselves with the runtime and they have a set
# of proxies that are, in a way, subscribed to their updates.
#
# Proxies share the UID with their remote counterparts and
# they also have to register themselves with the runtime to
# be notified by remote messages.
#
# Proxies are used to communicate things to the remote objects and
# to receive updates and results from them.
#
# For futures, the proxy can be used both to set a result or an exception,
# but also to ask for a result or an exception, to check the state of
# the future or to await it.
#
# Setting the result or the exception of a future will trigger the notification
# of the proxies.
#
# While tasks get their results (not their exceptions) lazily, only when
# we ask for them, futures happen immediately?
# Using all this at the level of comms seems like jumping levels of abstraction
# in the wrong direction, is that the right way to do it, or should we use another
# type of future and keep it simple?


class Future:
    def __init__(self, name='anon', loop=None):
        self._future = asyncio.Future()
        self._loop = loop or mosaic.get_event_loop()

        self._name = name
        self._uid = '%s-%s-%s' % ('fut',
                                  name,
                                  uuid.uuid4().hex)

    @property
    def uid(self):
        return self._uid

    @property
    def state(self):
        if self._future.cancelled():
            return 'cancelled'

        elif self._future.done():
            return 'done'

        else:
            return 'pending'

    @property
    def future(self):
        return self._future

    def __repr__(self):
        return "<%s object at %s, uid=%s, state=%s>" % \
               (self.__class__.__name__, id(self), self.uid, self.state)

    def __await__(self):
        return (yield from self._future.__await__())

    def result(self):
        return self._future.result()

    def exception(self):
        return self._future.exception()

    def set_result(self, result):
        self._future.set_result(result)

    def set_exception(self, exc):
        self._future.set_exception(exc)

    def done(self):
        return self._future.done()

    def cancelled(self):
        return self._future.cancelled()

    def add_done_callback(self, fun):
        self._future.add_done_callback(fun)


class EventLoop:
    def __init__(self, loop=None):  # Add option to make MainThread
        self._loop = loop or asyncio.new_event_loop()
        self._executor = None
        asyncio.set_event_loop(self._loop)

        self._within_loop = threading.local()
        self._thread = threading.Thread(target=self._run_loop,
                                        args=(self._loop,))
        self._thread.daemon = True
        self._thread.start()

        self._within_loop.flag = False

    @property
    def within_loop(self):
        try:
            return self._within_loop.flag is True

        except AttributeError:
            return False

    def get_event_loop(self):
        return self._loop

    def _run_loop(self, loop):
        self._loop = loop
        asyncio.set_event_loop(self._loop)

        # TODO Figure out the best way to set this
        num_workers = int(os.environ.get('OMP_NUM_THREADS', 2))
        self._executor = concurrent.futures.ThreadPoolExecutor(1)

        self._within_loop.flag = True

        self._loop.run_forever()

    def stop(self):
        self._loop.call_soon_threadsafe(self._executor.shutdown)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._executor.shutdown(wait=True)

        if not self.within_loop:
            self._thread.join()

    def __del__(self):
        self.stop()

    def run(self, coro, args=(), kwargs=None, wait=False):
        kwargs = kwargs or {}

        if not inspect.iscoroutine(coro) and not inspect.iscoroutinefunction(coro):
            coro = asyncio.coroutine(coro)

        future = asyncio.run_coroutine_threadsafe(coro(*args, **kwargs), self._loop)

        if wait is True:
            return future.result()

        else:
            return future

    def run_in_executor(self, callback, args=(), kwargs=None):
        callback = functools.partial(callback, *args, **kwargs)
        future = self._loop.run_in_executor(self._executor, callback)

        return future

    def run_async(self, coro, args=(), kwargs=None):
        kwargs = kwargs or {}

        if not inspect.iscoroutine(coro) and not inspect.iscoroutinefunction(coro):
            coro = asyncio.coroutine(coro)

        return self._loop.create_task(coro(*args, **kwargs))

    def wrap_future(self, future):
        return asyncio.wrap_future(future, loop=self._loop)

    def timeout(self, coro, timeout, args=(), kwargs=None):
        kwargs = kwargs or {}

        async def _timeout():
            await asyncio.sleep(timeout)
            await self.run_async(coro, args=args, kwargs=kwargs)

        future = asyncio.run_coroutine_threadsafe(_timeout(), self._loop)
        return future

    def interval(self, coro, interval, args=(), kwargs=None):
        kwargs = kwargs or {}

        async def _interval():
            while True:
                await asyncio.sleep(interval)
                await self.run_async(coro, args=args, kwargs=kwargs)

        future = asyncio.run_coroutine_threadsafe(_interval(), self._loop)
        return future

    def set_main_thread(self):
        self._loop.call_soon_threadsafe(set_main_thread)


def gather(tasks):
    if not isinstance(tasks, list):
        return tasks

    else:
        return asyncio.gather(*tasks)
