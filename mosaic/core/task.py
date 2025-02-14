
import sys
import uuid
import tblib
import inspect
import asyncio
import weakref
import operator
from cached_property import cached_property

import mosaic
from .. import types
from .base import Base, RemoteBase, ProxyBase, RuntimeDisconnectedError
from ..types import WarehouseObject
from ..utils import Future, MultiError, sizeof, remote_sizeof, memory_used


__all__ = ['Task', 'TaskArray', 'TaskProxy', 'TaskArrayProxy',
           'TaskOutputGenerator', 'TaskOutput', 'TaskDone']


class Task(RemoteBase):
    """
    When we call a method on a remote tessera, two things will happen:

    - a Task will be generated on the remote tessera and queued to be executed by it;
    - and a TaskProxy is generated on the calling side as a pointer to that remote task.

    We can use the task proxy to wait for the completion of the task (``await task_proxy``),
    as an argument to other tessera method calls, or to retrieve the result of the
    task (``await task_proxy.result()``).

    It is also possible to access references to the individual outputs of the task by
    using ``task_proxy.outputs``. Outputs can be accessed through their position: ``task_proxy.outputs[0]``
    will reference the first output of the task.

    A reference to the termination of the task is also available through ``task_proxy.outputs.done``,
    which can be used to create explicit dependencies between tasks, thus controlling the order
    of execution.

    Tasks on a particular tessera are guaranteed to be executed in the order in which they were called,
    but no such guarantees exist for tasks on different tesserae.

    A completed task is kept in memory at the worker for as long as there are proxy references to
    it. If none exist, it will be made available for garbage collection.

    Objects of class Task should not be instantiated directly by the user.

    Parameters
    ----------
    uid : str
        UID of the task.
    sender_id : str
        UID of the caller.
    tessera : Tessera
        Tessera on which the task is to be executed.
    method : callable
        Method associated with the task.
    args : tuple, optional
        Arguments to pass to the method.
    kwargs : optional
        Keyword arguments to pass to the method.

    """

    type = 'task'
    is_remote = True

    def __init__(self, uid, sender_id, tessera, method, *args, **kwargs):
        super().__init__(uid, *args, **kwargs)

        self._sender_id = sender_id
        self._tessera = weakref.proxy(tessera)
        self._silence = kwargs.pop('silence', [])
        self._silence = [self._silence] if not isinstance(self._silence, list) else self._silence

        kwargs = self._fill_config(**kwargs)

        self.method = method
        self.args = args
        self.kwargs = kwargs

        self._tic = 0
        self._elapsed = None

        self._arg_size = 0
        self._args_pending = set()
        self._kwargs_pending = set()

        self._args_value = dict()
        self._kwargs_value = dict()

        self._args_state = dict()
        self._kwargs_state = dict()

        self._ready_future = Future()
        self._done_future = Future()
        self._exception_future = Future()
        self._result = None
        self._exception = None

        self._state = None
        self.runtime.register(self)
        self.state_changed('pending')
        self.register_proxy(self._sender_id)

    @property
    def sender_id(self):
        """
        Caller UID.

        """
        return self._sender_id

    @cached_property
    def tessera_id(self):
        """
        Tessera UID.

        """
        try:
            return self._tessera.uid
        except ReferenceError:
            return None

    @cached_property
    def remote_runtime(self):
        """
         Proxies that have references to this task.

        """
        return {self.proxy(each) for each in list(self._proxies)}

    @property
    def collectable(self):
        """
        Whether the object is ready for collection.

        """
        return self._state in ['failed', 'done']

    @classmethod
    def remote_cls(cls):
        """
        Class of the remote.

        """
        return TaskProxy

    def _fill_config(self, **kwargs):
        kwargs['max_retries'] = kwargs.get('max_retries', self._tessera.max_retries)

        return super()._fill_config(**kwargs)

    def args_value(self):
        """
        Processed value of the args of the task.

        Returns
        -------
        tuple

        """
        args = [value for key, value in sorted(self._args_value.items(), key=operator.itemgetter(0))]

        return tuple(args)

    def kwargs_value(self):
        """
        Processed value of the args of the task.

        Returns
        -------
        dict

        """
        return self._kwargs_value

    def add_event(self, event_name, **kwargs):
        kwargs['tessera_id'] = self.tessera_id
        return super().add_event(event_name, **kwargs)

    def add_profile(self, profile, **kwargs):
        kwargs['tessera_id'] = self.tessera_id
        return super().add_profile(profile, **kwargs)

    async def prepare_args(self):
        """
        Prepare the arguments of the task for execution.

        Returns
        -------
        Future

        """
        tasks = []

        async def await_size(_arg):
            self._arg_size += await _arg.size(pending=True)

        for index in range(len(self.args)):
            arg = self.args[index]

            if type(arg) in types.awaitable_types:
                self._args_state[index] = arg.state
                if not isinstance(arg, TaskDone):
                    self._args_value[index] = None

                if arg.state != 'done':
                    self._args_pending.add(arg)

                    def callback(_index, _arg):
                        def _callback(fut):
                            self.loop.run(self._set_arg_done, fut, _index, _arg)

                        return _callback

                    arg.add_done_callback(callback(index, arg))

                else:
                    tasks.append(
                        await_size(arg)
                    )

            else:
                self._args_state[index] = 'ready'
                self._args_value[index] = arg

        for key, value in self.kwargs.items():
            if type(value) in types.awaitable_types:
                self._kwargs_state[key] = value.state
                if not isinstance(value, TaskDone):
                    self._kwargs_value[key] = None

                if value.state != 'done':
                    self._kwargs_pending.add(value)

                    def callback(_key, _arg):
                        def _callback(fut):
                            self.loop.run(self._set_kwarg_done, fut, _key, _arg)

                        return _callback

                    value.add_done_callback(callback(key, value))

                else:
                    tasks.append(
                        await_size(value)
                    )

            else:
                self._kwargs_state[key] = 'ready'
                self._kwargs_value[key] = value

        await asyncio.gather(*tasks)
        await self._check_ready()

        return self._ready_future

    async def set_result(self, result):
        """
        Set task result.

        Parameters
        ----------
        result

        Returns
        -------

        """
        if not isinstance(result, (tuple, dict)):
            result = (result,)

        min_size = 1024**1
        if isinstance(result, tuple):
            async def store(_value):
                return await self.runtime.put(_value, reply=True)

            async def noop(_value):
                return _value

            tasks = []
            for value in result:
                obj_size = sizeof(value)

                if obj_size > min_size:
                    tasks.append(store(value))
                else:
                    tasks.append(noop(value))

            stored_result = await asyncio.gather(*tasks)
            stored_result = tuple(stored_result)

        elif isinstance(result, dict):
            async def store(_key, _value):
                return _key, await self.runtime.put(_value, reply=True)

            async def noop(_key, _value):
                return _key, _value

            tasks = []
            for key, value in result.items():
                obj_size = sizeof(value)

                if obj_size > min_size:
                    tasks.append(store(key, value))
                else:
                    tasks.append(noop(key, value))

            stored_result = {}
            tasks = await asyncio.gather(*tasks)
            for key, value in tasks:
                stored_result[key] = value

        else:
            assert False

        await self.cmd_async(method='set_result', result=stored_result, silence=self._silence)
        self._result = stored_result

        await self.set_done()

    def check_result(self):
        """
        Check if the result is present.

        Returns
        -------
        str
            State of the task.
        Exception or None
            Exception if task has failed, None otherwise.

        """
        if self._state == 'failed':
            return 'failed', self._exception

        else:
            return self._state, self._result

    async def set_exception(self, exc):
        """
        Set task exception

        Parameters
        ----------
        exc : Exception

        Returns
        -------

        """
        self.state_changed('failed')
        self._exception = exc

        await self.cmd_async(method='set_exception', exc=exc, silence=self._silence)

        try:
            self._exception_future.set_result(True)
        except asyncio.InvalidStateError:
            pass

        # Once done release local copy of the arguments
        self._cleanup()

    async def set_done(self):
        """
        Set task as done.

        Returns
        -------

        """
        self.state_changed('done')

        try:
            self._done_future.set_result(True)
        except asyncio.InvalidStateError:
            pass

        # Once done release local copy of the arguments
        self._cleanup()

    def add_done_callback(self, fun):
        """
        Add done callback.

        Parameters
        ----------
        fun : callable

        Returns
        -------

        """
        self._done_future.add_done_callback(fun)

    def add_exception_callback(self, fun):
        """
        Add exception callback.

        Parameters
        ----------
        fun : callable

        Returns
        -------

        """
        self._exception_future.add_done_callback(fun)

    def _cleanup(self):
        self.args = None
        self.kwargs = None

        self._args_pending = weakref.WeakSet()
        self._kwargs_pending = weakref.WeakSet()

        self._args_value = dict()
        self._kwargs_value = dict()

        self._args_state = dict()
        self._kwargs_state = dict()

    async def _set_arg_done(self, fut, index, arg):
        if not (await self._check_exception(fut, arg)):
            return

        self._arg_size += await arg.size(pending=True)
        self._args_state[index] = 'done'

        try:
            self._args_pending.remove(arg)
        except KeyError:
            pass
        await self._check_ready()

    async def _set_kwarg_done(self, fut, index, arg):
        if not (await self._check_exception(fut, arg)):
            return

        self._arg_size += await arg.size(pending=True)
        self._kwargs_state[index] = 'done'

        try:
            self._kwargs_pending.remove(arg)
        except KeyError:
            pass
        await self._check_ready()

    async def _check_exception(self, fut, arg):
        try:
            exc = fut.exception()
        except asyncio.CancelledError:
            exc = None

        if exc is not None:
            exc = MultiError(exc)

            try:
                raise RuntimeError('Task failed due to failed argument: %s' % arg)
            except Exception as fail:
                exc.add(fail)

            try:
                raise exc
            except MultiError:
                et, ev, tb = sys.exc_info()

            tb = tblib.Traceback(tb)

            await self.set_exception((et, ev, tb))

            try:
                self._ready_future.set_result(True)
            except asyncio.InvalidStateError:
                pass

            return False

        else:
            return True

    async def _check_ready(self):
        if len(self._args_pending) or len(self._kwargs_pending):
            return

        # make sure there's enough memory to pull the arguments
        wait = 1
        while not self.runtime.fits_in_memory(self._arg_size):
            if self.runtime._running_tasks <= 0:
                await asyncio.sleep(wait)
                await self.runtime.maintenance()
                if self.runtime.fits_in_memory(self._arg_size):
                    break

                try:
                    raise MemoryOverflowError('Not enough memory to allocate %d bytes '
                                              'for task %s. Runtime mem limit: %d bytes, '
                                              'mem used: %d' % (self._arg_size, self,
                                                                self.runtime.memory_limit(), memory_used()))
                except MemoryOverflowError:
                    et, ev, tb = sys.exc_info()
                    tb = tblib.Traceback(tb)

                    await self.set_exception((et, ev, tb))

                self._ready_future.set_result(True)
                return

            await asyncio.sleep(wait)
            wait *= 1.2

        self.runtime.inc_committed_mem(self._arg_size)  # reserve memory to pull args
        self.runtime.dec_pending_tasks()
        self.runtime.inc_running_tasks()

        # pull all arguments
        awaitable_args = []

        for index in range(len(self.args)):
            arg = self.args[index]

            if type(arg) in types.awaitable_types:
                self._args_state[index] = 'ready'

                async def _await_arg(_index, _arg):
                    _result = await _arg.result()
                    _attr = self._args_value if not isinstance(_arg, TaskDone) else None
                    return _attr, _index, _result

                awaitable_args.append(
                    _await_arg(index, arg)
                )

        for key, value in self.kwargs.items():
            if type(value) in types.awaitable_types:
                self._kwargs_state[key] = 'ready'

                async def _await_kwarg(_key, _arg):
                    _result = await _arg.result()
                    _attr = self._kwargs_value if not isinstance(_arg, TaskDone) else None
                    return _attr, _key, _result

                awaitable_args.append(
                    _await_kwarg(key, value)
                )

        for task in asyncio.as_completed(awaitable_args):
            attr, key, result = await task
            if attr is not None:
                attr[key] = result

        self.runtime.dec_committed_mem(self._arg_size)  # return reserved memory

        # set task ready
        if not self._ready_future.done():
            self._ready_future.set_result(True)

    async def deregister(self):
        dereg_args = await super().deregister()

        drops = []
        result = self._result
        if isinstance(result, tuple):
            for value in result:
                if isinstance(value, WarehouseObject):
                    drops.append(value.drop())

        elif isinstance(result, dict):
            for value in result.values():
                if isinstance(value, WarehouseObject):
                    drops.append(value.drop())

        await asyncio.gather(*drops)

        return dereg_args


class TaskArray(Task):

    def __init__(self, uid, sender_id, tasks, **kwargs):
        if uid not in tasks:
            uid = list(tasks.keys())[-1]
        self_task = tasks.pop(uid)

        super().__init__(uid, sender_id, self_task['tessera'], self_task['method'],
                         *self_task['args'], **self_task['kwargs'], silence=sender_id)

        self._tasks = {}
        for t_uid, task in tasks.items():
            self._tasks[t_uid] = Task(t_uid, sender_id, task['tessera'], task['method'],
                                      *task['args'], **task['kwargs'], silence=sender_id)
        self._tasks[uid] = self

        self._done_count = {uid: 0 for uid in self._tasks.keys()}
        for t_uid, task in self._tasks.items():
            task.add_done_callback(self._task_done(t_uid))
            task.add_exception_callback(self._task_exception(t_uid))

        self._sender_id = sender_id

    def _task_done(self, uid):
        self_ref = weakref.ref(self)

        def task_done(*args):
            async def _task_done():
                if self_ref()._state == 'failed':
                    return

                self_ref()._done_count[uid] = 1
                if sum(self_ref()._done_count.values()) < len(self_ref()._tasks):
                    return

                stored_result = {}
                for t_uid, task in self_ref()._tasks.items():
                    stored_result[t_uid] = task._result

                await self_ref().cmd_async(method='set_result', result=stored_result,
                                           restrict=[self._sender_id])
                self._tasks = {}

            self_ref().loop.run(_task_done)

        return task_done

    def _task_exception(self, uid):
        self_ref = weakref.ref(self)

        def task_exception(*args):
            async def _task_exception(*args):
                try:
                    exc = self_ref()._tasks[uid]._exception
                except KeyError:
                    return

                for t_uid, task in self_ref()._tasks.items():
                    task.state_changed('failed')
                    task._exception = exc

                    try:
                        task._exception_future.set_result(True)
                    except asyncio.InvalidStateError:
                        pass

                    # Once done release local copy of the arguments
                    task._cleanup()

                await self_ref().cmd_async(method='set_exception', exc=exc,
                                           restrict=[self._sender_id])
                self._tasks = {}

            self_ref().loop.run(_task_exception)

        return task_exception

    @property
    def tasks(self):
        return list(self._tasks.values())

    @property
    def tesseras(self):
        return [task._tessera for task in self._tasks.values()]


class TaskProxy(ProxyBase):
    """
    Proxy pointing to a remote task that has been or will be executed.

    """

    type = 'task_proxy'

    def __init__(self, proxy, method, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._uid = '%s-%s-%s-%s' % ('task',
                                     proxy._cls.cls.__name__.lower(),
                                     method,
                                     uuid.uuid4().hex)
        self._tessera_proxy = proxy

        self._fill_config(**kwargs)
        self._eager = True

        self.method = method
        self.args = args
        self.kwargs = kwargs

        self._state = None
        self._result = None
        self._done_future = Future()
        self._outputs = None
        self._remote = None

        self.state_changed('pending')

        try:
            self.remote_runtime
        except ValueError:
            pass

    async def init(self):
        """
        Asynchronous correlate of ``__init__``.

        Returns
        -------

        """
        self.runtime.register(self)

        task = {
            'tessera_id': self.tessera_id,
            'method': self.method,
            'args': self.args,
            'kwargs': self.kwargs,
        }

        await self.remote_runtime.init_task(task=task, uid=self._uid,
                                            reply=True)

        if self._state == 'pending':
            self.state_changed('queued')

    def deregister_runtime(self, uid):
        if uid != self.runtime_id:
            return

        super().deregister_runtime(uid)

        self.state_changed('failed')

        try:
            self._done_future.set_exception(
                RuntimeDisconnectedError('Remote runtime %s became disconnected' % uid)
            )
        except asyncio.InvalidStateError:
            pass
        else:
            # Once done release local copy of the arguments
            self._cleanup()

    @cached_property
    def runtime_id(self):
        """
        UID of the runtime where the task lives.

        """
        try:
            return self._tessera_proxy.runtime_id
        except ReferenceError:
            return None

    @cached_property
    def tessera_id(self):
        """
        Tessera UID.

        """
        try:
            return self._tessera_proxy.uid
        except ReferenceError:
            return None

    @cached_property
    def remote_runtime(self):
        """
        Proxy to the runtime where the task lives.

        """
        return self._tessera_proxy.remote_runtime

    @property
    def init_future(self):
        """
        Future that will be completed when the remote task is initiated remotely.

        """
        return self._init_future

    @property
    def done_future(self):
        """
        Future that will be completed when the remote task is done.

        """
        return self._done_future

    @classmethod
    def remote_cls(cls):
        """
        Class of the remote.

        """
        return Task

    @property
    def done(self):
        """
        Access to TaskDone of this task.

        """
        return self.outputs.done

    @property
    def outputs(self):
        """
        Access individual outputs of the task.

        """
        if self._outputs is None or self._outputs() is None:
            outputs = TaskOutputGenerator(self)
            self._outputs = weakref.ref(outputs)
        else:
            outputs = self._outputs()

        return outputs

    @property
    def remote(self):
        """
        Execute remote operations on the task.

        """
        if self._remote is None or self._remote() is None:
            remote = TaskRemote(self)
            self._remote = weakref.ref(remote)
        else:
            remote = self._remote()

        return remote

    @property
    def collectable(self):
        """
        Whether the object is ready for collection.

        """
        return self._state in ['failed', 'done']

    def set_done(self):
        """
        Set task as done.

        Returns
        -------

        """
        self.state_changed('done')

        try:
            self._done_future.set_result(True)
        except asyncio.InvalidStateError:
            pass

        # Once done release local copy of the arguments
        self._cleanup()

    def set_result(self, result):
        """
        Set task result.

        Parameters
        ----------
        result

        Returns
        -------

        """
        self._result = result
        self.set_done()

    def set_exception(self, exc):
        """
        Set exception during task execution.

        Parameters
        ----------
        exc : Exception description

        Returns
        -------

        """
        self.state_changed('failed')

        exc = exc[1].with_traceback(exc[2].as_traceback())
        try:
            self._done_future.set_exception(exc)
        except asyncio.InvalidStateError:
            pass
        else:
            # Once done release local copy of the arguments
            self._cleanup()

    def wait(self):
        """
        Wait on the task to be completed.

        Returns
        -------

        """
        return self._done_future.result()

    def add_done_callback(self, fun):
        """
        Add done callback.

        Parameters
        ----------
        fun : callable

        Returns
        -------

        """
        self._done_future.add_done_callback(fun)

    def _cleanup(self):
        self.args = None
        self.kwargs = None
        # Release the strong reference to the tessera proxy once the task is complete
        # so that it can be garbage collected if necessary
        try:
            self._tessera_proxy = weakref.proxy(self._tessera_proxy)
        except TypeError:
            pass

    def add_event(self, event_name, **kwargs):
        kwargs['tessera_id'] = self.tessera_id
        return super().add_event(event_name, **kwargs)

    def add_profile(self, profile, **kwargs):
        kwargs['tessera_id'] = self.tessera_id
        return super().add_profile(profile, **kwargs)

    async def size(self, pending=False):
        """
        Size of the task result in bytes.

        Returns
        -------

        """
        await self

        if hasattr(self, '_retrieved'):
            if pending:
                return 0
            else:
                return sizeof(self._retrieved)

        return await remote_sizeof(self._result, pending=pending)

    async def result(self):
        """
        Gather remote result from the task.

        Returns
        -------
        Task result

        """
        await self

        if hasattr(self, '_retrieved'):
            return self._retrieved

        result = self._result
        if isinstance(result, tuple):
            async def retrieve(_value):
                return await self.runtime.get(_value)

            async def noop(_value):
                return _value

            tasks = []
            for value in result:
                if isinstance(value, WarehouseObject):
                    tasks.append(retrieve(value))
                else:
                    tasks.append(noop(value))

            retrieved = await asyncio.gather(*tasks)
            retrieved = tuple(retrieved)
            if len(retrieved) == 1:
                retrieved = retrieved[0]

        elif isinstance(result, dict):
            async def retrieve(_key, _value):
                return _key, await self.runtime.get(_value)

            async def noop(_key, _value):
                return _key, _value

            tasks = []
            for key, value in result.items():
                if isinstance(value, WarehouseObject):
                    tasks.append(retrieve(key, value))
                else:
                    tasks.append(noop(key, value))

            tasks = await asyncio.gather(*tasks)
            retrieved = {}
            for key, value in tasks:
                retrieved[key] = value

        else:
            assert False

        setattr(self, '_retrieved', retrieved)

        return retrieved

    async def check_result(self):
        """
        Check the remote result.

        Returns
        -------

        """
        if self._state != 'done' and self._state != 'failed':
            state, result = await self.cmd_recv_async(method='check_result')

            if state == 'done':
                self.set_result(result)

            elif state == 'failed':
                self.set_exception(result)

    def __await__(self):
        if not self._eager:
            loop = mosaic.get_event_loop()
            loop.run(self.__init_async__)

        yield from self._done_future.__await__()
        return self

    _serialisation_attrs = ProxyBase._serialisation_attrs + ['_eager', '_tessera_proxy', 'method']

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = super()._deserialisation_helper(state)

        if not hasattr(instance, 'args'):
            instance.args = None
            instance.kwargs = None

        if not hasattr(instance, '_result'):
            instance._result = None
            instance._done_future = Future()
            if instance.state == 'done':
                instance.set_done()

        # Synchronise the task state, in case something has happened between
        # the moment when it was pickled until it has been re-registered on
        # this side
        instance.loop.run(instance.check_result)

        return instance

    async def deregister(self):
        dereg_args = await super().deregister()

        drops = []
        result = self._result
        if isinstance(result, tuple):
            for value in result:
                if isinstance(value, WarehouseObject):
                    drops.append(value.drop())

        elif isinstance(result, dict):
            for value in result.values():
                if isinstance(value, WarehouseObject):
                    drops.append(value.drop())

        await asyncio.gather(*drops)

        return dereg_args


class AnonTaskProxy(TaskProxy):

    @cached_property
    def tessera_id(self):
        """
        Tessera UID.

        """
        return 'tess-anon'


class TaskArrayProxy(TaskProxy):

    def __init__(self, proxy, method, *args, **kwargs):
        super().__init__(proxy, method, *args, **kwargs)

        def _add_dependencies(_args, deps):
            proxies = []
            for dep in _args:
                if isinstance(dep, TaskProxy):
                    proxy = dep
                elif isinstance(dep, (TaskOutput, TaskDone)):
                    proxy = dep._task_proxy
                else:
                    continue
                deps[proxy.uid] = proxy
                proxies.append(proxy)

            return proxies

        def _add_sub_dependencies(proxies, deps):
            for proxy in proxies:
                for runtime_deps in proxy._dependencies.values():
                    for proxy_dep in runtime_deps.values():
                        deps[proxy_dep.uid] = proxy_dep

        self._dependencies = {'_': {}}
        proxies = _add_dependencies(self.args, self._dependencies['_'])
        proxies += _add_dependencies(self.kwargs.values(), self._dependencies['_'])
        _add_sub_dependencies(proxies, self._dependencies['_'])

        self._eager = False

    async def init(self):
        """
        Asynchronous correlate of ``__init__``.

        Returns
        -------

        """
        def _sort_dependencies(proxies, deps):
            for proxy in proxies:
                if proxy.runtime_id not in deps:
                    deps[proxy.runtime_id] = {}
                deps[proxy.runtime_id][proxy.uid] = proxy

        proxies = self._dependencies['_'].values()
        tessera_inits = []
        for proxy in list(proxies) + [self]:
            tessera_inits.append(proxy._tessera_proxy.init_future)

        await asyncio.gather(*tessera_inits)

        self._dependencies = {self.runtime_id: {}}
        _sort_dependencies(proxies, self._dependencies)
        self._dependencies[self.runtime_id][self.uid] = self
        self._dependencies = dict(reversed(list(self._dependencies.items())))

        for runtime_id, proxies in self._dependencies.items():
            proxies = list(proxies.values())

            tasks = {}
            for proxy in proxies:
                self.runtime.register(proxy)
                try:
                    proxy.remote_runtime
                except ValueError:
                    pass

                task = {
                    'tessera_id': proxy.tessera_id,
                    'method': proxy.method,
                    'args': proxy.args,
                    'kwargs': proxy.kwargs,
                }

                tasks[proxy.uid] = task

            await proxies[0].remote_runtime.init_task_array(uid=self.uid, tasks=tasks, reply=True)

            for proxy in proxies:
                if proxy.state == 'pending':
                    proxy.state_changed('queued')

                if proxy.uid == self.uid:
                    continue

                if proxy.init_future.done():
                    exc = proxy.init_future.exception()
                    if exc is not None:
                        raise exc

                proxy.init_future.set_result(True)

    def set_done(self):
        """
        Set task as done.

        Returns
        -------

        """
        for runtime_id, proxies in self._dependencies.items():
            proxies = list(proxies.values())

            for proxy in proxies:
                proxy.state_changed('done')

                try:
                    proxy._done_future.set_result(True)
                except asyncio.InvalidStateError:
                    pass

                # Once done release local copy of the arguments
                proxy._cleanup()

    def set_result(self, result):
        """
        Set task result.

        Parameters
        ----------
        result

        Returns
        -------

        """
        for runtime_id, proxies in self._dependencies.items():
            proxies = list(proxies.values())

            for proxy in proxies:
                try:
                    proxy._result = result[proxy.uid]
                    proxy.set_done()
                except KeyError:
                    pass

    def set_exception(self, exc):
        """
        Set exception during task execution.

        Parameters
        ----------
        exc : Exception description

        Returns
        -------

        """
        exc = exc[1].with_traceback(exc[2].as_traceback())

        for runtime_id, proxies in self._dependencies.items():
            proxies = list(proxies.values())

            for proxy in proxies:
                proxy.state_changed('failed')
                try:
                    proxy._done_future.set_exception(exc)
                except asyncio.InvalidStateError:
                    pass
                else:
                    # Once done release local copy of the arguments
                    proxy._cleanup()

    def _cleanup(self):
        for runtime_id, proxies in self._dependencies.items():
            proxies = list(proxies.values())

            for proxy in proxies:
                if proxy.uid != self.uid:
                    proxy._cleanup()

        super()._cleanup()
        self._dependencies = {}

    @classmethod
    def _deserialisation_helper(cls, state):
        inst = TaskProxy._deserialisation_helper(state)
        inst._eager = True
        return inst


class AnonTaskArrayProxy(TaskArrayProxy):

    @cached_property
    def tessera_id(self):
        """
        Tessera UID.

        """
        return 'tess-anon'


class TaskRemote:
    """
    Class that enables executing methods on remote task outputs,

    """

    def __init__(self, task_proxy):
        self._task_proxy = task_proxy

    async def _init_task(self, task_proxy, *args, **kwargs):
        await self._task_proxy.init_future

        for arg in args:
            if hasattr(arg, 'init_future'):
                await arg.init_future

        for arg in kwargs.values():
            if hasattr(arg, 'init_future'):
                await arg.init_future

        return await task_proxy.__init_async__()

    def _get_remote_method(self, item):
        def remote_method(*args, **kwargs):
            kwargs.pop('runtime', None)

            async def run(*_args, **_kwargs):
                output = _args[0]
                _args = _args[1:]
                f = getattr(output, item)
                if inspect.iscoroutine(f) or inspect.iscoroutinefunction(f):
                    return await f(*_args, **_kwargs)
                else:
                    return f(*_args, **_kwargs)

            args = (mosaic.core.PickleClass(run), self._task_proxy,) + args

            eager = kwargs.pop('eager', False)
            if eager:
                task_proxy = AnonTaskProxy(self._task_proxy._tessera_proxy, 'run', *args, **kwargs)

                loop = mosaic.get_event_loop()
                loop.run(self._init_task, task_proxy, *args, **kwargs)
                # return self._init_task(task_proxy, *args, **kwargs)
            else:
                task_proxy = AnonTaskArrayProxy(self._task_proxy._tessera_proxy, 'run', *args, **kwargs)

            return task_proxy

        return remote_method

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            return self._get_remote_method(item)

    def __getitem__(self, item):
        return self.__getattribute__(item)


class TaskOutputGenerator:
    """
    Class that generates pointers to specific outputs of a remote task,

    """

    def __init__(self, task_proxy):
        self._task_proxy = task_proxy

        self._generated_outputs = weakref.WeakValueDictionary()

    def __repr__(self):
        runtime_id = self._task_proxy.runtime_id

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self._task_proxy.uid, runtime_id, self._task_proxy.state)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            if item not in self._generated_outputs:
                if item == 'done':
                    generated_output = TaskDone(self._task_proxy)
                else:
                    generated_output = TaskOutput(item, self._task_proxy)

                self._generated_outputs[item] = generated_output

            return self._generated_outputs[item]

    def __getitem__(self, item):
        if item not in self._generated_outputs:
            if item == 'done':
                generated_output = TaskDone(self._task_proxy)
            else:
                generated_output = TaskOutput(item, self._task_proxy)

            self._generated_outputs[item] = generated_output

        return self._generated_outputs[item]


class TaskOutputBase(Base):
    """
    Base class for outputs of a task.

    """

    def __init__(self, task_proxy):
        self._task_proxy = task_proxy
        self._result = None

    @property
    def uid(self):
        return self._task_proxy.uid

    @property
    def state(self):
        return self._task_proxy.state

    @cached_property
    def runtime_id(self):
        return self._task_proxy.runtime_id

    @cached_property
    def remote_runtime(self):
        return self._task_proxy.remote_runtime

    @property
    def init_future(self):
        return self._task_proxy.init_future

    @property
    def done_future(self):
        return self._task_proxy.done_future

    def wait(self):
        return self._task_proxy.wait()

    async def result(self):
        pass

    async def size(self, pending=False):
        pass

    def add_done_callback(self, fun):
        self._task_proxy.add_done_callback(fun)

    def __await__(self):
        return (yield from self._task_proxy.__await__())


class TaskOutput(TaskOutputBase):
    """
    Pointer to specific remote output of a class.

    """

    def __init__(self, key, task_proxy):
        super().__init__(task_proxy)

        self._key = key

    def __repr__(self):
        runtime_id = self.runtime_id

        return "<%s object [%s] at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, self._key, id(self),
                self.uid, runtime_id, self.state)

    def _select_result(self, result):
        if not isinstance(result, tuple) and not isinstance(result, dict):
            result = (result,)

        return result[self._key]

    async def size(self, pending=False):
        """
        Size of the task result in bytes.

        Returns
        -------

        """
        await self

        if self._result is not None:
            if pending:
                return 0
            else:
                return sizeof(self._result)
        elif hasattr(self._task_proxy, '_retrieved'):
            if pending:
                return 0
            else:
                result = self._select_result(self._task_proxy._retrieved)
                return sizeof(result)

        result = self._select_result(self._task_proxy._result)
        return await remote_sizeof(result, pending=pending)

    async def result(self):
        """
        Gather output from the remote task.

        Returns
        -------
        Output

        """
        await self

        if self._result is None:
            result = await self._task_proxy.result()
            self._result = self._select_result(result)

        return self._result


class TaskDone(TaskOutputBase):
    """
    Reference to the termination of a remote task.

    """

    def __repr__(self):
        runtime_id = self.runtime_id

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self.state)

    async def size(self, pending=False):
        return 0

    async def result(self):
        """
        Wait for task termination.

        Returns
        -------

        """
        await self

        self._result = True

        return self._result


class MemoryOverflowError(Exception):
    pass


types.awaitable_types += (TaskProxy, TaskArrayProxy, AnonTaskProxy, AnonTaskArrayProxy, TaskOutput, TaskDone)
types.remote_types += (Task, TaskArray)
types.proxy_types += (TaskProxy, TaskArrayProxy, AnonTaskProxy, AnonTaskArrayProxy)
