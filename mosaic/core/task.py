
import uuid
import time
import weakref
import operator
from cached_property import cached_property

from .base import Base, RemoteBase, ProxyBase, MonitoredBase
from ..utils import Future


__all__ = ['Task', 'TaskProxy', 'TaskOutputGenerator', 'TaskOutput', 'TaskDone', 'MonitoredTask']


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

        kwargs = self._fill_config(**kwargs)

        self.method = method
        self.args = args
        self.kwargs = kwargs

        self._tic = None
        self._elapsed = None

        self._args_pending = set()
        self._kwargs_pending = set()

        self._args_value = dict()
        self._kwargs_value = dict()

        self._args_state = dict()
        self._kwargs_state = dict()

        self._ready_future = Future()
        self._result = None
        self._exception = None

        self._state = 'init'
        self.runtime.register(self)
        self.register_proxy(self._sender_id)

    @property
    def sender_id(self):
        """
        Caller UID.

        """
        return self._sender_id

    @cached_property
    def remote_runtime(self):
        """
         Proxies that have references to this task.

        """
        return {self.proxy(each) for each in list(self._proxies)}

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

    def get_result(self, key=None):
        """
        Get task result.

        Parameters
        ----------
        key : optional
            Access particular item within the result, defaults to None.

        Returns
        -------

        """
        if self._state == 'failed':
            raise Exception('Tried to get the result on failed task %s' % self._uid)

        if self._state != 'done':
            raise Exception('Tried to get result of task not done, this should never happen!')

        if key is None:
            return self._result

        else:
            result = self._result

            if not isinstance(result, tuple) and not isinstance(result, dict):
                result = (result,)

            return result[key]

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
            return self._state, None

    def _cleanup(self):
        self.args = None
        self.kwargs = None

        self._args_pending = weakref.WeakSet()
        self._kwargs_pending = weakref.WeakSet()

        self._args_value = dict()
        self._kwargs_value = dict()

        self._args_state = dict()
        self._kwargs_state = dict()

    # TODO Await all of the remote results together using gather
    async def prepare_args(self):
        """
        Prepare the arguments of the task for execution.

        Returns
        -------
        Future

        """
        waitable_types = [TaskProxy, TaskOutput, TaskDone]

        for index in range(len(self.args)):
            arg = self.args[index]

            if type(arg) in waitable_types:
                self._args_state[index] = arg.state

                if arg.state != 'done':
                    if not isinstance(arg, TaskDone):
                        self._args_value[index] = None
                    self._args_pending.add(arg)

                    def callback(_index, _arg):
                        def _callback(fut):
                            self.loop.run(self._set_arg_done, args=(_index, _arg))

                        return _callback

                    arg.add_done_callback(callback(index, arg))

                else:
                    result = await arg.result()
                    if not isinstance(arg, TaskDone):
                        self._args_value[index] = result

            else:
                self._args_state[index] = 'ready'
                self._args_value[index] = arg

        for key, value in self.kwargs.items():
            if type(value) in waitable_types:
                self._kwargs_state[key] = value.state

                if value.state != 'done':
                    if not isinstance(value, TaskDone):
                        self._kwargs_value[key] = None
                    self._kwargs_pending.add(value)

                    def callback(_key, _arg):
                        def _callback(fut):
                            self.loop.run(self._set_kwarg_done, args=(_key, _arg))

                        return _callback

                    value.add_done_callback(callback(key, value))

                else:
                    result = await value.result()
                    if not isinstance(value, TaskDone):
                        self._kwargs_value[key] = result

            else:
                self._kwargs_state[key] = 'ready'
                self._kwargs_value[key] = value

        await self._check_ready()

        return self._ready_future

    async def set_exception(self, exc):
        """
        Set task exception

        Parameters
        ----------
        exc : Exception

        Returns
        -------

        """
        await self.state_changed('failed')
        self._exception = exc

        await self.cmd_async(method='set_exception', exc=exc)

        # Once done release local copy of the arguments
        self._cleanup()

    async def set_done(self):
        """
        Set task as done.

        Returns
        -------

        """
        await self.state_changed('done')

        await self.cmd_async(method='set_done')

        # Once done release local copy of the arguments
        self._cleanup()

    async def _set_arg_done(self, index, arg):
        result = await arg.result()

        self._args_state[index] = 'ready'
        if not isinstance(arg, TaskDone):
            self._args_value[index] = result

        try:
            self._args_pending.remove(arg)
        except KeyError:
            pass
        await self._check_ready()

    async def _set_kwarg_done(self, index, arg):
        result = await arg.result()

        self._kwargs_state[index] = 'ready'
        if not isinstance(arg, TaskDone):
            self._kwargs_value[index] = result

        try:
            self._kwargs_pending.remove(arg)
        except KeyError:
            pass
        await self._check_ready()

    async def _check_ready(self):
        if not len(self._args_pending) and not len(self._kwargs_pending):
            await self.state_changed('ready')
            self._ready_future.set_result(True)

    async def state_changed(self, state):
        """
        Signal change in task state.

        Parameters
        ----------
        state : str
            New state of the task.

        Returns
        -------

        """
        self._state = state

        if state == 'running':
            self._tic = time.time()

        elapsed = None
        if state == 'done' or state == 'failed':
            self._elapsed = elapsed = time.time() - self._tic

        await self.runtime.task_state_changed(self, elapsed=elapsed)

    def __del__(self):
        self.logger.debug('Garbage collected object %s' % self)
        self.loop.run(self.state_changed, args=('collected',))


class TaskProxy(ProxyBase):
    """
    Proxy pointing to a remote task that has been or will be executed.

    """

    type = 'task_proxy'

    def __init__(self, proxy, method, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._uid = '%s-%s-%s' % ('task',
                                  method,
                                  uuid.uuid4().hex)
        self._tessera_proxy = proxy

        self._fill_config(**kwargs)

        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.outputs = TaskOutputGenerator(self)

        self._state = 'pending'
        self._result = None
        self._done_future = Future()

    async def init(self):
        """
        Asynchronous correlate of ``__init__``.

        Returns
        -------

        """
        await self.monitor.init_task(uid=self._uid,
                                     tessera_id=self._tessera_proxy.uid,
                                     runtime_id=self.runtime_id)

        self.runtime.register(self)

        task = {
            'tessera_id': self._tessera_proxy.uid,
            'method': self.method,
            'args': self.args,
            'kwargs': self.kwargs,
        }

        await self.remote_runtime.init_task(task=task, uid=self._uid,
                                            reply=True)

        self._state = 'queued'

    @property
    def runtime_id(self):
        """
        UID of the runtime where the task lives.

        """
        return self._tessera_proxy.runtime_id

    @cached_property
    def remote_runtime(self):
        """
        Proxy to the runtime where the task lives.

        """
        return self._tessera_proxy.remote_runtime

    @property
    def done_future(self):
        """
        Future that will be completed when the remote task is done.

        """
        return self._done_future

    def set_done(self):
        """
        Set task as done.

        Returns
        -------

        """
        self._state = 'done'

        self._done_future.set_result(True)

        # Once done release local copy of the arguments
        self._cleanup()

    def set_exception(self, exc):
        """
        Set exception during task execution.

        Parameters
        ----------
        exc : Exception description

        Returns
        -------

        """
        self._state = 'failed'

        exc = exc[1].with_traceback(exc[2].as_traceback())
        self._done_future.set_exception(exc)

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
        self._tessera_proxy = weakref.proxy(self._tessera_proxy)

    async def result(self):
        """
        Gather remote result from the task.

        Returns
        -------
        Task result

        """
        await self

        if self._result is not None:
            return self._result

        self._result = await self.cmd_recv_async(method='get_result')

        return self._result

    async def check_result(self):
        """
        Check the remote result.

        Returns
        -------

        """
        if self._state != 'done' and self._state != 'failed':
            state, exc = await self.cmd_recv_async(method='check_result')

            if state == 'done':
                self.set_done()

            elif state == 'failed':
                self.set_exception(exc)

    def __await__(self):
        return (yield from self._done_future.__await__())

    _serialisation_attrs = ProxyBase._serialisation_attrs + ['_tessera_proxy', 'method']

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

        # TODO Unsure about the need for this
        # Synchronise the task state, in case something has happened between
        # the moment when it was pickled until it has been re-registered on
        # this side
        instance.loop.run(instance.check_result)

        return instance


class TaskOutputGenerator:
    """
    Class that generates pointers to specific outputs of a remote task,

    """

    def __init__(self, task_proxy):
        self._task_proxy = weakref.ref(task_proxy)

        self._generated_outputs = weakref.WeakValueDictionary()

    def __repr__(self):
        runtime_id = self._task_proxy().runtime_id

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self._task_proxy().uid, runtime_id, self._task_proxy().state)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            if item not in self._generated_outputs:
                if item == 'done':
                    generated_output = TaskDone(self._task_proxy())
                else:
                    generated_output = TaskOutput(item, self._task_proxy())

                self._generated_outputs[item] = generated_output

            return self._generated_outputs[item]

    def __getitem__(self, item):
        if item not in self._generated_outputs:
            if item == 'done':
                generated_output = TaskDone(self._task_proxy())
            else:
                generated_output = TaskOutput(item, self._task_proxy())

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

    @property
    def runtime_id(self):
        return self._task_proxy.runtime_id

    @cached_property
    def remote_runtime(self):
        return self._task_proxy.remote_runtime

    @property
    def done_future(self):
        return self._task_proxy.done_future

    def wait(self):
        return self._task_proxy.wait()

    async def result(self):
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

    async def result(self):
        """
        Gather output from the remote task.

        Returns
        -------
        Output

        """
        await self

        if self._result is None and self._task_proxy._result is not None:
            self._result = self._select_result(self._task_proxy._result)

        if self._result is not None:
            return self._result

        self._result = await self._task_proxy.cmd_recv_async(method='get_result', key=self._key)

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

    async def result(self):
        """
        Wait for task termination.

        Returns
        -------

        """
        await self

        self._result = True

        return self._result


class MonitoredTask(MonitoredBase):
    """
    Information container on the state of a task.

    """

    def __init__(self, uid, tessera_id, runtime_id):
        super().__init__(uid, runtime_id)

        self.tessera_id = tessera_id
        self.elapsed = None
