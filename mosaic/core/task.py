
import uuid
import time
import weakref
import operator
from cached_property import cached_property

from .base import Base, RemoteBase, ProxyBase, MonitoredBase
from ..utils import Future


__all__ = ['Task', 'TaskOutputGenerator', 'TaskOutput', 'TaskDone', 'MonitoredTask']


class Task(RemoteBase):

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

        self._args_pending = weakref.WeakSet()
        self._kwargs_pending = weakref.WeakSet()

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
        return self._sender_id

    @cached_property
    def remote_runtime(self):
        return {self.proxy(each) for each in list(self._proxies)}

    def _fill_config(self, **kwargs):
        kwargs['max_retries'] = kwargs.get('max_retries', self._tessera.max_retries)

        return super()._fill_config(**kwargs)

    def args_value(self):
        args = [value for key, value in sorted(self._args_value.items(), key=operator.itemgetter(1))]

        return args

    def kwargs_value(self):
        return self._kwargs_value

    def set_result(self, result):
        self._result = result

    def get_result(self, key=None):
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
        waitable_types = [TaskProxy, TaskOutput, TaskDone]

        for index in range(len(self.args)):
            arg = self.args[index]

            if type(arg) in waitable_types:
                self._args_state[index] = arg.state

                if arg.state != 'done':
                    if not isinstance(arg, TaskDone):
                        self._args_value[index] = None
                    self._args_pending.add(arg)

                    def callback(fut):
                        self.loop.run(self._set_arg_done, args=(index, arg))

                    arg.add_done_callback(callback)

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

                    def callback(fut):
                        self.loop.run(self._set_kwarg_done, args=(key, value))

                    value.add_done_callback(callback)

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
        await self.state_changed('failed')
        self._exception = exc

        await self.cmd_async(method='set_exception', exc=exc)

        # Once done release local copy of the arguments
        self._cleanup()

    async def set_done(self):
        await self.state_changed('done')

        await self.cmd_async(method='set_done')

        # Once done release local copy of the arguments
        self._cleanup()

    async def _set_arg_done(self, index, arg):
        result = await arg.result()

        self._args_state[index] = 'ready'
        if not isinstance(arg, TaskDone):
            self._args_value[index] = result

        self._args_pending.remove(arg)
        await self._check_ready()

    async def _set_kwarg_done(self, index, arg):
        result = await arg.result()

        self._kwargs_state[index] = 'ready'
        if not isinstance(arg, TaskDone):
            self._kwargs_value[index] = result

        self._kwargs_pending.remove(arg)
        await self._check_ready()

    async def _check_ready(self):
        if not len(self._args_pending) and not len(self._kwargs_pending):
            await self.state_changed('ready')
            self._ready_future.set_result(True)

    async def state_changed(self, state):
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
        return self._tessera_proxy.runtime_id

    @cached_property
    def remote_runtime(self):
        return self._tessera_proxy.remote_runtime

    @property
    def done_future(self):
        return self._done_future

    def set_done(self):
        self._state = 'done'

        self._done_future.set_result(True)

        # Once done release local copy of the arguments
        self._cleanup()

    def set_exception(self, exc):
        self._state = 'failed'

        exc = exc[1].with_traceback(exc[2].as_traceback())
        self._done_future.set_exception(exc)

        # Once done release local copy of the arguments
        self._cleanup()

    def wait(self):
        return self._done_future.result()

    def add_done_callback(self, fun):
        self._done_future.add_done_callback(fun)

    def check_result(self):
        self.loop.run(self.check_result_async, wait=True)

    def _cleanup(self):
        self.args = None
        self.kwargs = None
        # Release the strong reference to the tessera proxy once the task is complete
        # so that it can be garbage collected if necessary
        self._tessera_proxy = weakref.proxy(self._tessera_proxy)

    async def result(self):
        await self

        if self._result is not None:
            return self._result

        self._result = await self.cmd_recv_async(method='get_result')

        return self._result

    async def check_result_async(self):
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
        # instance.check_result()

        return instance


class TaskOutputGenerator:

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
        await self

        if self._result is None and self._task_proxy._result is not None:
            self._result = self._select_result(self._task_proxy._result)

        if self._result is not None:
            return self._result

        self._result = await self._task_proxy.cmd_recv_async(method='get_result', key=self._key)

        return self._result


class TaskDone(TaskOutputBase):

    def __repr__(self):
        runtime_id = self.runtime_id

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self.state)

    async def result(self):
        await self

        self._result = True

        return self._result


class MonitoredTask(MonitoredBase):

    def __init__(self, uid, tessera_id, runtime_id):
        super().__init__(uid, runtime_id)

        self.tessera_id = tessera_id
        self.elapsed = None
