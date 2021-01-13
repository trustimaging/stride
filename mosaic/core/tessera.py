
import sys
import uuid
import tblib
import asyncio
import contextlib
import cloudpickle
from cached_property import cached_property

from .task import TaskProxy
from .base import Base, CMDBase, RemoteBase, ProxyBase, MonitoredBase


__all__ = ['Tessera', 'TesseraProxy', 'MonitoredTessera', 'tessera']


class Tessera(RemoteBase):

    type = 'tessera'

    def __init__(self, cls, uid, *args, **kwargs):
        super().__init__(uid, *args, **kwargs)
        kwargs = self._fill_config(**kwargs)

        self._state = 'init'
        self._task_queue = asyncio.Queue()
        self._task_lock = asyncio.Lock()

        self._cls = cls
        self._init_cls(*args, **kwargs)

        self.runtime.register(self)
        self.listen(wait=False)

    @cached_property
    def remote_runtime(self):
        return {self.proxy(each) for each in list(self._proxies)}

    def _init_cls(self, *args, **kwargs):
        try:
            self._obj = self._cls(*args, **kwargs)

        except Exception:
            self.retries += 1

            if self.retries > self.max_retries:
                raise

            else:
                self.logger.error('Tessera %s failed, attempting '
                                  'retry %d out of %s' % (self.uid,
                                                          self.retries, self.max_retries))

                return self._init_cls(*args, **kwargs)

    def queue_task(self, task):
        self._task_queue.put_nowait(task)

    def listen(self, wait=False):
        if self._state != 'init':
            return

        self.loop.run(self.listen_async, wait=wait)

    async def listen_async(self):
        if self._state != 'init':
            return

        while True:
            await self.state_changed('listening')

            sender_id, task = await self._task_queue.get()

            if type(task) is str and task == 'stop':
                break

            await asyncio.sleep(0)
            future = await task.prepare_args()
            await future

            method = getattr(self._obj, task.method, False)

            async with self.send_exception(task=task):
                if method is False:
                    raise AttributeError('Class %s does not have method %s' % (self._obj.__class__.__name__,
                                                                               task.method))

                if not callable(method):
                    raise ValueError('Method %s of class %s is not callable' % (task.method,
                                                                                self._obj.__class__.__name__))

            await asyncio.sleep(0)
            await self.state_changed('running')
            await task.state_changed('running')
            await self.call_safe(sender_id, method, task)

            # Make sure that the loop does not keep implicit references to the task until the
            # next task arrives in the queue
            self._task_queue.task_done()
            del task

    async def call_safe(self, sender_id, method, task):
        async with self._task_lock:
            async with self.send_exception(sender_id, method, task):
                future = self.loop.run_in_executor(method,
                                                   args=task.args_value(),
                                                   kwargs=task.kwargs_value())
                result = await future
                # TODO Dodgy
                await asyncio.sleep(0.1)

                task.set_result(result)
                await task.set_done()

    @contextlib.asynccontextmanager
    async def send_exception(self, sender_id=None, method=None, task=None):
        try:
            yield

        except Exception:
            task.retries += 1

            if task.retries > task.max_retries or sender_id is None:
                et, ev, tb = sys.exc_info()
                tb = tblib.Traceback(tb)

                await task.set_exception((et, ev, tb))

            else:
                self.logger.error('Task %s at %s failed, attempting '
                                  'retry %d out of %s' % (task.uid, self.uid,
                                                          task.retries, task.max_retries))
                await self.call_safe(sender_id, method, task)

        finally:
            pass

    async def state_changed(self, state):
        self._state = state
        await self.runtime.tessera_state_changed(self)

    def __del__(self):
        self.logger.debug('Garbage collected object %s' % self)
        self.loop.run(self.state_changed, args=('collected',))


class TesseraProxy(ProxyBase):

    type = 'tessera_proxy'

    def __init__(self, cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cls = PickleClass(cls)
        self._runtime_id = None

        self._uid = '%s-%s-%s' % ('tess',
                                  self._cls.__name__.lower(),
                                  uuid.uuid4().hex)

    async def init(self, *args, **kwargs):
        kwargs = self._fill_config(**kwargs)

        self._runtime_id = await self.monitor.select_worker(reply=True)

        self._state = 'pending'
        await self.monitor.init_tessera(uid=self._uid,
                                        runtime_id=self._runtime_id)

        self.runtime.register(self)
        await self.remote_runtime.init_tessera(cls=self._cls, uid=self._uid, args=args,
                                               reply=True, **kwargs)
        self._state = 'listening'

    @property
    def runtime_id(self):
        return self._runtime_id

    @cached_property
    def remote_runtime(self):
        return self.proxy(self._runtime_id)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:

            if not hasattr(self._cls, item):
                raise AttributeError('Class %s does not have method %s' % (self._cls.__name__, item))

            if not callable(getattr(self._cls, item)):
                raise ValueError('Method %s of class %s is not callable' % (item, self._cls.__name__))

            async def remote_method(*args, **kwargs):
                task_proxy = TaskProxy(self, item, *args, **kwargs)
                await task_proxy.init()

                return task_proxy

            return remote_method

    def __getitem__(self, item):
        return self.__getattribute__(item)

    _serialisation_attrs = ProxyBase._serialisation_attrs + ['_cls', '_runtime_id']


class ArrayProxy(CMDBase):

    type = 'tessera_proxy_array'

    def __init__(self, cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cls = PickleClass(cls)
        self._len = kwargs.pop('len', 1)

        self._state = 'pending'

        self._proxies = []
        self._locks = []
        self._runtime_id = []
        for _ in range(self._len):
            proxy = TesseraProxy(cls, *args, **kwargs)
            self._proxies.append(proxy)
            self._locks.append(asyncio.Lock())

        self._uid = '%s-%s-%s' % ('array',
                                  self._cls.__name__.lower(),
                                  uuid.uuid4().hex)

    async def init(self, *args, **kwargs):
        for proxy in self._proxies:
            await proxy.init(*args, **kwargs)
            self._runtime_id.append(proxy.runtime_id)

        self.runtime.register(self)

        self._state = 'listening'

    @property
    def runtime_id(self):
        return self._runtime_id

    @cached_property
    def remote_runtime(self):
        return [self.proxy(each) for each in self._runtime_id]

    @classmethod
    def remote_type(cls):
        return cls.type.split('_')[0]

    def _remotes(self):
        return self.remote_runtime

    async def _map_tasks(self, fun, elements, *args, **kwargs):
        proxy_queue = asyncio.Queue()
        for proxy in self._proxies:
            await proxy_queue.put(proxy)

        async def call(_element):
            async with self._proxy(proxy_queue) as _proxy:
                res = await fun(_element, _proxy, *args, **kwargs)

            return res

        tasks = [call(element) for element in elements]

        return tasks

    @contextlib.asynccontextmanager
    async def _proxy(self, proxy_queue):
        proxy = await proxy_queue.get()

        try:
            yield proxy

        finally:
            await proxy_queue.put(proxy)

    async def map(self, fun, elements, *args, **kwargs):
        tasks = await self._map_tasks(fun, elements, *args, **kwargs)

        return await asyncio.gather(*tasks)

    async def map_as_completed(self, fun, elements, *args, **kwargs):
        tasks = await self._map_tasks(fun, elements, *args, **kwargs)

        for task in asyncio.as_completed(tasks):
            res = await task
            yield res

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:

            if not hasattr(self._cls, item):
                raise AttributeError('Class %s does not have method %s' % (self._cls.__name__, item))

            if not callable(getattr(self._cls, item)):
                raise ValueError('Method %s of class %s is not callable' % (item, self._cls.__name__))

            async def remote_method(*args, **kwargs):
                tasks = []
                for proxy in self._proxies:
                    tasks.append(proxy[item](*args, **kwargs))

                return await asyncio.gather(*tasks)

            return remote_method

    def __getitem__(self, item):
        return self._proxies[item]

    def __repr__(self):
        runtime_id = ', '.join([str(each) for each in self.runtime_id])

        return "<%s object at %s, uid=%s, runtime=(%s), state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self._state)

    _serialisation_attrs = CMDBase._serialisation_attrs + []


class MonitoredTessera(MonitoredBase):
    pass


class PickleClass:

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            return getattr(self.cls, item)

    def _serialisation_helper(self):
        state = {
            'cls': cloudpickle.dumps(self.cls)
        }

        return state

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = cls.__new__(cls)
        instance.cls = cloudpickle.loads(state['cls'])

        return instance

    def __reduce__(self):
        state = self._serialisation_helper()
        return self._deserialisation_helper, (state,)


def tessera(*args, **cmd_config):

    def tessera_wrapper(cls):

        @classmethod
        async def remote(_, *args, **kwargs):
            kwargs.update(cmd_config)

            array_len = kwargs.pop('len', None)

            if array_len is None:
                proxy = TesseraProxy(cls, *args, **kwargs)
                await proxy.init(*args, **kwargs)

            else:
                proxy = ArrayProxy(cls, *args, len=array_len, **kwargs)
                await proxy.init(*args, **kwargs)

            return proxy

        @classmethod
        def tessera(_, *args, **kwargs):
            kwargs.update(cmd_config)
            return Tessera(cls, *args, **kwargs)

        cls.remote = remote
        cls.tessera = tessera

        method_list = [func for func in dir(Base) if not func.startswith("__")]
        for method in method_list:
            setattr(cls, method, getattr(Base, method))

        return cls

    if len(args) == 1 and len(cmd_config) == 0 and callable(args[0]):
        return tessera_wrapper(args[0])

    if len(args) != 0 or len(cmd_config) < 1:
        raise ValueError('@tessera should be applied to a class without brackets'
                         'or with configuration options within brackets.')

    return tessera_wrapper
