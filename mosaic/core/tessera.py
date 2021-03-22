
import sys
import uuid
import tblib
import asyncio
import contextlib
import cloudpickle
from cached_property import cached_property

from .task import TaskProxy
from .base import Base, CMDBase, RemoteBase, ProxyBase, MonitoredBase


__all__ = ['Tessera', 'TesseraProxy', 'ArrayProxy', 'MonitoredTessera', 'tessera']


class Tessera(RemoteBase):
    """
    A tessera is an actor in the mosaic parallelism model.

    A tessera represents an object that is instantiated in a remote portion of
    the network, and which we reference through a proxy. This proxy
    allows us to execute methods on that remote object simply by calling the
    method.

    A tessera is kept in memory at the worker for as long as there are proxy references to
    it. If none exist, it will be made available for garbage collection.

    Objects of class Tessera should not be instantiated directly by the user
    and the ``@mosaic.tessera`` decorator should be used instead.

    Parameters
    ----------
    cls : type
        Class of the remote object.
    uid : str
        UID assigned to the tessera.

    """

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
        """
        Proxies that have references to this tessera.

        """
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
        """
        Add a task to the queue of the tessera.

        Parameters
        ----------
        task : Task

        Returns
        -------

        """
        self._task_queue.put_nowait(task)

    def listen(self, wait=False):
        """
        Start the listening loop that consumes tasks.

        Parameters
        ----------
        wait : bool, optional
            Whether or not to wait for the loop end, defaults to False.

        Returns
        -------

        """
        if self._state != 'init':
            return

        self.loop.run(self.listen_async, wait=wait)

    async def listen_async(self):
        """
        Listening loop that consumes tasks from the tessera queue.

        Returns
        -------

        """
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
        """
        Call a method while handling exceptions, which will be sent back to the
        sender if they arise.

        Parameters
        ----------
        sender_id : str
            UID of the original caller.
        method : callable
            Method to execute.
        task : Task
            Task that has asked for the execution of the method.

        Returns
        -------

        """
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
        """
        Context manager that handles exceptions by sending them
        back to the ``uid``.

        Parameters
        ----------
        sender_id : str
            Remote UID.
        method : callable
            Method being executed.
        task : Task
            Task that has asked for the execution of the method.

        Returns
        -------

        """
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
        """
        Signal state changed.

        Parameters
        ----------
        state : str
            New state.

        Returns
        -------

        """
        self._state = state
        await self.runtime.tessera_state_changed(self)

    def __del__(self):
        self.logger.debug('Garbage collected object %s' % self)
        self.loop.run(self.state_changed, args=('collected',))


class TesseraProxy(ProxyBase):
    """
    Objects of this class represent connections to remote tessera, allowing us to
    call methods on them.

    Objects of class TesseraProxy should not be instantiated directly by the user
    and the ``@mosaic.tessera`` decorator should be used instead.

    Parameters
    ----------
    cls : type
        Class of the remote object.
    args : tuple, optional
        Arguments for the instantiation of the remote tessera.
    kwargs : optional
        Keyword arguments for the instantiation of the remote tessera.

    """

    type = 'tessera_proxy'

    def __init__(self, cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cls = PickleClass(cls)
        self._runtime_id = None

        self._uid = '%s-%s-%s' % ('tess',
                                  self._cls.__name__.lower(),
                                  uuid.uuid4().hex)

    async def init(self, *args, **kwargs):
        """
        Asynchronous correlate of ``__init__``.

        """
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
        """
        UID of the runtime where the tessera lives.

        """
        return self._runtime_id

    @cached_property
    def remote_runtime(self):
        """
        Proxy to the runtime where the tessera lives.

        """
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
    """
    Objects of this class represent more a set of remote tesserae that may live on one or
    more remote runtimes. An array proxy allows us to reference all of them together
    through a common interface, as well as map calls to them.

    Objects of class ArrayProxy should not be instantiated directly by the user
    and the ``@mosaic.tessera`` decorator should be used instead.

    Parameters
    ----------
    cls : type
        Class of the remote object.
    args : tuple, optional
        Arguments for the instantiation of the remote tessera.
    len : int, optional
        Length of the array, defaults to 1.
    kwargs : optional
        Keyword arguments for the instantiation of the remote tessera.

    """

    type = 'tessera_proxy_array'

    def __init__(self, cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cls = PickleClass(cls)
        self._len = kwargs.pop('len', 1)

        self._state = 'pending'

        self._proxies = []
        self._runtime_id = []
        for _ in range(self._len):
            proxy = TesseraProxy(cls, *args, **kwargs)
            self._proxies.append(proxy)

        self._uid = '%s-%s-%s' % ('array',
                                  self._cls.__name__.lower(),
                                  uuid.uuid4().hex)

    async def init(self, *args, **kwargs):
        """
        Asynchronous correlate of ``__init__``.

        """
        for proxy in self._proxies:
            await proxy.init(*args, **kwargs)
            self._runtime_id.append(proxy.runtime_id)

        self.runtime.register(self)

        self._state = 'listening'

    @property
    def runtime_id(self):
        """
        UID of the runtime where the tessera lives.

        """
        return self._runtime_id

    @cached_property
    def remote_runtime(self):
        """
        Proxy to the runtime where the tessera lives.

        """
        return [self.proxy(each) for each in self._runtime_id]

    @classmethod
    def remote_type(cls):
        """
        Type of mosaic object.

        """
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

        yield proxy

        await proxy_queue.put(proxy)

    async def map(self, fun, elements, *args, **kwargs):
        """
        Map a function to an iterable, distributed across the proxies
        of the proxy array.

        The function is given control over a certain proxy for as long as
        it takes to be executed. Once all mappings have completed, the
        results are returned together

        Parameters
        ----------
        fun : callable
            Function to execute
        elements : iterable
            Iterable to map.
        args : tuple, optional
            Arguments to the function.
        kwargs : optional
            Keyword arguments to the function.

        Returns
        -------
        list
            Results of the mapping.

        """
        tasks = await self._map_tasks(fun, elements, *args, **kwargs)

        return await asyncio.gather(*tasks)

    async def map_as_completed(self, fun, elements, *args, **kwargs):
        """
        Generator which maps a function to an iterable,
        distributed across the proxies of the proxy array.

        The function is given control over a certain proxy for as long as
        it takes to be executed. Once a function is completed, the result
        to that function is yielded immediately.

        Parameters
        ----------
        fun : callable
            Function to execute
        elements : iterable
            Iterable to map.
        args : tuple, optional
            Arguments to the function.
        kwargs : optional
            Keyword arguments to the function.

        Returns
        -------
        object
            Result of each execution as they are completed.

        """
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
    """
    Information container on the state of a tessera.

    """
    pass


class PickleClass:
    """
    A wrapper for a class that can be pickled safely.

    Parameters
    ----------
    cls : type
        Class to wrap.

    """

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
    """
    Decorator that transforms a standard class into a tessera-capable class.

    The resulting class can still be instantiated as usual ``Klass(...)``, which
    will generate a standard local instance, or onto the mosaic runtime ``await Klass.remote(...)``,
    which will instantiate the class in a remote endpoint and return a proxy to the user.

    TODO - Better explanations and more examples.

    Parameters
    ----------

    Returns
    -------
    Enriched class

    Examples
    --------

    >>> @tessera
    >>> class Klass:
    >>>     def __init__(self, value):
    >>>         self.value = value
    >>>
    >>>     def add(self, other):
    >>>         self.value += other
    >>>         return self.value
    >>>
    >>> # We can still generate a standard local instance
    >>> local_instance = Klass(10)
    >>>
    >>> # but also a remote instance by invoking remote.
    >>> remote_proxy = await Klass.remote(10)
    >>>
    >>> # The resulting proxy can be used to call the instance methods,
    >>> task = await remote_proxy.add(5)
    >>> # which will return immediately.
    >>>
    >>> # We can do some work while the remote method is executed
    >>> # and then wait for it to end
    >>> await task
    >>>
    >>> # We can retrieve the result of the task invoking result on the task
    >>> await task.result()
    15

    """

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
