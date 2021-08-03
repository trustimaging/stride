
import sys
import uuid
import tblib
import asyncio
import inspect
import weakref
import functools
import contextlib
import cloudpickle
from cached_property import cached_property

import mosaic
from .task import TaskProxy
from .base import Base, CMDBase, RemoteBase, ProxyBase, MonitoredBase
from ..utils.event_loop import AwaitableOnly


__all__ = ['Tessera', 'TesseraProxy', 'ArrayProxy', 'MonitoredTessera', 'tessera']


def _extract_methods(cls, exclude):
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    cls_methods = [method for method in methods if not method[0].startswith('__')
                   and method[0] not in exclude]
    cls_magic_methods = [method for method in methods if method[0].startswith('__')
                         and method[0] not in exclude]

    cls_method_names = [method[0] for method in cls_methods]
    cls_magic_method_names = [method[0] for method in cls_magic_methods]

    return cls_methods, cls_magic_methods, cls_method_names, cls_magic_method_names


def _extract_attrs(cls, exclude):
    attrs = inspect.getmembers(cls, predicate=lambda x: not inspect.isfunction(x) and not inspect.ismethod(x))
    cls_attrs = [attr for attr in attrs if attr[0] not in exclude]
    cls_attrs_names = [attr[0] for attr in cls_attrs]

    return cls_attrs, cls_attrs_names


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
        self._is_parameter = False
        self._task_queue = asyncio.Queue()
        self._task_lock = asyncio.Lock()

        self._cls = cls
        self._obj = None

        self._cls_methods = []
        self._cls_magic_methods = []
        self._cls_method_names = []
        self._cls_magic_method_names = []
        self._cls_attrs = []

        self._init_cls(*args, **kwargs)
        self._set_cls()

        self.runtime.register(self)
        self.listen()

    def _set_cls(self):
        obj = self.obj
        exclude = dir(self)

        cls_methods, cls_magic_methods, \
            cls_method_names, cls_magic_method_names = _extract_methods(obj, exclude)
        cls_attrs, cls_attr_names = _extract_attrs(obj, exclude)

        self._cls_methods = cls_methods
        self._cls_magic_methods = cls_magic_methods
        self._cls_attrs = cls_attrs

        self._cls_method_names = cls_method_names
        self._cls_magic_method_names = cls_magic_method_names
        self._cls_attr_names = cls_attr_names

    @cached_property
    def remote_runtime(self):
        """
        Proxies that have references to this tessera.

        """
        return {self.proxy(each) for each in list(self._proxies)}

    @classmethod
    def remote_cls(cls):
        """
        Class of the remote.

        """
        return TesseraProxy

    @property
    def obj(self):
        """
        Internal object.

        """
        if self.is_parameter:
            return self._obj()
        else:
            return self._obj

    @property
    def is_parameter(self):
        return self._is_parameter

    def _init_cls(self, *args, **kwargs):
        try:
            self._obj = self._cls(*args, **kwargs)
            setattr(self._obj, '_tessera', self)

        except Exception:
            self.retries += 1

            if self.retries > self.max_retries:
                raise

            else:
                self.logger.error('Tessera %s failed, attempting '
                                  'retry %d out of %s' % (self.uid,
                                                          self.retries, self.max_retries))

                return self._init_cls(*args, **kwargs)

    def make_parameter(self):
        """
        Transform the tessera into a parameter and return the object.

        Returns
        -------
        object

        """
        self._is_parameter = True

        obj = self._obj
        self._obj = weakref.ref(obj, self._dec_parameter_ref())

        return obj

    def get_attr(self, item):
        """
        Access attributes from the underlying object.

        Parameters
        ----------
        item : str
            Name of the attribute.

        Returns
        -------

        """
        return getattr(self.obj, item)

    def set_attr(self, item, value):
        """
        Set attributes on the underlying object.

        Parameters
        ----------
        item : str
            Name of the attribute.
        value : object

        Returns
        -------

        """
        return setattr(self.obj, item, value)

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

    def listen(self):
        """
        Start the listening loop that consumes tasks.

        Parameters
        ----------

        Returns
        -------

        """
        if self._state != 'init':
            return

        self.loop.run(self.listen_async)

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
            # Make sure that the loop does not keep implicit references to the task until the
            # next task arrives in the queue
            self._task_queue.task_done()

            if type(task) is str and task == 'stop':
                break

            await asyncio.sleep(0)
            future = await task.prepare_args()
            await future

            method = getattr(self.obj, task.method, False)

            async with self.send_exception(task=task):
                if method is False:
                    raise AttributeError('Class %s does not have method %s' % (self.obj.__class__.__name__,
                                                                               task.method))

                if not callable(method):
                    raise ValueError('Method %s of class %s is not callable' % (task.method,
                                                                                self.obj.__class__.__name__))

            await asyncio.sleep(0)
            await self.state_changed('running')
            await task.state_changed('running')
            await self.call_safe(sender_id, method, task)

            del task
            del method

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
                if inspect.iscoroutinefunction(method):
                    future = self.loop.run(method,
                                           *task.args_value(),
                                           **task.kwargs_value())

                else:
                    future = self.loop.run_in_executor(method,
                                                       *task.args_value(),
                                                       **task.kwargs_value())

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

    def _dec_parameter_ref(self):
        def _dec_ref(*args):
            self.dec_ref()

        return _dec_ref

    _serialisation_attrs = RemoteBase._serialisation_attrs + ['_cls_attr_names']

    def _serialisation_helper(self):
        state = super()._serialisation_helper()
        state['_cls'] = PickleClass(self._cls)
        state['_runtime_id'] = mosaic.runtime().uid

        return state

    def __del__(self):
        self.logger.debug('Garbage collected object %s' % self)
        self.loop.run(self.state_changed, 'collected')


class ParameterMixin:
    """
    A parameter is a Python object that, when moved across different runtimes
    in the mosaic network, will retain a reference to its original, root object.

    This reference means that changes to a local object can be propagated to
    the root object using ``obj.method(..., propagate=True)``.

    It also means that attributes of the root object can be changed using
    ``obj.set('attr', value)``. The latest value of an attribute can be pulled
    from the root object by doing ``await obj.get('attr')``.

    Objects of class Parameter should not be instantiated directly by the user
    and the ``@mosaic.tessera`` decorator should be used instead.

    """

    @property
    def has_tessera(self):
        return hasattr(self, '_tessera') \
               and self._tessera is not None \
               and isinstance(self._tessera, (Tessera, TesseraProxy))

    async def get(self, item):
        if self.has_tessera:
            value = await self._tessera.get_attr(item)
            super().__setattr__(item, value)

        return super().__getattribute__(item)

    def set(self, item, value):
        if self.has_tessera:
            self._tessera.set_attr(item, value)

        super().__setattr__(item, value)

    def __getattribute__(self, item):
        member = super().__getattribute__(item)

        has_tessera = False
        try:
            tess = super().__getattribute__('_tessera')
            has_tessera = tess is not None and isinstance(tess, (Tessera, TesseraProxy))
        except AttributeError:
            pass

        if has_tessera:
            if inspect.iscoroutine(member) or inspect.iscoroutinefunction(member):
                @functools.wraps(member)
                async def remote_method(*args, **kwargs):
                    propagate = kwargs.pop('propagate', False)
                    if propagate:
                        await tess[item](*args, **kwargs)

                    return await member(*args, **kwargs)

                return remote_method

            elif inspect.isfunction(member) or inspect.ismethod(member):
                @functools.wraps(member)
                def remote_method(*args, **kwargs):
                    propagate = kwargs.pop('propagate', False)
                    if propagate:
                        tess[item](*args, **kwargs)

                    return member(*args, **kwargs)

                return remote_method

        return member

    def __del__(self):
        if self.has_tessera and isinstance(self._tessera, Tessera):
            self._tessera.dec_ref()


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

        runtime = kwargs.pop('runtime', None)
        self._runtime_id = runtime.uid if hasattr(runtime, 'uid') else runtime

        self._uid = '%s-%s-%s' % ('tess',
                                  self._cls.__name__.lower(),
                                  uuid.uuid4().hex)

        self._cls_attr_names = None
        self._set_cls()

        self._state = 'pending'

    def _set_cls(self):
        if self.__class__.__name__ == '_TesseraProxy':
            return

        cls = self._cls.cls
        exclude = dir(self)
        cls_methods, cls_magic_methods, \
            cls_method_names, cls_magic_method_names = _extract_methods(cls, exclude)

        for method in cls_methods:
            method_getter = self._get_method_getter(method[0])
            method_getter = functools.wraps(method[1])(method_getter)

            setattr(self, method[0], method_getter)

        __dict__ = {}
        for method in cls_magic_methods:
            method_getter = self._get_magic_method_getter(method[0])
            method_getter = functools.wraps(method[1])(method_getter)

            __dict__[method[0]] = method_getter

        _TesseraProxy = type('_TesseraProxy', (TesseraProxy,), __dict__)
        self.__class__ = _TesseraProxy

    async def init(self, *args, **kwargs):
        """
        Asynchronous correlate of ``__init__``.

        """
        kwargs = self._fill_config(**kwargs)
        kwargs.pop('runtime', None)

        self._runtime_id = await self.select_worker(self._runtime_id)

        self._state = 'pending'
        await self.monitor.init_tessera(uid=self._uid,
                                        runtime_id=self._runtime_id)

        self.runtime.register(self)
        cls_attrs = await self.remote_runtime.init_tessera(cls=self._cls,
                                                           uid=self._uid, args=args,
                                                           reply=True, **kwargs)
        self._cls_attr_names = cls_attrs

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

    @classmethod
    def remote_cls(cls):
        """
        Class of the remote.

        """
        return Tessera

    @classmethod
    async def select_worker(cls, runtime=None):
        """
        Select an available worker.

        Parameters
        ----------
        runtime : str or Runtime, optional
            If a valid runtime is given, this will be selected as the target worker.

        Returns
        -------
        str
            UID of the target worker.

        """
        if hasattr(runtime, 'uid'):
            runtime = runtime.uid

        if runtime is None:
            monitor = mosaic.get_monitor()
            runtime = await monitor.select_worker(reply=True)

        return runtime

    def get_attr(self, item):
        """
        Get at attribute from the remote tessera.

        Parameters
        ----------
        item : str

        Returns
        -------
        AwaitableOnly

        """
        return self._get_remote_attr(item)

    def set_attr(self, item, value):
        """
        Set an attribute on the remote tessera.

        Parameters
        ----------
        item : str
        value : object

        Returns
        -------

        """
        return self._set_remote_attr(item, value)

    async def _init_task(self, task_proxy, *args, **kwargs):
        kwargs.pop('runtime', None)

        await self._init_future

        for arg in args:
            if hasattr(arg, 'init_future'):
                await arg.init_future

        for arg in kwargs.values():
            if hasattr(arg, 'init_future'):
                await arg.init_future

        return await task_proxy.__init_async__()

    def _get_remote_method(self, item):
        def remote_method(*args, **kwargs):
            task_proxy = TaskProxy(self, item, *args, **kwargs)

            loop = mosaic.get_event_loop()
            loop.run(self._init_task, task_proxy, *args, **kwargs)

            return task_proxy

        return remote_method

    def _get_remote_attr(self, item):
        async def remote_attr():
            await self._init_future
            attr = await self.cmd_recv_async(method='get_attr', item=item)

            return attr

        return AwaitableOnly(remote_attr)

    def _set_remote_attr(self, item, value):
        async def remote_attr():
            await self._init_future
            await self.cmd_recv_async(method='set_attr', item=item, value=value)

        return remote_attr()

    def _get_method_getter(self, method):
        def method_getter(*args, **kwargs):
            return self._get_remote_method(method)(*args, **kwargs)

        return method_getter

    def _get_magic_method_getter(self, method):
        def method_getter(_self, *args, **kwargs):
            return self._get_remote_method(method)(*args, **kwargs)

        return method_getter

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            if self._cls_attr_names is None or item in self._cls_attr_names:
                return self._get_remote_attr(item)
            else:
                raise AttributeError('Class %s or TesseraProxy has no attribute %s' %
                                     (self._cls.cls.__class__.__name__, item))

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __await__(self):
        yield from self._init_future.__await__()
        return self

    _serialisation_attrs = ProxyBase._serialisation_attrs + ['_cls',
                                                             '_runtime_id',
                                                             '_cls_attr_names']

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = super()._deserialisation_helper(state)
        instance._set_cls()

        return instance


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
        self._cls_attr_names = None

        self._proxies = []
        self._runtime_id = []

        self._uid = '%s-%s-%s' % ('array',
                                  self._cls.__name__.lower(),
                                  uuid.uuid4().hex)

        available_workers = [uid for uid in mosaic.get_workers().keys()]

        for index in range(self._len):
            worker = available_workers[index % len(available_workers)]
            proxy = TesseraProxy(cls, *args, runtime=worker, **kwargs)

            self._proxies.append(proxy)

        self._set_cls()

        self._state = 'pending'

    async def init(self, *args, **kwargs):
        """
        Asynchronous correlate of ``__init__``.

        """
        for proxy in self._proxies:
            await proxy.__init_async__(*args, **kwargs)
            self._runtime_id.append(proxy.runtime_id)
            self._cls_attr_names = proxy._cls_attr_names

        self.runtime.register(self)

        self._state = 'listening'

    def _set_cls(self):
        if self.__class__.__name__ == '_TesseraProxy':
            return

        cls = self._cls.cls
        exclude = dir(self)
        cls_methods, cls_magic_methods, \
            cls_method_names, cls_magic_method_names = _extract_methods(cls, exclude)

        for method in cls_methods:
            method_getter = self._get_method_getter(method[0])
            method_getter = functools.wraps(method[1])(method_getter)

            setattr(self, method[0], method_getter)

        __dict__ = {}
        for method in cls_magic_methods:
            method_getter = self._get_method_getter(method[0])
            method_getter = functools.wraps(method[1])(method_getter)

            __dict__[method[0]] = method_getter

        _ArrayProxy = type('_ArrayProxy', (ArrayProxy,), __dict__)
        self.__class__ = _ArrayProxy

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

    def get_attr(self, item):
        """
        Get at attribute from the remote tessera.

        Parameters
        ----------
        item : str

        Returns
        -------
        AwaitableOnly

        """
        attrs = [each._get_remote_attr(item) for each in self._proxies]
        return asyncio.gather(*attrs)

    def set_attr(self, item, value):
        """
        Set an attribute on the remote tessera.

        Parameters
        ----------
        item : str
        value : object

        Returns
        -------

        """
        return self._get_remote_method('set_attr')(item, value)

    def _get_remote_method(self, item):
        # TODO There should be an equivalent Task array proxy
        def remote_method(*args, **kwargs):
            runtime = kwargs.pop('runtime', None)
            runtime = runtime.uid if hasattr(runtime, 'uid') else runtime

            if runtime is None:
                task_proxies = []
                for proxy in self._proxies:
                    task_proxies.append(proxy[item](*args, **kwargs))

            else:
                task_proxies = None
                for proxy in self._proxies:
                    if proxy.runtime_id == runtime:
                        task_proxies = proxy[item](*args, **kwargs)
                        break

                if task_proxies is None:
                    raise RuntimeError('Runtime %s is no contained in the ArrayProxy' % runtime)

            return task_proxies

        return remote_method

    def _get_remote_attr(self, item):
        async def remote_attr():
            await self._init_future

            attrs = [each.cmd_recv_async(method='get_attr', item=item) for each in self._proxies]

            return await asyncio.gather(*attrs)

        return AwaitableOnly(remote_attr)

    @staticmethod
    def _get_method_getter(method):
        def method_getter(_self, *args, **kwargs):
            return _self._get_remote_method(method)(*args, **kwargs)

        return method_getter

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            if self._cls_attr_names is None or item in self._cls_attr_names:
                return self._get_remote_attr(item)
            else:
                raise AttributeError('Class %s or ArrayProxy has no attribute %s' %
                                     (self._cls.cls.__class__.__name__, item))

    def __getitem__(self, item):
        return self._proxies[item]

    def __repr__(self):
        runtime_id = ', '.join([str(each) for each in self.runtime_id])

        return "<%s object at %s, uid=%s, runtime=(%s), state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self._state)

    def __await__(self):
        yield from self._init_future.__await__()
        return self

    _serialisation_attrs = CMDBase._serialisation_attrs + ['_cls',
                                                           '_proxies',
                                                           '_runtime_id',
                                                           '_cls_attr_names',
                                                           '_len']


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
    will generate a standard local instance, or onto the mosaic runtime ``Klass.remote(...)``,
    which will instantiate the class in a remote endpoint and return a proxy to the user.

    A mosaic Parameter can also be instantiated by using ``Klass.parameter(...)``.

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
    >>> remote_proxy = Klass.remote(10)
    >>>
    >>> # The resulting proxy can be used to call the instance methods,
    >>> task = remote_proxy.add(5)
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

    @functools.wraps(tessera)
    def tessera_wrapper(cls):

        @classmethod
        def remote(_, *args, **kwargs):
            kwargs.update(cmd_config)

            array_len = kwargs.pop('len', None)

            if array_len is None:
                proxy = TesseraProxy(cls, *args, **kwargs)

            else:
                proxy = ArrayProxy(cls, *args, len=array_len, **kwargs)

            loop = mosaic.get_event_loop()
            loop.run(proxy.__init_async__, *args, **kwargs)

            return proxy

        @classmethod
        def local(_, *args, uid=None, **kwargs):
            if uid is None:
                uid = '%s-%s-%s' % ('tess',
                                    cls.__name__.lower(),
                                    uuid.uuid4().hex)

            kwargs.update(cmd_config)
            return Tessera(cls, uid, *args, **kwargs)

        @classmethod
        def parameter(_, *args, uid=None, **kwargs):
            if uid is None:
                uid = '%s-%s-%s' % ('tess',
                                    cls.__name__.lower(),
                                    uuid.uuid4().hex)

            kwargs.update(cmd_config)
            tess = Tessera(cls, uid, *args, **kwargs)
            param = tess.make_parameter()

            _Parameter = type('_%s' % param.__class__.__name__, (ParameterMixin, param.__class__), {})
            param.__class__ = _Parameter

            return param

        cls.remote = remote
        cls.local = local
        cls.parameter = parameter
        cls.select_worker = TesseraProxy.select_worker

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
