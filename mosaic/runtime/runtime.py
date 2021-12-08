
import gc
import zmq
import zmq.asyncio
import asyncio
import contextlib
import weakref

import mosaic
from ..utils.event_loop import EventLoop
from ..comms import CommsManager
from ..core import Task
from ..profile import profiler, global_profiler


__all__ = ['Runtime', 'RuntimeProxy']


class BaseRPC:
    """
    Base class representing either a mosaic runtime or a proxy to that runtime.

    Runtimes represent network endpoints, and proxies represent references to those
    endpoints. Using proxies, runtimes can  be addressed transparently through
    remote procedural calls. That is, calling a method on the proxy will execute it
    in the remote runtime.

    Runtimes also keep a series of resident mosaic objects (such as tessera or tasks),
    and can direct remote commands to those objects.

    A runtime has a name and an (optional) set of indices, which together produce
    a unique ID associated with that endpoint in the network. The runtime UID
    is used to direct messages across the network.

    A name ``runtime`` and indices ``(0, 0)`` will result in a UID ``runtime:0:0``,
    while the same name with no indices will result in a UID ``runtime``.

    Parameters
    ----------
    name : str, optional
        Name of the runtime, defaults to None. If no name is provided, the UID has
        to be given.
    indices : tuple or int, optional
        Indices associated with the runtime, defaults to none.
    uid : str
        UID from which to find the name and indices, defaults to None.

    """

    def __init__(self, name=None, indices=(), uid=None):
        if uid is not None:
            uid = uid.split(':')
            name = uid[0]

            if len(uid) > 1:
                indices = tuple([int(each) for each in uid[1:]])
            else:
                indices = ()

        elif name is None:
            raise ValueError('Either name and indices or UID are required to instantiate the RPC')

        indices = () if indices is None else indices

        if type(indices) is not tuple:
            indices = (indices,)

        self._name = name
        self._indices = indices

    @property
    def name(self):
        """
        Runtime name.

        """
        return self._name

    @property
    def indices(self):
        """
        Runtime indices.

        """
        return self._indices

    @property
    def uid(self):
        """
        Runtime UID.

        """
        if len(self.indices):
            indices = ':'.join([str(each) for each in self.indices])
            return '%s:%s' % (self.name, indices)

        else:
            return self.name

    @property
    def address(self):
        """
        Runtime IP address.

        """
        return None

    @property
    def port(self):
        """
        Runtime port.

        """
        return None


class Runtime(BaseRPC):
    """
    Class representing a local runtime of any possible type.

    The runtime handles the mosaic life cycle:

    - it handles the comms manager, the event loop, the logger and keeps proxies to existing remote runtimes;
    - it keeps track of resident mosaic objects (tessera, task) and proxies to those;
    - it routes remote commands to these resident mosaic objects.

    For referece on accepted parameters, check `mosaic.init`.

    """

    is_head = False
    is_monitor = False
    is_node = False
    is_worker = False

    def __init__(self, **kwargs):
        runtime_indices = kwargs.pop('runtime_indices', ())
        super().__init__(name=self.__class__.__name__.lower(), indices=runtime_indices)

        self.mode = kwargs.get('mode', 'local')

        self._comms = None
        self._head = None
        self._monitor = None
        self._node = None
        self._nodes = dict()
        self._worker = None
        self._workers = dict()
        self._zmq_context = None
        self._loop = None

        self.logger = None

        self._tessera = dict()
        self._tessera_proxy = weakref.WeakValueDictionary()
        self._tessera_proxy_array = weakref.WeakValueDictionary()
        self._task = dict()
        self._task_proxy = weakref.WeakValueDictionary()

    async def init(self, **kwargs):
        """
        Asynchronous counterpart of ``__init__``.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        # Start comms
        address = kwargs.pop('address', None)
        port = kwargs.pop('port', None)
        self.set_comms(address, port)

        # Start logger
        self.set_logger()

        # Connect to parent if necessary
        parent_id = kwargs.pop('parent_id', None)
        parent_address = kwargs.pop('parent_address', None)
        parent_port = kwargs.pop('parent_port', None)
        if parent_id is not None and parent_address is not None and parent_port is not None:
            await self._comms.handshake(parent_id, parent_address, parent_port)

        # Connect to monitor if necessary
        monitor_address = kwargs.get('monitor_address', None)
        monitor_port = kwargs.get('monitor_port', None)
        if not self.is_monitor and monitor_address is not None and monitor_port is not None:
            await self._comms.handshake('monitor', monitor_address, monitor_port)

        # Start listening
        self._comms.listen()

        # Set up profiling
        profile = kwargs.get('profile', False)
        if profile:
            self.set_profiler()

    def wait(self, wait=False):
        """
        Wait on the comms loop until done.

        Parameters
        ----------
        wait : bool
            Whether or not to wait, defaults to False.

        Returns
        -------

        """
        if wait is True:
            self._comms.wait()

    async def barrier(self, timeout=None):
        """
        Wait until all pending tasks are done. If no timeout is
        provided, the barrier will wait indefinitely.

        Parameters
        ----------
        timeout : float, optional

        Returns
        -------

        """
        monitor = self.get_monitor()
        await monitor.barrier(timeout=timeout, reply=True)

    def async_for(self, *iterables, **kwargs):

        async def _async_for(func):
            worker_queue = asyncio.Queue()
            for worker in self._workers.values():
                await worker_queue.put(worker)

            async def call(*iters):
                async with self._exclusive_proxy(worker_queue) as _worker:
                    res = await func(_worker, *iters)

                return res

            tasks = [call(*each) for each in zip(*iterables)]
            gather = await asyncio.gather(*tasks)
            await self.barrier()

            return gather

        return _async_for

    @contextlib.asynccontextmanager
    async def _exclusive_proxy(self, queue):
        proxy = await queue.get()

        yield proxy

        await queue.put(proxy)

    @property
    def address(self):
        """
        IP address of the runtime.

        """
        return self._comms.address

    @property
    def port(self):
        """
        Port of the runtime.

        """
        return self._comms.port

    @property
    def num_nodes(self):
        """
        Number of nodes on the network.

        """
        return len(self._nodes.keys())

    @property
    def num_workers(self):
        """
        Number of workers on the network.

        """
        return len(self._workers.keys())

    @property
    def nodes(self):
        """
        Nodes on the network.

        """
        return list(self._nodes.values())

    @property
    def workers(self):
        """
        Workers on the network.

        """
        return list(self._workers.values())

    # Interfaces to global objects

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        pass

    def set_profiler(self):
        """
        Set up profiling.

        Returns
        -------

        """
        pass
        # self._loop.interval(self.send_profile, interval=25)

    def set_comms(self, address=None, port=None):
        """
        Set up comms manager.

        Parameters
        ----------
        address : str, optional
            Address to use, defaults to None. If None, the comms will try to
            guess the address.
        port : int, optional
            Port to use, defaults to None. If None, the comms will test ports
            until one becomes available.

        Returns
        -------

        """
        if self._comms is None:
            self._comms = CommsManager(runtime=self, address=address, port=port)
            self._comms.connect_recv()

    def get_comms(self):
        """
        Access comms.

        Returns
        -------

        """
        return self._comms

    def get_zmq_context(self):
        """
        Access ZMQ socket context.

        Returns
        -------

        """
        if self._zmq_context is None:
            self._zmq_context = zmq.asyncio.Context()

        return self._zmq_context

    def get_event_loop(self, asyncio_loop=None):
        """
        Access event loop.

        Parameters
        ----------
        asyncio_loop: object, optional
            Async loop to use in our mosaic event loop, defaults to new loop.

        Returns
        -------

        """
        if self._loop is None:
            self._loop = EventLoop(loop=asyncio_loop)

        return self._loop

    def get_head(self):
        """
        Access head runtime.

        Returns
        -------

        """
        return self._head

    def get_monitor(self):
        """
        Access monitor runtime.

        Returns
        -------

        """
        return self._monitor

    def get_node(self, uid=None):
        """
        Access specific node runtime.

        Parameters
        ----------
        uid : str

        Returns
        -------

        """
        if uid is not None:
            return self._nodes[uid]

        else:
            return self._node

    def get_nodes(self):
        """
        Access all node runtimes.

        Parameters
        ----------

        Returns
        -------

        """
        return self._nodes

    def get_worker(self, uid=None):
        """
        Access specific worker runtime.

        Parameters
        ----------
        uid : str

        Returns
        -------

        """
        if uid is not None:
            return self._workers[uid]

        else:
            return self._worker

    def get_workers(self):
        """
        Access all worker runtimes.

        Parameters
        ----------

        Returns
        -------

        """
        return self._workers

    def proxy_from_uid(self, uid, proxy=None):
        """
        Generate a proxy from a UID.

        Parameters
        ----------
        uid : str
        proxy : BaseProxy

        Returns
        -------
        BaseProxy

        """
        proxy = proxy or self.proxy(uid=uid)

        found_proxy = None
        if hasattr(self, '_' + proxy.name + 's'):
            found_proxy = getattr(self, '_' + proxy.name + 's').get(uid, None)

        elif hasattr(self, '_' + proxy.name):
            found_proxy = getattr(self, '_' + proxy.name)

        if found_proxy is None:
            if hasattr(self, '_' + proxy.name + 's'):
                getattr(self, '_' + proxy.name + 's')[uid] = proxy

            elif hasattr(self, '_' + proxy.name):
                setattr(self, '_' + proxy.name, proxy)

            return proxy

        else:
            return found_proxy

    @staticmethod
    def proxy(name=None, indices=(), uid=None):
        """
        Generate proxy from name, indices or UID.

        Parameters
        ----------
        name : str, optional
        indices : tuple, optional
        uid : str, optional

        Returns
        -------

        """
        return RuntimeProxy(name=name, indices=indices, uid=uid)

    # Network management methods

    def log_info(self, sender_id, buf):
        """
        Log remote message from ``sender_id`` on info stream.

        Parameters
        ----------
        sender_id : str
        buf : str

        Returns
        -------

        """
        if self.logger is None:
            return

        self.logger.info(buf, uid=sender_id)

    def log_debug(self, sender_id, buf):
        """
        Log remote message from ``sender_id`` on debug stream.

        Parameters
        ----------
        sender_id : str
        buf : str

        Returns
        -------

        """
        if self.logger is None:
            return

        self.logger.debug(buf, uid=sender_id)

    def log_error(self, sender_id, buf):
        """
        Log remote message from ``sender_id`` on error stream.

        Parameters
        ----------
        sender_id : str
        buf : str

        Returns
        -------

        """
        if self.logger is None:
            return

        self.logger.error(buf, uid=sender_id)

    def log_warning(self, sender_id, buf):
        """
        Log remote message from ``sender_id`` on warning stream.

        Parameters
        ----------
        sender_id : str
        buf : str

        Returns
        -------

        """
        if self.logger is None:
            return

        self.logger.warning(buf, uid=sender_id)

    def raise_exception(self, sender_id, exc):
        """
        Raise remote exception that ocurred on ``sender_id``.

        Parameters
        ----------
        sender_id : str
        exc : Exception description

        Returns
        -------

        """
        self.log_error(sender_id, 'Endpoint raised exception "%s"' % str(exc[1]))
        raise exc[1].with_traceback(exc[2].as_traceback())

    def hand(self, sender_id, address, port):
        """
        Handle incoming handshake petition.

        Parameters
        ----------
        sender_id : str
        address : str
        port : int

        Returns
        -------

        """
        self.proxy_from_uid(sender_id)

    def shake(self, sender_id, network):
        """
        Handle handshake response.

        Parameters
        ----------
        sender_id : str
        network : dict

        Returns
        -------

        """
        for uid, address in network.items():
            self.connect(sender_id, uid, *address)

    def connect(self, sender_id, uid, address, port):
        """
        Connect to a specific remote runtime.

        Parameters
        ----------
        sender_id : str
        uid : str
        address : str
        port : int

        Returns
        -------

        """
        self.hand(uid, address, port)

    def disconnect(self, sender_id, uid):
        """
        Disconnect specific remote runtime.

        Parameters
        ----------
        sender_id : str
        uid : str

        Returns
        -------

        """
        pass

    def stop(self, sender_id=None):
        """
        Stop runtime.

        Parameters
        ----------
        sender_id : str

        Returns
        -------

        """
        if profiler.tracing:
            profiler.stop()

        if self._comms is not None:
            self._loop.run(self._comms.stop, sender_id)

        if self._loop is not None:
            self._loop._stop.set()

    def recv_profile(self, sender_id, profiler_update):
        """
        Process a profiler update.

        Parameters
        ----------
        sender_id : str
        profiler_update : dict

        Returns
        -------

        """
        global_profiler.recv_profile(sender_id, profiler_update)

    def request_profile(self, sender_id):
        """
        Return a profiler update.

        Parameters
        ----------
        sender_id : str

        Returns
        -------

        """
        return global_profiler.get_profile()

    async def send_profile(self):
        """
        Send profiler update to monitor.

        Returns
        -------

        """
        global_profiler.send_profile()

    # Command and task management methods

    def register(self, obj):
        """
        Register CMD object with runtime.

        Parameters
        ----------
        obj : BaseCMD

        Returns
        -------

        """
        obj_type = obj.type
        obj_uid = obj.uid

        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            obj_store[obj_uid] = obj
            obj._registered = True

        return obj_store[obj_uid]

    def needs_registering(self, obj):
        obj_type = obj.type
        obj_uid = obj.uid
        obj_store = getattr(self, '_' + obj_type)

        return obj_uid not in obj_store.keys()

    def deregister(self, obj):
        """
        Deregister CMD object from runtime.

        Parameters
        ----------
        obj : BaseCMD

        Returns
        -------

        """
        obj_type = obj.type
        obj_uid = obj.uid
        obj_store = getattr(self, '_' + obj_type)

        obj = obj_store.pop(obj_uid, None)

        if hasattr(obj, 'queue_task'):
            obj.queue_task((None, 'stop', None))

        gc.collect()

    def cmd(self, sender_id, cmd):
        """
        Process incoming command address to one of the resident objects.

        Parameters
        ----------
        sender_id : str
        cmd : CMD

        Returns
        -------

        """
        obj_type = cmd.type
        obj_uid = cmd.uid
        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            self.logger.warning('Runtime %s does not own object %s of type %s' % (self.uid, obj_uid, obj_type))
            return

        obj = obj_store[obj_uid]

        method = getattr(obj, cmd.method)
        result = method(*cmd.args, **cmd.kwargs)

        return result

    def inc_ref(self, sender_id, uid, type):
        """
        Increase reference count for a resident object.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        uid : str
            UID of the object being referenced.
        type : str
            Type of the object being referenced.

        Returns
        -------

        """
        self.logger.debug('Increased ref count for object %s' % uid)

        obj_type = type
        obj_uid = uid
        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            self.logger.warning('Runtime %s does not own object %s of type %s' % (self.uid, obj_uid, obj_type))
            return

        obj = obj_store[obj_uid]
        obj.inc_ref()
        obj.register_proxy(uid=sender_id)

    def dec_ref(self, sender_id, uid, type):
        """
        Decrease reference count for a resident object.

        If reference count decreases below 1, deregister the object.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        uid : str
            UID of the object being referenced.
        type : str
            Type of the object being referenced.

        Returns
        -------

        """
        self.logger.debug('Decreased ref count for object %s' % uid)

        obj_type = type
        obj_uid = uid
        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            # self.logger.warning('Runtime %s does not own object %s of type %s' % (self.uid, obj_uid, obj_type))
            return

        obj = obj_store[obj_uid]
        obj.dec_ref()
        obj.deregister_proxy(uid=sender_id)

    # Tessera and task management methods

    async def init_tessera(self, sender_id, cls, uid, args, **kwargs):
        """
        Create tessera in this worker.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        cls : type
            Class of the tessera.
        uid : str
            UID of the new tessera.
        args : tuple, optional
            Arguments for the initialisation of the tessera.
        kwargs : optional
            Keyword arguments for the initialisation of the tessera.

        Returns
        -------

        """
        tessera = cls.local(*args, uid=uid, **kwargs)
        tessera.register_proxy(sender_id)

        return tessera._cls_attr_names

    async def init_task(self, sender_id, task, uid):
        """
        Create new task for a tessera in this worker.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        task : dict
            Task configuration.
        uid : str
            UID of the new task.

        Returns
        -------

        """
        obj_uid = task['tessera_id']
        obj_store = self._tessera
        tessera = obj_store[obj_uid]

        task = Task(uid, sender_id, tessera,
                    task['method'], *task['args'], **task['kwargs'])

        frame_info = profiler.frame_info()

        tessera.queue_task((sender_id, task, frame_info))
        await task.state_changed('pending')


class RuntimeProxy(BaseRPC):
    """
    This class represents a proxy to a remote running runtime.

    This proxy can be used to execute methods and commands on the remote runtime simply by
    calling methods on it.

    The proxy uses the comms to direct messages to the correct endpoint using its UID.

    Parameters
    ----------
    name : str, optional
        Name of the runtime, defaults to None. If no name is provided, the UID has
        to be given.
    indices : tuple or int, optional
        Indices associated with the runtime, defaults to none.
    uid : str
        UID from which to find the name and indices, defaults to None.
    comms : CommsManager
        Comms instance to use, defaults to global comms.

    """

    def __init__(self, name=None, indices=(), uid=None, comms=None):
        super().__init__(name=name, indices=indices, uid=uid)

        self._comms = comms or mosaic.get_comms()
        self._subprocess = None

    @property
    def address(self):
        """
        Remote runtime IP address.

        """
        return self._comms.uid_address(self.uid)

    @property
    def port(self):
        """
        Remote runtime port.

        """
        return self._comms.uid_port(self.uid)

    @property
    def subprocess(self):
        """
        Subprocess on which remote runtime lives, if any.

        """
        return self._subprocess

    @subprocess.setter
    def subprocess(self, subprocess):
        """
        Set remote runtime subprocess.

        Parameters
        ----------
        subprocess : Subprocess

        Returns
        -------

        """
        self._subprocess = subprocess

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            def remote_method(**kwargs):
                reply = kwargs.pop('reply', False)
                as_async = kwargs.pop('as_async', True)

                if item.startswith('cmd'):
                    send_method = 'cmd'

                else:
                    send_method = 'send'
                    kwargs['method'] = item

                if reply is True:
                    send_method += '_recv'

                if as_async is True:
                    send_method += '_async'

                send_method = getattr(self._comms, send_method)
                return send_method(self.uid, **kwargs)

            return remote_method

    def __getitem__(self, item):
        return self.__getattribute__(item)
