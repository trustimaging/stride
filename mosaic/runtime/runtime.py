
import os
import zmq
import zmq.asyncio
import psutil
import asyncio
import contextlib
import weakref
from zict import LRU
from cached_property import cached_property

import mosaic
from ..types import WarehouseObject
from ..utils import subprocess, memory_limit, memory_used, sizeof
from ..utils.event_loop import EventLoop
from ..comms import CommsManager
from ..core import Task, RuntimeDisconnectedError
from ..profile import profiler, global_profiler
from ..utils.utils import cpu_count


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
    is_warehouse = False

    def __init__(self, **kwargs):
        runtime_indices = kwargs.pop('runtime_indices', ())
        super().__init__(name=self.__class__.__name__.lower(), indices=runtime_indices)

        self.mode = kwargs.get('mode', 'local')
        self.reuse_head = kwargs.get('reuse_head', False)

        self._comms = None
        self._head = None
        self._monitor = None
        self._node = None
        self._nodes = dict()
        self._worker = None
        self._workers = dict()
        self._zmq_context = None
        self._loop = None
        self._remote_warehouse = None
        self._local_warehouse = None

        cache_fraction = float(os.environ.get('MOSAIC_RUNTIME_CACHE_MEM', 0.01))
        cache_size = min(cache_fraction*memory_limit(), 10*1024**3)
        self._warehouse_cache = LRU(cache_size, {}, weight=lambda k, v: sizeof(v))
        self._warehouse_pending = set()

        self.logger = None

        self._tessera = dict()
        self._tessera_proxy = weakref.WeakValueDictionary()
        self._tessera_proxy_array = weakref.WeakValueDictionary()
        self._task = dict()
        self._task_proxy = weakref.WeakValueDictionary()
        self._pending_tasks = 0
        self._running_tasks = 0
        self._committed_mem = 0

        self._dealloc_queue = []
        self._maintenance_queue = []
        self._maintenance_msgs = {}

        self._inside_async_for = False

        if self.uid == 'warehouse':
            self._mem_fraction = float(os.environ.get('MOSAIC_WAREHOUSE_MEM', 0.85))
        elif 'worker' in self.uid:
            self._mem_fraction = float(os.environ.get('MOSAIC_WORKER_MEM', 0.85))
        else:
            self._mem_fraction = float(os.environ.get('MOSAIC_RUNTIME_MEM', 0.85))

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

        # Start warehouse
        self._remote_warehouse = self.proxy('warehouse')
        if len(self.indices):
            self._local_warehouse = self.proxy('warehouse', indices=self.indices[0])
        else:
            self._local_warehouse = self._remote_warehouse

        # Connect to monitor
        monitor_address = kwargs.get('monitor_address', None)
        monitor_port = kwargs.get('monitor_port', None)
        pubsub_port = kwargs.get('pubsub_port', None)
        if not self.is_monitor and monitor_address is not None and monitor_port is not None \
                and pubsub_port is not None:
            await self._comms.handshake('monitor', monitor_address, monitor_port, pubsub_port)

        # Start listening
        self._comms.listen()

        # Set up profiling
        profile = kwargs.get('profile', False)
        if profile:
            self.set_profiler()

        # Start maintenance loop
        if self.uid == 'head' or 'worker' in self.uid:
            maintenance_interval = max(0.5, min(len(self._workers)*0.5, 60))
        else:
            maintenance_interval = 0.5
        self._loop.interval(self.maintenance, interval=maintenance_interval)

    async def init_warehouse(self, **kwargs):
        """
        Init warehouse process.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        warehouse_indices = kwargs.pop('indices', -1)
        if warehouse_indices < 0:
            warehouse_proxy = RuntimeProxy(name='warehouse')
        else:
            warehouse_proxy = RuntimeProxy(name='warehouse', indices=warehouse_indices)
        indices = warehouse_proxy.indices

        def start_warehouse(*args, **extra_kwargs):
            kwargs.update(extra_kwargs)
            kwargs['runtime_indices'] = indices
            mosaic.init('warehouse', *args, **kwargs, wait=True)

        warehouse_subprocess = subprocess(start_warehouse)(name=warehouse_proxy.uid, daemon=False)
        warehouse_subprocess.start_process()
        warehouse_proxy.subprocess = warehouse_subprocess

        self._local_warehouse = warehouse_proxy
        if self.uid == 'monitor':
            self._remote_warehouse = warehouse_proxy
        await self._comms.wait_for(warehouse_proxy.uid)

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

    @cached_property
    def ps_process(self):
        return psutil.Process(os.getpid())

    def memory_limit(self):
        """
        Amount of RSS memory available to the runtime.

        Returns
        -------
        float
            RSS memory.

        """
        mem = memory_limit()
        return mem*self._mem_fraction

    def fits_in_memory(self, nbytes):
        mem_used = memory_used()
        mem_limit = self.memory_limit()
        return mem_used + self._committed_mem + nbytes < mem_limit

    def cpu_load(self):
        """
        CPU load of this runtime as a percentage.

        Returns
        -------
        float
            CPU load.

        """
        # OSX does not allow accessing information on external processes
        try:
            return self.ps_process.cpu_percent(interval=None)
        except psutil.AccessDenied:
            pass

        return 0

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
        await self.maintenance()
        monitor = self.get_monitor()
        await monitor.barrier(timeout=timeout, reply=True)

    def async_for(self, *iterables, **kwargs):
        assert not self._inside_async_for, 'async_for cannot be nested'

        safe = kwargs.pop('safe', True)
        timeout = kwargs.pop('timeout', None)
        max_await = kwargs.pop('max_await', None)

        async def _async_for(func):
            self._inside_async_for = True

            available_workers = self.num_workers
            if available_workers <= 0:
                raise RuntimeError('No workers available to complete async workload')

            worker_queue = asyncio.Queue()
            for worker in self._workers.values():
                await worker_queue.put(worker)

            async def call(*iters):
                res = None

                async with self._exclusive_proxy(worker_queue, safe=safe) as _worker:
                    res = await func(_worker, *iters)

                return res

            tasks = [asyncio.create_task(call(*each)) for each in zip(*iterables)]

            gather = []
            for task in asyncio.as_completed(tasks, timeout=timeout):
                try:
                    res = await task
                    gather.append(res)
                except RuntimeDisconnectedError as exc:
                    if safe:
                        self.logger.warn('Runtime failed, retiring worker: %s' % exc)
                        available_workers -= 1
                        if available_workers <= 0:
                            for other_task in tasks:
                                other_task.cancel()
                                try:
                                    await other_task
                                except (RuntimeDisconnectedError, asyncio.CancelledError):
                                    pass
                            raise RuntimeError('No workers available to complete async workload')
                    else:
                        raise

                if max_await is not None and len(gather) > max_await:
                    for other_task in tasks:
                        other_task.cancel()
                        try:
                            await other_task
                        except (RuntimeDisconnectedError, asyncio.CancelledError):
                            pass
                    break

            await self.barrier()

            self._inside_async_for = False

            return gather

        return _async_for

    @contextlib.asynccontextmanager
    async def _exclusive_proxy(self, queue, safe=False):
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
    def pubsub_port(self):
        """
        Pub-sub port of the runtime.

        """
        return self._comms.pubsub_port

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

    def get_warehouse(self):
        """
        Access warehouse.

        Returns
        -------

        """
        return self._local_warehouse

    def get_local_warehouse(self):
        """
        Access local warehouse.

        Returns
        -------

        """
        return self._local_warehouse

    def get_zmq_context(self):
        """
        Access ZMQ socket context.

        Returns
        -------

        """
        if self._zmq_context is None:
            self._zmq_context = zmq.asyncio.Context()

            # Set thread pool for ZMQ
            try:
                try:
                    import numa
                    available_cpus = numa.info.numa_hardware_info()['node_cpu_info']
                    num_cpus = min([len(cpus) for cpus in available_cpus.values()])//4
                except Exception:
                    num_cpus = cpu_count()//8

                num_cpus = os.environ.get('MOSAIC_ZMQ_NUM_THREADS', num_cpus)
                num_cpus = max(1, int(num_cpus))

                self._zmq_context.set(zmq.IO_THREADS, num_cpus)
            except AttributeError:
                pass

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
        if self.uid == 'monitor':
            return self
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

    def remove_proxy_from_uid(self, uid, proxy=None):
        """
        Remove a proxy from a UID.

        Parameters
        ----------
        uid : str
        proxy : BaseProxy

        Returns
        -------

        """
        proxy = proxy or self.proxy(uid=uid)

        if hasattr(self, '_' + proxy.name + 's'):
            try:
                del getattr(self, '_' + proxy.name + 's')[uid]
            except KeyError:
                pass

        elif hasattr(self, '_' + proxy.name):
            setattr(self, '_' + proxy.name, None)

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

    def log_perf(self, sender_id, buf):
        """
        Log remote message from ``sender_id`` on perf stream.

        Parameters
        ----------
        sender_id : str
        buf : str

        Returns
        -------

        """
        if self.logger is None:
            return

        self.logger.perf(buf, uid=sender_id)

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
        # deregister if remote uid held a proxy to a local tessera
        for obj in self._tessera.values():
            obj.deregister_proxy(uid)

        # deregister if remote uid held a proxy to a local task
        for obj in self._task.values():
            obj.deregister_proxy(uid)

        # deregister if local tessera proxy points to remote uid
        for obj in self._tessera_proxy.values():
            obj.deregister_runtime(uid)

        # deregister if local tessera proxy array points to remote uid
        for obj in self._tessera_proxy_array.values():
            obj.deregister_runtime(uid)

        # deregister if local task proxy points to remote uid
        for obj in self._task_proxy.values():
            obj.deregister_runtime(uid)

        # remove remote runtime from local runtime
        self.remove_proxy_from_uid(uid)

    async def stop(self, sender_id=None):
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

        await self.maintenance()

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

    async def put(self, obj, publish=False, reply=False):
        """
        Put an object into the warehouse.

        Parameters
        ----------
        obj
        publish
        reply

        Returns
        -------

        """
        if hasattr(obj, 'has_tessera') and obj.is_proxy:
            await obj.push(publish=publish)
            return obj.ref

        else:
            warehouse = self._local_warehouse
            warehouse_obj = WarehouseObject(obj)
            if self.uid != 'head':
                self._warehouse_cache[warehouse_obj.uid] = obj

            await warehouse.put_remote(obj=obj, uid=warehouse_obj.uid,
                                       publish=publish, reply=reply or publish)

            return warehouse_obj

    async def get(self, uid, cache=True):
        """
        Retrieve an object from the warehouse.

        Parameters
        ----------
        uid
        cache

        Returns
        -------

        """
        if self.uid == 'head' or not cache:
            return await self._local_warehouse.get_remote(uid=uid, reply=True)

        obj_uid = uid.uid if hasattr(uid, 'uid') else uid

        while obj_uid in self._warehouse_pending:
            await asyncio.sleep(0.1)
        self._warehouse_pending.add(obj_uid)

        try:
            obj = self._warehouse_cache[obj_uid]
        except KeyError:
            obj = await self._local_warehouse.get_remote(uid=uid, reply=True)
            self._warehouse_cache[obj_uid] = obj

        self._warehouse_pending.remove(obj_uid)

        return obj

    async def drop(self, uid):
        """
        Delete an object from the warehouse.

        Parameters
        ----------
        uid

        Returns
        -------

        """
        await self._local_warehouse.drop_remote(uid=uid)

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

    def needs_registering(self, obj_type, obj_uid):
        obj_store = getattr(self, '_' + obj_type)

        obj = None
        needs_registering = obj_uid not in obj_store.keys()

        if not needs_registering:
            try:
                obj = obj_store[obj_uid]
            except KeyError:
                pass

        return needs_registering, obj

    def deregister(self, obj):
        """
        Deregister CMD object from runtime.

        Parameters
        ----------
        obj : BaseCMD

        Returns
        -------

        """
        if obj is not None:
            self._dealloc_queue.append(obj)

    async def maintenance(self):
        """
        Task handling maintenance processes such as object
        deallocation.

        Returns
        -------

        """
        tasks = []

        if len(self._dealloc_queue):
            uncollectables = []
            deregisters = {}
            dec_refs = {}
            for obj in self._dealloc_queue:
                # If not collectable, defer until next cycle
                if not obj.collectable:
                    uncollectables.append(obj)
                    continue

                # Remove from internal storage
                obj_type = obj.type
                obj_uid = obj.uid
                obj_store = getattr(self, '_' + obj_type)
                obj_store.pop(obj_uid, None)

                # Stop tessera loop
                if hasattr(obj, 'queue_task'):
                    obj.queue_task((None, 'stop'))

                # Remove from local warehouse
                if 'warehouse' in self.uid and obj_uid in self._local_warehouse:
                    if self._local_warehouse[obj_uid] is obj:
                        del self._local_warehouse[obj_uid]

                # Execute object deregister
                if hasattr(obj, 'deregister'):
                    try:
                        runtime_uid, dec_ref, kwargs = await obj.deregister()
                        if runtime_uid not in deregisters:
                            deregisters[runtime_uid] = []
                            dec_refs[runtime_uid] = dec_ref
                        deregisters[runtime_uid].append(kwargs)
                    except TypeError:
                        pass

            self._dealloc_queue = uncollectables

            for runtime_uid, dec_ref in dec_refs.items():
                tasks.append(dec_ref(msgs=deregisters[runtime_uid]))

        if len(self._maintenance_queue):
            for task in self._maintenance_queue:
                tasks.append(task())

            self._maintenance_queue = []

        if len(self._maintenance_msgs):
            for method_name, msgs in self._maintenance_msgs.items():
                method = getattr(self._monitor, method_name)
                tasks.append(method(msgs=msgs))

            self._maintenance_msgs = {}

        await asyncio.gather(*tasks)

    def maintenance_queue(self, fun):
        """
        Add callable to maintenance queue

        Parameters
        ----------
        fun : callable

        Returns
        -------

        """
        self._maintenance_queue.append(fun)

    def maintenance_msg(self, method, msg):
        """
        Add message to maintenance queue

        Parameters
        ----------
        method : callable
        msg : dict

        Returns
        -------

        """
        if method not in self._maintenance_msgs:
            self._maintenance_msgs[method] = []
        self._maintenance_msgs[method].append(msg)

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
            self.logger.warning('cmd %s: Runtime %s does not own object %s of type %s' %
                                (cmd.method, self.uid, obj_uid, obj_type))
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
            self.logger.warning('inc_ref: Runtime %s does not own object %s of type %s' % (self.uid, obj_uid, obj_type))
            return

        obj = obj_store[obj_uid]
        obj.inc_ref()
        obj.register_proxy(uid=sender_id)

    def dec_refs(self, sender_id, msgs):
        """
        Decrease reference count for multiple resident objects.

        If reference count decreases below 1, deregister the object.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        msgs : list
            UIDs of the object being referenced.

        Returns
        -------

        """
        msgs = [msgs] if not isinstance(msgs, list) else msgs
        for msg in msgs:
            self.dec_ref(sender_id, **msg)

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

        tessera.queue_task((sender_id, task))
        self.inc_pending_tasks()

    async def init_tasks(self, sender_id, tasks):
        """
        Create new set of tasks for tesseras in this worker.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        tasks : list
            Tasks configuration.

        Returns
        -------

        """
        for uid, task in tasks:
            await self.init_task(sender_id, task, uid)

    def inc_pending_tasks(self):
        self._pending_tasks += 1

    def dec_pending_tasks(self):
        self._pending_tasks -= 1
        self._pending_tasks = max(0, self._pending_tasks)

    def inc_running_tasks(self):
        self._running_tasks += 1

    def dec_running_tasks(self):
        self._running_tasks -= 1
        self._running_tasks = max(0, self._running_tasks)

    def inc_committed_mem(self, nbytes):
        self._committed_mem += nbytes

    def dec_committed_mem(self, nbytes):
        self._committed_mem -= nbytes
        self._committed_mem = max(0, self._committed_mem)


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

        self._subprocess = None

    @property
    def comms(self):
        return mosaic.get_comms()

    @property
    def address(self):
        """
        Remote runtime IP address.

        """
        return self.comms.uid_address(self.uid)

    @property
    def port(self):
        """
        Remote runtime port.

        """
        return self.comms.uid_port(self.uid)

    @property
    def pubsub_port(self):
        """
        Remote pub-sub port.

        """
        return self.comms.pubsub_port

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

                send_method = getattr(self.comms, send_method)
                return send_method(self.uid, **kwargs)

            return remote_method

    def __getitem__(self, item):
        return self.__getattribute__(item)
