
import gc
import zmq
import zmq.asyncio
import weakref

import mosaic
from ..utils.event_loop import EventLoop
from ..comms import CommsManager


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
    name : str
        Name of the runtime.
    indices : tuple or int, optional
        Indices associated with the runtime, defaults to none.

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

    The runtime handles the mosaic life cycle

    - It handles the comms manager, the event loop, the logger and
    keeps proxies to existing remote runtimes.
    - It keeps track of resident mosaic objects (tessera, task) and proxies to those
    - It routes remote commands to these resident mosaic objects.

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
        # Start event loop
        self._loop.set_main_thread()

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

    def wait(self, wait=False):
        if wait is True:
            self._comms.wait()

    @property
    def address(self):
        return self._comms.address

    @property
    def port(self):
        return self._comms.port

    @property
    def num_nodes(self):
        return len(self._nodes.keys())

    @property
    def num_workers(self):
        return len(self._workers.keys())

    # Interfaces to global objects

    def set_logger(self):
        pass

    def set_comms(self, address=None, port=None):
        if self._comms is None:
            self._comms = CommsManager(runtime=self, address=address, port=port)
            self._comms.connect_recv()

    def get_comms(self):
        return self._comms

    def get_zmq_context(self):
        if self._zmq_context is None:
            self._zmq_context = zmq.asyncio.Context()

        return self._zmq_context

    def get_event_loop(self):
        if self._loop is None:
            self._loop = EventLoop()

        return self._loop

    def get_head(self):
        return self._head

    def get_monitor(self):
        return self._monitor

    def get_node(self, uid=None):
        if uid is not None:
            return self._nodes[uid]

        else:
            return self._node

    def get_worker(self, uid=None):
        if uid is not None:
            return self._workers[uid]

        else:
            return self._worker

    def proxy_from_uid(self, uid, proxy=None):
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
        return RuntimeProxy(name=name, indices=indices, uid=uid)

    # Network management methods

    def log_info(self, sender_id, buf):
        if self.logger is None:
            return

        self.logger.info(buf, uid=sender_id)

    def log_debug(self, sender_id, buf):
        if self.logger is None:
            return

        self.logger.debug(buf, uid=sender_id)

    def log_error(self, sender_id, buf):
        if self.logger is None:
            return

        self.logger.error(buf, uid=sender_id)

    def log_warning(self, sender_id, buf):
        if self.logger is None:
            return

        self.logger.warning(buf, uid=sender_id)

    def raise_exception(self, sender_id, exc):
        self.log_error(sender_id, 'Endpoint raised exception "%s"' % str(exc[1]))
        raise exc[1].with_traceback(exc[2].as_traceback())

    def hand(self, sender_id, address, port):
        self.proxy_from_uid(sender_id)

    def shake(self, sender_id, network):
        for uid, address in network.items():
            self.connect(sender_id, uid, *address)

    def connect(self, sender_id, uid, address, port):
        self.hand(uid, address, port)

    def disconnect(self, sender_id, uid):
        pass

    def stop(self, sender_id=None):
        if self._comms is not None:
            self._loop.run(self._comms.stop, args=(sender_id,))

    # Command and task management methods

    def register(self, obj):
        obj_type = obj.type
        obj_uid = obj.uid
        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            obj_store[obj_uid] = obj

        return obj_store[obj_uid]

    def unregister(self, obj):
        obj_type = obj.type
        obj_uid = obj.uid
        obj_store = getattr(self, '_' + obj_type)

        obj = obj_store.pop(obj_uid, None)

        if hasattr(obj, 'queue_task'):
            obj.queue_task((None, 'stop'))

        gc.collect()

    def cmd(self, sender_id, cmd):
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


class RuntimeProxy(BaseRPC):

    def __init__(self, name=None, indices=(), uid=None, comms=None):
        super().__init__(name=name, indices=indices, uid=uid)

        self._comms = comms or mosaic.get_comms()
        self._subprocess = None

    @property
    def address(self):
        return self._comms.uid_address(self.uid)

    @property
    def port(self):
        return self._comms.uid_port(self.uid)

    @property
    def subprocess(self):
        return self._subprocess

    @subprocess.setter
    def subprocess(self, subprocess):
        self._subprocess = subprocess

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)

        except AttributeError:
            def remote_method(**kwargs):
                wait = kwargs.pop('wait', False)
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

                else:
                    kwargs['wait'] = wait

                send_method = getattr(self._comms, send_method)
                return send_method(self.uid, **kwargs)

            return remote_method

    def __getitem__(self, item):
        return self.__getattribute__(item)
