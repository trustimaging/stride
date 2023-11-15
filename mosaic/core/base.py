import asyncio
import functools
import time

import mosaic
from ..utils import Future
from ..profile import profiler


__all__ = ['RemoteBase', 'ProxyBase', 'RuntimeDisconnectedError']


class RuntimeDisconnectedError(Exception):
    pass


class Base:

    @property
    def runtime(self):
        return mosaic.runtime()

    @property
    def comms(self):
        return mosaic.get_comms()

    @property
    def zmq_context(self):
        return mosaic.get_zmq_context()

    @property
    def loop(self):
        return mosaic.get_event_loop()

    @property
    def head(self):
        return mosaic.get_head()

    @property
    def monitor(self):
        return mosaic.get_monitor()

    @property
    def node(self):
        return mosaic.get_node()

    @property
    def worker(self):
        return mosaic.get_worker()

    @property
    def logger(self):
        if self.runtime:
            return self.runtime.logger
        else:
            return mosaic.logger()


class CMDBase(Base):
    """
    Base class for objects that accept remote commands, such as tesserae and tasks, and their proxies.

    """

    type = 'none'
    is_proxy = False
    is_remote = False

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._uid = None
        self._state = ''
        self._registered = False
        self._init_future = Future()

        # CMD specific config
        self.retries = 0
        self.max_retries = None
        self.is_async = False

    async def __init_async__(self, *args, **kwargs):
        await self.init(*args, **kwargs)

        if self._init_future.done():
            exc = self._init_future.exception()
            if exc is not None:
                raise exc

        self._init_future.set_result(True)

        return self

    async def init(self, *args, **kwargs):
        pass

    def deregister_runtime(self, uid):
        if uid != self.runtime_id:
            return

        if self._init_future.done():
            self._init_future = Future()

        self.init_future.set_exception(
            RuntimeDisconnectedError('Remote runtime %s became disconnected' % uid)
        )

    def __repr__(self):
        NotImplementedError('Unimplemented Base method __repr__')

    @property
    def uid(self):
        """
        Object UID.

        """
        return self._uid

    @property
    def state(self):
        """
        Object state.

        """
        return self._state

    @property
    def init_future(self):
        """
        Init state of the object.

        """
        return self._init_future

    @property
    def remote_runtime(self):
        """
        Proxy to runtime where remote counterpart(s) is(are).

        """
        raise NotImplementedError('Unimplemented Base property remote_runtime')

    @property
    def runtime_id(self):
        """
        Runtime ID where remote object resides.

        """
        raise NotImplementedError('Unimplemented CMDBase property runtime_id')

    @property
    def collectable(self):
        """
        Whether the object is ready for collection.

        """
        return True

    @classmethod
    def remote_type(cls):
        """
        Type of the remote.

        """
        NotImplementedError('Unimplemented Base method remote_type')

    @classmethod
    def remote_cls(cls):
        """
        Class of the remote.

        """
        NotImplementedError('Unimplemented Base method remote_cls')

    def _fill_config(self, **kwargs):
        self.max_retries = kwargs.pop('max_retries', 0)
        self.is_async = kwargs.pop('is_async', False)

        return kwargs

    def _remotes(self):
        NotImplementedError('Unimplemented Base method _remotes')

    def proxy(self, uid):
        """
        Generate proxy for specific UID.

        Parameters
        ----------
        uid : str

        Returns
        -------
        ProxyBase

        """
        return self.runtime.proxy(uid)

    def _prepare_cmd(self, method, *args, **kwargs):
        obj_type = self.remote_type()
        remotes = self._remotes()

        cmd = {
            'type': obj_type,
            'uid': self._uid,
            'method': method,
            'args': args,
            'kwargs': kwargs,
        }

        return remotes, cmd

    def cmd(self, method, *args, **kwargs):
        """
        Send command to remote counterparts.

        Parameters
        ----------
        method : str
            Method of the command.
        args : tuple, optional
            Arguments for the command.
        kwargs : optional
            Keyword arguments for the command.

        Returns
        -------
        concurrent.futures.Future

        """
        wait = kwargs.pop('wait', False)

        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(remote.cmd(**cmd, wait=wait, as_async=False))

        if len(result) == 1:
            result = result[0]

        return result

    def cmd_recv(self, method, *args, **kwargs):
        """
        Send command to remote counterparts and await reply.

        Parameters
        ----------
        method : str
            Method of the command.
        args : tuple, optional
            Arguments for the command.
        kwargs : optional
            Keyword arguments for the command.

        Returns
        -------
        reply

        """
        wait = kwargs.pop('wait', False)

        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(remote.cmd(**cmd, wait=wait, reply=True, as_async=False))

        if len(result) == 1:
            result = result[0]

        return result

    async def cmd_async(self, method, *args, **kwargs):
        """
        Send async command to remote counterparts.

        Parameters
        ----------
        method : str
            Method of the command.
        args : tuple, optional
            Arguments for the command.
        kwargs : optional
            Keyword arguments for the command.

        Returns
        -------
        asyncio.Future

        """
        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(await remote.cmd(**cmd))

        if len(result) == 1:
            result = result[0]

        return result

    async def cmd_recv_async(self, method, *args, **kwargs):
        """
        Send async command to remote counterparts and await reply.

        Parameters
        ----------
        method : str
            Method of the command.
        args : tuple, optional
            Arguments for the command.
        kwargs : optional
            Keyword arguments for the command.

        Returns
        -------
        asyncio.Future

        """
        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(await remote.cmd(**cmd, reply=True))

        if len(result) == 1:
            result = result[0]

        return result

    async def start_trace(self):
        profiler.start_trace(self._uid)

    async def stop_trace(self):
        profiler.stop_trace(self._uid)

    def state_changed(self, state):
        """
        Signal state changed.

        Parameters
        ----------
        state : str
            New state.

        Returns
        -------

        """
        if self.runtime.uid == 'monitor':
            return

        self._state = state
        return self.add_event(state)

    def add_event(self, event_name, **kwargs):
        if self.runtime.uid == 'monitor' or self.is_proxy:
            return

        obj_type = self.type.split('_')[0]
        method_name = 'add_%s_event' % obj_type

        event_type = 'proxy' if self.is_proxy else 'remote'
        event_t = time.time()

        # *sender_id
        # runtime_id (remote_runtime_id)
        # uid,
        # *tessera_id
        # event_type
        # event_name
        # **kwargs
        event = dict(runtime_id=self.runtime_id,
                     uid=self._uid,
                     event_type=event_type,
                     event_name=event_name,
                     event_t=event_t, **kwargs)

        runtime = mosaic.runtime()
        runtime.maintenance_msg(method_name, event)

    def add_profile(self, profile, **kwargs):
        if self.runtime.uid == 'monitor' or self.is_proxy:
            return

        obj_type = self.type.split('_')[0]
        method_name = 'add_%s_profile' % obj_type

        profile_type = 'proxy' if self.is_proxy else 'remote'

        # *sender_id
        # runtime_id (remote_runtime_id)
        # uid,
        # *tessera_id
        # profile_type
        # profile
        profile_update = dict(runtime_id=self.runtime_id,
                              uid=self._uid,
                              profile_type=profile_type,
                              profile=profile,
                              **kwargs)

        runtime = mosaic.runtime()
        runtime.maintenance_msg(method_name, profile_update)

    _serialisation_attrs = ['_uid', '_state']

    def _serialisation_helper(self):
        state = {}

        for attr in self._serialisation_attrs:
            state[attr] = getattr(self, attr)

        return state

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = cls.__new__(cls)
        instance._init_future = Future()

        for attr, value in state.items():
            setattr(instance, attr, value)

        instance._init_future.set_result(True)

        return instance

    def __reduce__(self):
        state = self._serialisation_helper()
        return self._deserialisation_helper, (state,)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    async def deregister(self):
        try:
            self.logger.debug('Garbage collected object %s' % self)
            self.state_changed('collected')
        except AttributeError:
            pass


class RemoteBase(CMDBase):
    """
    Base class for CMD objects that live in a remote runtime (e.g. tesserae and tasks).

    """

    is_proxy = False
    is_remote = True

    def __init__(self, uid, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._uid = uid
        self._ref_count = 1
        self._proxies = dict()

        self._init_future.set_result(True)

    def __repr__(self):
        runtime_id = self.runtime_id

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self._state)

    @property
    def runtime_id(self):
        """
        Runtime ID where remote object resides.

        """
        return self.runtime.uid

    @property
    def proxies(self):
        """
        Set of proxies that keep references to this remote.

        """
        proxies = set(self._proxies.keys())
        raise proxies

    @property
    def remote_runtime(self):
        raise NotImplementedError('Unimplemented RemoteBase property remote_runtime')

    @classmethod
    def remote_type(cls):
        return cls.type + '_proxy'

    def _remotes(self):
        return list(self.remote_runtime)

    def register_proxy(self, uid):
        """
        Register a new proxy pointing to this remote.

        Parameters
        ----------
        uid : str

        Returns
        -------

        """
        if uid not in self._proxies:
            self._proxies[uid] = 0

        self._proxies[uid] += 1

    def deregister_proxy(self, uid):
        """
        Deregister proxy pointing to this remote.

        Parameters
        ----------
        uid : str

        Returns
        -------

        """
        if uid not in self._proxies:
            return

        self._proxies[uid] -= 1

        if self._proxies[uid] < 1:
            del self._proxies[uid]

    def inc_ref(self):
        """
        Increase reference count.

        Returns
        -------

        """
        self._ref_count += 1

    def dec_ref(self):
        """
        Decrease reference count and deregister from runtime if needed.

        Returns
        -------

        """
        self._ref_count -= 1

        if self._ref_count < 1 and self.runtime is not None:
            self.runtime.deregister(self)

    _serialisation_attrs = CMDBase._serialisation_attrs + []

    @classmethod
    def _deserialisation_helper(cls, state):
        remote_cls = cls.remote_cls()
        return remote_cls._deserialisation_helper(state)


class ProxyBase(CMDBase):
    """
    Base class for CMD objects that represent proxies to remote objects (e.g. tessera proxies and task proxies).

    """

    is_proxy = True
    is_remote = False

    def __repr__(self):
        runtime_id = self.runtime_id

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self._state)

    @property
    def runtime_id(self):
        raise NotImplementedError('Unimplemented ProxyBase property runtime_id')

    @property
    def remote_runtime(self):
        raise NotImplementedError('Unimplemented ProxyBase property remote_runtime')

    @classmethod
    def remote_type(cls):
        return cls.type.split('_')[0]

    def _remotes(self):
        return [self.remote_runtime]

    _serialisation_attrs = CMDBase._serialisation_attrs + []

    @classmethod
    def _deserialisation_helper(cls, state):
        obj_type = cls.type
        obj_uid = state.get('_uid', None)

        runtime = mosaic.runtime()
        needs_registering, reg_instance = runtime.needs_registering(obj_type, obj_uid)

        if not needs_registering:
            return reg_instance

        instance = super()._deserialisation_helper(state)
        instance._registered = False

        if instance.runtime.uid == 'monitor':
            return instance

        obj_type = cls.remote_type()

        reg_instance = runtime.register(instance)
        if instance.is_proxy and instance._registered:
            reg_instance.remote_runtime.inc_ref(uid=reg_instance.uid, type=obj_type, as_async=False)
            reg_instance.state_changed('listening')

        return reg_instance

    def __del__(self):
        if self._registered and self.runtime:
            self.runtime.deregister(self)

    async def deregister(self):
        await super().deregister()
        return self.remote_runtime.uid, \
            self.remote_runtime.dec_refs, \
            dict(uid=self.uid, type=self.remote_type())
        # await self.remote_runtime.dec_ref(uid=self.uid, type=self.remote_type())
