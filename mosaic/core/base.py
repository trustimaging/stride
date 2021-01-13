
import datetime

import mosaic


__all__ = ['RemoteBase', 'ProxyBase', 'MonitoredBase']


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
        return self.runtime.logger


class CMDBase(Base):

    type = 'none'
    is_proxy = False
    is_remote = False

    def __init__(self, *args, **kwargs):
        super().__init__()

        self._uid = None
        self._state = ''

        # CMD specific config
        self.retries = 0
        self.max_retries = None

    def __repr__(self):
        NotImplementedError('Unimplemented Base method __repr__')

    @property
    def uid(self):
        return self._uid

    @property
    def state(self):
        return self._state

    @property
    def remote_runtime(self):
        raise NotImplementedError('Unimplemented Base property remote_runtime')

    @classmethod
    def remote_type(cls):
        NotImplementedError('Unimplemented Base method remote_type')

    def _fill_config(self, **kwargs):
        self.max_retries = kwargs.pop('max_retries', 0)

        return kwargs

    def _remotes(self):
        NotImplementedError('Unimplemented Base method _remotes')

    def proxy(self, uid):
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
        wait = kwargs.pop('wait', False)

        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(remote.cmd(**cmd, wait=wait, as_async=False))

        if len(result) == 1:
            result = result[0]

        return result

    def cmd_recv(self, method, *args, **kwargs):
        wait = kwargs.pop('wait', False)

        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(remote.cmd(**cmd, wait=wait, reply=True, as_async=False))

        if len(result) == 1:
            result = result[0]

        return result

    async def cmd_async(self, method, *args, **kwargs):
        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(await remote.cmd(**cmd))

        if len(result) == 1:
            result = result[0]

        return result

    async def cmd_recv_async(self, method, *args, **kwargs):
        remotes, cmd = self._prepare_cmd(method, *args, **kwargs)

        result = []
        for remote in remotes:
            result.append(await remote.cmd(**cmd, reply=True))

        if len(result) == 1:
            result = result[0]

        return result

    _serialisation_attrs = ['_uid', '_state']

    def _serialisation_helper(self):
        state = {}

        for attr in self._serialisation_attrs:
            state[attr] = getattr(self, attr)

        return state

    @classmethod
    def _deserialisation_helper(cls, state):
        instance = cls.__new__(cls)

        for attr, value in state.items():
            setattr(instance, attr, value)

        return instance

    def __reduce__(self):
        state = self._serialisation_helper()
        return self._deserialisation_helper, (state,)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class RemoteBase(CMDBase):

    is_proxy = False
    is_remote = True

    def __init__(self, uid, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._uid = uid
        self._ref_count = 1
        self._proxies = set()

    def __repr__(self):
        runtime_id = self.runtime.uid

        return "<%s object at %s, uid=%s, runtime=%s, state=%s>" % \
               (self.__class__.__name__, id(self),
                self.uid, runtime_id, self._state)

    @property
    def proxies(self):
        raise self._proxies

    @property
    def remote_runtime(self):
        raise NotImplementedError('Unimplemented RemoteBase property remote_runtime')

    @classmethod
    def remote_type(cls):
        return cls.type + '_proxy'

    def _remotes(self):
        return list(self.remote_runtime)

    def register_proxy(self, uid):
        self._proxies.add(uid)

    def unregister_proxy(self, uid):
        self._proxies.remove(uid)

    def inc_ref(self):
        self._ref_count += 1

    def dec_ref(self):
        self._ref_count -= 1

        if self._ref_count < 1:
            self.runtime.unregister(self)


class ProxyBase(CMDBase):

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
        instance = super()._deserialisation_helper(state)

        obj_type = cls.remote_type()

        instance = instance.runtime.register(instance)
        instance.remote_runtime.inc_ref(uid=instance.uid, type=obj_type, as_async=False)

        return instance

    def __del__(self):
        self.remote_runtime.dec_ref(uid=self.uid, type=self.remote_type(), as_async=False)
        self.runtime.unregister(self)


class MonitoredBase:

    def __init__(self, uid, runtime_id):
        self.uid = uid
        self.state = 'init'
        self.runtime_id = runtime_id

        self.time = -1
        self.history = []

    def update(self, **update):
        self.time = str(datetime.datetime.now())

        for key, value in update.items():
            setattr(self, key, value)

    def update_history(self, **update):
        self.update(**update)

        update['time'] = self.time
        self.history.append(update)

    def get_update(self):
        update = dict(
            state=self.state,
        )

        return update
