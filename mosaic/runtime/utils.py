
import uuid
import copy
import time
from collections import OrderedDict

import mosaic
from .. import types
from ..file_manipulation import h5


__all__ = ['MonitoredResource', 'MonitoredObject', 'WarehouseObject']


class MonitoredResource:
    """
    Base class for those that keep track of the state of a Mosaic runtime,

    """

    def __init__(self, uid):
        self.uid = self.name = uid
        self.history = []
        self.sub_resources = dict()

        self._last_update = 0
        self._last_append = 0

    @property
    def state(self):
        try:
            return self.history[-1]['state']
        except IndexError:
            return None

    @state.setter
    def state(self, value):
        last_event = copy.deepcopy(self.history[-1])
        last_event['state'] = value
        self.update(last_event)

    def update(self, update, **kwargs):
        if not isinstance(update, list):
            update = [update]

        self.history.extend(update)

        for group_name, group_update in kwargs.items():
            self.add_group(group_name)

            group = self.sub_resources[group_name]

            for name, resource in group_update.items():
                self.add_resource(group_name, name)

                group[name].update(resource)

    def get_update(self):
        history = self.history[self._last_update:]
        self._last_update = len(self.history)

        sub_resources = OrderedDict()
        for group_name, group in self.sub_resources.items():
            sub_resources[group_name] = OrderedDict()

            for name, resource in group.items():
                sub_resources[group_name][name] = resource.get_update()[0]

        return history, sub_resources

    def append(self, filename=None):
        history = self.history[self._last_append:]

        node_description = {}
        for index, item in enumerate(history):
            label = 'history_%08d' % (self._last_append + index)
            node_description[label] = item

        self._last_append = len(self.history)

        sub_description = {}
        for group_name, group in self.sub_resources.items():
            group_description = dict()

            for name, resource in group.items():
                resource_description = resource.append()
                group_description[name] = resource_description

            sub_description[group_name] = group_description

        description = {
            'history': node_description,
            'sub_resources': sub_description,
        }

        if filename is None:
            return description

        if not h5.file_exists(filename=filename):
            with h5.HDF5(filename=filename, mode='w') as file:
                file.dump(description)

        else:
            with h5.HDF5(filename=filename, mode='a') as file:
                file.append(description)

    def add_group(self, group_name):
        if group_name not in self.sub_resources:
            self.sub_resources[group_name] = OrderedDict()

        return self.sub_resources[group_name]

    def add_resource(self, group_name, name):
        group = self.sub_resources[group_name]

        if name not in group:
            group[name] = MonitoredResource(name)

        return group[name]

    def sort_resources(self, group_name, key, desc=False):
        group = self.sub_resources[group_name]

        self.sub_resources[group_name] = OrderedDict(sorted(group.items(),
                                                     key=lambda x: x[1].history[-1][key],
                                                     reverse=desc))


class MonitoredObject:
    """
    Base class for those that keep track of the state of a Mosaic object,

    """

    def __init__(self, runtime_id, uid, tessera_id=None):
        self.uid = uid
        self.runtime_id = runtime_id

        if tessera_id is not None:
            self.tessera_id = tessera_id

        self.proxy_events = OrderedDict()
        self.proxy_profiles = OrderedDict()
        self.remote_events = []
        self.remote_profiles = []

        self._last_proxy_event = dict()
        self._last_proxy_profile = dict()
        self._last_remote_event = 0
        self._last_remote_profile = 0

    @property
    def state(self):
        try:
            return self.remote_events[-1]['name']
        except IndexError:
            return None

    @state.setter
    def state(self, value):
        event = {
            'name': 'collected',
            'event_t': time.time(),
        }

        self.remote_events.append(event)

    def collect(self):
        event = {
            'name': 'collected',
            'event_t': time.time(),
        }

        if self.state != 'collected':
            self.remote_events.append(event)

        for runtime_id, proxy in self.proxy_events.items():
            if proxy[-1]['name'] != 'collected':
                proxy.append(event)

    def add_event(self, runtime_id, event_type, event_name, **kwargs):
        event_uid = uuid.uuid4().hex
        event = dict(name=event_name, event_uid=event_uid, **kwargs)

        if event_type == 'proxy':
            if runtime_id not in self.proxy_events:
                self.proxy_events[runtime_id] = []

            self.proxy_events[runtime_id].append(event)
        elif event_type == 'remote':
            self.remote_events.append(event)

    def add_profile(self, runtime_id, profile_type, profile):
        if profile_type == 'proxy':
            if runtime_id not in self.proxy_profiles:
                self.proxy_profiles[runtime_id] = []

            self.proxy_profiles[runtime_id].append(profile)
        elif profile_type == 'remote':
            self.remote_profiles.append(profile)

    def append(self, filename=None):
        remote_events = self.remote_events[self._last_remote_event:]
        remote_profiles = self.remote_profiles[self._last_remote_profile:]

        remote_events_description = {}
        for index, item in enumerate(remote_events):
            label = 'history_%08d' % (self._last_remote_event + index)
            remote_events_description[label] = item

        remote_profiles_description = {}
        for index, item in enumerate(remote_profiles):
            label = 'history_%08d' % (self._last_remote_profile + index)
            remote_profiles_description[label] = item

        self._last_remote_event = len(self.remote_events)
        self._last_remote_profile = len(self.remote_profiles)

        proxy_events_description = {}
        for proxy_name, proxy in self.proxy_events.items():
            try:
                last_event = self._last_proxy_event[proxy_name]
            except KeyError:
                last_event = 0

            proxy_events = proxy[last_event:]

            proxy_description = {}
            for index, item in enumerate(proxy_events):
                label = 'history_%08d' % (last_event + index)
                proxy_description[label] = item

            proxy_events_description[proxy_name] = proxy_description

            self._last_proxy_event[proxy_name] = len(proxy)

        proxy_profiles_description = {}
        for proxy_name, proxy in self.proxy_profiles.items():
            try:
                last_profile = self._last_proxy_profile[proxy_name]
            except KeyError:
                last_profile = 0

            proxy_profiles = proxy[last_profile:]

            proxy_description = {}
            for index, item in enumerate(proxy_profiles):
                label = 'history_%08d' % (last_profile + index)
                proxy_description[label] = item

            proxy_profiles_description[proxy_name] = proxy_description

            self._last_proxy_profile[proxy_name] = len(proxy)

        description = {
            'uid': self.uid,
            'runtime_id': self.runtime_id,
            'remote_events': remote_events_description,
            'remote_profiles': remote_profiles_description,
            'proxy_events': proxy_events_description,
            'proxy_profiles': proxy_profiles_description,
        }

        if hasattr(self, 'tessera_id'):
            description['tessera_id'] = self.tessera_id

        if filename is None:
            return description

        if not h5.file_exists(filename=filename):
            with h5.HDF5(filename=filename, mode='w') as file:
                file.dump(description)

        else:
            with h5.HDF5(filename=filename, mode='a') as file:
                file.append(description)


class WarehouseObject:
    """
    Represents a reference to an object that is stored into the warehouse.

    Parameters
    ----------
    obj : object
        Object being added.

    """

    def __init__(self, obj=None, uid=None):
        """

        Parameters
        ----------
        obj
        uid

        """
        if uid is None:
            self._name = obj.__class__.__name__.lower()
            self._uid = '%s-%s-%s' % ('ware',
                                      self._name,
                                      uuid.uuid4().hex)
        else:
            self._name = uid
            self._uid = uid

        self._tessera = None

    @property
    def state(self):
        return 'done'

    @property
    def uid(self):
        """
        UID of the object.

        """
        return self._uid

    @property
    def runtime(self):
        """
        Current runtime object.

        """
        return mosaic.runtime()

    @property
    def has_tessera(self):
        return hasattr(self, '_tessera') \
               and self._tessera is not None

    @property
    def is_tessera(self):
        return False

    @property
    def is_proxy(self):
        return True

    async def value(self):
        """
        Pull the underlying value from the warehouse.

        """
        return await self.runtime.get(self.uid)

    async def result(self):
        """
        Alias for WarehouseObject.value.

        """
        return await self.value()

    def __await__(self):
        return self.value().__await__()

    def __repr__(self):
        return "<warehouse object uid=%s>" % self.uid


types.awaitable_types += (WarehouseObject,)
