
import os
import uuid
import time
import copy
import psutil
import asyncio
import datetime
from collections import OrderedDict
import subprocess as cmd_subprocess

import mosaic
from .runtime import Runtime, RuntimeProxy
from .strategies import RoundRobin
from ..file_manipulation import h5
from ..utils import subprocess
from ..utils.logger import LoggerManager, _stdout, _stderr
from ..profile import profiler, global_profiler


__all__ = ['Monitor', 'MonitoredResource', 'MonitoredObject', 'monitor_strategies']


monitor_strategies = {
    'round-robin': RoundRobin
}


class Monitor(Runtime):
    """
    The monitor takes care of keeping track of the state of the network
    and collects statistics about it.

    It also handles the allocation of tesserae to certain workers.

    """

    is_monitor = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.strategy_name = kwargs.get('monitor_strategy', 'round-robin')
        self._monitor_strategy = monitor_strategies[self.strategy_name](self)

        self._monitored_nodes = dict()
        self._monitored_tessera = dict()
        self._monitored_tasks = dict()

        self._dirty_tessera = set()
        self._dirty_tasks = set()

        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self._profile_filename = '%s.profile.h5' % now

        self._start_t = time.time()
        self._end_t = None

    async def init(self, **kwargs):
        """
        Asynchronous counterpart of ``__init__``.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        await super().init(**kwargs)

        # if self.mode == 'local':
        #     available_cpus = list(range(psutil.cpu_count()))
        #     psutil.Process().cpu_affinity([available_cpus[0]])

        # Start local cluster
        if self.mode in ['local', 'interactive']:
            await self.init_local(**kwargs)

        else:
            await self.init_cluster(**kwargs)

    async def init_local(self, **kwargs):
        """
        Init nodes in local mode.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        def start_node(*args, **extra_kwargs):
            kwargs.update(extra_kwargs)
            kwargs['runtime_indices'] = 0

            mosaic.init('node', *args, **kwargs, wait=True)

        node_proxy = RuntimeProxy(name='node', indices=0)
        node_subprocess = subprocess(start_node)(name=node_proxy.uid, daemon=False)
        node_subprocess.start_process()
        node_proxy.subprocess = node_subprocess

        self._nodes[node_proxy.uid] = node_proxy
        await self._comms.wait_for(node_proxy.uid)

    async def init_cluster(self, **kwargs):
        """
        Init nodes in cluster mode.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        node_list = kwargs.get('node_list', None)

        if node_list is None:
            raise ValueError('No node_list was provided to initialise mosaic in cluster mode')

        num_nodes = len(node_list)
        num_workers = kwargs.get('num_workers', 1)
        num_threads = kwargs.get('num_threads', None)
        log_level = kwargs.get('log_level', 'info')
        runtime_address = self.address
        runtime_port = self.port

        ssh_flags = os.environ.get('SSH_FLAGS', None)
        ssh_flags = ssh_flags + ';' if ssh_flags else ''

        for node_index, node_address in zip(range(num_nodes), node_list):
            node_proxy = RuntimeProxy(name='node', indices=node_index)

            cmd = (f'ssh {node_address} '
                   f'"{ssh_flags} ' 
                   f'mrun --node -i {node_index} '
                   f'--monitor-address {runtime_address} --monitor-port {runtime_port} '
                   f'-n {num_nodes} -nw {num_workers} -nth {num_threads} '
                   f'--cluster --{log_level}"')

            node_subprocess = cmd_subprocess.Popen(cmd,
                                                   shell=True,
                                                   stdout=_stdout,
                                                   stderr=_stderr)
            node_proxy.subprocess = node_subprocess

            self._nodes[node_proxy.uid] = node_proxy
            await self._comms.wait_for(node_proxy.uid)

            self.logger.info('Started node %d at %s' % (node_index, node_address))

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        self.logger = LoggerManager()

        if self.mode == 'interactive':
            self.logger.set_remote(runtime_id='head', format=self.mode)
        else:
            self.logger.set_local(format=self.mode)

    def set_profiler(self):
        """
        Set up profiling.

        Returns
        -------

        """
        global_profiler.set_local()

        self._loop.interval(self.append_description, interval=10)

    def update_node(self, sender_id, update, sub_resources):
        if sender_id not in self._monitored_nodes:
            self._monitored_nodes[sender_id] = MonitoredResource(sender_id)

        node = self._monitored_nodes[sender_id]
        node.update(update, **sub_resources)
        self._monitor_strategy.update_node(node)

        # if profiler.tracing:
        #     node_description = node.append()
        #     description = {
        #         'monitored_nodes': {
        #             sender_id: node_description
        #         }
        #     }
        #
        #     self._append_description(description)

    def add_tessera_event(self, sender_id, runtime_id, uid, **kwargs):
        if uid not in self._monitored_tessera:
            self._monitored_tessera[uid] = MonitoredObject(runtime_id, uid)

        obj = self._monitored_tessera[uid]
        obj.add_event(sender_id, **kwargs)
        self._monitor_strategy.update_tessera(obj)
        self._dirty_tessera.add(uid)

        # if profiler.tracing:
        #     obj_description = obj.append()
        #     description = {
        #         'monitored_tessera': {
        #             uid: obj_description
        #         }
        #     }
        #
        #     self._append_description(description)

    def add_task_event(self, sender_id, runtime_id, uid, tessera_id, **kwargs):
        if uid not in self._monitored_tasks:
            self._monitored_tasks[uid] = MonitoredObject(runtime_id, uid, tessera_id=tessera_id)

        obj = self._monitored_tasks[uid]
        obj.add_event(sender_id, **kwargs)
        self._monitor_strategy.update_task(obj)
        self._dirty_tasks.add(uid)

        # if profiler.tracing:
        #     obj_description = obj.append()
        #     description = {
        #         'monitored_tasks': {
        #             uid: obj_description
        #         }
        #     }
        #
        #     self._append_description(description)

    def add_tessera_profile(self, sender_id, runtime_id, uid, profile):
        if uid not in self._monitored_tessera:
            self._monitored_tessera[uid] = MonitoredObject(runtime_id, uid)

        obj = self._monitored_tessera[uid]
        obj.add_profile(sender_id, profile)
        self._dirty_tessera.add(uid)

        # if profiler.tracing:
        #     obj_description = obj.append()
        #     description = {
        #         'monitored_tessera': {
        #             uid: obj_description
        #         }
        #     }
        #
        #     self._append_description(description)

    def add_task_profile(self, sender_id, runtime_id, uid, tessera_id, profile):
        if uid not in self._monitored_tasks:
            self._monitored_tasks[uid] = MonitoredObject(runtime_id, uid, tessera_id=tessera_id)

        obj = self._monitored_tasks[uid]
        obj.add_profile(sender_id, profile)
        self._dirty_tasks.add(uid)

        # if profiler.tracing:
        #     obj_description = obj.append()
        #     description = {
        #         'monitored_tasks': {
        #             uid: obj_description
        #         }
        #     }
        #
        #     self._append_description(description)

    def append_description(self):
        if not profiler.tracing:
            return

        description = {
            'monitored_tessera': {},
            'monitored_tasks': {},
        }

        for uid in self._dirty_tessera:
            tessera = self._monitored_tessera[uid]
            description['monitored_tessera'][uid] = tessera.append()

        for uid in self._dirty_tasks:
            task = self._monitored_tasks[uid]
            description['monitored_tasks'][uid] = task.append()

        self._append_description(description)

        self._dirty_tessera = set()
        self._dirty_tasks = set()

    def _append_description(self, description):
        start = time.time()
        if not h5.file_exists(filename=self._profile_filename):
            description['start_t'] = self._start_t

            with h5.HDF5(filename=self._profile_filename, mode='w') as file:
                file.dump(description)

        else:
            with h5.HDF5(filename=self._profile_filename, mode='a') as file:
                file.append(description)

        print('====> SAVING!', time.time()-start)

    async def stop(self, sender_id=None):
        """
        Stop runtime.

        Parameters
        ----------
        sender_id : str

        Returns
        -------

        """
        # Get final profile updates before closing
        if profiler.tracing:
            profiler.stop()
            self._end_t = time.time()

            description = {
                'end_t': self._end_t,
                'monitored_tessera': {},
                'monitored_tasks': {},
            }

            for uid, tessera in self._monitored_tessera.items():
                tessera.collect()
                description['monitored_tessera'][tessera.uid] = tessera.append()

            for uid, task in self._monitored_tasks.items():
                task.collect()
                description['monitored_tasks'][task.uid] = task.append()

            self._append_description(description)

        for node_id, node in self._nodes.items():
            await node.stop()

            if hasattr(node.subprocess, 'stop_process'):
                node.subprocess.join_process()

            if isinstance(node.subprocess, cmd_subprocess.Popen):
                ps_process = psutil.Process(node.subprocess.pid)

                for child in ps_process.children(recursive=True):
                    child.kill()

                ps_process.kill()

        super().stop(sender_id)

    async def select_worker(self, sender_id):
        """
        Select appropriate worker to allocate a tessera.

        Parameters
        ----------
        sender_id : str

        Returns
        -------
        str
            UID of selected worker.

        """
        while not len(self._monitored_nodes.keys()):
            await asyncio.sleep(0.1)

        return self._monitor_strategy.select_worker(sender_id)

    async def barrier(self, sender_id, timeout=None):
        """
        Wait until all pending tasks are done. If no timeout is
        provided, the barrier will wait indefinitely.

        Parameters
        ----------
        timeout : float, optional

        Returns
        -------

        """
        pending_tasks = []
        for task in self._monitored_tasks.values():
            if task.state in ['done', 'failed', 'collected']:
                continue

            pending_tasks.append(task)

        tic = time.time()
        while pending_tasks:
            await asyncio.sleep(0.01)

            for task in pending_tasks:
                if task.state in ['done', 'failed', 'collected']:
                    pending_tasks.remove(task)

            if timeout is not None and (time.time() - tic) > timeout:
                break


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
