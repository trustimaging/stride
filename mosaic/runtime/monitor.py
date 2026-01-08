
import os
import time
import psutil
import asyncio
import datetime
import subprocess as cmd_subprocess
from collections import defaultdict

import mosaic
from .runtime import Runtime, RuntimeProxy
from .utils import MonitoredResource, MonitoredObject
from .strategies import RoundRobin
from ..file_manipulation import h5
from ..utils import subprocess, at_exit
from ..utils.utils import memory_limit, cpu_count
from ..utils.logger import LoggerManager, _stdout, _stderr
from ..profile import profiler, global_profiler


__all__ = ['Monitor', 'monitor_strategies']


monitor_strategies = {
    'round-robin': RoundRobin
}


def _cpu_mask(num_workers, worker_index, num_threads):
    # Work out the first core ID for this subjob
    startid = (worker_index - 1) * num_threads

    # This is the process CPU ID
    valsum = {}
    for j in range(0, num_threads):
        # Thread CPU ID
        threadid = startid + j
        # Convert to bitmask components
        pos = int(threadid / 4)
        offset = threadid - pos * 4
        val = 2 ** offset
        # This is a fat bitmask so add up the thread values in the right position
        valsum[pos] = valsum.get(pos, 0) + val

    valmask = ''
    # Generate the hex repreesntation of the fat bitmask
    for j in range(max(valsum.keys()), -1, -1):
        valmask = f'{valmask}{valsum.get(j, 0):X}'

    # Append to the list of masks in the appropriate way for this subjob
    mask = '0x' + f'{valmask}'

    return mask


class Monitor(Runtime):
    """
    The monitor takes care of keeping track of the state of the network
    and collects statistics about it.

    It also handles the allocation of tesserae to certain workers.

    """

    is_monitor = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._memory_limit = memory_limit()

        self.strategy_name = kwargs.get('monitor_strategy', 'round-robin')
        self._monitor_strategy = monitor_strategies[self.strategy_name](self)

        self._monitored_nodes = dict()
        self._monitored_tessera = dict()
        self._monitored_tasks = dict()
        self._disconnected_runtimes = set()

        self._runtime_tessera = defaultdict(list)
        self._runtime_tasks = defaultdict(list)

        self._dirty_tessera = set()
        self._dirty_tasks = set()

        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self._profile_filename = '%s.profile.h5' % now
        self._init_filename = None

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

        # Maybe dump file
        if kwargs.get('dump_init', False):
            self.init_file({})

        # Start local cluster
        await self.init_warehouse(**kwargs)

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
        num_workers = kwargs.get('num_workers', 1)

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

        while node_proxy.uid not in self._monitored_nodes:
            await asyncio.sleep(0.1)

        self._comms.start_heartbeat(node_proxy.uid)

        self.logger.info('Listening at <NODE:0 | WORKER:0-%d>' % num_workers)

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

        num_cpus = cpu_count()
        num_nodes = len(node_list)
        num_workers = kwargs.get('num_workers', 1)
        num_threads = kwargs.get('num_threads', None) or num_cpus // num_workers
        log_level = kwargs.get('log_level', 'info')
        runtime_address = self.address
        runtime_port = self.port
        pubsub_port = self.pubsub_port

        ssh_flags = os.environ.get('SSH_FLAGS', '')
        ssh_commands = os.environ.get('SSH_COMMANDS', None)
        ssh_commands = ssh_commands + ';' if ssh_commands else ''
        reuse_head = '--reuse-head' if self.reuse_head else '--free-head'

        in_slurm = os.environ.get('SLURM_NODELIST', None) is not None

        tasks = []

        for node_index, node_address in zip(range(num_nodes), node_list):
            node_proxy = RuntimeProxy(name='node', indices=node_index)

            remote_cmd = (f'{ssh_commands} '
                          f'mrun --node -i {node_index} '
                          f'--monitor-address {runtime_address} --monitor-port {runtime_port} '
                          f'--pubsub-port {pubsub_port} '
                          f'-n {num_nodes} -nw {num_workers} -nth {num_threads} '
                          f'--cluster --{log_level} {reuse_head}')

            if in_slurm:
                cpu_mask = _cpu_mask(1, 1, num_cpus)

                cmd = (f'srun {ssh_flags} --nodes=1 --ntasks=1 --tasks-per-node={num_cpus} '
                       f'--cpu-bind=mask_cpu:{cpu_mask} --mem-bind=local '
                       f'--oversubscribe '
                       f'--distribution=block:block '
                       f'--hint=nomultithread --no-kill '
                       f'--nodelist={node_address} '
                       f'{remote_cmd}')

            else:
                cmd = (f'ssh {ssh_flags} {node_address} '
                       f'"{remote_cmd}"')

            node_subprocess = cmd_subprocess.Popen(cmd,
                                                   shell=True,
                                                   stdout=_stdout,
                                                   stderr=_stderr)
            node_proxy.subprocess = node_subprocess
            self._nodes[node_proxy.uid] = node_proxy

            async def wait_for(proxy):
                await self._comms.wait_for(proxy.uid)

                while proxy.uid not in self._monitored_nodes:
                    await asyncio.sleep(0.1)

                return proxy

            tasks.append(wait_for(node_proxy))

        for node_proxy in asyncio.as_completed(tasks):
            node_proxy = await node_proxy
            self.logger.debug('Started node %s' % node_proxy.uid)

        for node_uid in self._nodes.keys():
            self._comms.start_heartbeat(node_uid)
            self.logger.debug('Started heartbeat with node %s' % node_uid)

        self.logger.info('Listening at <NODE:%d-%d | '
                         'WORKER:0-%d address=%s>' % (0, num_nodes, num_workers,
                                                      ', '.join(node_list)))

    def init_file(self, runtime_config):
        runtime_id = self.uid
        runtime_address = self.address
        runtime_port = self.port
        pubsub_port = self.pubsub_port

        # Store runtime ID, address and port in a tmp file for the
        # head to use
        path = os.path.join(os.getcwd(), 'mosaic-workspace')
        if not os.path.exists(path):
            os.makedirs(path)

        self._init_filename = os.path.join(path, 'monitor.key')
        with open(self._init_filename, 'w') as file:
            file.write('[ADDRESS]\n')
            file.write('UID=%s\n' % runtime_id)
            file.write('ADD=%s\n' % runtime_address)
            file.write('PRT=%s\n' % runtime_port)
            file.write('PUB=%s\n' % pubsub_port)
            file.write('[ARGS]\n')

            for key, value in runtime_config.items():
                if key in ['runtime_indices', 'address', 'port',
                           'monitor_address', 'monitor_port', 'node_list']:
                    continue
                if isinstance(value, str):
                    file.write('%s="%s"\n' % (key, value))
                else:
                    file.write('%s=%s\n' % (key, value))

        def _rm_dirs():
            os.remove(self._init_filename)

        at_exit.add(_rm_dirs)

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
        if sender_id in self._disconnected_runtimes:
            return
        if sender_id not in self._monitored_nodes:
            self._monitored_nodes[sender_id] = MonitoredResource(sender_id)

        node = self._monitored_nodes[sender_id]
        node.update(update, **sub_resources)
        self._monitor_strategy.update_node(node)

    def add_tessera_event(self, sender_id, msgs):
        if sender_id in self._disconnected_runtimes:
            return
        msgs = [msgs] if not isinstance(msgs, list) else msgs
        for msg in msgs:
            self._add_tessera_event(sender_id, **msg)

    def add_task_event(self, sender_id, msgs):
        if sender_id in self._disconnected_runtimes:
            return
        msgs = [msgs] if not isinstance(msgs, list) else msgs
        for msg in msgs:
            self._add_task_event(sender_id, **msg)

    def add_tessera_profile(self, sender_id, msgs):
        if sender_id in self._disconnected_runtimes:
            return
        msgs = [msgs] if not isinstance(msgs, list) else msgs
        for msg in msgs:
            self._add_tessera_profile(sender_id, **msg)

    def add_task_profile(self, sender_id, msgs):
        if sender_id in self._disconnected_runtimes:
            return
        msgs = [msgs] if not isinstance(msgs, list) else msgs
        for msg in msgs:
            self._add_task_profile(sender_id, **msg)

    def _add_tessera_event(self, sender_id, runtime_id, uid, **kwargs):
        if runtime_id in self._disconnected_runtimes:
            return
        if uid not in self._monitored_tessera:
            self._monitored_tessera[uid] = MonitoredObject(runtime_id, uid)
            self._runtime_tessera[runtime_id].append(uid)

        obj = self._monitored_tessera[uid]
        obj.add_event(sender_id, **kwargs)
        self._monitor_strategy.update_tessera(obj)
        self._dirty_tessera.add(uid)

    def _add_task_event(self, sender_id, runtime_id, uid, tessera_id, **kwargs):
        if runtime_id in self._disconnected_runtimes:
            return
        if uid not in self._monitored_tasks:
            self._monitored_tasks[uid] = MonitoredObject(runtime_id, uid, tessera_id=tessera_id)
            self._runtime_tasks[runtime_id].append(uid)

        obj = self._monitored_tasks[uid]
        obj.add_event(sender_id, **kwargs)
        self._monitor_strategy.update_task(obj)
        self._dirty_tasks.add(uid)

    def _add_tessera_profile(self, sender_id, runtime_id, uid, profile):
        if runtime_id in self._disconnected_runtimes:
            return
        if uid not in self._monitored_tessera:
            self._monitored_tessera[uid] = MonitoredObject(runtime_id, uid)
            self._runtime_tessera[runtime_id].append(uid)

        obj = self._monitored_tessera[uid]
        obj.add_profile(sender_id, profile)
        self._dirty_tessera.add(uid)

    def _add_task_profile(self, sender_id, runtime_id, uid, tessera_id, profile):
        if runtime_id in self._disconnected_runtimes:
            return
        if uid not in self._monitored_tasks:
            self._monitored_tasks[uid] = MonitoredObject(runtime_id, uid, tessera_id=tessera_id)
            self._runtime_tasks[runtime_id].append(uid)

        obj = self._monitored_tasks[uid]
        obj.add_profile(sender_id, profile)
        self._dirty_tasks.add(uid)

    def append_description(self):
        if not profiler.tracing:
            return

        if not len(self._dirty_tessera) and not len(self._dirty_tasks):
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
        if not h5.file_exists(filename=self._profile_filename):
            description['start_t'] = self._start_t

            with h5.HDF5(filename=self._profile_filename, mode='w') as file:
                file.dump(description)

        else:
            with h5.HDF5(filename=self._profile_filename, mode='a') as file:
                file.append(description)

    async def stop(self, sender_id=None):
        """
        Stop runtime.

        Parameters
        ----------
        sender_id : str

        Returns
        -------

        """
        # Delete files
        if self._init_filename is not None:
            try:
                os.remove(self._init_filename)
            except Exception:
                pass

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

        # Close warehouse
        await self._local_warehouse.stop()
        self._local_warehouse.subprocess.join_process()

        # Close nodes
        for node_id, node in self._nodes.items():
            await node.stop()

            if hasattr(node.subprocess, 'stop_process'):
                node.subprocess.join_process()

            if isinstance(node.subprocess, cmd_subprocess.Popen):
                ps_process = psutil.Process(node.subprocess.pid)

                for child in ps_process.children(recursive=True):
                    child.kill()

                ps_process.kill()

        await super().stop(sender_id)

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
        super().disconnect(sender_id, uid)
        self._disconnected_runtimes.add(uid)

        # remove runtime from monitored nodes
        try:
            del self._monitored_nodes[uid]
        except KeyError:
            pass

        # remove monitored tessera
        for obj_uid in self._runtime_tessera[uid]:
            try:
                del self._monitored_tessera[obj_uid]
            except KeyError:
                pass
        del self._runtime_tessera[uid]

        # remove monitored tasks
        for obj_uid in self._runtime_tasks[uid]:
            try:
                del self._monitored_tasks[obj_uid]
            except KeyError:
                pass
        del self._runtime_tasks[uid]

        # disconnect associated workers
        if uid in self._monitored_nodes:
            for worker_id in self._monitored_nodes[uid].sub_resources['workers'].keys():
                self.disconnect(sender_id, worker_id)

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
        self.logger.info('Pending barrier tasks %d' % len(pending_tasks))

        num_tasks = len(self._monitored_tasks)

        tic = time.time()
        while pending_tasks:
            await asyncio.sleep(0.1)

            for task in pending_tasks:
                if task.state in ['done', 'failed', 'collected']:
                    pending_tasks.remove(task)

            if len(self._monitored_tasks) > num_tasks:
                for task in self._monitored_tasks.values():
                    if task.state not in ['done', 'failed', 'collected'] and task not in pending_tasks:
                        pending_tasks.append(task)

            if timeout is not None and (time.time() - tic) > timeout:
                break

        await self._local_warehouse.run_barrier_tasks(reply=True)

        self._monitored_tasks = dict()
        self._runtime_tasks = defaultdict(list)
        self._dirty_tasks = set()
