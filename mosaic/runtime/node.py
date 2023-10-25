
import psutil

import mosaic
from .runtime import Runtime, RuntimeProxy
from .monitor import MonitoredResource
from ..utils import LoggerManager
from ..utils import subprocess
from ..utils.utils import memory_limit, cpu_count
from ..profile import profiler, global_profiler


__all__ = ['Node']


class Node(Runtime):
    """
    A node represents a physically independent portion of the network,
    such as a separate cluster node. Nodes contain one or more
    workers, which they initialise and manage.

    """

    is_node = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        num_workers = kwargs.pop('num_workers', None)
        num_workers = num_workers or 1

        self._own_workers = dict()
        self._num_workers = num_workers
        self._num_threads = None
        self._memory_limit = memory_limit()

        self._monitored_node = MonitoredResource(self.uid)

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

        # Start local workers
        await self.init_workers(**kwargs)

    async def init_workers(self, **kwargs):
        """
        Init workers in the node.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        if self.mode == 'cluster':
            pass

        num_cpus = cpu_count()

        num_workers = self._num_workers
        num_threads = kwargs.pop('num_threads', None)
        num_threads = num_threads or num_cpus // num_workers
        self._num_threads = num_threads

        if num_workers*num_threads > num_cpus:
            raise ValueError('Requested number of CPUs per node (%d - num_workers*num_threads) '
                             'is greater than the number of available CPUs (%d)' % (num_workers*num_threads, num_cpus))

        # Find all available NUMA nodes and CPUs per node
        try:
            import numa
            numa_available = numa.info.numa_available()
        except Exception:
            numa_available = False

        if numa_available:
            available_cpus = numa.info.numa_hardware_info()['node_cpu_info']
        else:
            available_cpus = {worker_index: list(range(num_threads*worker_index,
                                                       num_threads*(worker_index+1)))
                              for worker_index in range(self._num_workers)}

        # Eliminate cores corresponding to hyperthreading
        for node_index, node_cpus in available_cpus.items():
            node_cpus = [each for each in node_cpus if each < num_cpus]
            available_cpus[node_index] = node_cpus

        node_ids = list(available_cpus.keys())
        num_nodes = len(available_cpus)
        num_cpus_per_node = min([len(cpus) for cpus in available_cpus.values()])

        # Distribute cores across workers
        worker_cpus = {}
        worker_nodes = {}
        if num_nodes >= self._num_workers:
            nodes_per_worker = num_nodes // self._num_workers
            for worker_index in range(self._num_workers):
                node_s = worker_index*nodes_per_worker
                node_e = min((worker_index+1)*nodes_per_worker, num_nodes)
                worker_cpus[worker_index] = sum([available_cpus[node_index]
                                                 for node_index in node_ids[node_s:node_e]], [])
                worker_nodes[worker_index] = node_ids[node_s:node_e]

        else:
            workers_per_node = self._num_workers // num_nodes
            cpus_per_worker = num_cpus_per_node // workers_per_node
            for node_index, node_cpus in available_cpus.items():
                worker_s = node_index*workers_per_node
                worker_e = min((node_index+1)*workers_per_node, self._num_workers)
                worker_chunk = {}
                for worker_index in range(worker_s, worker_e):
                    cpu_s = worker_index*cpus_per_worker
                    cpu_e = min((worker_index+1)*cpus_per_worker, len(node_cpus))
                    worker_chunk[worker_index] = node_cpus[cpu_s:cpu_e]
                    worker_nodes[worker_index] = [node_index]
                worker_cpus.update(worker_chunk)

        # Initialise workers
        for worker_index in range(self._num_workers):
            indices = self.indices + (worker_index,)

            def start_worker(*args, **extra_kwargs):
                kwargs.update(extra_kwargs)
                kwargs['runtime_indices'] = indices
                kwargs['num_workers'] = num_workers
                kwargs['num_threads'] = num_threads

                mosaic.init('worker', *args, **kwargs, wait=True)

            worker_proxy = RuntimeProxy(name='worker', indices=indices)
            worker_subprocess = subprocess(start_worker)(name=worker_proxy.uid,
                                                         daemon=False,
                                                         cpu_affinity=worker_cpus.get(worker_index, None),
                                                         mem_affinity=worker_nodes.get(worker_index, None))
            worker_subprocess.start_process()
            worker_proxy.subprocess = worker_subprocess

            self._workers[worker_proxy.uid] = worker_proxy
            self._own_workers[worker_proxy.uid] = worker_proxy
            await self._comms.wait_for(worker_proxy.uid)

        self.resource_monitor()

        await self.update_monitored_node()

        self._loop.interval(self.resource_monitor, interval=0.1)
        self._loop.interval(self.update_monitored_node, interval=1)

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        self.logger = LoggerManager()

        if self.mode == 'local':
            self.logger.set_local(format=self.mode)
        else:
            runtime_id = 'head' if self.mode == 'interactive' else 'monitor'
            self.logger.set_remote(runtime_id=runtime_id, format=self.mode)

    def set_profiler(self):
        """
        Set up profiling.

        Returns
        -------

        """
        global_profiler.set_remote('monitor')
        super().set_profiler()

    def resource_monitor(self):
        """
        Monitor reseources available for workers, and worker state.

        Returns
        -------

        """
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
        except (ImportError, ValueError):
            gpus = []

        cpu_load = 0.
        memory_fraction = 0.

        self._monitored_node.add_group('gpus')
        self._monitored_node.add_group('workers')

        for gpu in gpus:
            gpu_id = str(gpu.id)
            resource = self._monitored_node.add_resource('gpus', gpu_id)

            resource.update(dict(gpu_load=gpu.load,
                                 memory_limit=gpu.memoryTotal*1024**2,
                                 memory_fraction=gpu.memoryUtil))

        for worker_id, worker in self._own_workers.items():
            resource = self._monitored_node.add_resource('workers', worker_id)

            worker_cpu_load = worker.subprocess.cpu_load()
            worker_memory_fraction = worker.subprocess.memory() / self._memory_limit

            cpu_load += worker_cpu_load
            memory_fraction += worker_memory_fraction

            resource.update(dict(state=worker.subprocess.state,
                                 cpu_load=worker_cpu_load,
                                 memory_fraction=worker_memory_fraction))

        self._monitored_node.sort_resources('workers', 'memory_fraction', desc=True)
        sub_resources = self._monitored_node.sub_resources['workers']

        if memory_fraction > 0.95:
            self._monitored_node.sort_resources('workers', 'memory_fraction', desc=True)

            for worker_id, worker in sub_resources.items():
                if self._own_workers[worker_id].subprocess.paused():
                    continue

                self._own_workers[worker_id].subprocess.pause_process()
                sub_resources[worker_id].state = self._own_workers[worker_id].subprocess.state
                break

        else:
            self._monitored_node.sort_resources('workers', 'memory_fraction', desc=False)

            for worker_id, worker in sub_resources.items():
                if self._own_workers[worker_id].subprocess.running():
                    continue

                self._own_workers[worker_id].subprocess.start_process()
                sub_resources[worker_id].state = self._own_workers[worker_id].subprocess.state
                break

        # TODO Dynamic constraints and shared resources

        self._monitored_node.update(dict(num_cpus=psutil.cpu_count(),
                                         num_gpus=len(gpus),
                                         num_workers=self._num_workers,
                                         num_threads=self._num_threads,
                                         cpu_load=cpu_load,
                                         memory_limit=self._memory_limit,
                                         memory_fraction=memory_fraction))

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

        for worker_id, worker in self._own_workers.items():
            await worker.stop()
            worker.subprocess.join_process()

        await super().stop(sender_id)

    async def update_monitored_node(self):
        """
        Send status update to monitor.

        Returns
        -------

        """
        history, sub_resources = self._monitored_node.get_update()

        await self._comms.send_async('monitor',
                                     method='update_node',
                                     update=history,
                                     sub_resources=sub_resources)
