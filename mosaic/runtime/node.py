
import psutil
import datetime
from collections import OrderedDict

import mosaic
from .runtime import Runtime, RuntimeProxy
from ..utils import LoggerManager
from ..utils import subprocess
from ..utils.utils import memory_limit


__all__ = ['Node', 'MonitoredNode', 'MonitoredWorker', 'MonitoredGPU']


class Node(Runtime):

    is_node = True

    def __init__(self, num_workers=None, num_threads=None, **kwargs):
        super().__init__(**kwargs)

        if num_workers is None:
            num_workers = 1

        if num_threads is None:
            num_threads = psutil.cpu_count() // num_workers

        self._num_workers = num_workers
        self._num_threads = num_threads
        self._memory_limit = memory_limit()

        self._monitored_node = MonitoredNode(self.uid)

    async def init(self, **kwargs):
        await super().init(**kwargs)

        # Start local workers
        await self.init_workers(**kwargs)

    async def init_workers(self, **kwargs):
        num_workers = self._num_workers
        num_threads = self._num_threads

        available_cpus = list(range(psutil.cpu_count()))

        if self.mode == 'local':
            psutil.Process().cpu_affinity([available_cpus[2]])
            available_cpus = available_cpus[3:]
        else:
            psutil.Process().cpu_affinity([available_cpus[0]])
            available_cpus = available_cpus[1:]

        cpus_per_worker = len(available_cpus) // self._num_workers

        for worker_index in range(self._num_workers):
            indices = self.indices + (worker_index,)

            if worker_index < num_workers - 1:
                cpu_affinity = available_cpus[worker_index*cpus_per_worker:(worker_index+1)*cpus_per_worker]

            else:
                cpu_affinity = available_cpus[worker_index*cpus_per_worker:]

            def start_worker(*args, **extra_kwargs):
                kwargs.update(extra_kwargs)
                kwargs['runtime_indices'] = indices
                kwargs['num_workers'] = num_workers
                kwargs['num_threads'] = num_threads

                mosaic.init('worker', *args, **kwargs, wait=True)

            worker_proxy = RuntimeProxy(name='worker', indices=indices)
            worker_subprocess = subprocess(start_worker)(name=worker_proxy.uid,
                                                         cpu_affinity=cpu_affinity,
                                                         daemon=self.mode != 'local')
            worker_subprocess.start_process()
            worker_proxy.subprocess = worker_subprocess

            self._workers[worker_proxy.uid] = worker_proxy
            await self._comms.wait_for(worker_proxy.uid)

        self.resource_monitor()

        await self.update_monitored_node()

        self._loop.interval(self.resource_monitor, interval=0.1)
        self._loop.interval(self.update_monitored_node, interval=0.1)

    def set_logger(self):
        self.logger = LoggerManager()

        if self.mode == 'local':
            self.logger.set_local()
        else:
            self.logger.set_remote()

    def resource_monitor(self):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
        except ImportError:
            gpus = []

        cpu_load = 0.
        memory_fraction = 0.

        for gpu in gpus:
            if gpu.id not in self._monitored_node.gpu_info:
                self._monitored_node.gpu_info[gpu.id] = MonitoredGPU(gpu.id)

            self._monitored_node.gpu_info[gpu.id].update(gpu_load=gpu.load,
                                                         memory_limit=gpu.memoryTotal*1024**2,
                                                         memory_fraction=gpu.memoryUtil)

        for worker_id, worker in self._workers.items():
            if worker_id not in self._monitored_node.worker_info:
                self._monitored_node.worker_info[worker_id] = MonitoredWorker(worker_id)

            worker_cpu_load = worker.subprocess.cpu_load()
            worker_memory_fraction = worker.subprocess.memory() / self._memory_limit

            cpu_load += worker_cpu_load
            memory_fraction += worker_memory_fraction

            self._monitored_node.worker_info[worker_id].update(state=worker.subprocess.state,
                                                               cpu_load=worker_cpu_load,
                                                               memory_fraction=worker_memory_fraction)

        if memory_fraction > 0.95:
            self._monitored_node.sort_workers(desc=True)

            for worker_id, worker in self._monitored_node.worker_info.items():
                if self._workers[worker_id].subprocess.paused():
                    continue

                self._workers[worker_id].subprocess.pause_process()
                self._monitored_node.worker_info[worker_id].state = self._workers[worker_id].subprocess.state
                break

        else:
            self._monitored_node.sort_workers(desc=False)

            for worker_id, worker in self._monitored_node.worker_info.items():
                if self._workers[worker_id].subprocess.running():
                    continue

                self._workers[worker_id].subprocess.start_process()
                self._monitored_node.worker_info[worker_id].state = self._workers[worker_id].subprocess.state
                break

        # How do we introduce dynamic constraints on shared resources? By pre-emptively asking for
        # permission to run. If permission is not granted, then they will wait. Otherwise, we will take
        # some resources and lock them for a worker. When the worker finishes with it, we can keep going, that goes back
        # to the general resource pool.
        # When the region exceeds the memory it asked for, it should probably be stopped somehow
        # Info should be sent to the head node

        self._monitored_node.update(num_cpus=psutil.cpu_count(),
                                    num_gpus=len(gpus),
                                    num_workers=self._num_workers,
                                    num_threads=self._num_threads,
                                    cpu_load=cpu_load,
                                    memory_limit=self._memory_limit,
                                    memory_fraction=memory_fraction)

    async def stop(self, sender_id=None):
        for worker_id, worker in self._workers.items():
            await worker.stop()
            worker.subprocess.join_process()

        if self.mode != 'local':
            super().stop(sender_id)

    async def update_monitored_node(self):
        await self._comms.send_async('monitor',
                                     method='update_monitored_node',
                                     monitored_node=self._monitored_node.get_update())


class MonitoredGPU:

    def __init__(self, uid):
        self.uid = self.name = uid
        self.time = -1

        self.gpu_load = -1
        self.memory_limit = -1
        self.memory_fraction = -1

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
            gpu_load=self.gpu_load,
            memory_limit=self.memory_limit,
            memory_fraction=self.memory_fraction,
        )

        return update


class MonitoredWorker:

    def __init__(self, uid):
        self.uid = self.name = uid
        self.state = 'running'
        self.time = -1

        self.cpu_load = -1
        self.memory_fraction = -1

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
            cpu_load=self.cpu_load,
            memory_fraction=self.memory_fraction,
        )

        return update


class MonitoredNode:

    def __init__(self, uid):
        self.uid = self.name = uid
        self.state = 'running'
        self.time = -1

        self.num_cpus = -1
        self.num_gpus = -1
        self.num_workers = -1
        self.num_threads = -1
        self.memory_limit = -1
        self.cpu_load = -1
        self.memory_fraction = -1

        self.gpu_info = OrderedDict()
        self.worker_info = OrderedDict()

        self.history = []

    def update(self, **update):
        if 'gpu_info' in update:
            for gpu_id, gpu in update.pop('gpu_info').items():
                if gpu_id not in self.gpu_info:
                    self.gpu_info[gpu_id] = MonitoredGPU(gpu_id)

                self.gpu_info[gpu_id].update(**gpu)

        if 'worker_info' in update:
            for worker_id, worker in update.pop('worker_info').items():
                if worker_id not in self.gpu_info:
                    self.worker_info[worker_id] = MonitoredWorker(worker_id)

                self.worker_info[worker_id].update(**worker)

        self.time = str(datetime.datetime.now())

        for key, value in update.items():
            setattr(self, key, value)

    def update_history(self, **update):
        if 'gpu_info' in update:
            for gpu_id, gpu in update.pop('gpu_info').items():
                if gpu_id not in self.gpu_info:
                    self.gpu_info[gpu_id] = MonitoredGPU(gpu_id)

                self.gpu_info[gpu_id].update_history(**gpu)

        if 'worker_info' in update:
            for worker_id, worker in update.pop('worker_info').items():
                if worker_id not in self.gpu_info:
                    self.worker_info[worker_id] = MonitoredWorker(worker_id)

                self.worker_info[worker_id].update_history(**worker)

        self.update(**update)

        update['time'] = self.time
        self.history.append(update)

    def get_update(self):
        update = dict(
            state=self.state,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            num_workers=self.num_workers,
            num_threads=self.num_threads,
            memory_limit=self.memory_limit,
            cpu_load=self.cpu_load,
            memory_fraction=self.memory_fraction,
        )

        update['gpu_info'] = dict()
        update['worker_info'] = dict()

        for gpu_id, gpu in self.gpu_info.items():
            update['gpu_info'][gpu_id] = gpu.get_update()

        for worker_id, worker in self.worker_info.items():
            update['worker_info'][worker_id] = worker.get_update()

        return update

    def sort_workers(self, desc=False):
        self.worker_info = OrderedDict(sorted(self.worker_info.items(),
                                              key=lambda x: x[1].memory_fraction,
                                              reverse=desc))

    def sort_gpus(self, desc=False):
        self.gpu_info = OrderedDict(sorted(self.gpu_info.items(),
                                           key=lambda x: x[1].memory_fraction,
                                           reverse=desc))
