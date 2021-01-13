
import psutil
import asyncio

import mosaic
from .runtime import Runtime, RuntimeProxy
from .node import MonitoredNode
from .strategies import RoundRobin
from ..core.tessera import MonitoredTessera
from ..core.task import MonitoredTask
from ..utils import LoggerManager
from ..utils import subprocess


__all__ = ['Monitor', 'monitor_strategies']


monitor_strategies = {
    'round-robin': RoundRobin
}


class Monitor(Runtime):

    is_monitor = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.strategy_name = kwargs.get('monitor_strategy', 'round-robin')
        self._monitor_strategy = monitor_strategies[self.strategy_name](self)

        self._monitored_nodes = dict()
        self._monitored_tessera = dict()
        self._monitored_tasks = dict()

    async def init(self, **kwargs):
        await super().init(**kwargs)

        if self.mode == 'local':
            available_cpus = list(range(psutil.cpu_count()))
            psutil.Process().cpu_affinity([available_cpus[0]])

        # Start local cluster
        if self.mode == 'local':
            await self.init_local(**kwargs)

    async def init_local(self, **kwargs):

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

    def set_logger(self):
        self.logger = LoggerManager()
        self.logger.set_local()

    def update_monitored_node(self, sender_id, monitored_node):
        if sender_id not in self._monitored_nodes:
            self._monitored_nodes[sender_id] = MonitoredNode(sender_id)

        self._monitored_nodes[sender_id].update_history(**monitored_node)

        self._monitor_strategy.update_node(self._monitored_nodes[sender_id])

    def init_tessera(self, sender_id, uid, runtime_id):
        monitored = MonitoredTessera(uid, runtime_id)
        self._monitored_tessera[uid] = monitored

        self._monitor_strategy.update_tessera(self._monitored_tessera[uid])

    def init_task(self, sender_id, uid, tessera_id, runtime_id):
        monitored = MonitoredTask(uid, tessera_id, runtime_id)
        self._monitored_tasks[uid] = monitored

        self._monitor_strategy.update_task(self._monitored_tasks[uid])

    def tessera_state_changed(self, sender_id, uid, state):
        if uid not in self._monitored_tasks:
            return

        self._monitored_tasks[uid].update_history(state=state)

        self._monitor_strategy.update_tessera(self._monitored_tessera[uid])

    def task_state_changed(self, sender_id, uid, state, elapsed=None):
        if uid not in self._monitored_tasks:
            return

        self._monitored_tasks[uid].update_history(state=state)

        if elapsed is not None:
            self._monitored_tasks[uid].elapsed = elapsed

        self._monitor_strategy.update_task(self._monitored_tasks[uid])

    async def stop(self, sender_id=None):
        for node_id, node in self._nodes.items():
            await node.stop()

            if hasattr(node.subprocess, 'stop_process'):
                node.subprocess.join_process()

        super().stop(sender_id)

    async def select_worker(self, sender_id):
        while not len(self._monitored_nodes.keys()):
            await asyncio.sleep(0.1)

        return self._monitor_strategy.select_worker(sender_id)
