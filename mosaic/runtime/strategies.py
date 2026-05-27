

class MonitorStrategy:
    """
    Base class for the strategies used to allocate tesserae to
    workers.

    """

    def __init__(self, monitor):
        self._monitor = monitor

    def update_node(self, updated):
        """
        Update inner record of node state.

        Parameters
        ----------
        updated : MonitoredNode

        Returns
        -------

        """
        pass

    def update_tessera(self, updated):
        """
        Update inner record of tesserae state.

        Parameters
        ----------
        updated : MonitoredTessera

        Returns
        -------

        """
        pass

    def update_task(self, updated):
        """
        Update inner record of task state.

        Parameters
        ----------
        updated : MonitoredTask

        Returns
        -------

        """
        pass

    def select_worker(self, sender_id):
        """
        Select an appropriate worker.

        Parameters
        ----------
        sender_id : str

        Returns
        -------

        """
        pass


class RoundRobin(MonitorStrategy):
    """
    Round robin strategy for allocating tesserae.

    """

    def __init__(self, monitor):
        super().__init__(monitor)

        self._worker_list = set()
        self._num_workers = -1
        self._last_worker = -1

    def update_node(self, updated):

        # Stale worker eviction (replacement may join before heartbeat timeout)
        node_idx = int(updated.uid.split(':')[1])
        incoming = set(updated.sub_resources.get('workers', {}).keys())
        prefix = f'worker:{node_idx}'
        stale = {w for w in self._worker_list if w.startswith(prefix) and w not in incoming}

        if stale:
            self._worker_list -= stale
            self._num_workers = len(self._worker_list)
            self._monitor.logger.debug(f'Evicted stale workers: {stale}')

        before = set(self._worker_list)
        for worker_id in incoming:
            self._worker_list.add(worker_id)

        self._num_workers = len(self._worker_list)

        new_workers = self._worker_list - before
        if new_workers:
            self._monitor.logger.debug(
                f'New workers added: {new_workers} '
                f"(pool now: {sorted(self._worker_list)}"
            )

    def select_worker(self, sender_id):
        self._last_worker = (self._last_worker + 1) % self._num_workers

        return list(self._worker_list)[self._last_worker]
