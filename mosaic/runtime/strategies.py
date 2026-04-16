

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

    def remove_worker(self, uid):
        """
        Remove a worker from the strategy's pool.

        Parameters
        ----------
        uid : str

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
        before = set(self._worker_list)
        for worker_id in updated.sub_resources['workers'].keys():
            self._worker_list.add(worker_id)

        self._num_workers = len(self._worker_list)

        new_workers = self._worker_list - before
        if new_workers:
            self._monitor.logger.info(
                'STRATEGY-POOL: added workers %s from node %s '
                '(pool now %d: %s)'
                % (sorted(new_workers), updated.uid,
                   self._num_workers, sorted(self._worker_list)))

    def select_worker(self, sender_id):
        self._last_worker = (self._last_worker + 1) % self._num_workers

        return list(self._worker_list)[self._last_worker]

    def remove_worker(self, uid):
        if uid in self._worker_list:
            self._worker_list.discard(uid)
            self._num_workers = len(self._worker_list)
            if self._num_workers > 0:
                self._last_worker = self._last_worker % self._num_workers
            else:
                self._last_worker = -1
