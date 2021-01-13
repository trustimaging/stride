

class MonitorStrategy:

    def __init__(self, monitor):
        self._monitor = monitor

    def update_node(self, updated):
        pass

    def update_tessera(self, updated):
        pass

    def update_task(self, updated):
        pass

    def select_worker(self, sender_id):
        pass


class RoundRobin(MonitorStrategy):

    def __init__(self, monitor):
        super().__init__(monitor)

        self._worker_list = set()
        self._num_workers = -1
        self._last_worker = -1

    def update_node(self, updated):
        for worker_id in updated.worker_info.keys():
            self._worker_list.add(worker_id)

        self._num_workers = len(self._worker_list)

    def select_worker(self, sender_id):
        self._last_worker = (self._last_worker + 1) % self._num_workers

        return list(self._worker_list)[self._last_worker]
