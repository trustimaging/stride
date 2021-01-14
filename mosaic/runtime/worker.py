
import os
import psutil

from .runtime import Runtime
from ..core import Task
from ..utils import LoggerManager


__all__ = ['Worker']


class Worker(Runtime):

    is_worker = True

    def __init__(self, num_workers=None, num_threads=None, **kwargs):
        super().__init__(**kwargs)

        if num_threads is None:
            num_threads = psutil.cpu_count()
        self._num_threads = num_threads

        os.environ['OMP_NUM_THREADS'] = str(self._num_threads)

    def set_logger(self):
        self.logger = LoggerManager()

        if self.mode == 'local':
            self.logger.set_local()
        else:
            self.logger.set_remote()

    async def stop(self, sender_id=None):
        pass

    # Command and task management methods

    async def init_tessera(self, sender_id, cls, uid, args, **kwargs):
        tessera = cls.tessera(uid, *args, **kwargs)
        tessera.register_proxy(sender_id)

    async def init_task(self, sender_id, task, uid):
        obj_uid = task['tessera_id']
        obj_store = self._tessera
        tessera = obj_store[obj_uid]

        task = Task(uid, sender_id, tessera,
                    task['method'], *task['args'], **task['kwargs'])

        tessera.queue_task((sender_id, task))
        await task.state_changed('pending')

    async def tessera_state_changed(self, tessera):
        monitor = self.get_monitor()
        await monitor.tessera_state_changed(uid=tessera.uid,
                                            state=tessera.state)

    async def task_state_changed(self, task, elapsed=None):
        monitor = self.get_monitor()
        await monitor.task_state_changed(uid=task.uid,
                                         state=task.state,
                                         elapsed=elapsed)

    def inc_ref(self, sender_id, uid, type):
        self.logger.debug('Increased ref count for object %s' % uid)

        obj_type = type
        obj_uid = uid
        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            raise KeyError('Runtime %s does not own object %s of type %s' % (self.uid, obj_uid, obj_type))

        obj = obj_store[obj_uid]
        obj.inc_ref()
        obj.register_proxy(uid=sender_id)

    def dec_ref(self, sender_id, uid, type):
        self.logger.debug('Decreased ref count for object %s' % uid)

        obj_type = type
        obj_uid = uid
        obj_store = getattr(self, '_' + obj_type)

        if obj_uid not in obj_store.keys():
            raise KeyError('Runtime %s does not own object %s of type %s' % (self.uid, obj_uid, obj_type))

        obj = obj_store[obj_uid]
        obj.dec_ref()
        obj.unregister_proxy(uid=sender_id)