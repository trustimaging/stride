
import copy
import asyncio

from .runtime import Runtime
from .utils import WarehouseObject
from ..utils import LoggerManager
from ..profile import global_profiler


__all__ = ['Warehouse']


class Warehouse(Runtime):
    """
    A warehouse represents a key-value storage that is located in a specific runtime,
    and is accessible from all other runtimes.

    Values written into the warehouse cannot be re-assigned, are not writable.

    When a runtime pulls a value from the warehouse, it stores a local cache copy for later use.

    When the amount of memory (usually 25% of total memory) used by the warehouse is exceeded, its contents
    will spill to disk, where they can be retrieved later on.

    """

    is_warehouse = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        self.logger = LoggerManager()
        self.logger.set_local(format=self.mode)

    def set_profiler(self):
        """
        Set up profiling.

        Returns
        -------

        """
        global_profiler.set_remote('monitor')
        super().set_profiler()

    async def put(self, obj, uid=None, publish=False):
        raise NotImplementedError('Cannot put directly into the warehouse runtime')

    async def get(self, uid):
        raise NotImplementedError('Cannot get directly from the warehouse runtime')

    async def drop(self, uid):
        raise NotImplementedError('Cannot drop directly from the warehouse runtime')

    async def force_put(self, sender_id, obj, uid=None):
        raise NotImplementedError('Cannot force put directly into the warehouse runtime')

    async def put_remote(self, sender_id, obj, uid=None, publish=False):
        """
        Put an object into the warehouse.

        Parameters
        ----------
        sender_id
        obj
        uid
        publish

        Returns
        -------

        """
        if uid in self._local_warehouse:
            raise RuntimeError('Warehouse values are not writable')

        self._local_warehouse[uid] = obj

        if publish:
            await self.publish(sender_id, uid)

    async def get_remote(self, sender_id, uid):
        """
        Retrieve an object from the warehouse.

        Parameters
        ----------
        sender_id
        uid

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            uid = uid.uid

        if uid in self._local_warehouse:
            return self._local_warehouse[uid]
        else:
            raise KeyError('%s is not available in the warehouse' % uid)

    async def drop_remote(self, sender_id, uid):
        """
        Delete an object from the warehouse.

        Parameters
        ----------
        sender_id
        uid

        Returns
        -------

        """
        if uid in self._local_warehouse:
            del self._local_warehouse[uid]

    async def push_remote(self, sender_id, __dict__, uid=None, publish=False):
        """
        Push changes into warehouse object.

        Parameters
        ----------
        sender_id
        __dict__
        uid
        publish

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            uid = uid.uid

        if uid not in self._local_warehouse:
            raise KeyError('%s is not available in the warehouse' % uid)

        obj = self._local_warehouse[uid]
        for key, value in __dict__.items():
            setattr(obj, key, value)

        if publish:
            tasks = []

            for worker in self._workers.values():
                tasks.append(worker.force_push(__dict__=__dict__, uid=uid, reply=True))

            await asyncio.gather(*tasks)

    async def pull_remote(self, sender_id, uid):
        """
        Pull changes from warehouse object.

        Parameters
        ----------
        sender_id
        uid

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            uid = uid.uid

        if uid not in self._local_warehouse:
            raise KeyError('%s is not available in the warehouse' % uid)

        __dict__ = copy.copy(self._local_warehouse[uid].__dict__)
        try:
            del __dict__['_tessera']
        except KeyError:
            pass

        return __dict__

    async def publish(self, sender_id, uid):
        """
        Publish object to all workers.

        Parameters
        ----------
        sender_id
        uid

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            uid = uid.uid

        if uid not in self._local_warehouse:
            raise KeyError('%s is not available in the warehouse' % uid)

        obj = self._local_warehouse[uid]
        tasks = []

        for worker in self._workers.values():
            tasks.append(worker.force_put(obj=obj, uid=uid, reply=True))

        await asyncio.gather(*tasks)

    async def init_parameter(self, sender_id, cls, uid, args, **kwargs):
        """
        Create parameter in this worker.

        Parameters
        ----------
        sender_id : str
            Caller UID.
        cls : type
            Class of the tessera.
        uid : str
            UID of the new tessera.
        args : tuple, optional
            Arguments for the initialisation of the tessera.
        kwargs : optional
            Keyword arguments for the initialisation of the tessera.

        Returns
        -------

        """
        param = cls.local_parameter(*args, uid=uid, **kwargs)
        tessera = param._tessera

        self._local_warehouse[uid] = param
        tessera.register_proxy(sender_id)

        return tessera._cls_attr_names
