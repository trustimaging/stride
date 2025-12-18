
import os
import copy
import shutil
import asyncio

from .runtime import Runtime
from ..types import WarehouseObject
from ..utils import LoggerManager, SpillBuffer, at_exit
from ..utils.utils import memory_limit
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

        self._warehouses = dict()
        self._spill_directory = None

    async def init(self, **kwargs):
        await super().init(**kwargs)

        # Set up local warehouse
        self._spill_directory = os.path.join(os.getcwd(), 'mosaic-workspace', '%s-storage' % self.uid)
        self._spill_directory.replace(':', '-')
        if not os.path.exists(self._spill_directory):
            os.makedirs(self._spill_directory)

        def _rm_dirs():
            shutil.rmtree(self._spill_directory, ignore_errors=True)

        at_exit.add(_rm_dirs)

        warehouse_memory_fraction = kwargs.pop('warehouse_memory_fraction', 0.80)
        warehouse_memory = memory_limit() * warehouse_memory_fraction

        self._local_warehouse = SpillBuffer(self._spill_directory, warehouse_memory)

    def set_logger(self):
        """
        Set up logging.

        Returns
        -------

        """
        self.logger = LoggerManager()

        if not len(self.indices) or self.mode == 'local':
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

    async def put(self, obj, uid=None, publish=False, **kwargs):
        return await self.put_remote(self.uid, obj, uid=uid)

    async def get(self, uid, **kwargs):
        return await self.get_remote(self.uid, uid=uid)

    async def drop(self, uid, **kwargs):
        return await self.drop_remote(self.uid, uid=uid)

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
        if isinstance(uid, WarehouseObject):
            uid = uid.uid

        if uid in self._local_warehouse:
            raise RuntimeError('Warehouse values are not writable')

        self._local_warehouse[uid] = obj

        if publish:
            await self.publish(sender_id, uid)

    async def get_remote(self, sender_id, uid, warehouse_id=None, node_id=None):
        """
        Retrieve an object from the warehouse.

        Parameters
        ----------
        sender_id
        uid
        warehouse_id
        node_id

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            node_id = uid.node_id
            warehouse_id = uid.warehouse_id
            obj_id = uid.uid
        else:
            obj_id = uid

        if obj_id in self._local_warehouse:
            return self._local_warehouse[obj_id]

        if warehouse_id == self.uid or (node_id is None and warehouse_id is None):
            retries = 0
            wait = 1
            while obj_id not in self._local_warehouse:
                await asyncio.sleep(wait)
                wait *= 1.2
                retries += 1
                if retries > 20:
                    break

            if obj_id in self._local_warehouse:
                return self._local_warehouse[obj_id]

            raise KeyError('%s is not available in %s' % (obj_id, self.uid))

        if node_id not in self._warehouses:
            self._warehouses[node_id] = self.proxy(uid=warehouse_id)

        obj = await self._warehouses[node_id].get_remote(uid=uid, reply=True)

        if hasattr(obj, 'cached') and obj.cached:
            self._local_warehouse[obj_id] = obj

        return obj

    async def exec_remote(self, sender_id, uid, func, func_args=None, func_kwargs=None):
        """
        Retrieve an object from the warehouse and execute function on it.

        Parameters
        ----------
        sender_id
        uid
        func
        func_args
        func_kwargs

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            obj_id = uid.uid
        else:
            obj_id = uid

        try:
            obj = self._local_warehouse[obj_id]
        except KeyError:
            obj = None

        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        obj = await func(obj, *func_args, **func_kwargs)

        self._local_warehouse[obj_id] = obj

        warehouse_obj = WarehouseObject(obj, uid=obj_id)
        return warehouse_obj

    async def drop_remote(self, sender_id, uid, propagate=False):
        """
        Delete an object from the warehouse.

        Parameters
        ----------
        sender_id
        uid
        propagate

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            obj_id = uid.uid

            if propagate:
                node_id = uid.node_id
                warehouse_id = uid.warehouse_id

                if node_id is not None and warehouse_id is not None:
                    if node_id not in self._warehouses:
                        self._warehouses[node_id] = self.proxy(uid=warehouse_id)

                    await self._warehouses[node_id].drop_remote(uid=uid)

        if obj_id in self._local_warehouse:
            del self._local_warehouse[obj_id]

    async def push_remote(self, sender_id, __dict__,
                          uid=None, warehouse_id=None, node_id=None,
                          publish=False):
        """
        Push changes into warehouse object.

        Parameters
        ----------
        sender_id
        __dict__
        uid
        warehouse_id
        node_id
        publish

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            node_id = uid.node_id
            warehouse_id = uid.warehouse_id
            obj_id = uid.uid
        else:
            obj_id = uid

        if obj_id in self._local_warehouse:
            obj = self._local_warehouse[obj_id]
            for key, value in __dict__.items():
                setattr(obj, key, value)

        else:
            if warehouse_id == self.uid or (node_id is None and warehouse_id is None):
                raise KeyError('%s is not available in %s' % (obj_id, self.uid))

            await self.get_remote(sender_id, uid=uid)

        if publish:
            tasks = []

            for node in self._nodes.values():
                if node.uid not in self._warehouses:
                    self._warehouses[node.uid] = self.proxy('warehouse',
                                                            indices=node.indices[0])
                tasks.append(self._warehouses[node.uid].push_remote(__dict__=__dict__, uid=uid, reply=True))

            if len(self.indices):
                if 'head' not in self._warehouses:
                    self._warehouses['head'] = self.proxy('warehouse')
                tasks.append(self._warehouses['head'].push_remote(__dict__=__dict__, uid=uid, reply=True))

            await asyncio.gather(*tasks)

    async def pull_remote(self, sender_id, uid, warehouse_id=None, node_id=None, attr=None):
        """
        Pull changes from warehouse object.

        Parameters
        ----------
        sender_id
        uid
        warehouse_id
        node_id
        attr

        Returns
        -------

        """
        if isinstance(uid, WarehouseObject):
            node_id = uid.node_id
            warehouse_id = uid.warehouse_id
            obj_id = uid.uid
        else:
            obj_id = uid

        if obj_id in self._local_warehouse:
            obj = self._local_warehouse[obj_id]

        else:
            if warehouse_id == self.uid or (node_id is None and warehouse_id is None):
                raise KeyError('%s is not available in %s' % (obj_id, self.uid))

            if node_id not in self._warehouses:
                self._warehouses[node_id] = self.proxy(uid=warehouse_id)

            obj = await self._warehouses[node_id].get_remote(uid=uid, reply=True)

            if not hasattr(obj, 'cached') or obj.cached:
                self._local_warehouse[obj_id] = obj

        if attr is None:
            __dict__ = copy.copy(obj.__dict__)
        else:
            if not isinstance(attr, list):
                attr = [attr]

            __dict__ = dict()
            for key in attr:
                __dict__[key] = getattr(obj, key)

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
            obj_id = uid.uid
        else:
            obj_id = uid

        if obj_id not in self._local_warehouse:
            raise KeyError('%s is not available in %s' % (obj_id, self.uid))

        obj = self._local_warehouse[obj_id]
        tasks = []

        for node in self._nodes.values():
            if node.uid not in self._warehouses:
                self._warehouses[node.uid] = self.proxy('warehouse',
                                                        indices=node.indices[0])
            tasks.append(self._warehouses[node.uid].put_remote(obj=obj, uid=obj_id, reply=True))

        if len(self.indices):
            if 'head' not in self._warehouses:
                self._warehouses['head'] = self.proxy('warehouse')
            tasks.append(self._warehouses['head'].put_remote(obj=obj, uid=obj_id, reply=True))

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

    async def stop(self, sender_id=None):
        await super().stop(sender_id=sender_id)
        shutil.rmtree(self._spill_directory, ignore_errors=True)
