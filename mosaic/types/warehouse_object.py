
import uuid

import mosaic
from ..utils import sizeof


__all__ = ['WarehouseObject']


class WarehouseObject:
    """
    Represents a reference to an object that is stored into the warehouse.

    Parameters
    ----------
    obj : object
        Object being added.

    """

    def __init__(self, obj=None, uid=None, size=None):
        if uid is None:
            self._name = obj.__class__.__name__.lower()
            self._uid = '%s-%s-%s' % ('ware',
                                      self._name,
                                      uuid.uuid4().hex)
        else:
            self._name = uid
            self._uid = uid

        runtime = self.runtime
        if 'worker' in runtime.uid:
            node_id = 'node:%d' % runtime.indices[0]
        else:
            node_id = runtime.uid

        self._node_id = node_id
        try:
            self._warehouse_id = runtime._local_warehouse.uid
        except AttributeError:
            self._warehouse_id = runtime.uid
        self._tessera = None
        self._size = sizeof(obj) if size is None else size

    @property
    def state(self):
        return 'done'

    @property
    def uid(self):
        """
        UID of the object.

        """
        return self._uid

    @property
    def runtime(self):
        """
        Current runtime object.

        """
        return mosaic.runtime()

    @property
    def warehouse_id(self):
        """
        UID of the owner warehouse.

        """
        return self._warehouse_id

    @property
    def node_id(self):
        """
        UID of the warehouse node.

        """
        return self._node_id

    @property
    def has_tessera(self):
        return hasattr(self, '_tessera') \
               and self._tessera is not None

    @property
    def is_tessera(self):
        return False

    @property
    def is_proxy(self):
        return True

    async def value(self):
        """
        Pull the underlying value from the warehouse.

        """
        return await self.runtime.get(self, cache=False)

    async def result(self):
        """
        Alias for WarehouseObject.value.

        """
        return await self.value()

    async def drop(self, **kwargs):
        """
        Delete object from the warehouse.

        """
        return await self.runtime.drop(self, **kwargs)

    async def size(self, pending=False):
        """
        Size of the object in bytes.

        """
        return self._size

    def __await__(self):
        return self.value().__await__()

    def __repr__(self):
        return "<%s object uid=%s>" % (self.warehouse_id, self.uid)
