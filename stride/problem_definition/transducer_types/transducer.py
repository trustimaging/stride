
from abc import ABC

from ..base import GriddedSaved


__all__ = ['Transducer']


class Transducer(ABC, GriddedSaved):
    """
    The transducer holds information about its location in space, its type, as well as
    other things such as IRs.

    Parameters
    ----------
    id : int
        Identifier assigned to this transducer.
    """

    type = ''

    def __init__(self, id, name=None, *args, **kwargs):
        name = name or 'transducer_%05d' % id
        super().__init__(name, *args, **kwargs)

        if id < 0:
            raise ValueError('The transducer needs a positive ID')

        self.id = id
        self.transmit_ir = None
        self.receive_ir = None

        self._coordinates = None

    @property
    def coordinates(self):
        return self._coordinates

    def sub_problem(self, shot, sub_problem):
        return self

    def __get_desc__(self):
        description = {
            'id': self.id,
            'type': self.type,
        }

        return description

    def __set_desc__(self, description):
        self.id = description.id
