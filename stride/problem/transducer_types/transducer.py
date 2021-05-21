
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
    name : str
        Optional name for the shot.
    grid : Grid or any of Space or Time
        Grid on which the Transducer is defined

    """

    type = ''
    """Type of transducer, e.g. point_transducer"""

    def __init__(self, id, name=None, *args, **kwargs):
        name = name or 'transducer_%05d' % id
        super().__init__(*args, name=name, **kwargs)

        if id < 0:
            raise ValueError('The transducer needs a positive ID')

        self.id = id
        self.transmit_ir = None
        self.receive_ir = None

        self._coordinates = None

    @property
    def coordinates(self):
        """
        Coordinates of points in the transducer, relative to its centre.

        Returns
        -------
        ndarray
            Coordinate array.

        """
        return self._coordinates

    def sub_problem(self, shot, sub_problem):
        """
        Create a subset object for a certain shot.

        A SubProblem contains everything that is needed to fully determine how to run a particular shot.
        This method has no effect for this particular case.

        Parameters
        ----------
        shot : Shot
            Shot for which the SubProblem is being generated.
        sub_problem : SubProblem
            Container for the sub-problem being generated.

        Returns
        -------
        Transducer
            Transducer instance.

        """
        return self

    def __get_desc__(self):
        description = {
            'id': self.id,
            'type': self.type,
        }

        return description

    def __set_desc__(self, description):
        self.id = description.id
