

import numpy as np

from .transducer import Transducer


__all__ = ['PointTransducer']


class PointTransducer(Transducer):
    """
    This class describes a point transducers, in which a single point represents the
    effect of the device.

    """

    type = 'point_transducer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def coordinates(self):
        """
        Coordinates of points in the transducer, relative to its centre.

        Returns
        -------
        ndarray
            Coordinate array.

        """
        if self._coordinates is None:
            self._coordinates = np.zeros((self.space.dim,))

        return self._coordinates
