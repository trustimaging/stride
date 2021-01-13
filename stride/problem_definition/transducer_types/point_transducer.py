

import numpy as np

from .transducer import Transducer


__all__ = ['PointTransducer']


class PointTransducer(Transducer):

    type = 'point_transducer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = np.zeros((self.space.dim,))

        return self._coordinates
