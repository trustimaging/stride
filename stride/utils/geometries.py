
import numpy as np


__all__ = ['elliptical']


def elliptical(num, radius, centre):
    angles = np.linspace(0, 2*np.pi, num, endpoint=False)

    geometry = np.zeros((num, 2))
    for index, angle in zip(range(num), angles):
        geometry[index, 0] = radius[0] * np.cos(angle) + centre[0]
        geometry[index, 1] = radius[1] * np.sin(angle) + centre[1]

    return geometry
