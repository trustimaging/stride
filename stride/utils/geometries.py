
import numpy as np


__all__ = ['elliptical']


def elliptical(num, radius, centre):
    """
    Generate a 2D elliptical geometry for a number of points ``num``, centred
    on ``centre`` and with radius ``radius``.

    Parameters
    ----------
    num : int
        Number of points on the geometry.
    radius : array-like
        List or array with each of the two radii of the ellipsis.
    centre : array-like
        List or array with the coordinates of the centre of the ellipsis.

    Returns
    -------
    2d-array
        Array containing the coordinates of points in the geometry, with shape (num, 2).

    """
    angles = np.linspace(0, 2*np.pi, num, endpoint=False)

    geometry = np.zeros((num, 2))
    for index, angle in zip(range(num), angles):
        geometry[index, 0] = radius[0] * np.cos(angle) + centre[0]
        geometry[index, 1] = radius[1] * np.sin(angle) + centre[1]

    return geometry
