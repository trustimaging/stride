
import numpy as np
import scipy.linalg


__all__ = ['elliptical', 'ellipsoidal']


def _rot_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis / np.linalg.norm(axis) * theta))


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


def ellipsoidal(num, radius, centre, theta=0., axis=None, threshold=0.):
    """
    Generate a 3D ellipsoidal geometry for a number of points ``num``, centred
    on ``centre`` and with radius ``radius``. The geometry can be rotated by
    an amount ``theta``, and thresholded by eliminating ``threshold`` percent of it.

    Parameters
    ----------
    num : int
        Number of points on the geometry.
    radius : array-like
        List or array with each of the two radii of the ellipsis.
    centre : array-like
        List or array with the coordinates of the centre of the ellipsis.
    theta
    axis
    threshold

    Returns
    -------
    3d-array
        Array containing the coordinates of points in the geometry, with shape (num, 3).

    """
    num = int(np.round(num / (1 - threshold)))

    offset = 2. / num
    increment = np.pi * (3. - np.sqrt(5.))
    axis = axis or [1, 0, 0]

    geometry = np.zeros((num, 3))

    index = 0
    for sample in range(num):
        z = ((sample * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - z ** 2)

        phi = (sample % num) * increment

        y = np.cos(phi) * r
        x = np.sin(phi) * r

        if z + 1 < threshold*2:
            continue

        x *= radius[0]
        y *= radius[1]
        z *= radius[2]

        [x, y, z] = np.dot(_rot_matrix(axis, theta), [x, y, z])

        point = np.array([x, y, z])
        point[0] += centre[0]
        point[1] += centre[1]
        point[2] += centre[2]

        geometry[index, :] = point
        index += 1

    geometry = geometry[:index, :]

    return geometry
