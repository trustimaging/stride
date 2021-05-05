
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


def ellipsoidal(num, radius, centre, theta=0., axis=None, threshold=0., angle_range=np.pi):
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
    angle_range

    Returns
    -------
    3d-array
        Array containing the coordinates of points in the geometry, with shape (num, 3).

    """

    increment = np.pi * (3. - np.sqrt(5.))
    start_angle = np.pi / 2 - angle_range / 2
    start_offset = np.sin(start_angle)
    threshold = threshold or start_offset

    num = int(np.round(num / (1 - threshold)))
    offset = 2. / num

    axis = axis or [1, 0, 0]

    geometry = np.zeros((num, 3))

    index = 0
    for sample in range(num):
        z = ((sample * offset) - 1) + offset / 2
        r = np.sqrt(1 - z ** 2)

        phi = ((sample + 1) % num) * increment

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


def disk(num, radius, centre, orientation, boundary_points=2):
    """
    Generate a 3D disk for a number of points ``num``, centred
    on ``centre``, with orientation vector ``orientation``, and with radius ``radius``.

    Parameters
    ----------
    num : int
        Number of points on the geometry.
    radius : array-like
        List or array with each of the two radii of the ellipsis.
    centre : array-like
        List or array with the coordinates of the centre of the ellipsis.
    boundary_points
    orientation

    Returns
    -------
    3d-array
        Array containing the coordinates of points in the geometry, with shape (num, 3).

    """

    golden_angle = (np.sqrt(5) + 1) / 2
    boundary_points = boundary_points or 2.
    boundary_points = round(boundary_points * np.sqrt(num))
    theta = 2 * np.pi * np.arange(num, dtype=np.float32) / golden_angle**2

    # Calculate discretisation in polar coordinates
    r = np.zeros_like(theta)

    for sample in range(1, num):
        r[sample] = 1 if sample > num - boundary_points \
            else np.sqrt(sample - 0.5) / np.sqrt(num - (boundary_points/0.99 + 1) / 2)

    # Change to Cartesian coordinates
    r *= radius
    x = np.zeros((num,))
    y = np.cos(theta) * r
    z = np.sin(theta) * r

    geometry = np.stack((x, y, z))

    # Calculate orthonormal basis of orientation
    x_prime = np.array(orientation)
    x_prime = x_prime / np.linalg.norm(x_prime)

    if x_prime[0] == 0.:
        y_prime = np.array([0., x_prime[2], -x_prime[1]])

    elif x_prime[1] == 0.:
        y_prime = np.array([x_prime[2], 0., -x_prime[0]])

    elif x_prime[2] == 0.:
        y_prime = np.array([x_prime[1], -x_prime[0], 0.])

    else:
        y_prime = np.array([x_prime[2], 0., -x_prime[0]])

    z_prime = np.cross(y_prime, x_prime)
    y_prime = y_prime / np.linalg.norm(y_prime)
    z_prime = z_prime / np.linalg.norm(z_prime)

    # Construct transformation matrix
    T = np.vstack((x_prime, y_prime, z_prime))

    # Change back to normal basis
    geometry = np.dot(np.transpose(T), geometry).T

    # Displace shot to its centre
    geometry[:, 0] += centre[0]
    geometry[:, 1] += centre[1]
    geometry[:, 2] += centre[2]

    return geometry
