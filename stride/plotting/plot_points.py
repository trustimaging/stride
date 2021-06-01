
import os
import numpy as np

try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    import matplotlib.pyplot as plt

    ENABLED_2D_PLOTTING = True
except ModuleNotFoundError:
    ENABLED_2D_PLOTTING = False

try:
    if not os.environ.get('DISPLAY', None):
        raise ModuleNotFoundError

    from mayavi import mlab
    from mayavi.core.ui.api import MlabSceneModel

    ENABLED_3D_PLOTTING = True

except ModuleNotFoundError:
    ENABLED_3D_PLOTTING = False


__all__ = ['plot_points', 'plot_points_2d', 'plot_points_3d']


def plot_points_2d(coordinates, axis=None, colour='red', size=15, title=None):
    """
    Utility function to plot 2D scattered points using matplotlib.

    Parameters
    ----------
    coordinates : 2-dimensional array
        Coordinates of the points to be plotted, shape should be (n_points, 2).
    axis : matplotlib axis, optional
        Axis in which to make the plotting, defaults to new empty one.
    colour : str
        Colour to apply to the points, defaults to red.
    size : float
        Size of the plotted points, defaults to 15.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    matplotlib axis
        Generated matplotlib axis

    """
    if not ENABLED_2D_PLOTTING:
        return None

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    if len(coordinates.shape) == 1:
        coordinates = coordinates.reshape((1, coordinates.shape[0]))

    space_scale = 1e-3

    im = axis.scatter(coordinates[:, 0]/space_scale, coordinates[:, 1]/space_scale,
                      s=size, c=colour)

    if title is not None:
        axis.set_title(title)

    return axis


def plot_points_3d(coordinates, axis=None, colour='red', size=15, title=None):
    """
    Utility function to plot 3D scattered points using MayaVi.

    Parameters
    ----------
    coordinates : 2-dimensional array
        Coordinates of the points to be plotted, shape should be (n_points, 3).
    axis : MayaVi axis, optional
        Axis in which to make the plotting, defaults to new empty one.
    colour : str
        Colour to apply to the points, defaults to red.
    size : float
        Size of the plotted points, defaults to 15.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    MayaVi figure
        Generated MayaVi figure

    """
    if not ENABLED_3D_PLOTTING:
        return None

    if axis is None:
        axis = MlabSceneModel()

    colour_map = {
        'red': (1., 0., 0.),
        'green': (0., 1., 0.),
        'blue': (0., 0., 1.),
    }

    scale_factor = 100 * size / np.max(coordinates)

    transducers = mlab.pipeline.scalar_scatter(coordinates[:, 0],
                                               coordinates[:, 1],
                                               coordinates[:, 2],
                                               figure=axis.scene3d.mayavi_scene)
    mlab.pipeline.glyph(transducers,
                        mode='sphere', color=colour_map[colour], scale_factor=scale_factor,
                        figure=axis.scene3d.mayavi_scene)

    return axis


def plot_points(coordinates, axis=None, colour='red', size=15, title=None):
    """
    Utility function to plot scattered points using matplotlib (2D) or MayaVi (3D).

    Parameters
    ----------
    coordinates : 2-dimensional array
        Coordinates of the points to be plotted, shape should be (n_points, dimensions).
    axis : axis, optional
        Axis in which to make the plotting, defaults to new empty one.
    colour : str
        Colour to apply to the points, defaults to red.
    size : float
        Size of the plotted points, defaults to 15.
    title : str, optional
        Figure title, defaults to empty title.

    Returns
    -------
    matplotlib or MayaVi axis
        Generated axis

    """
    if coordinates.shape[-1] > 2:
        axis = plot_points_3d(coordinates,
                              axis=axis, colour=colour, size=size, title=title)

    else:
        axis = plot_points_2d(coordinates,
                              axis=axis, colour=colour, size=size, title=title)

    return axis
